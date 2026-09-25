#![allow(non_upper_case_globals)]
#![allow(clippy::excessive_precision)]

//! We use the [OPC model](https://pubs.acs.org/doi/10.1021/jz501780a) for solvent by default.
//! See also, the Amber Reference Manual. Other rigid 3 and 4-site models can be configured using
//! `WaterModel`.
//!
//! This is a rigid model that includes an "EP" or "M" massless charge-only molecule (No LJ terms),
//! and no charge on the Oxygen. We integrate it using standard Amber-style forces.
//! Amber strongly recommends using this model when their ff19SB foces for proteins.
//!
//! Amber RM: "OPC is a non-polarizable, 4-point, 3-charge rigid solvent model. Geometrically, it
//! resembles TIP4P-like mod-
//! els, although the values of OPC point charges and charge-charge distances are quite different.
//! The model has a single VDW center on the oxygen nucleus."
//!
//! Note: The original paper uses the term "M" for the massless charge; Amber calls it "EP".
//!
//! We integrate the molecule's internal rigid geometry using the `SETTLE` algorithm. This is likely
//! to be cheaper, and more robust than Shake/Rattle. It's less general, but it works here.
//! Settle is specifically tailored for three-atom rigid bodies.
//!
//! This module, in particular, contains structs, constants, and the integrator.
//!
//! Note: H bond average maintenance time: 1-20ps: Use this to validate your solvent model

use std::{
    borrow::Cow,
    fmt,
    fmt::{Display, Formatter},
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16};
use lin_alg::{
    f32::{Quaternion as QuaternionF32, Vec3 as Vec3F32, X_VEC, Z_VEC},
    f64::Vec3,
};
use na_seq::Element;

use crate::{AtomDynamics, KCAL_TO_NATIVE, MolDynamics, SimBox, non_bonded::CHARGE_UNIT_SCALER};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use crate::{AtomDynamicsx8, AtomDynamicsx16};

pub(crate) mod init;
pub(crate) mod octanol;
pub(crate) mod opc_settle;
pub(crate) mod shrinking_box;
pub(crate) mod template_creation;

// For converting from R_star to eps. See notes in bio_files's `LjParams`.
const SIGMA_FACTOR: f32 = 2. / 1.122_462_048_309_373;

/// Parameters for a rigid water model: O, two H, and a massless charge site M (also called EP) on
/// the H-O-H bisector. O carries no charge; M carries -2 × the H charge. O has LJ parameters, and
/// H may too. (e.g. CHARMM's TIP3P) A 3-site model such as TIP3P is represented with
/// `o_m_dist = 0`, so M coincides with O and carries its charge.
///
/// Note: Molecules placed from a template keep the template's geometry until `reset_angle` runs,
/// so templates should match the model's geometry.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Clone, Debug, PartialEq)]
pub struct WaterModel {
    /// amu
    pub mass_o: f32,
    /// amu
    pub mass_h: f32,
    /// Å
    pub o_h_dist: f32,
    /// Å. From O to M, along the H-O-H bisector. 0 for 3-site models.
    pub o_m_dist: f32,
    /// Radians
    pub h_o_h_angle: f32,
    /// Elementary charge units.
    pub q_h: f32,
    /// Å
    pub lj_sigma_o: f32,
    /// kcal/mol
    pub lj_eps_o: f32,
    /// Å. Unused if `lj_eps_h` is 0.
    pub lj_sigma_h: f32,
    /// kcal/mol. 0 for most models.
    pub lj_eps_h: f32,
    /// We add these counter-ions to neutralize the system. Ion parameters are generally tuned for
    /// a specific water model.
    pub cation: IonParams,
    pub anion: IonParams,
}

impl WaterModel {
    /// The OPC model (JPCL, 2014, 5 (21), pp 3863-3871); values taken directly from Amber 2025's
    /// `frcmod.opc`. Ions use Joung–Cheatham parameters tuned for OPC (`frcmod.ionsjc_opc`),
    /// with sigma = 2 * R_MIN_HALF / 2^(1/6).
    pub const OPC: Self = Self {
        mass_o: 16.,
        mass_h: 1.008,
        o_h_dist: 0.872_433_13,
        o_m_dist: 0.159_398_33,
        h_o_h_angle: 1.808_161_105_066, // 103.6°
        // See the OPC paper, Table 2.
        q_h: 0.6791,
        lj_sigma_o: 1.777_167_268 * SIGMA_FACTOR,
        lj_eps_o: 0.212_800_813_0,
        lj_sigma_h: 0.,
        lj_eps_h: 0.,
        cation: IonParams {
            ion: Ion::Sodium,
            ff_type: Cow::Borrowed("Na+"),
            mass: 22.99,
            lj_sigma: 2.439,
            lj_eps: 0.1065,
        },
        anion: IonParams {
            ion: Ion::Chloride,
            ff_type: Cow::Borrowed("Cl-"),
            mass: 35.45,
            lj_sigma: 4.478,
            lj_eps: 0.0073,
        },
    };

    /// CHARMM's modified TIP3P, with LJ on H (the TIPS3P form), as in `toppar_water_ions.str`.
    /// Ions are CHARMM's SOD and CLA; their NBFIX entries (e.g. with carboxylate O) come from
    /// the force field. sigma = 2 * R_MIN_HALF / 2^(1/6).
    pub const TIP3P_CHARMM: Self = Self {
        mass_o: 15.9994,
        mass_h: 1.008,
        o_h_dist: 0.9572,
        o_m_dist: 0.,
        h_o_h_angle: 1.824_218_134_6, // 104.52°
        q_h: 0.417,
        lj_sigma_o: 1.7682 * SIGMA_FACTOR,
        lj_eps_o: 0.1521,
        lj_sigma_h: 0.2245 * SIGMA_FACTOR,
        lj_eps_h: 0.046,
        cation: IonParams {
            ion: Ion::Sodium,
            ff_type: Cow::Borrowed("SOD"),
            mass: 22.989_77,
            lj_sigma: 1.410_75 * SIGMA_FACTOR,
            lj_eps: 0.0469,
        },
        anion: IonParams {
            ion: Ion::Chloride,
            ff_type: Cow::Borrowed("CLA"),
            mass: 35.45,
            lj_sigma: 2.27 * SIGMA_FACTOR,
            lj_eps: 0.150,
        },
    };

    /// amu
    pub fn mass(&self) -> f32 {
        self.mass_o + 2.0 * self.mass_h
    }

    /// Distance from O to the midpoint of the H-H line. Å
    pub(crate) fn ra(&self) -> f32 {
        self.o_h_dist * (0.5 * self.h_o_h_angle).cos()
    }

    /// Coefficients (O, each H) we use to project force on M to the real sites. For a bisector
    /// site at distance d_OM, with bond length d_OH and angle theta:
    /// c_H = (d_OM / (d_OH * cos(theta/2))) / 2.0. This conserves force and torque exactly.
    pub(crate) fn m_force_coeffs(&self) -> (f32, f32) {
        let c_h = (self.o_m_dist / self.ra()) / 2.;
        (1.0 - 2.0 * c_h, c_h)
    }

    /// (O, H). Converts force to acceleration, in our internal units.
    pub(crate) fn accel_conversions(&self) -> (f32, f32) {
        (KCAL_TO_NATIVE / self.mass_o, KCAL_TO_NATIVE / self.mass_h)
    }
}

impl Default for WaterModel {
    fn default() -> Self {
        Self::OPC
    }
}

/// A monatomic ion species, for neutralizing the system.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ion {
    Sodium,
    Potassium,
    Chloride,
}

impl Ion {
    pub fn element(self) -> Element {
        match self {
            Self::Sodium => Element::Sodium,
            Self::Potassium => Element::Potassium,
            Self::Chloride => Element::Chlorine,
        }
    }

    /// Elementary charge units.
    pub fn charge(self) -> f32 {
        match self {
            Self::Sodium | Self::Potassium => 1.,
            Self::Chloride => -1.,
        }
    }

    pub fn ff_type(self) -> &'static str {
        match self {
            Self::Sodium => "Na+",
            Self::Potassium => "K+",
            Self::Chloride => "Cl-",
        }
    }
}

// Decode is implemented manually below; the derive can't decode a `Cow<'static, str>`.
#[cfg_attr(feature = "encode", derive(Encode))]
#[derive(Clone, Debug, PartialEq)]
pub struct IonParams {
    pub ion: Ion,
    /// The force field type, e.g. "Na+" in Amber, and "SOD" in CHARMM. Pair-specific LJ
    /// parameters (NBFIX) match on it.
    pub ff_type: Cow<'static, str>,
    /// amu
    pub mass: f32,
    /// Å
    pub lj_sigma: f32,
    /// kcal/mol
    pub lj_eps: f32,
}

#[cfg(feature = "encode")]
impl<Context> Decode<Context> for IonParams {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            ion: Decode::decode(decoder)?,
            ff_type: Cow::Owned(String::decode(decoder)?),
            mass: Decode::decode(decoder)?,
            lj_sigma: Decode::decode(decoder)?,
            lj_eps: Decode::decode(decoder)?,
        })
    }
}

#[cfg(feature = "encode")]
bincode::impl_borrow_decode!(IonParams);

/// Used when configuring a MD Sim. We use OPC (rigid) water as a default, but can
/// use custom solvents as well, from arbitrary molecules using standard MD forcefields.
#[derive(Clone, Debug, Default)]
pub enum Solvent {
    None,
    /// Fill the entire sim box with rigid water molecules, at a realistic density.
    #[default]
    WaterOpc,
    /// Fill the sim box uniformly with water, but with a non-standard density.
    WaterOpcSpecifyMolCount(usize),
    /// Fill sub-regions of the initial sim box with rigid water molecules at a realistic density.
    /// Regions move with the cell when init recenters it and must remain inside the full sim box.
    WaterOpcCustomRegions(Vec<SimBox>),
    /// Fill the whole sim box with octanol, and a realistic saturation of rigid water molecules.
    OctanolWithWater,
    /// (Custom mols and their counts, OPC water count). Unlike for OPC water, we use standard
    /// MD force fields for these, as we do for other molecules. Their presense in solvents here
    /// is primarily for the purposes of initializing them on their own, or with rigid water. Compared
    /// to with other molecules, the intent here is to saturate the cell / SimBox at init, in a way
    /// which requires care with how we're packing.
    ///
    /// For now, we use GAFF2 (Small molecule) force fields for these non-water solvents.
    Custom((Vec<(MolDynamics, usize)>, usize)),
}

impl Display for Solvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let v = match self {
            Self::None => "None",
            Self::WaterOpc => "OPC water",
            Self::WaterOpcSpecifyMolCount(c) => &format!("Water OPC. {c} mols"),
            Self::WaterOpcCustomRegions(_) => "OPC water (Custom regions)",
            Self::OctanolWithWater => "Octanol with Water",
            Self::Custom(_) => "Custom",
        };

        write!(f, "{v}")
    }
}

impl PartialEq for Solvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::WaterOpc, Self::WaterOpc) | (Self::OctanolWithWater, Self::OctanolWithWater) => {
                true
            }
            (Self::WaterOpcSpecifyMolCount(a), Self::WaterOpcSpecifyMolCount(b)) => a == b,
            (Self::WaterOpcCustomRegions(a), Self::WaterOpcCustomRegions(b)) => a == b,
            (Self::Custom((_, water_a)), Self::Custom((_, water_b))) => water_a == water_b,
            _ => false,
        }
    }
}

// Manual Encode/Decode as MolDynamics doesn't impl it, so we can't derive. Need to encode,
// as it's part of MdConfig.
#[cfg(feature = "encode")]
impl bincode::Encode for Solvent {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        match self {
            Self::None | Self::WaterOpc => {
                0u32.encode(encoder)?;
            }
            Self::WaterOpcSpecifyMolCount(count) => {
                1u32.encode(encoder)?;
                count.encode(encoder)?;
            }
            Self::WaterOpcCustomRegions(regions) => {
                3u32.encode(encoder)?;
                regions.encode(encoder)?;
            }
            Self::Custom(_) | Self::OctanolWithWater => {
                0u32.encode(encoder)?;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "encode")]
impl<Context> bincode::Decode<Context> for Solvent {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let variant = u32::decode(decoder)?;

        match variant {
            0 => Ok(Self::WaterOpc),
            1 => {
                let count = usize::decode(decoder)?;
                Ok(Self::WaterOpcSpecifyMolCount(count))
            }
            2 => Err(bincode::error::DecodeError::OtherString(
                "Solvent variant 2 (pre-positioned OPC water) is no longer supported.".to_owned(),
            )),
            3 => Ok(Self::WaterOpcCustomRegions(Vec::<SimBox>::decode(decoder)?)),
            _ => Err(bincode::error::DecodeError::UnexpectedVariant {
                type_name: "Solvent",
                allowed: &bincode::error::AllowedEnumVariants::Allowed(&[0, 1, 3]),
                found: variant,
            }),
        }
    }
}

#[cfg(feature = "encode")]
impl<'de, Context> bincode::BorrowDecode<'de, Context> for Solvent {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let variant = u32::borrow_decode(decoder)?;

        match variant {
            0 => Ok(Self::WaterOpc),
            1 => {
                let count = usize::borrow_decode(decoder)?;
                Ok(Self::WaterOpcSpecifyMolCount(count))
            }
            2 => Err(bincode::error::DecodeError::OtherString(
                "Solvent variant 2 (pre-positioned OPC water) is no longer supported.".to_owned(),
            )),
            3 => Ok(Self::WaterOpcCustomRegions(Vec::<SimBox>::borrow_decode(
                decoder,
            )?)),
            _ => Err(bincode::error::DecodeError::UnexpectedVariant {
                type_name: "Solvent",
                allowed: &bincode::error::AllowedEnumVariants::Allowed(&[0, 1, 3]),
                found: variant,
            }),
        }
    }
}

#[cfg(all(test, feature = "encode"))]
mod codec_tests {
    use bincode::{config, decode_from_slice, encode_to_vec};
    use lin_alg::f32::Vec3;

    use super::{SimBox, Solvent};

    #[test]
    fn custom_regions_round_trip() {
        let solvent = Solvent::WaterOpcCustomRegions(vec![SimBox::new(
            Vec3::new(-3., -2., -1.),
            Vec3::new(3., 2., 1.),
        )]);
        let bytes = encode_to_vec(&solvent, config::standard()).unwrap();
        let (decoded, _): (Solvent, usize) = decode_from_slice(&bytes, config::standard()).unwrap();

        assert_eq!(decoded, solvent);
    }
}

// We use this encoding when passing to CUDA. We reserve 0 for non-solvent atoms.
#[derive(Copy, Clone, PartialEq)]
#[repr(u8)]
pub(crate) enum WaterSite {
    O = 1,
    M = 2,
    H0 = 3,
    H1 = 4,
}

/// Per-solvent, per-site force accumulator. Used transiently when applying nonbonded forces.
/// This is the force *on* each atom in the molecule.
#[derive(Clone, Copy, Default)]
pub struct ForcesOnWaterMol {
    // 64-bit as they're accumulators.
    pub f_o: Vec3,
    pub f_h0: Vec3,
    pub f_h1: Vec3,
    /// SETTLE/constraint will redistribute force on M/EP.
    pub f_m: Vec3,
}

#[allow(unused)]
// todo: Note: These are 32-bit due to limits on 64-bit with. Be careful; you use 64-bit elsewhere.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Default)]
pub struct ForcesOnWaterMolx8 {
    pub f_o: Vec3x8,
    pub f_h0: Vec3x8,
    pub f_h1: Vec3x8,
    pub f_m: Vec3x8,
}

#[allow(unused)]
// todo: Note: These are 32-bit due to limits on 64-bit with. Be careful; you use 64-bit elsewhere.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Default)]
pub struct ForcesOnWaterMolx16 {
    pub f_o: Vec3x16,
    pub f_h0: Vec3x16,
    pub f_h1: Vec3x16,
    pub f_m: Vec3x16,
}

/// Contains 4 atoms for each solvent molecules, at a given time step. Note that these
/// are not independent, but are useful in our general MD APIs, for compatibility with
/// non-solvent atoms.
///
/// Note: We currently don't use accel value on each atom directly, but use a `ForcesOnAtoms` abstraction.
///
/// Important: We repurpose the `accel` field of `AtomDynamics` to store forces instead. These differ
/// by a factor of mass.
/// todo: We may or may not change this A/R.
#[derive(Clone, Debug)]
pub struct WaterMolOpc {
    /// Chargeless; its charge is represented at the offset "M" or "EP".
    /// The only Lennard Jones/Vdw source. Has mass.
    pub o: AtomDynamics,
    /// Hydrogens: carries charge, but no VdW force; have mass.
    pub h0: AtomDynamics,
    pub h1: AtomDynamics,
    /// The massless, charged particle offset from O. Also known as EP.
    pub m: AtomDynamics,
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub struct WaterMolx8 {
    pub o: AtomDynamicsx8,
    pub h0: AtomDynamicsx8,
    pub h1: AtomDynamicsx8,
    pub m: AtomDynamicsx8,
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub struct WaterMolx16 {
    pub o: AtomDynamicsx16,
    pub h0: AtomDynamicsx16,
    pub h1: AtomDynamicsx16,
    pub m: AtomDynamicsx16,
}

impl WaterMolOpc {
    pub fn new(
        o_pos: Vec3F32,
        vel: Vec3F32,
        orientation: QuaternionF32,
        model: &WaterModel,
    ) -> Self {
        // Set up H and EP/M positions based on orientation.
        // Unit vectors defining the body frame
        let z_local = orientation.rotate_vec(Z_VEC);
        let e_local = orientation.rotate_vec(X_VEC);

        // Place Hs in the plane spanned by ex, ez with the right HOH angle.
        // Let the bisector be ez, and put the hydrogens symmetrically around it.
        let angle_half = 0.5 * model.h_o_h_angle;

        let h0_dir = (z_local * angle_half.cos() + e_local * angle_half.sin()).to_normalized();
        let h1_dir = (z_local * angle_half.cos() - e_local * angle_half.sin()).to_normalized();

        let h0_pos = o_pos + h0_dir * model.o_h_dist;
        let h1_pos = o_pos + h1_dir * model.o_h_dist;

        // EP on the HOH bisector at fixed O–EP distance
        let ep_pos = o_pos + (h0_pos - o_pos + h1_pos - o_pos).to_normalized() * model.o_m_dist;

        let q_h = model.q_h * CHARGE_UNIT_SCALER;

        let h0 = AtomDynamics {
            force_field_type: String::from("HW"),
            element: Element::Hydrogen,
            posit: h0_pos,
            vel,
            // This is actually force for our purposes, in the context of solvent molecules.
            mass: model.mass_h,
            partial_charge: q_h,
            lj_sigma: model.lj_sigma_h,
            lj_eps: model.lj_eps_h,
            ..Default::default()
        };

        Self {
            // Override LJ params, charge, and mass.
            o: AtomDynamics {
                force_field_type: String::from("OW"),
                posit: o_pos,
                element: Element::Oxygen,
                mass: model.mass_o,
                partial_charge: 0.,
                lj_sigma: model.lj_sigma_o,
                lj_eps: model.lj_eps_o,
                ..h0.clone()
            },
            h1: AtomDynamics {
                posit: h1_pos,
                ..h0.clone()
            },
            // Override charge and mass.
            m: AtomDynamics {
                force_field_type: String::from("EP"),
                posit: ep_pos,
                element: Element::Potassium, // Placeholder
                mass: 0.,
                partial_charge: -2. * q_h,
                lj_sigma: 0.,
                lj_eps: 0.,
                ..h0.clone()
            },
            h0,
        }
    }

    pub(crate) fn site(&self, site: WaterSite) -> &AtomDynamics {
        match site {
            WaterSite::O => &self.o,
            WaterSite::M => &self.m,
            WaterSite::H0 => &self.h0,
            WaterSite::H1 => &self.h1,
        }
    }

    pub(crate) fn site_mut(&mut self, site: WaterSite) -> &mut AtomDynamics {
        match site {
            WaterSite::O => &mut self.o,
            WaterSite::M => &mut self.m,
            WaterSite::H0 => &mut self.h0,
            WaterSite::H1 => &mut self.h1,
        }
    }

    /// Run this after updating force on the M/EP site; converts its force to the O and H sites,
    /// and leaves it at 0.
    pub(crate) fn project_ep_force(&mut self, model: &WaterModel) {
        let f_m = self.m.force;
        let (c_o, c_h) = model.m_force_coeffs();

        // Exact force conservation, exact torque conservation (for this geometry)
        self.o.force += f_m * c_o;
        self.h0.force += f_m * c_h;
        self.h1.force += f_m * c_h;

        self.m.force = Vec3F32::new_zero();
    }

    // todo: Experimenting
    /// Places the M (EP) site based on current O and H positions.
    /// Call this after Initialization, Settle, or Barostat scaling.
    pub(crate) fn update_virtual_site(&mut self, model: &WaterModel) {
        // Fast approximate bisector reconstruction
        let v_h0 = self.h0.posit - self.o.posit;
        let v_h1 = self.h1.posit - self.o.posit;

        // Unnormalized bisector
        let bis = v_h0 + v_h1;

        // This squareroot is unavoidable for exact distance,
        // but cheaper than the full geometry logic in your snippet.
        self.m.posit = self.o.posit + bis.to_normalized() * model.o_m_dist;

        // Interpolate velocity for M (important for thermostats)
        // M is approx midway between H's angularly, but closer to O.
        // A simple average of H's is often 'good enough' for temperature,
        // but strictly it depends on geometry. Your code used avg of H:
        self.m.vel = (self.h0.vel + self.h1.vel) * 0.5;
    }
}
