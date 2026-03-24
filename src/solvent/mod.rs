#![allow(non_upper_case_globals)]
#![allow(clippy::excessive_precision)]

//! We use the [OPC model](https://pubs.acs.org/doi/10.1021/jz501780a) for solvent.
//! See also, the Amber Reference Manual.
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

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16};
use lin_alg::{
    f32::{Quaternion as QuaternionF32, Vec3 as Vec3F32, X_VEC, Z_VEC},
    f64::Vec3,
};
use na_seq::Element;

use crate::{
    AtomDynamics, KCAL_TO_NATIVE, MolDynamics, barostat::SimBox, non_bonded::CHARGE_UNIT_SCALER,
};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use crate::{AtomDynamicsx8, AtomDynamicsx16};

pub(crate) mod init;
pub(crate) mod settle;

use settle::RA;

// Constant parameters below are for the OPC solvent (JPCL, 2014, 5 (21), pp 3863-3871)
// (Amber 2025, frcmod.opc) EP/M is the massless, 4th charge.
// These values are taken directly from `frcmod.opc`, in the Amber package. We have omitted
// values that are 0., or otherwise not relevant in this model. (e.g. EP mass, O charge, bonded params
// other than bond distances and the valence angle)
pub(crate) const O_MASS: f32 = 16.;
pub(crate) const H_MASS: f32 = 1.008;
pub(crate) const MASS_WATER_MOL: f32 = O_MASS + 2.0 * H_MASS;

// We have commented out flexible-bond parameters that are provided by Amber, but not
// used in this rigid model.

// Å; bond distance. (frcmod.opc, or Table 2.)
pub(crate) const O_EP_R: f32 = 0.159_398_33;
pub(crate) const O_H_R: f32 = 0.872_433_13;

// Angle bending angle, radians.
pub(crate) const H_O_H_θ: f32 = 1.808_161_105_066; // (103.6 degrees in frcmod.opc)
const H_O_H_θ_HALF: f32 = 0.5 * H_O_H_θ;

// For converting from R_star to eps. See notes in bio_files's `LjParams`.
const SIGMA_FACTOR: f32 = 2. / 1.122_462_048_309_373;

// Van der Waals / JL params. Only O carries this.
const O_RSTAR: f32 = 1.777_167_268;
pub const O_SIGMA: f32 = O_RSTAR * SIGMA_FACTOR;
pub const O_EPS: f32 = 0.212_800_813_0;

// Partial charges in elementary charge. See the OPC paper, Table 2. None on O.
const Q_H: f32 = 0.6791 * CHARGE_UNIT_SCALER;
const Q_EP: f32 = -2. * Q_H;

// Consts for force projection from the virtual site.
// For a bisector site at distance d_OM, with bond length d_OH and angle theta:
// c_H = (d_OM / (d_OH * cos(theta/2))) / 2.0;
// We pre-calculate the cos part.
const C_H: f32 = (O_EP_R / RA) / 2.;
const C_O: f32 = 1.0 - 2.0 * C_H;

// We use this to convert from force to acceleration, in the appropriate units.
pub(crate) const ACCEL_CONV_WATER_O: f32 = KCAL_TO_NATIVE / O_MASS;
pub(crate) const ACCEL_CONV_WATER_H: f32 = KCAL_TO_NATIVE / H_MASS;

/// Used when configuring a MD Sim. We use OPC (rigid) water as a default, but can
/// use custom solvents as well, from arbitrary molecules using standard MD forcefields.
///
/// todo: We may switch out these absolute count `usize` values for something else, like
/// todo a portion by mass, volume, or mol count.
#[derive(Clone, Debug, Default)]
pub enum Solvent {
    None,
    #[default]
    WaterOpc,
    WaterOpcSpecifyMolCount(usize),
    /// (Custom mols and their counts, OPC water count). Unlikje for OPC water, we use standard
    /// MD force fields for these, as we do for other molecules. Their presense in solvents here
    /// is primarily for the purposes of initializing them on their own, or with OPC water. Compared
    /// to with other molecules, the intent here is to saturate the cell / SimBox at init, in a way
    /// which requires care with how we're packing.
    ///
    /// For now, we use GAFF2 (Small molecule) force fields for these non-water solvents.
    Custom((Vec<(MolDynamics, usize)>, usize)),
}

impl PartialEq for Solvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::WaterOpc, Self::WaterOpc) => true,
            (Self::WaterOpcSpecifyMolCount(a), Self::WaterOpcSpecifyMolCount(b)) => a == b,
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
            Self::Custom(_) => {
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
            _ => Err(bincode::error::DecodeError::UnexpectedVariant {
                type_name: "Solvent",
                allowed: &bincode::error::AllowedEnumVariants::Range { min: 0, max: 1 },
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
            _ => Err(bincode::error::DecodeError::UnexpectedVariant {
                type_name: "Solvent",
                allowed: &bincode::error::AllowedEnumVariants::Range { min: 0, max: 1 },
                found: variant,
            }),
        }
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
pub struct WaterMol {
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

impl WaterMol {
    pub fn new(o_pos: Vec3F32, vel: Vec3F32, orientation: QuaternionF32) -> Self {
        // Set up H and EP/M positions based on orientation.
        // Unit vectors defining the body frame
        let z_local = orientation.rotate_vec(Z_VEC);
        let e_local = orientation.rotate_vec(X_VEC);

        // Place Hs in the plane spanned by ex, ez with the right HOH angle.
        // Let the bisector be ez, and put the hydrogens symmetrically around it.

        let h0_dir = (z_local * H_O_H_θ_HALF.cos() + e_local * H_O_H_θ_HALF.sin()).to_normalized();
        let h1_dir = (z_local * H_O_H_θ_HALF.cos() - e_local * H_O_H_θ_HALF.sin()).to_normalized();

        let h0_pos = o_pos + h0_dir * O_H_R;
        let h1_pos = o_pos + h1_dir * O_H_R;

        // EP on the HOH bisector at fixed O–EP distance
        let ep_pos = o_pos + (h0_pos - o_pos + h1_pos - o_pos).to_normalized() * O_EP_R;

        let h0 = AtomDynamics {
            force_field_type: String::from("HW"),
            element: Element::Hydrogen,
            posit: h0_pos,
            vel,
            // This is actually force for our purposes, in the context of solvent molecules.
            mass: H_MASS,
            partial_charge: Q_H,
            ..Default::default()
        };

        Self {
            // Override LJ params, charge, and mass.
            o: AtomDynamics {
                force_field_type: String::from("OW"),
                posit: o_pos,
                element: Element::Oxygen,
                mass: O_MASS,
                partial_charge: 0.,
                lj_sigma: O_SIGMA,
                lj_eps: O_EPS,
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
                partial_charge: Q_EP,
                ..h0.clone()
            },
            h0,
        }
    }

    /// Run this after updating force on the EP site; converts its force to the O and H sites,
    /// and leaves it at 0.
    pub(crate) fn _project_ep_force_to_real_sites(&mut self, cell: &SimBox) {
        // Geometry in O-centered frame
        let r_O_H0 = self.o.posit + cell.min_image(self.h0.posit - self.o.posit) - self.o.posit;
        let r_O_H1 = self.o.posit + cell.min_image(self.h1.posit - self.o.posit) - self.o.posit;

        let s = r_O_H0 + r_O_H1;
        let s_norm = s.magnitude();

        if s_norm < 1e-12 {
            // Degenerate geometry: drop EP force this step
            self.o.force += self.m.force;
            self.m.force = Vec3F32::new_zero();
            return;
        }

        let f_m = self.m.force;

        // Unit bisector and projection operator P = (I - uu^T)/|s|
        let u = s / s_norm;
        let fm_parallel = u * f_m.dot(u);
        let fm_perp = f_m - fm_parallel; // (I - uu^T) f_m
        let scale = O_EP_R / s_norm; // d / |s|

        // Chain rule: ∂rM/∂rO = I - 2 d P ;  ∂rM/∂rHk = d P
        // Because P is symmetric, (∂rM/∂ri)^T Fm == same expression with P acting on Fm.
        let fh = fm_perp * scale; // contribution that goes to each H
        let fo = f_m - fh * 2.0; // remaining force goes to O

        // Force on M/EP is now zero, and we've modified the forces on the other atoms from it.
        self.m.force = Vec3F32::new_zero();
        self.o.force += fo;
        self.h0.force += fh;
        self.h1.force += fh;
    }

    // todo: Experimenting
    /// Run this after updating force on the EP site; converts its force to the O and H sites,
    /// and leaves it at 0.
    pub(crate) fn project_ep_force_optimized(&mut self) {
        let f_m = self.m.force;

        // Exact force conservation, exact torque conservation (for this geometry)
        self.o.force += f_m * C_O;
        self.h0.force += f_m * C_H;
        self.h1.force += f_m * C_H;

        self.m.force = Vec3F32::new_zero();
    }

    // todo: Experimenting
    /// Places the M (EP) site based on current O and H positions.
    /// Call this after Initialization, Settle, or Barostat scaling.
    pub(crate) fn update_virtual_site(&mut self) {
        // Fast approximate bisector reconstruction
        let v_h0 = self.h0.posit - self.o.posit;
        let v_h1 = self.h1.posit - self.o.posit;

        // Unnormalized bisector
        let bis = v_h0 + v_h1;

        // This squareroot is unavoidable for exact distance,
        // but cheaper than the full geometry logic in your snippet.
        // O_EP_R_0 is the parameter distance (e.g. 0.15 A).
        self.m.posit = self.o.posit + bis.to_normalized() * O_EP_R;

        // Interpolate velocity for M (important for thermostats)
        // M is approx midway between H's angularly, but closer to O.
        // A simple average of H's is often 'good enough' for temperature,
        // but strictly it depends on geometry. Your code used avg of H:
        self.m.vel = (self.h0.vel + self.h1.vel) * 0.5;
    }
}
//
// /// Wrap molecule as a rigid unit. Wrap O, then translate Hs and ,EP so they're on the same
// /// side of the cell.
// pub(crate) fn wrap_water(mol: &mut WaterMol, cell: &SimBox) {
//     let new_o = cell.wrap(mol.o.posit);
//     let shift = new_o - mol.o.posit;
//
//     mol.o.posit = new_o;
//     mol.h0.posit += shift;
//     mol.h1.posit += shift;
//     mol.m.posit += shift;
// }
