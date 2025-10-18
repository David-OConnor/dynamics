//! For VDW and Coulomb forces

use std::{ops::AddAssign, time::Instant};

use ewald::force_coulomb_short_range;
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::gpu_interface::force_nonbonded_gpu;
use crate::{
    AtomDynamics, AtomDynamicsx8, AtomDynamicsx16, COMPUTATION_TIME_RATIO, ComputationDevice,
    MdState,
    ambient::{AMU_A2_PS2_TO_KCAL_PER_MOL_EXACT, SimBox},
    forces::force_e_lj,
    water_opc::{ForcesOnWaterMol, O_EPS, O_SIGMA, WaterMol, WaterSite},
};

// Å. 9-12 should be fine; there is very little VDW force > this range due to
// the ^-7 falloff.
pub const CUTOFF_VDW: f32 = 12.0;

// Ewald SPME approximation for Coulomb force

// Instead of a hard cutoff between short and long-range forces, these
// parameters control a smooth taper.
// Our neighbor list must use the same cutoff as this, so we use it directly.

// We don't use a taper, for now.
// const LONG_RANGE_SWITCH_START: f64 = 8.0; // start switching (Å)
pub const LONG_RANGE_CUTOFF: f32 = 10.0; // Å

// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
// reciprocal load.
// Common rule for α: erfc(α r_c) ≲ 10⁻⁴…10⁻⁵
pub const EWALD_ALPHA: f32 = 0.35; // Å^-1. 0.35 is good for cutoff = 10.
pub const PME_MESH_SPACING: f32 = 1.0;

// // SPME order‑4 B‑spline interpolation
// pub const SPME_N: usize = 64;

// Å. Smaller uses a higher-resolution mesh. 1 is a good default.
pub const SPME_MESH_SPACING: f32 = 1.;

// See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
// "Non-bonded interactions between atoms separated by three consecutive bonds... require a special
// treatment in Amber force fields."
// "By default, vdW 1-4 interactions are divided (scaled down) by a factor of 2.0, electrostatic 1-4 terms by a factor
// of 1.2."
const SCALE_LJ_14: f32 = 0.5;
pub const SCALE_COUL_14: f32 = 1.0 / 1.2;

// Multiply by this to convert partial charges from elementary charge (What we store in Atoms loaded from mol2
// files and amino19.lib.) to the self-consistent amber units required to calculate Coulomb force.
// We apply this to dynamic and static atoms when building Indexed params, and to water molecules
// on their construction. We do not apply this during integration.
pub const CHARGE_UNIT_SCALER: f32 = 18.2223;

/// We use this to load the correct data from LJ lookup tables. Since we use indices,
/// we must index correctly into the dynamic, or static tables. We have single-index lookups
/// for atoms acting on water, since there is only one O LJ type.
#[derive(Debug)]
pub enum LjTableIndices {
    /// (tgt, src)
    StdStd((usize, usize)),
    /// (dyn tgt or src))
    StdWater(usize),
    /// One value, stored as a constant (Water O -> Water O)
    WaterWater,
}

/// We cache σ and ε on the first step, then use it on the others. This increases
/// memory use, and reduces CPU use. We use indices, as they're faster than HashMaps.
/// The indices are flattened, of each interaction pair. Values are (σ, ε).
///
/// Water-water is not included, as it's a single, hard-coded parameter pair.
#[derive(Default)]
pub struct LjTables {
    /// Non-water, non-water interactions. Upper triangle.
    pub std: Vec<(f32, f32)>,
    /// Water, non-water interactions.
    pub water_std: Vec<(f32, f32)>,
    pub n_std: usize,
}

#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct LjTablesx8 {
    /// Non-water, non-water interactions. Upper triangle.
    pub std: Vec<(f32x8, f32x8)>,
    /// Water, non-water interactions.
    pub water_std: Vec<(f32x8, f32x8)>,
    pub n_std: [usize; 8],
}

#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct LjTablesx16 {
    /// Non-water, non-water interactions. Upper triangle.
    pub std: Vec<(f32x16, f32x16)>,
    /// Water, non-water interactions.
    pub water_std: Vec<(f32x16, f32x16)>,
    pub n_std: [usize; 8],
}

// todo note: On large systems, this can have very high memory use. Consider
// todo setting up your table by atom type, instead of by atom, if that proves to be a problem.
impl LjTables {
    /// Create an indexed table, flattened.
    pub fn new(atoms: &[AtomDynamics]) -> Self {
        let n_std = atoms.len();

        // Construct an upper triangle table, excluding reverse order, and self interactions.
        let mut std = Vec::with_capacity(n_std * (n_std - 1) / 2);

        for (i_0, atom_0) in atoms.iter().enumerate() {
            for (i_1, atom_1) in atoms.iter().enumerate() {
                if i_1 <= i_0 {
                    continue;
                }
                let (σ, ε) = combine_lj_params(atom_0, atom_1);
                std.push((σ, ε));
            }
        }

        // One LJ pair per dynamic atom vs water O:
        let mut water_std = Vec::with_capacity(n_std);
        for atom in atoms {
            let σ = 0.5 * (atom.lj_sigma + O_SIGMA);
            let ε = (atom.lj_eps * O_EPS).sqrt();
            water_std.push((σ, ε));
        }

        Self {
            std,
            water_std,
            n_std,
        }
    }

    /// Get (σ, ε)
    pub fn lookup(&self, i: &LjTableIndices) -> (f32, f32) {
        match i {
            LjTableIndices::StdStd((i_0, i_1)) => {
                // Map to (i<j), then index into the packed upper triangle (row-major).
                let (i, j) = if i_0 < i_1 {
                    (*i_0, *i_1)
                } else {
                    (*i_1, *i_0)
                };

                if i >= self.n_std {
                    println!("I > i: {i} std: {}", self.n_std);
                }

                if j >= self.n_std {
                    println!("J > J: {j} std: {}", self.n_std);
                }

                // todo temp
                assert!(i < self.n_std && j < self.n_std && i != j);

                // Elements before row i: sum_{r=0}^{i-1} (N-1-r) = i*(2N - i - 1)/2
                // Offset within row i: (j - i - 1)
                let idx = i * (2 * self.n_std - i - 1) / 2 + (j - i - 1);

                self.std[idx]
            }
            LjTableIndices::StdWater(ix) => self.water_std[*ix],
            LjTableIndices::WaterWater => (O_SIGMA, O_EPS),
        }
    }
}

impl AddAssign<Self> for ForcesOnWaterMol {
    fn add_assign(&mut self, rhs: Self) {
        self.f_o += rhs.f_o;
        self.f_h0 += rhs.f_h0;
        self.f_h1 += rhs.f_h1;
        self.f_m += rhs.f_m;
    }
}

#[derive(Copy, Clone)]
pub enum BodyRef {
    NonWater(usize),
    // Static(usize),
    Water { mol: usize, site: WaterSite },
}

impl BodyRef {
    pub(crate) fn get<'a>(
        &self,
        non_waters: &'a [AtomDynamics],
        waters: &'a [WaterMol],
    ) -> &'a AtomDynamics {
        match *self {
            BodyRef::NonWater(i) => &non_waters[i],
            BodyRef::Water { mol, site } => match site {
                WaterSite::O => &waters[mol].o,
                WaterSite::M => &waters[mol].m,
                WaterSite::H0 => &waters[mol].h0,
                WaterSite::H1 => &waters[mol].h1,
            },
        }
    }
}

pub struct NonBondedPair {
    pub tgt: BodyRef,
    pub src: BodyRef,
    pub scale_14: bool,
    pub lj_indices: LjTableIndices,
    pub calc_lj: bool,
    pub calc_coulomb: bool,
    pub symmetric: bool,
}

/// Add a force into the right accumulator (std or water). Static never accumulates.
fn add_to_sink(
    sink_non_water: &mut [Vec3F64],
    sink_wat: &mut [ForcesOnWaterMol],
    body_type: BodyRef,
    f: Vec3F64,
) {
    match body_type {
        BodyRef::NonWater(i) => sink_non_water[i] += f,
        BodyRef::Water { mol, site } => match site {
            WaterSite::O => sink_wat[mol].f_o += f,
            WaterSite::M => sink_wat[mol].f_m += f,
            WaterSite::H0 => sink_wat[mol].f_h0 += f,
            WaterSite::H1 => sink_wat[mol].f_h1 += f,
        },
        // BodyRef::Static(_) => (),
    }
}

/// Applies non-bonded force in parallel (CPU thread-pool) over a set of atoms, with indices assigned
/// upstream.
///
/// Return the virial pair component we accumulate. For use with the temp/barostat. (kcal/mol)
fn calc_force_cpu(
    pairs: &[NonBondedPair],
    atoms_std: &[AtomDynamics],
    water: &[WaterMol],
    cell: &SimBox,
    lj_tables: &LjTables,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64) {
    let n_std = atoms_std.len();
    let n_wat = water.len();

    pairs
        .par_iter()
        .fold(
            || {
                (
                    // Sums as f64.
                    vec![Vec3F64::new_zero(); n_std],
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64, // Virial sum
                    0.0_f64, // Energy sum
                )
            },
            |(mut acc_std, mut acc_w, mut virial, mut energy), p| {
                let a_t = p.tgt.get(atoms_std, water);
                let a_s = p.src.get(atoms_std, water);

                let (f, e_pair) = f_nonbonded_cpu(
                    &mut virial,
                    a_t,
                    a_s,
                    cell,
                    p.scale_14,
                    &p.lj_indices,
                    lj_tables,
                    p.calc_lj,
                    p.calc_coulomb,
                );

                // Convert to f64 prior to summing.
                let f: Vec3F64 = f.into();
                add_to_sink(&mut acc_std, &mut acc_w, p.tgt, f);
                if p.symmetric {
                    add_to_sink(&mut acc_std, &mut acc_w, p.src, -f);
                }

                // We are not interested, in this point, at energy that does not involve our dyanamic (ligand) atoms.
                // We skip water-water, and water-static interations.
                let involves_std =
                    matches!(p.tgt, BodyRef::NonWater(_)) || matches!(p.src, BodyRef::NonWater(_));

                if involves_std {
                    energy += e_pair as f64;
                }

                (acc_std, acc_w, virial, energy)
            },
        )
        .reduce(
            || {
                (
                    vec![Vec3F64::new_zero(); n_std],
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64,
                    0.0_f64,
                )
            },
            |(mut f_on_std, mut f_on_water, virial_a, e_a), (db, wb, virial_b, e_b)| {
                for i in 0..n_std {
                    f_on_std[i] += db[i];
                }
                for i in 0..n_wat {
                    f_on_water[i].f_o += wb[i].f_o;
                    f_on_water[i].f_m += wb[i].f_m;
                    f_on_water[i].f_h0 += wb[i].f_h0;
                    f_on_water[i].f_h1 += wb[i].f_h1;
                }

                (f_on_std, f_on_water, virial_a + virial_b, e_a + e_b)
            },
        )
}

// #[cfg(target_arch = "x86_64")]
// fn calc_force_x8(
//     pairs: &[NonBondedPair],
//     atoms_std: &[AtomDynamicsx8],
//     water: &[WaterMolx8],
//     cell: &SimBox,
//     lj_tables: &LjTablesx8,
// ) -> (Vec<Vec3x8>, Vec<ForcesOnWaterMol>, f64, f64) {
// }
//
// #[cfg(target_arch = "x86_64")]
// fn calc_force_x16(
//     pairs: &[NonBondedPair],
//     atoms_std: &[AtomDynamicsx16],
//     water: &[WaterMolx16],
//     cell: &SimBox,
//     lj_tables: &LjTablesx16,
// ) -> (Vec<Vec3x16>, Vec<ForcesOnWaterMol>, f64, f64) {
// }

impl MdState {
    /// Run the appropriate force-computation function to get force on non-water atoms, force
    /// on water atoms, and virial sum for the barostat. Uses GPU if available.
    ///
    /// Applies Coulomb and Van der Waals (Lennard-Jones) forces on non-water atoms, in place.
    /// We use the MD-standard [S]PME approach to handle approximated Coulomb forces. This function
    /// applies forces from non-water, and water sources.
    pub fn apply_nonbonded_forces(&mut self, dev: &ComputationDevice) {
        let (f_on_std, f_on_water, virial, energy) = match dev {
            ComputationDevice::Cpu => {
                if is_x86_feature_detected!("avx512f") {
                    // calc_force_x16(
                    //     &self.nb_pairs,
                    //     &self.atoms_x16,
                    //     &self.water,
                    //     &self.cell,
                    //     &self.lj_tables,
                    // )
                } else {
                    // calc_force_x8(
                    //     &self.nb_pairs,
                    //     &self.atoms_x8,
                    //     &self.water,
                    //     &self.cell,
                    //     &self.lj_tables,
                    // )
                }

                calc_force_cpu(
                    &self.nb_pairs,
                    &self.atoms,
                    &self.water,
                    &self.cell,
                    &self.lj_tables,
                )
            }
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu((stream, module)) => force_nonbonded_gpu(
                stream,
                module,
                self.gpu_kernel.as_ref().unwrap(),
                &self.nb_pairs,
                &self.atoms,
                &self.water,
                self.cell.extent,
                self.forces_posits_gpu.as_mut().unwrap(),
                self.per_neighbor_gpu.as_ref().unwrap(),
            ),
        };

        // `.into()` below converts accumulated forces to f32.
        for (i, tgt) in self.atoms.iter_mut().enumerate() {
            let f: Vec3 = f_on_std[i].into();
            tgt.accel += f;
            // println!("SHORT. i: {i}, f: {f:?}");
        }

        for (i, tgt) in self.water.iter_mut().enumerate() {
            let f = f_on_water[i];
            let f_0: Vec3 = f.f_o.into();
            let f_m: Vec3 = f.f_m.into();
            let f_h0: Vec3 = f.f_h0.into();
            let f_h1: Vec3 = f.f_h1.into();

            tgt.o.accel += f_0;
            tgt.m.accel += f_m;
            tgt.h0.accel += f_h0;
            tgt.h1.accel += f_h1;
        }

        self.barostat.virial_coulomb += virial;
        self.potential_energy += energy;
    }

    /// [Re] initialize non-bonded interaction pairs between atoms. Do this whenever we rebuild neighbors.
    /// Build the neighbors set prior to running this.
    pub(crate) fn setup_pairs(&mut self) {
        let n_std = self.atoms.len();
        let n_water_mols = self.water.len();

        let sites = [WaterSite::O, WaterSite::M, WaterSite::H0, WaterSite::H1];

        // todo: You can probably consolidate even further. Instead of calling apply_force
        // todo per each category, you can assemble one big set of pairs, and call it once.
        // todo: This has performance and probably code organization benefits. Maybe try
        // todo after you get the intial version working. Will have to add symmetric to pairs.

        // ------ Forces from other dynamic atoms on dynamic ones ------

        // Exclusions and scaling apply to std-std interactions only.
        let exclusions = &self.pairs_excluded_12_13;
        let scaled_set = &self.pairs_14_scaled;

        // Set up pairs ahead of time; conducive to parallel iteration. We skip excluded pairs,
        // and mark scaled ones. These pairs, in symmetric cases (e.g. std-std), only
        let pairs_std_std: Vec<_> = (0..n_std)
            .flat_map(|i_tgt| {
                self.neighbors_nb.std_std[i_tgt]
                    .iter()
                    .copied()
                    .filter(move |&j| j > i_tgt) // Ensure stable order
                    .filter_map(move |i_src| {
                        let key = (i_tgt, i_src);
                        if exclusions.contains(&key) {
                            return None;
                        }
                        let scale_14 = scaled_set.contains(&key);

                        Some(NonBondedPair {
                            tgt: BodyRef::NonWater(i_tgt),
                            src: BodyRef::NonWater(i_src),
                            scale_14,
                            lj_indices: LjTableIndices::StdStd(key),
                            calc_lj: true,
                            calc_coulomb: true,
                            symmetric: true,
                        })
                    })
            })
            .collect();

        // todo: Look at water_water
        // todo: In general, your static exclusions will get messed up with this logic.

        // Forces from water on non-water atoms, and vice-versa
        let mut pairs_std_water: Vec<_> = (0..n_std)
            .flat_map(|i_std| {
                self.neighbors_nb.std_water[i_std]
                    .iter()
                    .copied()
                    .flat_map(move |i_water| {
                        sites.into_iter().map(move |site| NonBondedPair {
                            tgt: BodyRef::NonWater(i_std),
                            src: BodyRef::Water { mol: i_water, site },
                            scale_14: false,
                            // todo: Ensure you reverse it.
                            lj_indices: LjTableIndices::StdWater(i_std),
                            calc_lj: site == WaterSite::O,
                            calc_coulomb: site != WaterSite::O,
                            symmetric: true,
                        })
                    })
            })
            .collect();

        // ------ Water on water ------
        let mut pairs_water_water = Vec::new();

        for i_0 in 0..n_water_mols {
            for &i_1 in &self.neighbors_nb.water_water[i_0] {
                if i_1 <= i_0 {
                    continue;
                }

                for &site_0 in &sites {
                    for &site_1 in &sites {
                        let calc_lj = site_0 == WaterSite::O && site_1 == WaterSite::O;
                        let calc_coulomb = site_0 != WaterSite::O && site_1 != WaterSite::O;

                        if !(calc_lj || calc_coulomb) {
                            continue;
                        }

                        pairs_water_water.push(NonBondedPair {
                            tgt: BodyRef::Water {
                                mol: i_0,
                                site: site_0,
                            },
                            src: BodyRef::Water {
                                mol: i_1,
                                site: site_1,
                            },
                            scale_14: false,
                            lj_indices: LjTableIndices::WaterWater,
                            calc_lj,
                            calc_coulomb,
                            symmetric: true,
                        });
                    }
                }
            }
        }

        // todo: Consider just removing the functional parts above, and add to `pairs` directly.
        // Combine pairs into a single set; we compute in one parallel pass.
        let len_added = pairs_std_water.len() + pairs_water_water.len();

        let mut pairs = pairs_std_std;
        pairs.reserve(len_added);

        pairs.append(&mut pairs_std_water);
        pairs.append(&mut pairs_water_water);

        self.nb_pairs = pairs;
    }
}

/// Lennard Jones and (short-range) Coulomb forces. Used by water and non-water.
/// We run long-range SPME Coulomb force separately.
///
/// We use a hard distance cutoff for Vdw, enabled by its ^-7 falloff.
/// Returns energy as well.
pub fn f_nonbonded_cpu(
    virial_w: &mut f64,
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    cell: &SimBox,
    scale14: bool, // See notes earlier in this module.
    lj_indices: &LjTableIndices,
    lj_tables: &LjTables,
    // These flags are for use with forces on water.
    calc_lj: bool,
    calc_coulomb: bool,
) -> (Vec3, f32) {
    let diff = cell.min_image(tgt.posit - src.posit);

    // We compute these dist-related values once, and share them between
    // LJ and Coulomb.
    let dist_sq = diff.magnitude_squared();

    if dist_sq < 1e-12 {
        return (Vec3::new_zero(), 0.);
    }

    let dist = dist_sq.sqrt();
    let inv_dist = 1.0 / dist;
    let dir = diff * inv_dist;

    let (f_lj, energy_lj) = if !calc_lj || dist > CUTOFF_VDW {
        (Vec3::new_zero(), 0.)
    } else {
        let (σ, ε) = lj_tables.lookup(lj_indices);

        let (mut f, mut e) = force_e_lj(dir, inv_dist, σ, ε);
        if scale14 {
            f *= SCALE_LJ_14;
            e *= SCALE_LJ_14;
        }
        (f, e)
    };

    // We assume that in the AtomDynamics structs, charges are already scaled to Amber units.
    // (No longer in elementary charge)
    let (mut f_coulomb, mut energy_coulomb) = if !calc_coulomb {
        (Vec3::new_zero(), 0.)
    } else {
        force_coulomb_short_range(
            dir,
            dist,
            inv_dist,
            tgt.partial_charge,
            src.partial_charge,
            LONG_RANGE_CUTOFF,
            EWALD_ALPHA,
        )
    };

    // See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
    if scale14 {
        f_coulomb *= SCALE_COUL_14;
        energy_coulomb *= SCALE_COUL_14;
    }

    // todo: How do we prevent accumulating energy on static atoms and water?

    let force = f_lj + f_coulomb;
    let energy = energy_lj + energy_coulomb;

    *virial_w += diff.dot(force) as f64;

    (force, energy)
}

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
/// Note that this uses the traditional algorithm; not the Amber-specific version: We pre-set
/// atom-specific σ and ε to traditional versions on ingest, and when building water.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f32, f32) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
