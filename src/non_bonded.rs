//! For VDW and Coulomb forces

use std::ops::AddAssign;

use ewald::{PmeRecip, force_coulomb_short_range, get_grid_n};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::gpu_interface::force_nonbonded_gpu;
use crate::{
    AtomDynamics, ComputationDevice, MdOverrides, MdState,
    barostat::SimBox,
    forces::force_e_lj,
    solvent::{ForcesOnWaterMol, O_EPS, O_SIGMA, WaterMolOpc, WaterSite},
};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use crate::{AtomDynamicsx8, AtomDynamicsx16};

// // Å. 9-12 should be fine; there is very little VDW force > this range due to
// // the ^-7 falloff.
// pub const CUTOFF_VDW: f32 = 12.0;

// Ewald SPME approximation for Coulomb force

// Instead of a hard cutoff between short and long-range forces, these
// parameters control a smooth taper.
// Our neighbor list must use the same cutoff as this, so we use it directly.

// The distance beyond which we truncate the real-space erfc-screened interaction.
// This is not used for the reciprical part.
// We don't use a taper, for now.
// const LONG_RANGE_SWITCH_START: f64 = 8.0; // start switching (Å)

// pub const LONG_RANGE_CUTOFF: f32 = 12.0; // Å

// // A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
// // reciprocal load.
// // Common rule for α: erfc(α r_c) ≲ 10⁻⁴…10⁻⁵
// pub const EWALD_ALPHA: f32 = 0.35; // Å^-1. 0.35 is good for cutoff of 10–12 Å.

// See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
// "Non-bonded interactions between atoms separated by three consecutive bonds... require a special
// treatment in Amber force fields."
// "By default, vdW 1-4 interactions are divided (scaled down) by a factor of 2.0, electrostatic 1-4 terms by a factor
// of 1.2."
const SCALE_LJ_14: f32 = 0.5;
pub const SCALE_COUL_14: f32 = 1.0 / 1.2;

// Multiply by this to convert partial charges from elementary charge (What we store in Atoms loaded from mol2
// files and amino19.lib.) to the self-consistent amber units required to calculate Coulomb force.
// We apply this to dynamic and static atoms when building Indexed params, and to solvent molecules
// on their construction. We do not apply this during integration.
// Electrostatic constant: 332.0522 kcal·Å/(mol·e²). This is the square root of that.
pub const CHARGE_UNIT_SCALER: f32 = 18.2223;

/// We use this to load the correct data from LJ lookup tables. Since we use indices,
/// we must index correctly into the dynamic, or static tables. We have single-index lookups
/// for atoms acting on solvent, since there is only one O LJ type.
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
/// Water-solvent is not included, as it's a single, hard-coded parameter pair.
#[derive(Default)]
pub struct LjTables {
    /// Non-solvent, non-solvent interactions. Upper triangle.
    pub std: Vec<(f32, f32)>,
    /// Water, non-solvent interactions.
    pub water_std: Vec<(f32, f32)>,
    pub n_std: usize,
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct LjTablesx8 {
    /// Non-solvent, non-solvent interactions. Upper triangle.
    pub std: Vec<(f32x8, f32x8)>,
    /// Water, non-solvent interactions.
    pub water_std: Vec<(f32x8, f32x8)>,
    pub n_std: [usize; 8],
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct LjTablesx16 {
    /// Non-solvent, non-solvent interactions. Upper triangle.
    pub std: Vec<(f32x16, f32x16)>,
    /// Water, non-solvent interactions.
    pub water_std: Vec<(f32x16, f32x16)>,
    pub n_std: [usize; 8],
}

// todo note: On large systems, this can have very high memory use. Consider
// todo setting up your table by atom type, instead of by atom, if that proves to be a problem.
impl LjTables {
    /// Create an indexed table, flattened.
    pub fn new(atoms: &[AtomDynamics]) -> Self {
        let n_std = atoms.len();

        if n_std == 0 {
            // Otherwise, we will get an out-of-bounds error when subtracting.
            return Default::default();
        }

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

        // One LJ pair per dynamic atom vs solvent O:
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
        waters: &'a [WaterMolOpc],
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
    /// True when this pair is a cross interaction between the alchemical molecule
    /// and the rest of the system, and therefore should be scaled by `(1 - lambda)`.
    /// False unless using an alchemical free-energy computation.
    pub alch_interaction: bool,
}

/// Add a force into the right accumulator (std or solvent). Static never accumulates.
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
/// Returns (forces on non-solvent atoms, forces on solvent molecules, virial, potential energy total,
/// potential energy between molecule pairs. (kcal/mol)
fn calc_force_cpu(
    pairs: &[NonBondedPair],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    mol_start_indices: &[usize],
    // For alchemical free-energy computation. This is CPU-only for now.
    // Ignored unless the BodyRef has alchemical = true.
    lambda_alch: f64,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    let n_std = atoms_std.len();
    let n_wat = water.len();
    let n_mol = mol_start_indices.len();

    // Map flattened atom index -> molecule index. We use this for assigning per-molecule-pair
    // potential energy.
    let find_mol_idx = |atom_idx: usize, starts: &[usize]| -> usize {
        starts
            .binary_search(&atom_idx)
            .unwrap_or_else(|pos| if pos == 0 { 0 } else { pos - 1 })
    };

    pairs
        .par_iter()
        .fold(
            || {
                (
                    // Sums as f64.
                    vec![Vec3F64::new_zero(); n_std],
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64,                                       // Virial sum
                    0.0_f64,                                       // Energy sum
                    vec![0.0_f64; mol_start_indices.len().pow(2)], // Per-pair
                    0.0_f64, // Unscaled alchemical interaction energy
                )
            },
            |(
                mut f_std,
                mut f_wat,
                mut virial,
                mut energy,
                mut energy_between_mols,
                mut alch_energy,
            ),
             p| {
                let a_t = p.tgt.get(atoms_std, water);
                let a_s = p.src.get(atoms_std, water);

                let interaction_scale = if p.alch_interaction {
                    (1.0 - lambda_alch).clamp(0.0, 1.0) as f32
                } else {
                    1.0
                };

                let (f, e_pair, raw_e_pair) = f_nonbonded_cpu(
                    &mut virial,
                    a_t,
                    a_s,
                    cell,
                    p.scale_14,
                    &p.lj_indices,
                    lj_tables,
                    p.calc_lj,
                    p.calc_coulomb,
                    overrides,
                    spme_alpha,
                    coulomb_cutoff,
                    lj_cutoff,
                    interaction_scale,
                );

                // Convert to f64 prior to summing.
                let f: Vec3F64 = f.into();
                add_to_sink(&mut f_std, &mut f_wat, p.tgt, f);
                if p.symmetric {
                    add_to_sink(&mut f_std, &mut f_wat, p.src, -f);
                }

                // We are not interested, in this point, at potential energy that only involves solvent atoms.
                // We skip solvent-solvent.
                let involves_std =
                    matches!(p.tgt, BodyRef::NonWater(_)) || matches!(p.src, BodyRef::NonWater(_));

                if involves_std {
                    energy += e_pair as f64;
                }

                if p.alch_interaction {
                    alch_energy += raw_e_pair as f64;
                }

                // todo: QC this!
                // Experimenting with per-mol potential energy.
                if let (BodyRef::NonWater(i_tgt), BodyRef::NonWater(i_src)) = (p.tgt, p.src) {
                    let m_t = find_mol_idx(i_tgt, mol_start_indices);
                    let m_s = find_mol_idx(i_src, mol_start_indices);
                    let idx_ts = m_t * n_mol + m_s;
                    energy_between_mols[idx_ts] += e_pair as f64;

                    // make it symmetric so callers don't have to
                    if m_t != m_s {
                        let idx_st = m_s * n_mol + m_t;
                        energy_between_mols[idx_st] += e_pair as f64;
                    }
                }

                (
                    f_std,
                    f_wat,
                    virial,
                    energy,
                    energy_between_mols,
                    alch_energy,
                )
            },
        )
        .reduce(
            || {
                (
                    vec![Vec3F64::new_zero(); n_std],
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64,
                    0.0_f64,
                    vec![0.0_f64; mol_start_indices.len().pow(2)],
                    0.0_f64,
                )
            },
            |(mut f_on_std, mut f_on_water, virial_a, e_a, mut em_a, alch_a),
             (db, wb, virial_b, e_b, em_b, alch_b)| {
                for i in 0..n_std {
                    f_on_std[i] += db[i];
                }
                for i in 0..n_wat {
                    f_on_water[i].f_o += wb[i].f_o;
                    f_on_water[i].f_m += wb[i].f_m;
                    f_on_water[i].f_h0 += wb[i].f_h0;
                    f_on_water[i].f_h1 += wb[i].f_h1;
                }

                // Merge per-molecule energy
                for i in 0..em_a.len() {
                    em_a[i] += em_b[i];
                }

                // (f_on_std, f_on_water, virial_a + virial_b, e_a + e_b)
                (
                    f_on_std,
                    f_on_water,
                    virial_a + virial_b,
                    e_a + e_b,
                    em_a,
                    alch_a + alch_b,
                )
            },
        )
}

// #[cfg(target_arch = "x86_64")]
// fn calc_force_x8(
//     pairs: &[NonBondedPair],
//     atoms_std: &[AtomDynamicsx8],
//     solvent: &[WaterMolx8],
//     cell: &SimBox,
//     lj_tables: &LjTablesx8,
// ) -> (Vec<Vec3x8>, Vec<ForcesOnWaterMol>, f64, f64) {
// }
//
// #[cfg(target_arch = "x86_64")]
// fn calc_force_x16(
//     pairs: &[NonBondedPair],
//     atoms_std: &[AtomDynamicsx16],
//     solvent: &[WaterMolx16],
//     cell: &SimBox,
//     lj_tables: &LjTablesx16,
// ) -> (Vec<Vec3x16>, Vec<ForcesOnWaterMol>, f64, f64) {
// }

impl MdState {
    /// Run the appropriate force-computation function to get force on non-solvent atoms, force
    /// on solvent atoms, and virial sum for the barostat. Uses GPU if available.
    ///
    /// Applies Coulomb and Van der Waals (Lennard-Jones) forces on non-solvent atoms, in place.
    /// We use the MD-standard [S]PME approach to handle approximated Coulomb forces. This function
    /// applies forces from non-solvent, and solvent sources.
    pub fn apply_nonbonded_forces(&mut self, dev: &ComputationDevice) {
        let (f_on_non_water, f_on_water, virial, energy, energy_between_mols, alch_energy) =
            match dev {
                ComputationDevice::Cpu => {
                    if is_x86_feature_detected!("avx512f") {
                        // calc_force_x16(
                        //     &self.nb_pairs,
                        //     &self.atoms_x16,
                        //     &self.solvent,
                        //     &self.cell,
                        //     &self.lj_tables,
                        // )
                    } else {
                        // calc_force_x8(
                        //     &self.nb_pairs,
                        //     &self.atoms_x8,
                        //     &self.solvent,
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
                        &self.cfg.overrides,
                        &self.mol_start_indices,
                        self.lambda_alch,
                        self.cfg.spme_alpha,
                        self.cfg.coulomb_cutoff,
                        self.cfg.lj_cutoff,
                    )
                }
                #[cfg(feature = "cuda")]
                ComputationDevice::Gpu(stream) => {
                    let (f_std, f_wat, virial, energy, energy_between_mols) = force_nonbonded_gpu(
                        stream,
                        self.gpu_kernel.as_ref().unwrap(),
                        self.gpu_kernel_zero_f32.as_ref().unwrap(),
                        self.gpu_kernel_zero_f64.as_ref().unwrap(),
                        &self.nb_pairs,
                        &self.atoms,
                        &self.water,
                        self.cell.extent,
                        self.forces_posits_gpu.as_mut().unwrap(),
                        self.per_neighbor_gpu.as_ref().unwrap(),
                        &self.cfg.overrides,
                    );
                    (f_std, f_wat, virial, energy, energy_between_mols, 0.0)
                }
            };

        // println!("\nF short-range: {}", f_on_non_water[0]);

        // `.into()` below converts accumulated forces to f32.
        for (i, tgt) in self.atoms.iter_mut().enumerate() {
            let f: Vec3 = f_on_non_water[i].into();
            tgt.force += f;
        }

        for (i, tgt) in self.water.iter_mut().enumerate() {
            let f = f_on_water[i];
            let f_0: Vec3 = f.f_o.into();
            let f_m: Vec3 = f.f_m.into();
            let f_h0: Vec3 = f.f_h0.into();
            let f_h1: Vec3 = f.f_h1.into();

            tgt.o.force += f_0;
            tgt.m.force += f_m;
            tgt.h0.force += f_h0;
            tgt.h1.force += f_h1;
        }

        self.potential_energy += energy;
        self.potential_energy_nonbonded += energy;

        self.barostat.virial.nonbonded_short_range += virial;

        // todo; not sure. For one mol, we get 1 and 0.
        if energy_between_mols.len() == self.potential_energy_between_mols.len() {
            for (i, e) in self.potential_energy_between_mols.iter_mut().enumerate() {
                *e += energy_between_mols[i];
            }
        }

        self.alch_interaction_energy += alch_energy;
    }

    /// [Re] initialize non-bonded interaction pairs between atoms. Do this whenever we rebuild neighbors.
    /// Build the neighbors set prior to running this.
    pub(crate) fn setup_pairs(&mut self) {
        let atoms = &self.atoms;
        let n_std = self.atoms.len();
        let n_water_mols = self.water.len();
        let atom_to_mol = atom_to_mol_indices(n_std, &self.mol_start_indices);
        let atom_to_mol = atom_to_mol.as_slice();
        let alch_mol_idx = self.alch_mol_idx;

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
                        if atoms[i_src].bonded_only || atoms[i_tgt].bonded_only {
                            return None;
                        }

                        let key = (i_tgt, i_src);
                        if exclusions.contains(&key) {
                            return None;
                        }
                        let scale_14 = scaled_set.contains(&key);
                        let alch_interaction = alch_mol_idx.is_some_and(|m_alch| {
                            let tgt_is_alch = atom_to_mol[i_tgt] == m_alch;
                            let src_is_alch = atom_to_mol[i_src] == m_alch;
                            tgt_is_alch ^ src_is_alch
                        });

                        Some(NonBondedPair {
                            tgt: BodyRef::NonWater(i_tgt),
                            src: BodyRef::NonWater(i_src),
                            scale_14,
                            lj_indices: LjTableIndices::StdStd(key),
                            calc_lj: true,
                            calc_coulomb: true,
                            symmetric: true,
                            alch_interaction,
                        })
                    })
            })
            .collect();

        // todo: Look at water_water
        // todo: In general, your static exclusions will get messed up with this logic.

        // Forces from solvent on non-solvent atoms, and vice-versa
        let mut pairs_std_water: Vec<_> = (0..n_std)
            .flat_map(|i_std| {
                self.neighbors_nb.std_water[i_std]
                    .iter()
                    .copied()
                    .flat_map(move |i_water| {
                        let alch_interaction =
                            alch_mol_idx.is_some_and(|m_alch| atom_to_mol[i_std] == m_alch);
                        sites.into_iter().map(move |site| NonBondedPair {
                            tgt: BodyRef::NonWater(i_std),
                            src: BodyRef::Water { mol: i_water, site },
                            scale_14: false,
                            lj_indices: LjTableIndices::StdWater(i_std),
                            calc_lj: site == WaterSite::O,
                            calc_coulomb: site != WaterSite::O,
                            symmetric: true,
                            alch_interaction,
                        })
                    })
            })
            .collect();

        // ------ Water on solvent ------
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
                            alch_interaction: false,
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

    /// We return the values for the case of not running SPME every step; store them for application
    /// in future steps.
    pub(crate) fn handle_spme_recip(&mut self, dev: &ComputationDevice) -> (Vec<Vec3>, f64, f64) {
        let (pos_all, q_all) = self.pack_pme_pos_q();
        let scale = (1.0 - self.lambda_alch).clamp(0.0, 1.0);
        let alch_atom_range = self.alchemical_atom_range();

        let (f_recip, e_recip, virial_from_kspace, alch_cross_energy) = match &mut self.pme_recip {
            Some(pme_recip) => {
                let mut eval = |charges: &[f32]| -> (Vec<Vec3>, f64, f64) {
                    match dev {
                        ComputationDevice::Cpu => {
                            let (forces, energy, virial) =
                                pme_recip.forces_and_virial(&pos_all, charges);
                            (forces, energy as f64, virial)
                        }
                        #[cfg(feature = "cuda")]
                        #[allow(unused)]
                        ComputationDevice::Gpu(stream) => {
                            #[cfg(not(any(feature = "cufft", feature = "vkfft")))]
                            let (f, e) = pme_recip.forces(&pos_all, charges);
                            #[cfg(any(feature = "cufft", feature = "vkfft"))]
                            let (f, e) = pme_recip.forces_gpu(stream, &pos_all, charges);

                            (f, e as f64, 0.0_f64)
                        }
                    }
                };

                if let Some((start, end)) = alch_atom_range {
                    let (f_full, e_full, virial_full) = eval(&q_all);

                    let mut q_env = q_all.clone();
                    for q in &mut q_env[start..end] {
                        *q = 0.0;
                    }
                    let (f_env, e_env, virial_env) = eval(&q_env);

                    let mut q_alch = vec![0.0; q_all.len()];
                    q_alch[start..end].copy_from_slice(&q_all[start..end]);
                    let (f_alch, e_alch, virial_alch) = eval(&q_alch);

                    let f_scaled = f_full
                        .iter()
                        .zip(&f_env)
                        .zip(&f_alch)
                        .map(|((f_full, f_env), f_alch)| {
                            let cross = *f_full - *f_env - *f_alch;
                            *f_env + *f_alch + cross * scale as f32
                        })
                        .collect();

                    let cross_energy = e_full - e_env - e_alch;
                    let cross_virial = virial_full - virial_env - virial_alch;
                    let e_scaled = e_env + e_alch + scale * cross_energy;
                    let virial_scaled = virial_env + virial_alch + scale * cross_virial;

                    (f_scaled, e_scaled, virial_scaled, cross_energy)
                } else {
                    let (f, e, virial) = eval(&q_all);
                    (f, e, virial, 0.0)
                }
            }
            None => {
                panic!("No PME recip available; not computing SPME recip.");
            }
        };

        // println!("F Recip: {:.6?}", f_recip[0]);

        self.potential_energy += e_recip as f64;
        self.potential_energy_nonbonded += e_recip as f64;
        self.alch_interaction_energy += alch_cross_energy;

        // Apply forces; virial comes from the analytical k-space formula, not r·F.
        self.unpack_apply_pme_forces(&f_recip);
        let mut virial_lr_recip = virial_from_kspace;

        // 1–4 Coulomb scaling correction (vacuum correction)
        for &(i, j) in &self.pairs_14_scaled {
            let diff = self
                .cell
                .min_image(self.atoms[i].posit - self.atoms[j].posit);

            let r = diff.magnitude();
            if r.abs() < 1e-6 {
                continue;
            }

            let dir = diff / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            // Vacuum Coulomb force (K=1 if charges are Amber-scaled)
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let f_vac = dir * (qi * qj * inv_r2);

            let df = f_vac * (SCALE_COUL_14 - 1.0);

            self.atoms[i].force += df;
            self.atoms[j].force -= df;

            virial_lr_recip += (dir * r).dot(df) as f64; // r·F
        }

        self.barostat.virial.nonbonded_long_range += virial_lr_recip;

        (f_recip, e_recip as f64, virial_lr_recip)
    }

    /// Gather all particles that contribute to PME (non-solvent atoms, solvent sites).
    /// Returns positions wrapped to the primary box, and their charges. We pack (and unpack)
    /// in a predictable way: non-solvent atoms, then solvent, with order as defined below.
    fn pack_pme_pos_q(&self) -> (Vec<Vec3>, Vec<f32>) {
        let n_std = self.atoms.len();
        let n_wat = self.water.len();

        let mut pos = Vec::with_capacity(n_std + 3 * n_wat);
        let mut q = Vec::with_capacity(pos.capacity());

        // Non-solvent atoms.
        for a in &self.atoms {
            pos.push(self.cell.wrap(a.posit)); // [0,L) per axis
            q.push(a.partial_charge); // already scaled to Amber units
        }

        // Water sites. We omit O, as it has no charge.
        for w in &self.water {
            pos.push(self.cell.wrap(w.m.posit));
            q.push(w.m.partial_charge);

            pos.push(self.cell.wrap(w.h0.posit));
            q.push(w.h0.partial_charge);

            pos.push(self.cell.wrap(w.h1.posit));
            q.push(w.h1.partial_charge);
        }

        (pos, q)
    }

    /// Apply PME reciprocal forces to atoms and water sites. In the same order as pack_pme_pos_q.
    /// Virial is computed analytically in the ewald library (forces_and_virial), not here.
    pub(crate) fn unpack_apply_pme_forces(&mut self, forces: &[Vec3]) {
        let water_start = self.atoms.len();

        for (i, f) in forces.iter().enumerate() {
            if i < water_start {
                self.atoms[i].force += *f;
            } else {
                let i_wat = i - water_start;
                let i_wat_mol = i_wat / 3;
                match i_wat % 3 {
                    0 => self.water[i_wat_mol].m.force += *f,
                    1 => self.water[i_wat_mol].h0.force += *f,
                    _ => self.water[i_wat_mol].h1.force += *f,
                }
            }
        }
    }

    /// Re-initializes the SPME based on sim box dimensions. Run this at init, and whenever you
    /// update the sim box. Sets FFT planner dimensions.
    pub(crate) fn regen_pme(&mut self, dev: &ComputationDevice) {
        let [lx, ly, lz] = self.cell.extent.to_arr();
        let l = (lx, ly, lz);
        let n = get_grid_n(l, self.cfg.spme_mesh_spacing);

        self.pme_recip = Some(match dev {
            ComputationDevice::Cpu => {
                #[cfg(any(feature = "vkfft", feature = "cufft"))]
                let v = PmeRecip::new(None, n, l, self.cfg.spme_alpha);
                #[cfg(not(any(feature = "vkfft", feature = "cufft")))]
                let v = PmeRecip::new(n, l, self.cfg.spme_alpha);

                v
            }
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu(stream) => {
                #[cfg(any(feature = "vkfft", feature = "cufft"))]
                let v = PmeRecip::new(Some(stream), n, l, self.cfg.spme_alpha);

                #[cfg(not(any(feature = "vkfft", feature = "cufft")))]
                let v = PmeRecip::new(n, l, self.cfg.spme_alpha);

                v
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
/// Lennard Jones and (short-range) Coulomb forces. Used by solvent and non-solvent.
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
    // These flags are for use with forces on solvent.
    calc_lj: bool,
    calc_coulomb: bool,
    overrides: &MdOverrides,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    interaction_scale: f32,
) -> (Vec3, f32, f32) {
    let diff = cell.min_image(tgt.posit - src.posit);

    // We compute these dist-related values once, and share them between
    // LJ and Coulomb.
    let dist_sq = diff.magnitude_squared();

    if dist_sq < 1e-12 {
        return (Vec3::new_zero(), 0., 0.);
    }

    let dist = dist_sq.sqrt();
    let inv_dist = 1.0 / dist;
    let dir = diff * inv_dist;

    let (f_lj, energy_lj) = if !calc_lj || dist > lj_cutoff || overrides.lj_disabled {
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
    let (mut f_coulomb, mut energy_coulomb) = if !calc_coulomb || overrides.coulomb_disabled {
        (Vec3::new_zero(), 0.)
    } else {
        force_coulomb_short_range(
            dir,
            dist,
            inv_dist,
            tgt.partial_charge,
            src.partial_charge,
            coulomb_cutoff,
            spme_alpha,
        )
    };

    // println!("F Short range: {}", f_coulomb);
    // println!("\nQ: {:?}, dist: {:?}, f: {:?}", tgt.partial_charge, dist, f_coulomb.x);

    // See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
    if scale14 {
        f_coulomb *= SCALE_COUL_14;
        energy_coulomb *= SCALE_COUL_14;
    }

    // println!("F coulomb (CPU): {f_coulomb} LJ: {f_lj}");

    let raw_force = f_lj + f_coulomb;
    let raw_energy = energy_lj + energy_coulomb;
    let force = raw_force * interaction_scale;
    let energy = raw_energy * interaction_scale;

    *virial_w += diff.dot(force) as f64;

    (force, energy, raw_energy)
}

fn atom_to_mol_indices(n_atoms: usize, mol_start_indices: &[usize]) -> Vec<usize> {
    let mut atom_to_mol = vec![0; n_atoms];

    for (mol_idx, &start) in mol_start_indices.iter().enumerate() {
        let end = mol_start_indices
            .get(mol_idx + 1)
            .copied()
            .unwrap_or(n_atoms);
        for atom_idx in start..end {
            atom_to_mol[atom_idx] = mol_idx;
        }
    }

    atom_to_mol
}

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
/// Note that this uses the traditional algorithm; not the Amber-specific version: We pre-set
/// atom-specific σ and ε to traditional versions on ingest, and when building solvent.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f32, f32) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
