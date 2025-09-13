#![allow(non_snake_case)]

//! See the [https://github.com/David-OConnor/dynamics/blob/main/README.md](Readme) for a general overview.
//!
//! This module contains a traditional molecular dynamics approach
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//! [A summary paper](https://arxiv.org/pdf/1401.1181)
//!
//! [Amber Force Fields reference](https://ambermd.org/AmberModels.php)
//! [Small molucules using GAFF2](https://ambermd.org/downloads/amber_geostd.tar.bz2)
//! [Amber RM 2025](https://ambermd.org/doc12/Amber25.pdf)
//!
//! To download .dat files (GAFF2), download Amber source (Option 2) [here](https://ambermd.org/GetAmber.php#ambertools).
//! Files are in dat -> leap -> parm
//!
//! Base units: Å, ps (10^-12), Dalton (AMU), native charge units (derive from other base units;
//! not a traditional named unit).
//!
//! We are using f64, and CPU-only for now, unless we confirm f32 will work.
//! Maybe a mixed approach: Coordinates, velocities, and forces in 32-bit; sensitive global
//! reductions (energy, virial, integration) in 64-bit.
//!
//! We use Verlet integration. todo: Velocity verlet? Other techniques that improve and build upon it?
//!
//! Amber: ff19SB for proteins, gaff2 for ligands. (Based on recs from https://ambermd.org/AmberModels.php).
//!
//! We use the term "Non-bonded" interactions to refer to Coulomb, and Lennard Interactions, the latter
//! of which is an approximation for Van der Waals force.
//!
//! ## A broad list of components of this simulation:
//! - Water: Rigid OPC water molecules that have mutual non-bonded interactions with dynamic atoms and water
//! - Thermostat/barostat, with a way to specify temp, pressure, water density
//! - OPC water model
//! - Cell wrapping
//! - Verlet integration (Water and non-water)
//! - Amber parameters for mass, partial charge, VdW (via LJ), dihedral/improper, angle, bond len
//! - Optimizations for Coulomb: Ewald/SPME.
//! - Optimizations for LJ: Dist cutoff for now.
//! - Amber 1-2, 1-3 exclusions, and 1-4 scaling of covalently-bonded atoms.
//! - Rayon parallelization of non-bonded forces
//! - WIP SIMD and CUDA parallelization of non-bonded forces, depending on hardware availability. todo
//! - A thermostat and barostat
//! - An energy-measuring system.
//!
//! --------
//! A timing test, using bond-stretching forces between two atoms only. Measure the period
//! of oscillation for these atom combinations, e.g. using custom Mol2 files.
//! c6-c6: 35fs (correct).   os-os: 47fs        nc-nc: 34fs        hw-hw: 9fs
//! Our measurements, 2025-08-04
//! c6-c6: 35fs    os-os: 31fs        nc-nc: 34fs (Correct)       hw-hw: 6fs
//!
//! --------
//!
//! We use traditional MD non-bonded terms to maintain geometry: Bond length, valence angle between
//! 3 bonded atoms, dihedral angle between 4 bonded atoms (linear), and improper dihedral angle between
//! each hub and 3 spokes. (E.g. at ring intersections). We also apply Coulomb force between atom-centered
//! partial charges, and Lennard Jones potentials to simulate Van der Waals forces. These use spring-like
//! forces to retain most geometry, while allowing for flexibility.
//!
//! We use the OPC water model. (See `water_opc.rs`). For both maintaining the geometry of each water
//! molecule, and for maintaining Hydrogen atom positions, we do not apply typical non-bonded interactions:
//! We use SHAKE + RATTLE algorithms for these. In the case of water, it's required for OPC compliance.
//! For H, it allows us to maintain integrator stability with a greater timestep, e.g. 2fs instead of 1fs.
//!
//! On f32 vs f64 floating point precision: f32 may be good enough for most things, and typical MD packages
//! use mixed precision. Long-range electrostatics are a good candidate for using f64. Or, very long
//! runs.
//!
//! Note on performance: It appears that non-bonded forces dominate computation time. This is my observation,
//! and it's confirmed by an LLM. Both LJ and Coulomb take up most of the time; bonded forces
//! are comparatively insignificant. Building neighbor lists are also significant. These are the areas
//! we focus on for parallel computation (Thread pools, SIMD, CUDA)

// todo: You should keep more data on the GPU betwween time steps, instead of passing back and
// todo forth each time. If practical.

extern crate core;

mod add_hydrogens;
mod ambient;
mod bonded;
mod bonded_forces;
mod forces;
pub mod integrate;
mod neighbors;
mod non_bonded;
pub mod params;
mod prep;
pub mod snapshot;
mod util;
mod water_init;
mod water_opc;
mod water_settle;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::HashSet,
    fmt,
    fmt::{Display, Formatter},
    path::Path,
    time::Instant,
};

pub use add_hydrogens::{add_hydrogens_2::Dihedral, populate_hydrogens_dihedrals};
use ambient::SimBox;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, BondGeneric, md_params::ForceFieldParams, mmcif::MmCif, mol2::Mol2};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaModule, CudaStream};
use ewald::{PmeRecip, ewald_comp_force};
pub use integrate::Integrator;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{Vec3x4, f64x4};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::Element;
use neighbors::NeighborsNb;
pub use params::{ProtFFTypeChargeMap, ProtFfMap};
pub use prep::{HydrogenConstraint, merge_params};
use rand::Rng;
use rand_distr::StandardNormal;
pub use util::{load_snapshots, save_snapshots};
pub use water_opc::ForcesOnWaterMol;

use crate::{
    ambient::BerendsenBarostat,
    non_bonded::{
        CHARGE_UNIT_SCALER, EWALD_ALPHA, LjTableIndices, LjTables, SCALE_COUL_14, SPME_N,
    },
    params::{FfParamSet, ForceFieldParamsIndexed},
    snapshot::{FILE_SAVE_INTERVAL, SaveType, Snapshot, SnapshotHandler, append_dcd},
    util::build_adjacency_list,
    water_init::make_water_mols,
    water_opc::WaterMol,
};

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
pub const PTX: &str = include_str!("../dynamics.ptx");

/// Convert convert kcal mol⁻¹ Å⁻¹ (Values in the Amber parameter files) to amu Å ps⁻². Multiply all bonded
/// accelerations by this. TODO: we are currently multiplying *all* accelerations by this.
const ACCEL_CONVERSION: f64 = 418.4;
const ACCEL_CONVERSION_F32: f32 = 418.4;
const ACCEL_CONVERSION_INV: f64 = 1. / ACCEL_CONVERSION;
const ACCEL_CONVERSION_INV_F32: f32 = 1. / ACCEL_CONVERSION_F32;

// For assigning velocities from temperature, and other thermostat/barostat use.
const KB: f32 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹ (Amber-style units)

// Boltzmann constant in (amu · Å^2 / ps^2) K⁻¹
const KB_A2_PS2_PER_K_PER_AMU: f32 = 0.831_446_26;

// SHAKE tolerances for fixed hydrogens. These SHAKE constraints are for fixed hydrogens.
// The tolerance controls how close we get
// to the target value; lower values are more precise, but require more iterations. `SHAKE_MAX_ITER`
// constrains the number of iterations.
const SHAKE_TOL: f32 = 1.0e-4; // Å
const SHAKE_MAX_IT: usize = 100;

// Every this many steps, re-
const CENTER_SIMBOX_RATIO: usize = 30;

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu((Arc<CudaStream>, Arc<CudaModule>)),
}

/// Represents problems loading parameters. For example, if an atom is missing a force field type
/// or partial charge, or has a force field type that hasn't been loaded.
#[derive(Clone, Debug)]
pub struct ParamError {
    pub descrip: String,
}

impl ParamError {
    pub fn new(descrip: &str) -> Self {
        Self {
            descrip: descrip.to_owned(),
        }
    }
}

/// This is used to assign the correct force field parameters to a molecule.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FfMolType {
    /// Protein or other construct of amino acids
    Peptide,
    /// E.g. a ligand.
    SmallOrganic,
    Dna,
    Rna,
    Lipid,
    Carbohydrate,
}

/// Packages information required to perform dynamics on a Molecule. This is used to initialize
/// the simulation with atoms and related; one or more of these is passed at init.
#[derive(Clone, Debug)]
pub struct MolDynamics<'a> {
    pub ff_mol_type: FfMolType,
    /// These must hold force field type and partial charge.
    pub atoms: &'a [AtomGeneric],
    /// Separate from `atoms`; this may be more convenient than mutating the atoms
    /// as they may move! If None, we use the positions stored in the atoms.
    pub atom_posits: Option<&'a [Vec3]>,
    /// Not required if static.
    pub bonds: &'a [BondGeneric],
    /// If None, will be generated automatically from atoms and bonds. Use this
    /// if you wish to cache.
    pub adjacency_list: Option<&'a [Vec<usize>]>,
    /// If true, the atoms in the molecule don't move, but exert LJ and Coulomb forces
    /// on other atoms in the system.
    pub static_: bool,
    /// If present, any values here override molecule-type general parameters.
    pub mol_specific_params: Option<&'a ForceFieldParams>,
}

// impl MolDynamics<'_> {
//     /// Load an Amber Geostd molecule from an online database.
//     /// todo: Wonky due to use of refs.
//     pub fn from_amber_geostd(ident: &str) -> io::Result<Self> {
//         let data = bio_apis::amber_geostd::load_mol_files("CPB").map_err(|e| io::Error::new(io::ErrorKind::Other, "Error loading data"))?;
//         let mol = Mol2::new(&data.mol2)?;
//         let params = ForceFieldParams::from_frcmod(&data.frcmod.unwrap())?;
//
//         Ok(Self {
//             ff_mol_type: FfMolType::SmallOrganic,
//             atoms: &mol.atoms,
//             atom_posits: None,
//             bonds: &mol.bonds,
//             adjacency_list: None,
//             static_: false,
//             mol_specific_params: Some(&params)
//         })
//     }
// }

/// A trimmed-down atom for use with molecular dynamics. Contains parameters for single-atom,
/// but we use ParametersIndex for multi-atom parameters.
#[derive(Clone, Debug, Default)]
pub struct AtomDynamics {
    pub serial_number: u32,
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    /// and VDW) only.
    pub static_: bool,
    pub force_field_type: String,
    pub element: Element,
    // pub name: String,
    pub posit: Vec3F32,
    /// Å / ps
    pub vel: Vec3F32,
    /// Å / ps²
    pub accel: Vec3F32,
    /// Daltons
    /// todo: Move these 4 out of this to save memory; use from the params struct directly.
    pub mass: f32,
    /// Amber charge units. This is not the elementary charge units found in amino19.lib and gaff2.dat;
    /// it's scaled by a constant.
    pub partial_charge: f32,
    /// Å
    pub lj_sigma: f32,
    /// kcal/mol
    pub lj_eps: f32,
}

impl Display for AtomDynamics {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Atom {}: {}, {}. ff: {}, q: {}",
            self.serial_number,
            self.element.to_letter(),
            self.posit,
            self.force_field_type,
            self.partial_charge,
        )?;

        if self.static_ {
            write!(f, ", Static")?;
        }

        Ok(())
    }
}

impl AtomDynamics {
    pub fn new(
        atom: &AtomGeneric,
        atom_posits: &[Vec3F32],
        i: usize,
        static_: bool,
    ) -> Result<Self, ParamError> {
        let ff_type = match &atom.force_field_type {
            Some(ff_type) => ff_type.clone(),
            None => {
                return Err(ParamError::new(&format!(
                    "Atom missing FF type; can't run dynamics: {:?}",
                    atom
                )));
            }
        };

        Ok(Self {
            serial_number: atom.serial_number,
            static_,
            element: atom.element,
            // name: atom.type_in_res.clone().unwrap_or_default(),
            posit: atom_posits[i],
            force_field_type: ff_type,
            ..Default::default()
        })
    }

    /// Populate atom-specific parameters.
    /// E.g. we use this workflow if creating the atoms prior to the indexed FF.
    pub(crate) fn assign_data_from_params(
        &mut self,
        ff_params: &ForceFieldParamsIndexed,
        // ff_params: &ForceFieldParams,
        i: usize,
    ) {
        // mass: ff_params.mass.get(&i).unwrap().mass as f64,
        self.mass = ff_params.mass[&i].mass;
        // We get partial charge for ligands from (e.g. Amber-provided) Mol files, so we load it from the atom, vice
        // the loaded FF params. They are not in the dat or frcmod files that angle, bond-length etc params are from.
        self.partial_charge = CHARGE_UNIT_SCALER * self.partial_charge;
        // lj_sigma: ff_params.lennard_jones.get(&i).unwrap().sigma as f64,
        self.lj_sigma = ff_params.lennard_jones[&i].sigma;
        // lj_eps: ff_params.lennard_jones.get(&i).unwrap().eps as f64,
        self.lj_eps = ff_params.lennard_jones[&i].eps;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx4 {
    // pub posit: Vec3x8,
    // pub vel: Vec3x8,
    // pub accel: Vec3x8,
    // pub mass: f32x8,
    pub posit: Vec3x4,
    pub vel: Vec3x4,
    pub accel: Vec3x4,
    pub mass: f64x4,
    pub element: [Element; 4],
}

// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// impl AtomDynamicsx4 {
//     pub fn from_array(bodies: [AtomDynamics; 4]) -> Self {
//         let mut posits = [Vec3F32::new_zero(); 4];
//         let mut vels = [Vec3F32::new_zero(); 4];
//         let mut accels = [Vec3F32::new_zero(); 4];
//         let mut masses = [0.0; 4];
//         // Replace `Element::H` (for example) with some valid default for your `Element` type:
//         let mut elements = [Element::Hydrogen; 4];
//
//         for (i, body) in bodies.into_iter().enumerate() {
//             posits[i] = body.posit;
//             vels[i] = body.vel;
//             accels[i] = body.accel;
//             masses[i] = body.mass;
//             elements[i] = body.element;
//         }
//
//         Self {
//             posit: Vec3x4::from_array(posits),
//             vel: Vec3x4::from_array(vels),
//             accel: Vec3x4::from_array(accels),
//             mass: f64x4::from_array(masses),
//             element: elements,
//         }
//     }
// }

// todo: FIgure out how to apply this to python.
/// Note: The shortest edge should be > 2(r_cutoff + r_skin), to prevent atoms
/// from interacting with their own image in the real-space component.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone)]
pub enum SimBoxInit {
    /// Distance in Å from the edge to the molecule, at init.
    Pad(f32),
    /// Coordinate boundaries, at opposite corners
    Fixed((Vec3F32, Vec3F32)),
}

impl Default for SimBoxInit {
    fn default() -> Self {
        Self::Pad(10.)
    }
}

#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone)]
pub struct MdConfig {
    /// Defaults to Velocity Verlet.
    pub integrator: Integrator,
    /// If enabled, zero the drift in center of mass of the system.
    /// todo: Implement
    pub zero_com_drift: bool,
    /// Kelvin. Defaults to 310 K.
    pub temp_target: f32,
    /// Bar (Pa/100). Defaults to 1 bar.
    pub pressure_target: f32,
    /// Allows constraining Hydrogens to be rigid with their bonded atom, using SHAKE and RATTLE
    /// algorithms. This allows for higher time steps.
    pub hydrogen_constraint: HydrogenConstraint,
    pub snapshot_handlers: Vec<SnapshotHandler>,
    pub sim_box: SimBoxInit,
}

impl Default for MdConfig {
    fn default() -> Self {
        Self {
            integrator: Default::default(),
            zero_com_drift: false, // todo: True?
            temp_target: 310.,
            pressure_target: 1.,
            hydrogen_constraint: Default::default(),
            // snapshot_ratio_memory: 1,
            // snapshot_ratio_file: 2,
            // snapshot_path: None,
            snapshot_handlers: vec![SnapshotHandler {
                save_type: SaveType::Memory,
                ratio: 1,
            }],
            sim_box: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct MdState {
    // todo: Update how we handle mode A/R.
    // todo: You need to rework this state in light of arbitrary mol count.
    pub cfg: MdConfig,
    pub atoms: Vec<AtomDynamics>,
    pub adjacency_list: Vec<Vec<usize>>,
    // h_constraints: Vec<HydrogenConstraintInner>,
    // /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    // /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    // /// and VDW) only.
    // pub atoms_static: Vec<AtomDynamics>,
    // todo: Make this a vec. For each dynamic atom.
    // todo: We don't need it for static, as they use partial charge and LJ data, which
    // todo are assigned to each atom.
    pub force_field_params: ForceFieldParamsIndexed,
    // /// `lj_lut`, `lj_sigma`, and `lj_eps` are Lennard Jones parameters. Flat here, with outer loop receptor.
    // /// Flattened. Separate single-value array facilitate use in CUDA and SIMD, vice a tuple.
    // pub lj_sigma: Vec<f64>,
    // pub lj_eps: Vec<f64>,
    // todo: Implment these SIMD variants A/R, bearing in mind the caveat about our built-in ones vs
    // todo ones loaded from [e.g. Amber] files.
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_sigma_x8: Vec<f64x4>,
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_eps_x8: Vec<f64x4>,
    /// Current simulation time, in picoseconds.
    pub time: f64,
    pub step_count: usize, // increments.
    /// These are the snapshots we keep in memory, accumulating.
    pub snapshots: Vec<Snapshot>,
    pub cell: SimBox,
    pub neighbors_nb: NeighborsNb,
    // max_disp_sq: f64,           // track atom displacements²
    /// K
    barostat: BerendsenBarostat,
    /// Exclusions of non-bonded forces for atoms connected by 1, or 2 covalent bonds.
    /// I can't find this in the RM, but ChatGPT is confident of it, and references an Amber file
    /// called 'prmtop', which I can't find. Fishy, but we're going with it.
    pairs_excluded_12_13: HashSet<(usize, usize)>,
    /// See Amber RM, sectcion 15, "1-4 Non-Bonded Interaction Scaling"
    /// These are indices of atoms separated by three consecutive bonds
    pairs_14_scaled: HashSet<(usize, usize)>,
    water: Vec<WaterMol>,
    lj_tables: LjTables,
    // todo: Hmm... Is this DRY with forces_on_water? Investigate.
    pub water_pme_sites_forces: Vec<[Vec3; 3]>, // todo: A/R
    pme_recip: Option<PmeRecip>,
    /// kcal/mol
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    /// Every so many snapshots, write these to file, then clear from memory.
    snapshot_queue_for_file: Vec<Snapshot>,
}

impl fmt::Display for MdState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MdState. # Snapshots: {}. # steps: {}  Current time: {}. # of dynamic atoms: {}. # Water mols: {}",
            self.snapshots.len(),
            self.step_count,
            self.time,
            self.atoms.len(),
            self.water.len()
        )
    }
}

impl MdState {
    pub fn new(
        cfg: &MdConfig,
        mols: &[MolDynamics],
        param_set: &FfParamSet,
    ) -> Result<Self, ParamError> {
        // todo: QC how you handle hydrogen_md_type.

        if mols.iter().filter(|m| !m.static_).count() != 1 {
            return Err(ParamError::new(
                &"We currently only support exactly 1 dynamic molecule. Sorry!",
            ));
        }

        // We create a flattened atom list, which simplifies our workflow, and is conducive to
        // parallel operations.
        // These Vecs all share indices, and all include all molecules.
        let mut atoms_md = Vec::new();
        let mut adjacency_list = Vec::new();

        // let mut atoms_md_static: Vec<AtomDynamics> = Vec::new();

        // todo: Sort this out. One set for all molecules now. Assumes unique FF names between
        // todo differenet set types.
        // let mut ff_params: ForceFieldParamsIndexed = Default::default();

        // We combine all molecule general and specific params into this set, then
        // create Indexed params from it.
        let mut params = ForceFieldParams::default();

        // todo: Make sure you don't use atom SN anywhere in the MD pipeline.
        // let mut last_mol_sn = 0;

        // Used for updating indices for tracking purposes.
        let mut total_atom_count = 0;

        for (i, mol) in mols.iter().enumerate() {
            // Filter out hetero atoms in proteins. These are often example ligands that we do
            // not wish to model.
            // We must perform this filter prior to most of the other steps in this function.
            let atoms: Vec<AtomGeneric> = match mol.ff_mol_type {
                FfMolType::Peptide => mol
                    .atoms
                    .into_iter()
                    .filter(|a| !a.hetero)
                    .map(|a| a.clone())
                    .collect(),
                _ => mol.atoms.to_vec(),
            };

            // If the atoms list isn't already filtered by Hetero, and a manual
            // adjacency list or atom posits is passed, this will get screwed up.
            if mol.ff_mol_type == FfMolType::Peptide
                && atoms.len() != mol.atoms.len()
                && (mol.adjacency_list.is_some() || mol.atom_posits.is_some())
            {
                return Err(ParamError::new(
                    "Unable to perform MD on this peptide: If passing atom positions or an adjacency list,\
                 you must already have filtered out hetero atoms. We found one or more hetero atoms in the input.",
                ));
            }

            {
                let params_general = match mol.ff_mol_type {
                    FfMolType::Peptide => &param_set.peptide,
                    FfMolType::SmallOrganic => &param_set.small_mol,
                    FfMolType::Dna => &param_set.dna,
                    FfMolType::Rna => &param_set.rna,
                    FfMolType::Lipid => &param_set.lipids,
                    FfMolType::Carbohydrate => &param_set.carbohydrates,
                };

                let Some(params_general) = params_general else {
                    return Err(ParamError::new(&format!(
                        "Missing general parameters for {:?}",
                        mol.ff_mol_type
                    )));
                };

                // todo: If there are multiple molecules of a given type, this is unnecessary.
                // todo: Make sure overrides from one individual molecule don't affect others, todo,
                // todo and don't affect general params.
                params = merge_params(&params, &params_general);

                if let Some(p) = mol.mol_specific_params {
                    params = merge_params(&params, &p);
                }
            }

            let mut p: Vec<Vec3F32> = Vec::new(); // to store the ref.
            let atom_posits = match mol.atom_posits {
                Some(a) => {
                    p = a.iter().map(|p| (*p).into()).collect();
                    &p
                }
                None => {
                    p = mol.atoms.iter().map(|a| a.posit.into()).collect();
                    &p
                }
            };

            for (i, atom) in atoms.iter().enumerate() {
                // atom.serial_number += last_mol_sn;
                atoms_md.push(AtomDynamics::new(
                    &atom,
                    atom_posits,
                    // &params,
                    i,
                    mol.static_,
                )?);
            }

            // Use the included adjacency list if available. If not, construct it.
            let adjacency_list_ = match mol.adjacency_list {
                Some(a) => a,
                None => &build_adjacency_list(&atoms, mol.bonds)?,
            };

            for aj in adjacency_list_ {
                let mut updated = aj.clone();
                for neighbor in &mut updated {
                    *neighbor += total_atom_count;
                }

                adjacency_list.push(updated);
            }

            // for constraint in h_constraints_ {
            //
            // }

            // atoms_md.extend(atoms_md);

            // if mol.static_ {
            //     // atoms_md_static.extend(atoms_md);
            // } else {
            //     // Set up Indexed params. Merges general with atom-specific if available.
            //     // Not required for static atoms. (Only applies to bonded forces.)
            //     // ff_params = params;
            //     adjacency_list = adjacency_list_.to_vec();
            // }

            // h_constraints.push(h_constraints_);

            total_atom_count += atoms.len();
        }

        let force_field_params = ForceFieldParamsIndexed::new(
            &params,
            // mol.mol_specific_params,
            &atoms_md,
            // mol.bonds,
            &adjacency_list,
            // &mut h_constraints,
            cfg.hydrogen_constraint,
        )?;

        // Assign mass, LJ params, etc.
        for (i, atom) in atoms_md.iter_mut().enumerate() {
            atom.assign_data_from_params(&force_field_params, i);
        }

        // let cell = SimBox::new_padded(&atoms_dy);
        let cell = SimBox::new(&atoms_md, &cfg.sim_box);

        let mut result = Self {
            cfg: cfg.clone(),
            atoms: atoms_md,
            adjacency_list: adjacency_list.to_vec(),
            // h_constraints,
            // atoms_static: atoms_md_static,
            cell,
            pairs_excluded_12_13: HashSet::new(),
            pairs_14_scaled: HashSet::new(),
            force_field_params,
            ..Default::default()
        };

        result.barostat.pressure_target = cfg.pressure_target as f64;

        result.water = make_water_mols(
            &result.cell,
            cfg.temp_target,
            &result.atoms,
            // &result.atoms_static,
        );
        result.water_pme_sites_forces = vec![[Vec3::new_zero(); 3]; result.water.len()];

        result.setup_nonbonded_exclusion_scale_flags();
        result.init_neighbors();
        // Initializes the FFT planner[s], among other things.
        result.regen_pme();

        // Set up our LJ cache.
        result.lj_tables = LjTables::new(&result.atoms, &result.water);

        Ok(result)
    }

    /// Reset acceleration and virial pair. Do this each step after the first half-step and drift, and
    /// shaking the fixed hydrogens.
    /// We must reset the virial pair prior to accumulating it, which we do when calculating non-bonded
    /// forces. Also reset forces on water.
    fn reset_accels(&mut self) {
        for a in &mut self.atoms {
            a.accel = Vec3F32::new_zero();
        }
        for mol in &mut self.water {
            mol.o.accel = Vec3F32::new_zero();
            mol.m.accel = Vec3F32::new_zero();
            mol.h0.accel = Vec3F32::new_zero();
            mol.h1.accel = Vec3F32::new_zero();
        }

        self.barostat.virial_pair_kcal = 0.0;
        self.potential_energy = 0.;
    }

    fn apply_all_forces(&mut self, dev: &ComputationDevice) {
        // Bonded forces
        let mut start = Instant::now();
        self.apply_bond_stretching_forces();

        if self.step_count == 0 {
            let elapsed = start.elapsed();
            println!("Bond stretching time: {:?} μs", elapsed.as_micros());
        }

        if self.step_count == 0 {
            start = Instant::now();
        }
        self.apply_angle_bending_forces();

        if self.step_count == 0 {
            let elapsed = start.elapsed();
            println!("Angle bending time: {:?} μs", elapsed.as_micros());
        }

        if self.step_count == 0 {
            start = Instant::now();
        }

        self.apply_dihedral_forces(false);
        if self.step_count == 0 {
            let elapsed = start.elapsed();
            println!("Dihedral: {:?} μs", elapsed.as_micros());
        }

        if self.step_count == 0 {
            start = Instant::now();
        }

        self.apply_dihedral_forces(true);
        if self.step_count == 0 {
            let elapsed = start.elapsed();
            println!("Improper time: {:?} μs", elapsed.as_micros());
        }

        if self.step_count == 0 {
            start = Instant::now();
        }

        // Note: Non-bonded takes the vast majority of time.
        self.apply_nonbonded_forces(dev);
        if self.step_count == 0 {
            let elapsed = start.elapsed();
            println!("Non-bonded time: {:?} μs", elapsed.as_micros());
        }
    }

    /// Relaxes the molecules. Use this at the start of the simulation to control kinetic energy that
    /// arrises from differences between atom positions, and bonded parameters.
    fn minimize_energy(&mut self, dev: &ComputationDevice) {}

    // todo: For calling by user at the end (temp), don't force it to append the path.
    //todo: DRY with in the main step path (Doesn't call this) to avoid a dbl borrow.
    pub fn save_snapshots_to_file(&mut self, path: &Path) {
        if self.step_count % FILE_SAVE_INTERVAL == 0 {
            if let Err(e) = append_dcd(&self.snapshot_queue_for_file, &path) {
                eprintln!("Error saving snapshot as DCD: {e:?}");
            }
            self.snapshot_queue_for_file = Vec::new();
        }
    }

    /// Note: This is currently only for the dynamic atoms; does not take water kinetic energy into account.
    fn current_kinetic_energy(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 0.5 * (a.mass * a.vel.magnitude_squared()) as f64)
            .sum()
    }

    fn take_snapshot(&self) -> Snapshot {
        let mut water_o_posits = Vec::with_capacity(self.water.len());
        let mut water_h0_posits = Vec::with_capacity(self.water.len());
        let mut water_h1_posits = Vec::with_capacity(self.water.len());
        let mut water_velocities = Vec::with_capacity(self.water.len());

        for water in &self.water {
            water_o_posits.push(water.o.posit);
            water_h0_posits.push(water.h0.posit);
            water_h1_posits.push(water.h1.posit);
            water_velocities.push(water.o.vel); // Can be from any atom; they should be the same.
        }

        Snapshot {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            water_velocities,
            energy_kinetic: self.kinetic_energy as f32,
            energy_potential: self.potential_energy as f32,
        }
    }
}

#[inline]
/// Mutable aliasing helpers.
pub(crate) fn split2_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);

    let (low, high) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = v.split_at_mut(high);
    (&mut left[low], &mut right[0])
}

#[inline]
fn split3_mut<T>(v: &mut [T], i: usize, j: usize, k: usize) -> (&mut T, &mut T, &mut T) {
    let len = v.len();
    assert!(i < len && j < len && k < len, "index out of bounds");
    assert!(i != j && i != k && j != k, "indices must be distinct");

    // SAFETY: we just asserted that 0 <= i,j,k < v.len() and that they're all different.
    let ptr = v.as_mut_ptr();
    unsafe {
        let a = &mut *ptr.add(i);
        let b = &mut *ptr.add(j);
        let c = &mut *ptr.add(k);
        (a, b, c)
    }
}

#[inline]
pub(crate) fn split4_mut<T>(
    slice: &mut [T],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> (&mut T, &mut T, &mut T, &mut T) {
    // Safety gates
    let len = slice.len();
    assert!(
        i0 < len && i1 < len && i2 < len && i3 < len,
        "index out of bounds"
    );
    assert!(
        i0 != i1 && i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3 && i2 != i3,
        "indices must be pair-wise distinct"
    );

    unsafe {
        let base = slice.as_mut_ptr();
        (
            &mut *base.add(i0),
            &mut *base.add(i1),
            &mut *base.add(i2),
            &mut *base.add(i3),
        )
    }
}

// todo: Move this somewhere apt
#[derive(Clone, Copy, Debug)]
pub enum PMEIndex {
    // Dynamic atoms (protein, ligand, ions, etc.)
    Dyn(usize),

    // Water sites (by molecule index)
    WatO(usize),
    WatM(usize),
    WatH0(usize),
    WatH1(usize),

    // Static atoms (included in the field, but you won't update their accel)
    Static(usize),
}
