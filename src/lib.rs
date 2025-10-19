#![allow(non_snake_case)]

//! See the [https://github.com/David-OConnor/dynamics/blob/main/README.md](Readme) for a general overview.
//! The textual information here is informal, and aimed at code maintenance; not library use.
//!
//! This module contains a traditional molecular dynamics approach.
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
#[cfg(target_arch = "x86_64")]
mod simd;
pub mod snapshot;
mod util;
mod water_init;
mod water_opc;
mod water_settle;

mod com_zero;
#[cfg(feature = "cuda")]
mod gpu_interface;
pub mod minimize_energy;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::HashSet,
    fmt,
    fmt::{Display, Formatter},
    io,
    path::Path,
    time::Instant,
};

pub use add_hydrogens::{
    add_hydrogens_2::Dihedral,
    bond_vecs::{find_planar_posit, find_tetra_posit_final, find_tetra_posits},
    populate_hydrogens_dihedrals,
};
use ambient::SimBox;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, BondGeneric, Sdf, md_params::ForceFieldParams, mol2::Mol2};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
use ewald::PmeRecip;
pub use integrate::Integrator;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use na_seq::Element;
use neighbors::NeighborsNb;
pub use prep::{HydrogenConstraint, merge_params};
use rand::Rng;
pub use util::{load_snapshots, save_snapshots};
pub use water_opc::ForcesOnWaterMol;

#[cfg(feature = "cuda")]
use crate::gpu_interface::{ForcesPositsGpu, PerNeighborGpu};
use crate::{
    ambient::BerendsenBarostat,
    non_bonded::{CHARGE_UNIT_SCALER, EWALD_ALPHA, LONG_RANGE_CUTOFF, LjTables, NonBondedPair},
    params::{FfParamSet, ForceFieldParamsIndexed},
    snapshot::{FILE_SAVE_INTERVAL, SaveType, Snapshot, SnapshotHandler, append_dcd},
    util::{ComputationTime, ComputationTimeSums, build_adjacency_list},
    water_init::make_water_mols,
    water_opc::WaterMol,
};

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../dynamics.ptx");

/// Convert convert kcal mol⁻¹ Å⁻¹ (Values in the Amber parameter files) to amu Å ps⁻². Multiply all
/// accelerations by this. (Bonded, and nonbonded)
const ACCEL_CONVERSION: f32 = 418.4;
const ACCEL_CONVERSION_INV: f32 = 1. / ACCEL_CONVERSION;

// SHAKE tolerances for fixed hydrogens. These SHAKE constraints are for fixed hydrogens.
// The tolerance controls how close we get
// to the target value; lower values are more precise, but require more iterations. `SHAKE_MAX_ITER`
// constrains the number of iterations.
const SHAKE_TOL: f32 = 1.0e-4; // Å
const SHAKE_MAX_IT: usize = 100;

// Every this many steps, re-
const CENTER_SIMBOX_RATIO: usize = 30;

// Run SPME once every these steps. It's the slowest computation, and is comparatively
// smooth over time compared to Coulomb and LJ.
const SPME_RATIO: usize = 2;

// todo: This may not be necessary, other than having it be a multiple of SPME_RATIO.
// todo: This is because the recording is very fast. (ns order)
// Log computation time every this many steps. (Except for neighbor rebuild)
const COMPUTATION_TIME_RATIO: usize = 20;

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu(Arc<CudaStream>),
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
pub struct MolDynamics {
    pub ff_mol_type: FfMolType,
    /// These must hold force field type and partial charge.
    pub atoms: Vec<AtomGeneric>,
    /// Separate from `atoms`; this may be more convenient than mutating the atoms
    /// as they may move! If None, we use the positions stored in the atoms.
    pub atom_posits: Option<Vec<Vec3F64>>,
    /// Not required if static.
    pub bonds: Vec<BondGeneric>,
    /// A fast lookup for finding atoms, by index, covalently bonded to each atom.
    /// If None, will be generated automatically from atoms and bonds. Use this
    /// if you wish to cache.
    pub adjacency_list: Option<Vec<Vec<usize>>>,
    /// If true, the atoms in the molecule don't move, but exert LJ and Coulomb forces
    /// on other atoms in the system.
    pub static_: bool,
    /// If present, any values here override molecule-type general parameters.
    pub mol_specific_params: Option<ForceFieldParams>,
    /// todo experimentin
    /// If true, this atom exerts and experiences non-bonded forces only.
    /// This may be useful for protein atoms that aren't near a docking site.
    pub bonded_only: bool,
}

impl MolDynamics {
    // todo: from_mmcif?

    /// Load a molecule from a Mol2 file. Includes optional molecule-specific pararmeters.
    /// To work directly, this assumes that forcefield names, and partial charge are present
    /// in the `Mol2` struct for all atoms.
    ///
    /// You may wish to modify the `atom_posits` field after to position this relative to
    /// other molecules.
    pub fn from_mol2(mol: &Mol2, mol_specific_params: Option<ForceFieldParams>) -> Self {
        Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms.clone(),
            atom_posits: None,
            bonds: mol.bonds.clone(),
            adjacency_list: None,
            static_: false,
            mol_specific_params,
            bonded_only: false,
        }
    }

    /// Load a molecule from a SDF file. Includes optional molecule-specific pararmeters.
    /// To work directly, this assumes that forcefield names, and partial charge are present
    /// in the `Mol2` struct for all atoms. Note that these are not present in
    /// SDF files that come from most online databases.
    ///
    /// You may wish to modify the `atom_posits` field after to position this relative to
    /// other molecules.
    pub fn from_sdf(mol: &Sdf, mol_specific_params: Option<ForceFieldParams>) -> Self {
        Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms.clone(),
            atom_posits: None,
            bonds: mol.bonds.clone(),
            adjacency_list: None,
            static_: false,
            mol_specific_params,
            bonded_only: false,
        }
    }

    /// Load an Amber Geostd molecule from an online database, from its unique identifier. This
    /// includes molecule-specific parameters.
    ///
    /// You may wish to modify the `atom_posits` field after to position this relative to
    /// other molecules.
    pub fn from_amber_geostd(ident: &str) -> io::Result<Self> {
        let data = bio_apis::amber_geostd::load_mol_files(ident)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, "Error loading data"))?;

        let mol = Mol2::new(&data.mol2)?;
        let params = ForceFieldParams::from_frcmod(&data.frcmod.unwrap())?;

        Ok(Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms,
            atom_posits: None,
            bonds: mol.bonds,
            adjacency_list: None,
            static_: false,
            mol_specific_params: Some(params),
            bonded_only: false,
        })
    }
}

/// A trimmed-down atom for use with molecular dynamics. Contains parameters for single-atom,
/// but we use ParametersIndex for multi-atom parameters.
#[derive(Clone, Debug, Default)]
pub struct AtomDynamics {
    pub serial_number: u32,
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    /// and VDW) only.
    pub static_: bool,
    /// If true, this atom exerts and experiences non-bonded forces only.
    /// This may be useful for protein atoms that aren't near a docking site.
    pub bonded_only: bool,
    pub force_field_type: String,
    pub element: Element,
    // pub name: String,
    pub posit: Vec3,
    /// Å / ps
    pub vel: Vec3,
    /// Å / ps²
    pub accel: Vec3,
    /// We confirm this to acceleration prior to updating velocities.
    pub force: Vec3,
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
        atom_posits: &[Vec3],
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

        let partial_charge = match atom.partial_charge {
            Some(p) => p * CHARGE_UNIT_SCALER,
            None => return Err(ParamError::new("Missing partial charge on atom {i}")),
        };

        Ok(Self {
            serial_number: atom.serial_number,
            static_,
            element: atom.element,
            posit: atom_posits[i],
            force_field_type: ff_type,
            partial_charge,
            ..Default::default()
        })
    }

    /// Populate atom-specific parameters.
    /// E.g. we use this workflow if creating the atoms prior to the indexed FF.
    pub(crate) fn assign_data_from_params(
        &mut self,
        ff_params: &ForceFieldParamsIndexed,
        i: usize,
    ) {
        self.mass = ff_params.mass[&i].mass;
        self.lj_sigma = ff_params.lennard_jones[&i].sigma;
        self.lj_eps = ff_params.lennard_jones[&i].eps;
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx8 {
    pub serial_number: [u32; 8],
    pub static_: [bool; 8],
    pub bonded_only: [bool; 8],
    pub force_field_type: [String; 8],
    pub element: [Element; 8],
    pub posit: Vec3x8,
    pub vel: Vec3x8,
    pub accel: Vec3x8,
    pub mass: f32x8,
    pub partial_charge: f32x8,
    pub lj_sigma: f32x8,
    pub lj_eps: f32x8,
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx16 {
    pub serial_number: [u32; 16],
    pub static_: [bool; 16],
    pub bonded_only: [bool; 16],
    pub force_field_type: [String; 16],
    pub element: [Element; 16],
    pub posit: Vec3x16,
    pub vel: Vec3x16,
    pub accel: Vec3x16,
    pub mass: f32x16,
    pub partial_charge: f32x16,
    pub lj_sigma: f32x16,
    pub lj_eps: f32x16,
}

// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// impl AtomDynamicsx4 {
//     pub fn from_array(bodies: [AtomDynamics; 4]) -> Self {
//         let mut posits = [Vec3::new_zero(); 4];
//         let mut vels = [Vec3::new_zero(); 4];
//         let mut accels = [Vec3::new_zero(); 4];
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
    Fixed((Vec3, Vec3)),
}

impl Default for SimBoxInit {
    fn default() -> Self {
        Self::Pad(10.)
    }
}

#[derive(Clone, Default, Debug)]
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
/// These are primarily used for debugging and testing, but may be used
/// for specific scenarios as well, e.g. if wishing to speed up computations for real-time use
/// by removing long range forces.
pub struct MdOverrides {
    pub allow_missing_dihedral_params: bool,
    pub skip_water: bool,
    pub bonded_disabled: bool,
    pub coulomb_disabled: bool,
    pub lj_disabled: bool,
    pub long_range_recip_disabled: bool,
    pub thermo_disabled: bool,
    pub baro_disabled: bool,
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
    /// Prior to the first integrator step, we attempt to relax energy in the system.
    /// Use no more than this many iterations to do so. Higher can produce better results,
    /// but is slower. If None, don't relax.
    pub max_init_relaxation_iters: Option<usize>,
    /// Distance threshold, in Å, used to determine when we rebuild neighbor lists.
    /// 2-4Å are common values. Higher values rebuild less often, and have more computationally-intense
    /// rebuilds. Rebuild the list if an atom moved > skin/2.
    pub neighbor_skin: f32,
    pub overrides: MdOverrides,
}

impl Default for MdConfig {
    fn default() -> Self {
        Self {
            integrator: Default::default(),
            zero_com_drift: false, // todo: True?
            temp_target: 310.,
            pressure_target: 1.,
            hydrogen_constraint: Default::default(),
            snapshot_handlers: vec![SnapshotHandler {
                save_type: SaveType::Memory,
                ratio: 1,
            }],
            sim_box: Default::default(),
            max_init_relaxation_iters: Some(300), // todo: A/R
            neighbor_skin: 4.0,
            overrides: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct MdState {
    // todo: Update how we handle mode A/R.
    // todo: You need to rework this state in light of arbitrary mol count.
    pub cfg: MdConfig,
    pub atoms: Vec<AtomDynamics>,
    #[cfg(target_arch = "x86_64")]
    pub atoms_x8: Vec<AtomDynamicsx8>,
    #[cfg(target_arch = "x86_64")]
    pub atoms_x16: Vec<AtomDynamicsx16>,
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
    /// Rebuilt whenever nonbonded neighbors is.
    nb_pairs: Vec<NonBondedPair>,
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
    pub water_pme_sites_forces: Vec<[Vec3F64; 3]>, // todo: A/R
    pme_recip: Option<PmeRecip>,
    /// kcal/mol
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    /// Every so many snapshots, write these to file, then clear from memory.
    snapshot_queue_for_file: Vec<Snapshot>,
    #[cfg(feature = "cuda")]
    gpu_kernel: Option<CudaFunction>, // Option only due to not impling Default.
    #[cfg(feature = "cuda")]
    gpu_kernel_zero_f32: Option<CudaFunction>,
    #[cfg(feature = "cuda")]
    gpu_kernel_zero_f64: Option<CudaFunction>,
    #[cfg(feature = "cuda")]
    /// These store handles to data structures on the GPU. We pass them to the kernel each
    /// step, but don't transfer. Init to None. Populated during the run.
    forces_posits_gpu: Option<ForcesPositsGpu>,
    #[cfg(feature = "cuda")]
    per_neighbor_gpu: Option<PerNeighborGpu>,
    pub neighbor_rebuild_count: usize,
    /// A cache of accel_factor / mass, per atom. Built once, at init.
    mass_accel_factor: Vec<f32>,
    pub computation_time: ComputationTimeSums,
}

impl Display for MdState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
        dev: &ComputationDevice,
        cfg: &MdConfig,
        mols: &[MolDynamics],
        param_set: &FfParamSet,
    ) -> Result<Self, ParamError> {
        // We create a flattened atom list, which simplifies our workflow, and is conducive to
        // parallel operations.
        // These Vecs all share indices, and all include all molecules.
        let mut atoms_md = Vec::new();
        let mut adjacency_list = Vec::new();

        // todo: Allow a water-only sim, buut need to rework how your simbox sizes for that.
        if mols.is_empty() {
            return Err(ParamError::new(
                "No molecules to simulate. Please provide at least one molecule.",
            ));
        }

        // We combine all molecule general and specific params into this set, then
        // create Indexed params from it.
        let mut params = ForceFieldParams::default();

        // Used for updating indices for tracking purposes.
        let mut total_atom_count = 0;

        for mol in mols {
            // Filter out hetero atoms in proteins. These are often example ligands that we do
            // not wish to model.
            // We must perform this filter prior to most of the other steps in this function.
            let atoms: Vec<AtomGeneric> = match mol.ff_mol_type {
                FfMolType::Peptide => mol
                    .atoms
                    .iter()
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

                if let Some(p) = &mol.mol_specific_params {
                    params = merge_params(&params, &p);
                }
            }

            let mut p: Vec<Vec3> = Vec::new(); // to store the ref.
            let atom_posits = match &mol.atom_posits {
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
                atoms_md.push(AtomDynamics::new(&atom, atom_posits, i, mol.static_)?);
            }

            // Use the included adjacency list if available. If not, construct it.
            let adjacency_list_ = match &mol.adjacency_list {
                Some(a) => a,
                None => &build_adjacency_list(&atoms, &mol.bonds)?,
            };

            for aj in adjacency_list_ {
                let mut updated = aj.clone();
                for neighbor in &mut updated {
                    *neighbor += total_atom_count;
                }

                adjacency_list.push(updated);
            }

            total_atom_count += atoms.len();
        }

        if atoms_md.is_empty() {
            return Err(ParamError::new(
                "No atoms to simulate; please provide at least one.",
            ));
        }

        let force_field_params = ForceFieldParamsIndexed::new(
            &params,
            &atoms_md,
            &adjacency_list,
            cfg.hydrogen_constraint,
            cfg.overrides.allow_missing_dihedral_params,
        )?;

        let mut mass_accel_factor = Vec::with_capacity(atoms_md.len());

        // Assign mass, LJ params, etc.
        for (i, atom) in atoms_md.iter_mut().enumerate() {
            atom.assign_data_from_params(&force_field_params, i);
            mass_accel_factor.push(ACCEL_CONVERSION / atom.mass);
        }

        let cell = SimBox::new(&atoms_md, &cfg.sim_box);

        let mut result = Self {
            cfg: cfg.clone(),
            atoms: atoms_md,
            adjacency_list: adjacency_list.to_vec(),
            cell,
            pairs_excluded_12_13: HashSet::new(),
            pairs_14_scaled: HashSet::new(),
            force_field_params,
            mass_accel_factor,
            ..Default::default()
        };

        // Set up our LJ cache. Do this prior to building neighbors for the first time,
        // as that also sets up the GPU-struct LJ data.
        result.lj_tables = LjTables::new(&result.atoms);

        result.neighbors_nb = NeighborsNb::new(result.cfg.neighbor_skin);

        result.barostat.pressure_target = cfg.pressure_target as f64;

        result.water = if cfg.overrides.skip_water {
            Vec::new()
        } else {
            make_water_mols(&result.cell, cfg.temp_target, &result.atoms)
        };

        result.water_pme_sites_forces = vec![[Vec3F64::new_zero(); 3]; result.water.len()];

        result.setup_nonbonded_exclusion_scale_flags();

        result.build_all_neighbors(dev);

        // Initializes the FFT planner[s], among other things.
        result.regen_pme(dev);

        // Allocate force buffers on the GPU, and store a handle. Used for the entire run.
        // Initialize the per-neighbor data as well; we will do this again every time
        // we compute neighbors.
        #[cfg(feature = "cuda")]
        if let ComputationDevice::Gpu(stream) = dev {
            let ctx = CudaContext::new(0).unwrap();
            let module = ctx.load_module(Ptx::from_src(PTX)).unwrap();
            result.gpu_kernel = Some(module.load_function("nonbonded_force_kernel").unwrap());
            result.gpu_kernel_zero_f32 = Some(module.load_function("zero_f32").unwrap());
            result.gpu_kernel_zero_f64 = Some(module.load_function("zero_f64").unwrap());

            result.forces_posits_gpu = Some(ForcesPositsGpu::new(
                stream,
                result.atoms.len(),
                result.water.len(),
                LONG_RANGE_CUTOFF,
                EWALD_ALPHA,
            ));

            result.per_neighbor_gpu = Some(PerNeighborGpu::new(
                stream,
                &result.nb_pairs,
                &result.atoms,
                &result.water,
                &result.lj_tables,
            ));
        }

        println!("Init pair count: {:?}", result.nb_pairs.len());

        // todo: Move this AR
        // Pack SIMD once at init.
        #[cfg(target_arch = "x86_64")]
        result.pack_atoms();

        // todo: Add to config A/R,
        if let Some(max_iters) = cfg.max_init_relaxation_iters {
            result.minimize_energy(dev, max_iters);
        }

        // Reset computation time to negate anything that was applied by minimization, initial
        // neighbor rebuild, and anything else done here that may affect it.
        result.computation_time = Default::default();

        Ok(result)
    }

    pub fn computation_time(&self) -> io::Result<ComputationTime> {
        self.computation_time.time_per_step(self.step_count)
    }

    /// Reset acceleration and virial pair. Do this each step after the first half-step and drift, and
    /// shaking the fixed hydrogens.
    /// We must reset the virial pair prior to accumulating it, which we do when calculating non-bonded
    /// forces. Also reset forces on water.
    fn reset_accel_e(&mut self) {
        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
            a.force = Vec3::new_zero();
        }
        for mol in &mut self.water {
            mol.o.accel = Vec3::new_zero();
            mol.m.accel = Vec3::new_zero();
            mol.h0.accel = Vec3::new_zero();
            mol.h1.accel = Vec3::new_zero();

            mol.o.force = Vec3::new_zero();
            mol.m.force = Vec3::new_zero();
            mol.h0.force = Vec3::new_zero();
            mol.h1.force = Vec3::new_zero();
        }

        self.barostat.virial_coulomb = 0.0;
        self.barostat.virial_lj = 0.0;
        self.barostat.virial_bonded = 0.0;
        self.barostat.virial_constraints = 0.0;
        self.barostat.virial_nonbonded_long_range = 0.0;
        self.potential_energy = 0.;
    }

    fn apply_all_forces(&mut self, dev: &ComputationDevice) {
        let mut start = Instant::now();
        let log_time = self.step_count.is_multiple_of(COMPUTATION_TIME_RATIO);

        if log_time {
            start = Instant::now();
        }

        if !self.cfg.overrides.bonded_disabled {
            self.apply_bonded_forces();
        }

        if log_time {
            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.bonded_sum += elapsed;
        }

        self.apply_nonbonded_forces(dev);

        if log_time {
            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.non_bonded_short_range_sum += elapsed;
        }

        // Note: We currently set to skip these on energy minimization, but this
        // check may fail at step 0 anyway?
        // todo: When skipping long range forces, you may wish to use naive coulomb instead
        // todo of the short-range part of the recip. This depends on the application.
        if !self.cfg.overrides.long_range_recip_disabled
            && self.step_count.is_multiple_of(SPME_RATIO)
        {
            // Note: This relies on SPME_RATIO being divisible by COMPUTATION_TIME_RATIO.
            // It will produce inaccurate results otherwise.
            if log_time {
                start = Instant::now();
            }

            self.handle_spme_recip(dev);

            if log_time {
                let elapsed = start.elapsed().as_micros() as u64;
                self.computation_time.ewald_long_range_sum += elapsed;
            }
        }

        if self.step_count == 1 {
            let elapsed = start.elapsed();
            println!("SPME recip time: {:?} μs", elapsed.as_micros());
        }
    }

    pub(crate) fn take_snapshot_if_required(&mut self) {
        let mut updated_ke = false;
        let mut take_ss = false;
        let mut take_ss_file = false;

        for handler in &self.cfg.snapshot_handlers {
            if self.step_count % handler.ratio != 0 {
                continue;
            }

            // We currently only use kinetic energy in snapshots, so update it only when
            // calling a handler.
            if !updated_ke {
                updated_ke = true;
                self.kinetic_energy = self.current_kinetic_energy();
            }

            match &handler.save_type {
                // No action if multiple Memory savetypes are specified.
                SaveType::Memory => {
                    take_ss = true;
                }
                SaveType::Dcd(path) => {
                    take_ss_file = true;

                    // todo: Handle the case of the final step!
                    if self.step_count % FILE_SAVE_INTERVAL == 0 {
                        if let Err(e) = append_dcd(&self.snapshot_queue_for_file, &path) {
                            eprintln!("Error saving snapshot as DCD: {e:?}");
                        }
                        self.snapshot_queue_for_file = Vec::new();
                    }
                }
            }
        }

        if take_ss || take_ss_file {
            let snapshot = self.take_snapshot();

            if take_ss {
                // todo: DOn't clone.
                self.snapshots.push(snapshot.clone());
            }
            if take_ss_file {
                self.snapshot_queue_for_file.push(snapshot);
            }
        }
    }

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
            hydrogen_bonds: Vec::new(), // Populated later A/R.
        }
    }
}

/// Mutable aliasing helpers.
pub(crate) fn split2_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);

    let (low, high) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = v.split_at_mut(high);
    (&mut left[low], &mut right[0])
}

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
    NonWat(usize),

    // Water sites (by molecule index)
    WatO(usize),
    WatM(usize),
    WatH0(usize),
    WatH1(usize),

    // Static atoms (included in the field, but you won't update their accel)
    Static(usize),
}
