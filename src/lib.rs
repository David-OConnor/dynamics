#![allow(non_snake_case)]
#![allow(confusable_idents)]

//! See the [Readme](https://github.com/David-OConnor/dynamics/blob/main/README.md) for a general overview,
//! or [Molchanica docs, MD section](https://www.athanorlab.com/docs/md.html) for more information about
//! assumptions. Or see the [examples folder on Github](https://github.com/David-OConnor/dynamics/tree/main/examples)
//! for how to use this in your application.
//!
//! The textual information here is informal, and aimed at code maintenance; not library use.
//!
//! This module contains high-level tools for running Newtonian molecular dynamics simulations.
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//! [A summary  on molecular dynamics](https://arxiv.org/pdf/1401.1181)
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
//! Amber: ff19SB for proteins, gaff2 for ligands. (Based on recommendations from https://ambermd.org/AmberModels.php).
//!
//! We use the term "Non-bonded" interactions to refer to Coulomb, and Lennard Interactions, the latter
//! of which is an approximation for both Van der Waals force and exclusion.
//!
//! ## A broad list of components of this simulation:
//! - Water: Rigid OPC solvent molecules that have mutual non-bonded interactions with dynamic atoms and solvent
//! - Thermostat/barostat, with a way to specify temp, pressure, solvent density
//! - OPC solvent model
//! - Cell wrapping
//! - Velocity Verlet integration (Water and non-solvent)
//! - Amber parameters for mass, partial charge, VdW (via LJ), dihedral/improper, angle, bond len
//! - Optimizations for Coulomb: Ewald/SPME.
//! - Optimizations for LJ: Dist cutoff for now.
//! - Amber 1-2, 1-3 exclusions, and 1-4 scaling of covalently-bonded atoms.
//! - Rayon parallelization of non-bonded forces
//! - WIP SIMD and CUDA parallelization of non-bonded forces, depending on hardware availability. todo
//! - A thermostat and barostat
//! - An energy-measuring system.
//! - An integrated tool for inferring atom types, bonded-parameter overrides, and partial charges for arbitrary
//!   small organic molecules. (Similar to Amber's Antechamber)
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
//! We use the OPC solvent model. (See `water_opc.rs`). For both maintaining the geometry of each solvent
//! molecule, and for maintaining Hydrogen atom positions, we do not apply typical non-bonded interactions:
//! We use SHAKE + RATTLE algorithms for these. In the case of solvent, it's required for OPC compliance.
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

mod add_hydrogens;
mod barostat;
mod bonded;
mod bonded_forces;
mod config;
// mod dcd;
mod forces;
pub mod integrate;
mod neighbors;
mod non_bonded;
pub mod params;
pub mod partial_charge_inference;
mod prep;
#[cfg(target_arch = "x86_64")]
mod simd;
pub mod snapshot;
mod solvent;
mod thermostat;
mod util;

mod com_zero;
#[cfg(feature = "cuda")]
mod gpu_interface;
pub mod minimize_energy;

pub mod alchemical;
pub mod param_inference;
mod sa_surface;
// pub mod snapshot_mdt;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::HashSet,
    fmt,
    fmt::{Display, Formatter},
    io,
    path::Path,
};

pub use add_hydrogens::{
    add_hydrogens_2::Dihedral,
    bond_vecs::{find_planar_posit, find_tetra_posit_final, find_tetra_posits},
    populate_hydrogens_dihedrals,
};
use barostat::SimBox;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, Sdf, dcd::DcdTrajectory, md_params::ForceFieldParams, mol2::Mol2,
};
pub use bonded::{LINCS_ITER_DEFAULT, LINCS_ORDER_DEFAULT, SHAKE_TOL_DEFAULT};
pub use config::MdConfig;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, CudaStream};
use ewald::PmeRecip;
pub use integrate::Integrator;
#[allow(unused)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use na_seq::Element;
use neighbors::NeighborsNb;
pub use prep::{HydrogenConstraint, merge_params};
pub use solvent::{
    ForcesOnWaterMol, Solvent,
    init::{WATER_TEMPLATE_60A, WaterInitTemplate},
};

#[cfg(feature = "cuda")]
use crate::gpu_interface::{ForcesPositsGpu, PerNeighborGpu};
use crate::{
    barostat::Barostat,
    non_bonded::{CHARGE_UNIT_SCALER, LjTables, NonBondedPair},
    params::{FfParamSet, ForceFieldParamsIndexed},
    snapshot::Snapshot,
    solvent::{WaterMol, WaterMolx8, WaterMolx16},
    util::ComputationTimeSums,
};
pub use crate::{
    barostat::{BarostatCfg, PRESSURE_DEFAULT, TAU_PRESSURE_DEFAULT},
    thermostat::{LANGEVIN_GAMMA_DEFAULT, TAU_TEMP_DEFAULT},
};

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../dynamics.ptx");

// Multiply by this to convert from kcal/mol to amu • (Å/ps)²  Multiply all accelerations by this.
// Converts *into* our internal units.
const KCAL_TO_NATIVE: f32 = 418.4;

// Multiply by this to convert from amu • (Å/ps)² to kcal/mol.  We use this when accumulating kinetic
// energy, for example. This, in practice, is for temperature and pressure computations.
// Converts *out of * our internal units.
const NATIVE_TO_KCAL: f32 = 1. / KCAL_TO_NATIVE;

// Every this many steps, re-center the sim (solvent) box.
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
    /// This may have uses if "shooting" a molecule into a docking position?
    pub atom_init_velocities: Option<Vec<Vec3>>,
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

/// This is mainly for overriding, while specifying atoms, bonds, posits, and mol type explicitly.
impl Default for MolDynamics {
    fn default() -> Self {
        Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: Vec::new(),
            atom_posits: None,
            atom_init_velocities: None,
            bonds: Vec::new(),
            adjacency_list: None,
            static_: false,
            mol_specific_params: None,
            bonded_only: false,
        }
    }
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
            atom_init_velocities: None,
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
            atom_init_velocities: None,
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
            .map_err(|e| io::Error::other(format!("Error loading data: {e:?}")))?;

        let mol = Mol2::new(&data.mol2)?;
        let params = ForceFieldParams::from_frcmod(&data.frcmod.unwrap())?;

        Ok(Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms,
            atom_posits: None,
            atom_init_velocities: None,
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
    /// in docking, this might be a rigid receptor. They serve as sources for Coulomb and LJ (non-bonded)
    /// interactions, and as anchors for bonded ones.
    pub static_: bool,
    /// If true, this atom exerts and experiences non-bonded forces only.
    /// This may be useful for protein atoms that aren't near a docking site.
    pub bonded_only: bool,
    pub force_field_type: String,
    pub element: Element,
    pub posit: Vec3,
    /// Å / ps
    pub vel: Vec3,
    /// Å / ps²
    pub accel: Vec3,
    /// Å • amu / ps²
    pub force: Vec3,
    /// Daltons or amu
    pub mass: f32,
    /// Amber charge units. This is not the elementary charge units found in amino19.lib and gaff2.dat;
    /// it's scaled by the electrostatic constant.
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
        bonded_only: bool,
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
            bonded_only,
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

#[allow(unused)]
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

#[allow(unused)]
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
#[derive(Debug, Clone, PartialEq)]
pub enum SimBoxInit {
    /// Distance in Å from the edge to the molecule, at init.
    Pad(f32),
    /// Coordinate boundaries, at opposite corners
    Fixed((Vec3, Vec3)),
}

impl SimBoxInit {
    /// Centered at the origin; can be moved after, e.g. to center on molecules.
    pub fn new_cube(side_len: f32) -> Self {
        let l = side_len / 2.;
        Self::Fixed((Vec3::new(-l, -l, -l), Vec3::new(l, l, l)))
    }
}

impl Default for SimBoxInit {
    fn default() -> Self {
        Self::Pad(12.)
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
/// These are primarily used for debugging and testing, but may be used
/// for specific scenarios as well, e.g. if wishing to speed up computations for real-time use
/// by removing long range forces. These are not standard MD config parameters.
pub struct MdOverrides {
    /// Skips the initial solvent relaxation, where a simulation is run until
    /// hydrogen bonds are established, and temperature is initialized.
    pub skip_water_relaxation: bool,
    pub bonded_disabled: bool,
    pub coulomb_disabled: bool,
    pub lj_disabled: bool,
    pub long_range_recip_disabled: bool,
    /// Run this block if we wish to, for dev purposes, take snapshots during the
    /// solvent equilibration phase, e.g. for tuning it.
    pub snapshots_during_equilibration: bool,
    /// Take snapshots during the energy minimization phase. (Not solvent equilibration)
    /// This can be used to visually QC this process.
    pub snapshots_during_energy_min: bool,
}

#[derive(Default)]
pub struct MdState {
    // todo: Update how we handle mode A/R.
    // todo: You need to rework this state in light of arbitrary mol count.
    pub cfg: MdConfig,
    pub atoms: Vec<AtomDynamics>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) atoms_x8: Vec<AtomDynamicsx8>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) atoms_x16: Vec<AtomDynamicsx16>,
    pub water: Vec<WaterMol>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) water_x8: Vec<WaterMolx8>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) water_x16: Vec<WaterMolx16>,
    /// Note: We don't use bond structs once the simulation is set up; the adjacency list is the
    /// source of this.
    pub adjacency_list: Vec<Vec<usize>>,
    pub(crate) force_field_params: ForceFieldParamsIndexed,
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
    barostat: Barostat,
    /// Exclusions of non-bonded forces for atoms connected by 1, or 2 covalent bonds.
    /// I can't find this in the RM, but ChatGPT is confident of it, and references an Amber file
    /// called 'prmtop', which I can't find. Fishy, but we're going with it.
    pairs_excluded_12_13: HashSet<(usize, usize)>,
    /// See Amber RM, sectcion 15, "1-4 Non-Bonded Interaction Scaling"
    /// These are indices of atoms separated by three consecutive bonds
    pairs_14_scaled: HashSet<(usize, usize)>,
    lj_tables: LjTables,
    // todo: Hmm... Is this DRY with forces_on_water? Investigate.
    pub water_pme_sites_forces: Vec<[Vec3F64; 3]>, // todo: A/R
    pme_recip: Option<PmeRecip>,
    /// kcal/mol
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    /// A newer, simpler approach for energy between molecules, compared to `potential_energy_between_mols`.
    /// This is simply the potential energy from non-bonded interactions, and excludes that from bonded.
    pub potential_energy_nonbonded: f64,
    /// E.g. energy in covalent bonds, as modelled as oscillators.
    pub potential_energy_bonded: f64,
    /// Every so many snapshots, write these to file, then clear from memory.
    /// Used to track which molecule each atom is associated with in our flattened structures.
    /// This is the potential energy between every pair of molecules.
    pub potential_energy_between_mols: Vec<f64>,
    snapshot_queue_for_dcd: Vec<Snapshot>,
    snapshot_queue_for_trr: Vec<Snapshot>,
    snapshot_queue_for_xtc: Vec<Snapshot>,
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
    /// A cache. We don't run SPME every step; store the previous step's per-atom
    /// force values (Flattened; non-solvent, then solvent M, H0, H1), and apply them
    /// on the steps where we don't re-calculate. (Force, potential energy, virial energy)
    spme_force_prev: Option<(Vec<Vec3>, f64, f64)>,
    /// Cached at init; used for kinetic energy calculations.
    _num_static_atoms: usize,
    // todo: Sub-struct for ambient cache like num_static atoms and thermo_dof
    /// Degrees of freedom, used in temperature and kinetic energy calculations.
    thermo_dof: usize,
    // todo: Deprecate this if you deprecate per-atom posits and vels in snapshots?
    /// Used to track which molecule each atom is associated with in our flattened structures.
    pub mol_start_indices: Vec<usize>,
    /// A flag we set to disable certain things like snapshots during this MD phase.
    solvent_only_sim_at_init: bool,
    /// Index into `mol_start_indices` of the molecule being alchemically decoupled.
    ///
    /// When `Some(m)`, `take_snapshot` computes ∂H/∂λ for molecule m and stores it
    /// in each `Snapshot::dh_dl`.  Set `None` (the default) for ordinary MD.
    pub alch_mol_idx: Option<usize>,
    /// Current lambda value for alchemical simulations, in [0, 1].
    ///
    /// λ = 0: solute fully coupled; λ = 1: solute fully decoupled.
    /// For thermodynamic integration, hold this fixed for the duration of one
    /// simulation window and sweep across multiple windows.
    pub lambda: f64,
    /// Index assigned at the start of each MD run. Trajectory files are named
    /// `traj_N.dcd`, `traj_N.trr`, etc. so that successive runs never overwrite
    /// each other.  Chosen as the lowest N for which no such files exist yet.
    pub(crate) run_index: usize,
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

/// Set up with no solvent molecules or relaxation. Run one step to compute energies, then return
/// the snapshot taken.
pub fn compute_energy_snapshot(
    dev: &ComputationDevice,
    mols: &[MolDynamics],
    param_set: &FfParamSet,
) -> Result<Snapshot, ParamError> {
    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity { thermostat: None },
        hydrogen_constraint: HydrogenConstraint::Flexible,
        max_init_relaxation_iters: None,
        solvent: Solvent::None,
        ..Default::default()
    };

    let mut md_state = MdState::new(dev, &cfg, mols, param_set)?;

    // dt is arbitrary?
    let dt = 0.001;
    md_state.step(dev, dt, None);

    if md_state.snapshots.is_empty() {
        return Err(ParamError {
            descrip: String::from("Snapshots empty on energy compuptation"),
        });
    }

    Ok(md_state.snapshots[0].clone())
}

// todo: Investiate how to handle now that we revamped our snapshot/file system.
// /// For calling by the application. Loads snapshots from a file (e.g. DCD/XTC/MDT) into memory.
// pub fn load_snapshots_from_file(path: &Path) -> Result<Vec<Snapshot>, io::Error> {
//     let ext = path
//         .extension()
//         .and_then(|s| s.to_str())
//         .map(|s| s.to_ascii_lowercase())
//         .ok_or_else(|| io::Error::other("Input path must have a file extension"))?;
//
//     let result: io::Result<Vec<Snapshot>> = match ext.as_ref() {
//         "dcd" => {
//             let dcd = DcdTrajectory::load(path)?;
//             let snaps = Snapshot::from_dcd(&dcd);
//             Ok(snaps)
//         }
//         "xtc" => {
//             let dcd = DcdTrajectory::load_xtc(path)?;
//             let snaps = Snapshot::from_dcd(&dcd);
//             Ok(snaps)
//         } // "mdt" => load_mdt(path),
//         _ => Err(io::Error::new(
//             io::ErrorKind::InvalidInput,
//             "Invalid file extension for loading snapshots.",
//         )),
//     };
//
//     match result {
//         Ok(snaps) => Ok(snaps),
//         Err(e) => {
//             eprintln!("Error loading snapshots from file: {e:?}");
//             Err(e)
//         }
//     }
// }
