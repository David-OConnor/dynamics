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

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    fmt::{Display, Formatter},
    io,
    time::Instant,
};

pub use add_hydrogens::{
    add_hydrogens_2::Dihedral,
    bond_vecs::{find_planar_posit, find_tetra_posit_final, find_tetra_posits},
    populate_hydrogens_dihedrals,
};
pub use barostat::SimBox;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, Sdf,
    gromacs::gro::Gro,
    md_params::{ForceFieldParams, ForceFieldParamsIndexed, LjParams, MassParams},
    mol2::Mol2,
};
pub use bonded::{LINCS_ITER_DEFAULT, LINCS_ORDER_DEFAULT, SHAKE_TOL_DEFAULT};
pub use config::{ComMotionRemoval, MdConfig};
#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream},
    nvrtc::Ptx,
};
use ewald::PmeRecip;
pub use integrate::Integrator;
#[allow(unused)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};
use na_seq::Element;
use neighbors::NeighborsNb;
pub use prep::{HydrogenConstraint, merge_params};
pub use solvent::{
    ForcesOnWaterMol, Solvent,
    init::{OCTANOL_WATER_TEMPLATE, WATER_TEMPLATE_60A, WaterInitTemplate},
    template_creation::pack_solvent_with_shrinking_box,
};

#[cfg(feature = "cuda")]
use crate::gpu_interface::{ForcesPositsGpu, PerNeighborGpu};
use crate::{
    barostat::Barostat,
    non_bonded::{CHARGE_UNIT_SCALER, LjTables, NonBondedPair},
    param_inference::update_small_mol_params,
    params::FfParamSet,
    snapshot::Snapshot,
    solvent::{
        WaterMolOpc, WaterMolx8, WaterMolx16, init::water_mols_from_template,
        octanol::octanol_mols_from_gro,
    },
    util::{ComputationTime, ComputationTimeSums, build_adjacency_list},
};
pub use crate::{
    barostat::{BarostatCfg, PRESSURE_DEFAULT, TAU_PRESSURE_DEFAULT},
    solvent::octanol::make_octanol,
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

impl bio_files::md_params::AtomFfSource for AtomDynamics {
    fn ff_type(&self) -> Option<&str> {
        Some(&self.force_field_type)
    }
    fn element(&self) -> na_seq::Element {
        self.element
    }
    fn serial_number(&self) -> u32 {
        self.serial_number
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
    pub water: Vec<WaterMolOpc>,
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
    /// Unscaled non-bonded interaction energy between the alchemical molecule and the
    /// rest of the system for the current step, in kcal/mol.
    alch_interaction_energy: f64,
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
    pub lambda_alch: f64,
    /// Index assigned at the start of each MD run. Trajectory files are named
    /// `traj_N.dcd`, `traj_N.trr`, etc. so that successive runs never overwrite
    /// each other.  Chosen as the lowest N for which no such files exist yet.
    /// `None` until the first call to `handle_ss_file_writes`.
    pub run_index: Option<usize>,
}

impl MdState {
    /// Also returns any explicit solvent molecules added. This may be needed by applications
    /// in order to create molecule sets for rendering the trajectories. These are placed, in the
    /// trajectory, after all solute atoms.
    pub fn new(
        dev: &ComputationDevice,
        cfg: &MdConfig,
        mols: &[MolDynamics],
        param_set: &FfParamSet,
    ) -> Result<(Self, Vec<MolDynamics>), ParamError> {
        // We combine all molecule general and specific params into this set, then
        // create Indexed params from it.
        let mut params = ForceFieldParams::default();

        // Used for updating indices for tracking purposes.
        let mut atom_ct_prior_to_this_mol = 0;

        let mut mol_start_indices = Vec::new();

        let posits_solute = {
            let mut p = Vec::new();

            // todo: This is DRY /extra computation with the general atom posits building below, but may be required
            // todo when dealing with a custom solute.
            for mol in mols {
                let atom_posits: Vec<Vec3> = match &mol.atom_posits {
                    Some(a) => a.iter().map(|p| (*p).into()).collect(),
                    None => mol.atoms.iter().map(|a| a.posit.into()).collect(),
                };
                p.extend(atom_posits);
            }

            p
        };

        let octanol_water_template = if cfg.solvent == Solvent::OctanolWithWater {
            Some(Gro::new(OCTANOL_WATER_TEMPLATE).map_err(|e| {
                ParamError::new(&format!("Problem loading the octanol/water GRO file: {e}"))
            })?)
        } else {
            None
        };

        let cell = if let Some(gro) = &octanol_water_template {
            const NM_TO_ANGSTROM: f32 = 10.0;
            SimBox::new(
                (-gro.box_vec * NM_TO_ANGSTROM as f64 / 2.0).into(),
                (gro.box_vec * NM_TO_ANGSTROM as f64 / 2.0).into(),
            )
        } else {
            SimBox::from_solute_atoms(&posits_solute, &cfg.sim_box)
        };

        // let custom_solvent_packed = match &cfg.solvent {
        //     Solvent::Custom((mols_solvent, water_count)) => {
        //         let (mols, snaps) = pack_solvent_with_shrinking_box(
        //             dev,
        //             // todo: Handle cases where len of custom solvent isn't 1.
        //             &mols_solvent[0].0,
        //             mols_solvent[0].1,
        //             *water_count,
        //             cell,
        //             param_set,
        //         )?;
        //
        //         mols
        //     }
        //     _ => Vec::new(),
        // };

        let custom_solvent_packed = match &octanol_water_template {
            Some(gro) => octanol_mols_from_gro(gro)?,
            None => Vec::new(),
        };

        // Build a combined slice: caller-supplied mols first, then packed custom solvents.
        let combined_mols: Vec<MolDynamics>;
        let all_mols: &[MolDynamics] = if custom_solvent_packed.is_empty() {
            mols
        } else {
            combined_mols = mols
                .iter()
                .cloned()
                .chain(custom_solvent_packed.clone())
                .collect();
            &combined_mols
        };

        // We create a flattened atom list, which simplifies our workflow, and is conducive to
        // parallel operations.
        // These Vecs all share indices, and all include all molecules.
        let mut atoms_md = Vec::new();
        let mut adjacency_list = Vec::new();

        for mol in all_mols {
            if !mol.atoms.is_empty() {
                mol_start_indices.push(atoms_md.len());
            }

            // Filter out hetero atoms in proteins. These are often example ligands that we do
            // not wish to model.
            // We must perform this filter prior to most of the other steps in this function.
            let mut atoms: Vec<AtomGeneric> = match mol.ff_mol_type {
                FfMolType::Peptide => mol.atoms.iter().filter(|a| !a.hetero).cloned().collect(),
                _ => mol.atoms.to_vec(),
            };

            let mut mol_specific_params = mol.mol_specific_params.clone();

            // Update partial charge, FF names, and param overrides A/R.
            if mol.ff_mol_type == FfMolType::SmallOrganic {
                let mut needs_ff_type_or_q = false;
                for atom in &atoms {
                    if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                        needs_ff_type_or_q = true;
                        break;
                    }
                }

                if needs_ff_type_or_q {
                    // Note: This invalidates any passed by the user.
                    mol_specific_params = Some(
                        update_small_mol_params(
                            &mut atoms,
                            &mol.bonds,
                            Some(&adjacency_list),
                            param_set.small_mol.as_ref().unwrap(),
                        )
                        .map_err(|_| ParamError {
                            descrip: "Problem inferring params".to_string(),
                        })?,
                    );
                }
            }

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
                params = merge_params(&params, params_general);

                if let Some(p) = &mol_specific_params {
                    params = merge_params(&params, p);
                }
            }

            let atom_posits: Vec<Vec3> = match &mol.atom_posits {
                Some(a) => a.iter().map(|p| (*p).into()).collect(),
                None => mol.atoms.iter().map(|a| a.posit.into()).collect(),
            };

            for (i, atom) in atoms.iter().enumerate() {
                let mut atom =
                    AtomDynamics::new(atom, &atom_posits, i, mol.static_, mol.bonded_only)?;

                if let Some(vel) = &mol.atom_init_velocities {
                    if i >= vel.len() {
                        return Err(ParamError::new(
                            "Initial velocities passed, but don't match atom len.",
                        ));
                    }
                    atom.vel = vel[i];
                }

                atoms_md.push(atom);
            }

            // Use the included adjacency list if available. If not, construct it.
            let adjacency_list_ = match &mol.adjacency_list {
                Some(a) => a,
                None => &build_adjacency_list(&atoms, &mol.bonds)?,
            };

            // Update indices based on atoms from previously added molecules.
            for aj in adjacency_list_ {
                let mut updated = aj.clone();
                for neighbor in &mut updated {
                    *neighbor += atom_ct_prior_to_this_mol;
                }

                adjacency_list.push(updated);
            }

            atom_ct_prior_to_this_mol += atoms.len();
        }

        // Compute net charge from all solute atoms (internal Amber charge units → elementary).
        let net_q_e: f32 =
            atoms_md.iter().map(|a| a.partial_charge).sum::<f32>() / CHARGE_UNIT_SCALER;
        let n_ions = net_q_e.abs().round() as usize;

        let h_constrained = !matches!(cfg.hydrogen_constraint, HydrogenConstraint::Flexible);
        let force_field_params =
            ForceFieldParamsIndexed::new(&params, &atoms_md, &adjacency_list, h_constrained)
                .map_err(|e| ParamError::new(&e.to_string()))?;

        let mut mass_accel_factor = Vec::with_capacity(atoms_md.len());

        // Assign mass, LJ params, etc.
        for (i, atom) in atoms_md.iter_mut().enumerate() {
            atom.assign_data_from_params(&force_field_params, i);
            mass_accel_factor.push(KCAL_TO_NATIVE / atom.mass);
        }

        let num_static_atoms = atoms_md.iter().filter(|a| !a.static_).count();

        let potential_energy_between_mols = vec![0.; mol_start_indices.len().pow(2)];

        let mut result = Self {
            cfg: cfg.clone(),
            atoms: atoms_md,
            adjacency_list: adjacency_list.to_vec(),
            cell,
            pairs_excluded_12_13: HashSet::new(),
            pairs_14_scaled: HashSet::new(),
            force_field_params,
            mass_accel_factor,
            _num_static_atoms: num_static_atoms,
            mol_start_indices,
            potential_energy_between_mols,
            ..Default::default()
        };

        // Validate atom positions BEFORE recentering so we check against the box that was
        // used for placement (add_copies places atoms relative to the original Fixed bounds).
        // Recentering shifts bounds_low/bounds_high to the atom centroid, which can move
        // atoms that were legitimately near a wall to just outside the new bounds.
        //
        // `OctanolWithWater` is loaded from a pre-equilibrated GRO template whose atoms can
        // legitimately sit very close to the cell faces, so we skip this generic placement
        // validation and preserve the template box as-is.
        if cfg.solvent != Solvent::OctanolWithWater {
            result.check_for_overlaps_oob()?;
            if cfg.recenter_sim_box {
                result.cell.recenter(&result.atoms);
            }
        }

        // Set up our LJ cache. Do this prior to building neighbors for the first time,
        // as that also sets up the GPU-struct LJ data.
        result.lj_tables = LjTables::new(&result.atoms);
        result.neighbors_nb = NeighborsNb::new(result.cfg.neighbor_skin, result.cfg.coulomb_cutoff);

        // Custom solvent molecules were pre-packed and added to `all_mols` before the atom-
        // processing loop above, so their atoms are already in `result.atoms` and their
        // parameters are already included in `force_field_params`.  Water (below) will avoid
        // them automatically because `make_water_mols` checks against `result.atoms`.

        result.water = match cfg.solvent {
            Solvent::None => Vec::new(),
            Solvent::WaterOpc | Solvent::WaterOpcSpecifyMolCount(_) => {
                let count = match &cfg.solvent {
                    Solvent::WaterOpc => None,
                    Solvent::WaterOpcSpecifyMolCount(c) => Some(*c),
                    _ => unreachable!(),
                };

                water_mols_from_template(
                    &result.cell,
                    &result.atoms,
                    count,
                    &cfg.solvent_template_type,
                    cfg.skip_water_pbc_filter,
                )
            }
            Solvent::OctanolWithWater => {
                let Some(gro) = &octanol_water_template else {
                    return Err(ParamError::new(
                        "Missing octanol/water GRO template during initialization.",
                    ));
                };

                water_mols_from_gro(gro)?
            }
            Solvent::Custom(_) => Vec::new(), // todo: ?
        };

        add_ions(&mut result, net_q_e, n_ions);

        // Rebuild the LJ table to include any ions that were appended after the initial build.
        if n_ions > 0 && !result.water.is_empty() {
            result.lj_tables = LjTables::new(&result.atoms);
        }

        // Calc DOF only after all atoms and solvent are initialized.
        result.thermo_dof = result.dof_for_thermo();

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
                result.cfg.coulomb_cutoff,
                result.cfg.spme_alpha,
            ));

            result.per_neighbor_gpu = Some(PerNeighborGpu::new(
                stream,
                &result.nb_pairs,
                &result.atoms,
                &result.water,
                &result.lj_tables,
            ));
        }

        // todo: Move this AR
        // Pack SIMD once at init.
        #[cfg(target_arch = "x86_64")]
        result.pack_atoms();

        if !result.cfg.overrides.skip_water_relaxation {
            result.md_on_water_only(dev);
        }

        if let Some(max_iters) = result.cfg.max_init_relaxation_iters {
            result.minimize_energy(dev, max_iters, None);
        }

        // Reset computation time to negate anything that was applied by minimization, initial
        // neighbor rebuild, and anything else done here that may affect it.
        result.computation_time = Default::default();

        Ok((result, custom_solvent_packed))
    }

    /// This way of returning a Result isn't great semantically, but it works.
    fn check_for_overlaps_oob(&mut self) -> Result<(), ParamError> {
        const MIN_DIST_FROM_EDGE: f32 = 0.5; // Å
        const MIN_INTER_MOL_DIST: f32 = 0.5; // Å

        let lo = self.cell.bounds_low;
        let hi = self.cell.bounds_high;

        for (i, atom) in self.atoms.iter().enumerate() {
            let p = atom.posit;

            if p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z {
                return Err(ParamError::new(&format!(
                    "Atom index {i} is outside the sim box. \
                         Pos ({:.3}, {:.3}, {:.3}), box [{:.3}..{:.3}, {:.3}..{:.3}, {:.3}..{:.3}]",
                    p.x, p.y, p.z, lo.x, hi.x, lo.y, hi.y, lo.z, hi.z,
                )));
            }

            let dist_to_edge = (p.x - lo.x)
                .min(hi.x - p.x)
                .min(p.y - lo.y)
                .min(hi.y - p.y)
                .min(p.z - lo.z)
                .min(hi.z - p.z);

            if dist_to_edge < MIN_DIST_FROM_EDGE {
                return Err(ParamError::new(&format!(
                    "Atom index {i} is too close to the sim box edge ({dist_to_edge:.3} Å). \
                         Pos ({:.3}, {:.3}, {:.3})",
                    p.x, p.y, p.z,
                )));
            }
        }

        // Inter-molecular minimum-image overlap check.
        // This catches the periodic-image case: molecules that appear far apart in direct
        // space but whose images wrap to overlap on the other side of the cell.
        if self.mol_start_indices.len() > 1 {
            let mut atom_mol = vec![0usize; self.atoms.len()];
            for (mol_i, &start) in self.mol_start_indices.iter().enumerate() {
                let end = self
                    .mol_start_indices
                    .get(mol_i + 1)
                    .copied()
                    .unwrap_or(self.atoms.len());
                for idx in start..end {
                    atom_mol[idx] = mol_i;
                }
            }

            for i in 0..self.atoms.len() {
                for j in (i + 1)..self.atoms.len() {
                    if atom_mol[i] == atom_mol[j] {
                        continue;
                    }
                    let diff = self
                        .cell
                        .min_image(self.atoms[i].posit - self.atoms[j].posit);
                    let dist = diff.magnitude();
                    if dist < MIN_INTER_MOL_DIST {
                        let pi = self.atoms[i].posit;
                        let pj = self.atoms[j].posit;
                        eprintln!(
                            "check_for_overlaps_oob FAIL: atom {i} pos=({:.3},{:.3},{:.3}) \
                             atom {j} pos=({:.3},{:.3},{:.3}) direct_dist={:.3} Å  \
                             min_image_dist={dist:.3} Å  cell_extent=({:.3},{:.3},{:.3})",
                            pi.x,
                            pi.y,
                            pi.z,
                            pj.x,
                            pj.y,
                            pj.z,
                            (self.atoms[i].posit - self.atoms[j].posit).magnitude(),
                            self.cell.extent.x,
                            self.cell.extent.y,
                            self.cell.extent.z,
                        );
                        return Err(ParamError::new(&format!(
                            "Atoms from different molecules (indices {i} and {j}) are too \
                                 close in minimum-image distance ({dist:.3} Å). This would cause \
                                 immediate simulation malfunction.",
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn computation_time(&self) -> io::Result<ComputationTime> {
        self.computation_time.time_per_step(self.step_count)
    }

    /// Reset acceleration, force, potential energy, and virial. Do this each step after the first half-step and drift, and
    /// shaking the fixed hydrogens.
    /// We must reset the virial pair prior to accumulating it, which we do when calculating non-bonded
    /// forces. Also reset forces on solvent.
    pub(crate) fn reset_f_acc_pe_virial(&mut self) {
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

        self.barostat.virial = Default::default();

        self.potential_energy = 0.;
        self.potential_energy_nonbonded = 0.;
        self.potential_energy_bonded = 0.;
        self.potential_energy_between_mols = vec![0.; self.mol_start_indices.len().pow(2)];
        self.alch_interaction_energy = 0.0;
    }

    /// Entry point for force application. This includes bonded, non-bonded (LJ and Coulomb/SPME), and
    /// optionally external forces. (Indexed by atom).
    pub(crate) fn apply_all_forces(
        &mut self,
        dev: &ComputationDevice,
        external_force: &Option<Vec<Vec3>>,
    ) {
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
            start = Instant::now();
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
        if !self.cfg.overrides.long_range_recip_disabled {
            let compute_spme_every_step = self.alch_mol_idx.is_some();

            if compute_spme_every_step || self.step_count.is_multiple_of(SPME_RATIO) {
                // Compute SPME recip forces as usual, and cache for use in steps where we don't.

                // Note: This relies on SPME_RATIO being divisible by COMPUTATION_TIME_RATIO.
                // It will produce inaccurate results otherwise.
                if log_time {
                    start = Instant::now();
                }

                let data = self.handle_spme_recip(dev);

                if !compute_spme_every_step && SPME_RATIO != 1 {
                    self.spme_force_prev = Some(data);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ewald_long_range_sum += elapsed;
                }
            } else {
                // Use the previously-cached SPME forces.
                match &self.spme_force_prev {
                    Some((forces, potential_e, virial_e)) => {
                        // Unpack; forces were applied to flattened solvent and non-solvent
                        // due to how our GPU pipeline works.

                        // self.unpack_apply_pme_forces(forces, &[]);
                        // todo: This is a C+P from the unpack fn! We are getting a borrow error otherwise.
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

                        self.potential_energy += potential_e;
                        self.potential_energy_nonbonded += potential_e;

                        self.barostat.virial.nonbonded_long_range += virial_e;
                    }
                    None => {
                        eprintln!(
                            "Error! Attempting to use cached previous SPME forces, but it's not set"
                        );
                    }
                }
            }
        }

        if let Some(f_ext) = external_force {
            for (i, f) in f_ext.iter().enumerate() {
                self.atoms[i].force += *f;
            }
        }
    }
    /// Compute the instantaneous ∂H/∂λ in kcal/mol for the current alchemical molecule.
    ///
    /// For linear coupling/decoupling:
    ///   H(λ) = H_solvent + (1−λ)·U_alch_interact + H_solute_bonded
    ///   ∂H/∂λ = −U_alch_interact
    ///
    /// where U_alch_interact is the non-bonded interaction energy of molecule
    /// `alch_mol_idx` with all other molecules, accumulated while computing the
    /// short-range and reciprocal non-bonded interactions for the current step.
    ///
    /// Returns 0.0 when `alch_mol_idx` is `None`.
    ///
    /// # Physical correctness
    /// For true thermodynamic integration at intermediate λ values, the non-bonded
    /// forces on the alchemical molecule must be scaled by `(1 − λ)` in
    /// `apply_nonbonded_forces`.  Without that scaling every window samples the
    /// fully-coupled ensemble and TI is equivalent to a single-point FEP estimate.
    pub fn compute_dh_dl(&self) -> f64 {
        if self.alch_mol_idx.is_none() {
            return 0.0;
        }

        -self.alch_interaction_energy
    }

    pub(crate) fn alchemical_atom_range(&self) -> Option<(usize, usize)> {
        let mol_idx = self.alch_mol_idx?;
        let start = *self.mol_start_indices.get(mol_idx)?;
        let end = self
            .mol_start_indices
            .get(mol_idx + 1)
            .copied()
            .unwrap_or(self.atoms.len());

        Some((start, end))
    }
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

    let (mut md_state, _) = MdState::new(dev, &cfg, mols, param_set)?;

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

fn water_mols_from_gro(gro: &Gro) -> Result<Vec<WaterMolOpc>, ParamError> {
    const NM_TO_ANGSTROM: f32 = 10.0;

    #[derive(Default)]
    struct WaterGroMol {
        o: Option<(Vec3, Vec3)>,
        h0: Option<(Vec3, Vec3)>,
        h1: Option<(Vec3, Vec3)>,
    }

    let mut water_by_mol_id: BTreeMap<u32, WaterGroMol> = BTreeMap::new();

    for atom in &gro.atoms {
        if atom.mol_name != "SOL" {
            continue;
        }

        let Some(vel) = atom.velocity else {
            return Err(ParamError::new(&format!(
                "Missing velocity on water atom {} in octanol/water GRO template.",
                atom.serial_number
            )));
        };

        let posit: Vec3 = atom.posit.into();
        let vel: Vec3 = vel.into();
        let posit = posit * NM_TO_ANGSTROM;
        let vel = vel * NM_TO_ANGSTROM;

        let entry = water_by_mol_id.entry(atom.mol_id).or_default();

        match atom.atom_type.as_str() {
            "OW" => entry.o = Some((posit, vel)),
            "HW1" => entry.h0 = Some((posit, vel)),
            "HW2" => entry.h1 = Some((posit, vel)),
            _ => (),
        }
    }

    let mut water = Vec::with_capacity(water_by_mol_id.len());

    for (mol_id, sites) in water_by_mol_id {
        let Some((o_posit, o_vel)) = sites.o else {
            return Err(ParamError::new(&format!(
                "Water molecule {mol_id} is missing OW in octanol/water GRO template."
            )));
        };
        let Some((h0_posit, h0_vel)) = sites.h0 else {
            return Err(ParamError::new(&format!(
                "Water molecule {mol_id} is missing HW1 in octanol/water GRO template."
            )));
        };
        let Some((h1_posit, h1_vel)) = sites.h1 else {
            return Err(ParamError::new(&format!(
                "Water molecule {mol_id} is missing HW2 in octanol/water GRO template."
            )));
        };

        let mut mol = WaterMolOpc::new(o_posit, o_vel, Quaternion::new_identity());
        mol.o.posit = o_posit;
        mol.o.vel = o_vel;
        mol.h0.posit = h0_posit;
        mol.h0.vel = h0_vel;
        mol.h1.posit = h1_posit;
        mol.h1.vel = h1_vel;
        mol.update_virtual_site();

        water.push(mol);
    }

    Ok(water)
}

// todo: Move this A/R
fn add_ions(state: &mut MdState, net_q_e: f32, n_ions: usize) {
    // Add counter-ions to neutralize any net charge.
    // Joung–Cheatham parameters tuned for OPC water (Amber frcmod.ionsjc_opc).
    // sigma = 2 * R_MIN_HALF / 2^(1/6)
    if n_ions > 0 && !state.water.is_empty() {
        // Positive net → add Cl⁻;  negative net → add Na⁺.
        let (ff_type, elem, mass, q_scaled, sigma, eps): (&str, Element, f32, f32, f32, f32) =
            if net_q_e > 0.0 {
                (
                    "Cl-",
                    Element::Chlorine,
                    35.45,
                    -CHARGE_UNIT_SCALER,
                    4.478,
                    0.0073,
                )
            } else {
                (
                    "Na+",
                    Element::Sodium,
                    22.99,
                    CHARGE_UNIT_SCALER,
                    2.439,
                    0.1065,
                )
            };

        let stride = (state.water.len() / n_ions).max(1);
        let mut water_to_remove: Vec<usize> = (0..n_ions)
            .map(|i| (i * stride).min(state.water.len() - 1))
            .collect();

        for (k, &w_idx) in water_to_remove.iter().enumerate() {
            let atom_idx = state.atoms.len() + k;
            let posit = state.water[w_idx].o.posit;

            state.atoms.push(AtomDynamics {
                serial_number: atom_idx as u32,
                force_field_type: ff_type.to_string(),
                element: elem,
                posit,
                mass,
                partial_charge: q_scaled,
                lj_sigma: sigma,
                lj_eps: eps,
                ..Default::default()
            });
            state.force_field_params.mass.insert(
                atom_idx,
                MassParams {
                    atom_type: ff_type.to_string(),
                    mass,
                    comment: None,
                },
            );
            state.force_field_params.lennard_jones.insert(
                atom_idx,
                LjParams {
                    atom_type: ff_type.to_string(),
                    sigma,
                    eps,
                },
            );
            state.adjacency_list.push(Vec::new());
            state.mass_accel_factor.push(KCAL_TO_NATIVE / mass);
            state.mol_start_indices.push(atom_idx);
        }
        // Expand per-mol energy tracking for the new ion "molecules".
        let n_mols = state.mol_start_indices.len();
        state
            .potential_energy_between_mols
            .resize(n_mols.pow(2), 0.0);

        // Remove displaced water molecules (in reverse index order).
        water_to_remove.sort_unstable_by(|a, b| b.cmp(a));
        water_to_remove.dedup();
        for idx in water_to_remove {
            state.water.remove(idx);
        }

        eprintln!(
            "Added {n_ions} {} ion(s) to neutralize net charge ({net_q_e:+.3}e).",
            ff_type
        );
    } else if n_ions > 0 {
        eprintln!(
            "Warning: net charge {net_q_e:+.3}e detected but no solvent available to \
                 displace; skipping ion insertion."
        );
    }
}
