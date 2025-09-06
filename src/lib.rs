#![allow(non_snake_case)]

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
//! - Atoms are divided into three categories:
//! -- Dynamic: Atoms that move
//! -- Static: Atoms that don't move, but have mutual non-bonded interactions with dynamic atoms and water
//! -- Water: Rigid OPC water molecules that have mutual non-bonded interactions with dynamic atoms and water
//!
//! - Thermostat/barostat, with a way to specify temp, pressure, water density
//! - OPC water model
//! - Cell wrapping
//! - Verlet integration (Water and non-water)
//! - Amber parameters for mass, partial charge, VdW (via LJ), dihedral/improper, angle, bond len
//! - Optimizations for Coulomb: Ewald/PME/SPME?
//! - Optimizations for LJ: Dist cutoff for now.
//! - Amber 1-2, 1-3 exclusions, and 1-4 scaling of covalently-bonded atoms.
//! - Rayon parallelization of non-bonded forces
//! - WIP SIMD and CUDA parallelization of non-bonded forces, depending on hardware availability. todo
//! - A thermostat+barostat for the whole system. (Is water and dyn separate here?) todo
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
//! On f32 vs f64 floating point precision: f32 may be good enough fo rmost things, and typical MD packages
//! use mixed precision. Long-range electrostatics are a good candidate for using f64. Or, very long
//! runs.
//!
//! Note on performance: It appears that non-bonded forces dominate computation time. This is my observation,
//! and it's confirmed by an LLM. Both LJ and Coulomb take up most of the time; bonded forces
//! are comparatively insignificant. Building neighbor lists are also significant. These are the areas
//! we focus on for parallel computation (Thread pools, SIMD, CUDA)

// todo: Long-term, you will need to figure out what to run as f32 vice f64, especially
// todo for being able to run on GPU.

mod ambient;
mod bonded;
mod bonded_forces;
mod forces;
mod neighbors;
mod non_bonded;
pub mod params;
mod prep;
mod util;
mod water_init;
mod water_opc;
mod water_settle;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{collections::HashSet, time::Instant};

use ambient::SimBox;
pub use bio_files as files;
use bio_files::{AtomGeneric, BondGeneric, amber_params::ForceFieldParamsKeyed};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaModule, CudaStream};
use ewald::{PmeRecip, ewald_comp_force};
use lin_alg::f64::Vec3;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{Vec3x4, f64x4};
use na_seq::Element;
use neighbors::NeighborsNb;
pub use prep::{HydrogenMdType, merge_params};
pub use water_opc::ForcesOnWaterMol;

use crate::{
    ambient::BerendsenBarostat,
    non_bonded::{
        CHARGE_UNIT_SCALER, EWALD_ALPHA, LjTableIndices, LjTables, SCALE_COUL_14, SPME_N,
    },
    params::{FfParamSet, ForceFieldParamsIndexed},
    water_init::make_water_mols,
    water_opc::WaterMol,
};

const SNAPSHOT_RATIO: usize = 1;

/// Convert convert kcal mol⁻¹ Å⁻¹ (Values in the Amber parameter files) to amu Å ps⁻². Multiply all bonded
/// accelerations by this. TODO: we are currently multiplying *all* accelerations by this.
const ACCEL_CONVERSION: f64 = 418.4;
pub const ACCEL_CONVERSION_INV: f64 = 1. / ACCEL_CONVERSION;

// For assigning velocities from temperature, and other thermostat/barostat use.
pub const KB: f64 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹ (Amber-style units)

// SHAKE tolerances for fixed hydrogens. These SHAKE constraints are for fixed hydrogens.
// The tolerance controls how close we get
// to the target value; lower values are more precise, but require more iterations. `SHAKE_MAX_ITER`
// constrains the number of iterations.
const SHAKE_TOL: f64 = 1.0e-4; // Å
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

/// This stores the positions and velocities of all atoms in the system, and the total energy.
#[derive(Debug, Default)]
pub struct Snapshot {
    pub time: f64,
    // todo: You will need to store this by molecule, when we support that.
    pub atom_posits: Vec<Vec3>,
    pub atom_velocities: Vec<Vec3>,
    pub water_o_posits: Vec<Vec3>,
    pub water_h0_posits: Vec<Vec3>,
    pub water_h1_posits: Vec<Vec3>,
    /// Single velocity per water molecule, as it's rigid.
    pub water_velocities: Vec<Vec3>,
    pub energy_kinetic: f32,
    pub energy_potential: f32,
    // For now, I believe velocities are unused, but tracked here for non-water atoms.
    // We can add water velocities if needed.
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

/// Packages information required to perform dynamics on a Molecule. The significance
/// of breaking the simulation into molecules is due to a few things:
///-  The fixed covalent bond model we use.
#[derive(Clone, Debug)]
pub struct MolDynamics<'a> {
    pub ff_mol_type: FfMolType,
    /// These must hold force field type and partial charge.
    // pub atoms: Vec<AtomGeneric>,
    pub atoms: &'a [AtomGeneric],
    /// Separate from `atoms`; this may be more convenient than mutating the atoms
    /// as they may move! If None, we use the positions stored in the atoms.
    // pub atom_posits: Vec<Vec3>,
    pub atom_posits: Option<&'a [Vec3]>,
    pub bonds: &'a [BondGeneric],
    /// If None, will be generated automatically
    pub adjacency_list: Option<&'a [Vec<usize>]>,
    /// If true,
    pub static_: bool,
    /// If present, any values here override molecule-type general parameters.
    // pub mol_specific_params: Option<ForceFieldParamsKeyed>
    pub mol_specific_params: Option<&'a ForceFieldParamsKeyed>,
}

/// A trimmed-down atom for use with molecular dynamics. Contains parameters for single-atom,
/// but we use ParametersIndex for multi-atom parameters.
#[derive(Clone, Debug)]
pub struct AtomDynamics {
    pub serial_number: u32,
    pub force_field_type: String,
    pub element: Element,
    // pub name: String,
    pub posit: Vec3,
    /// Å / ps
    pub vel: Vec3,
    /// Å / ps²
    pub accel: Vec3,
    /// Daltons
    /// todo: Move these 4 out of this to save memory; use from the params struct directly.
    pub mass: f64,
    /// Amber charge units. This is not the elementary charge units found in amino19.lib and gaff2.dat;
    /// it's scaled by a constant.
    pub partial_charge: f64,
    /// Å
    pub lj_sigma: f64,
    /// kcal/mol
    pub lj_eps: f64,
}

impl AtomDynamics {
    pub fn new(
        atom: &AtomGeneric,
        atom_posits: &[Vec3],
        ff_params: &ForceFieldParamsIndexed,
        i: usize,
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
            element: atom.element,
            // name: atom.type_in_res.clone().unwrap_or_default(),
            posit: atom_posits[i],
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: ff_params.mass.get(&i).unwrap().mass as f64,
            // We get partial charge for ligands from (e.g. Amber-provided) Mol files, so we load it from the atom, vice
            // the loaded FF params. They are not in the dat or frcmod files that angle, bond-length etc params are from.
            partial_charge: CHARGE_UNIT_SCALER * atom.partial_charge.unwrap_or_default() as f64,
            lj_sigma: ff_params.lennard_jones.get(&i).unwrap().sigma as f64,
            lj_eps: ff_params.lennard_jones.get(&i).unwrap().eps as f64,
            force_field_type: ff_type,
        })
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl AtomDynamicsx4 {
    pub fn from_array(bodies: [AtomDynamics; 4]) -> Self {
        let mut posits = [Vec3::new_zero(); 4];
        let mut vels = [Vec3::new_zero(); 4];
        let mut accels = [Vec3::new_zero(); 4];
        let mut masses = [0.0; 4];
        // Replace `Element::H` (for example) with some valid default for your `Element` type:
        let mut elements = [Element::Hydrogen; 4];

        for (i, body) in bodies.into_iter().enumerate() {
            posits[i] = body.posit;
            vels[i] = body.vel;
            accels[i] = body.accel;
            masses[i] = body.mass;
            elements[i] = body.element;
        }

        Self {
            posit: Vec3x4::from_array(posits),
            vel: Vec3x4::from_array(vels),
            accel: Vec3x4::from_array(accels),
            mass: f64x4::from_array(masses),
            element: elements,
        }
    }
}

#[derive(Default)]
pub struct MdState {
    // todo: Update how we handle mode A/R.
    // todo: You need to rework this state in light of arbitrary mol count.
    pub atoms: Vec<AtomDynamics>,
    pub adjacency_list: Vec<Vec<usize>>,
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    /// and VDW) only.
    pub atoms_static: Vec<AtomDynamics>,
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
    /// This stores
    pub snapshots: Vec<Snapshot>,
    pub cell: SimBox,
    pub neighbors_nb: NeighborsNb,
    // max_disp_sq: f64,           // track atom displacements²
    /// K
    temp_target: f64,
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
    hydrogen_md_type: HydrogenMdType,
    // todo: Hmm... Is this DRY with forces_on_water? Investigate.
    pub water_pme_sites_forces: Vec<[Vec3; 3]>, // todo: A/R
    pme_recip: Option<PmeRecip>,
    /// kcal/mol
    pub potential_energy: f64,
}

impl MdState {
    pub fn new(
        // todo: Support multiple molecules.
        mols: &[MolDynamics],
        temp_target: f64,     // K
        pressure_target: f64, // k
        ff_params: &FfParamSet,
        mut hydrogen_md_type: HydrogenMdType,
    ) -> Result<Self, ParamError> {
        // todo: QC how you handle hydrogen_md_type.

        if mols.is_empty() {
            return Err(ParamError::new(&"We require at least one dynamic mol"));
        }

        // todo: For now, just a single molecule.

        for mol in mols {
            let params_general = match mol.ff_mol_type {
                FfMolType::Peptide => &ff_params.peptide,
                FfMolType::SmallOrganic => &ff_params.small_mol,
                FfMolType::Dna => &ff_params.dna,
                FfMolType::Rna => &ff_params.rna,
                FfMolType::Lipid => &ff_params.lipids,
                FfMolType::Carbohydrate => &ff_params.carbohydrates,
            };

            let Some(params_general) = params_general else {
                return Err(ParamError::new(&format!(
                    "Missing general parameters for {:?}",
                    mol.ff_mol_type
                )));
            };

            // Set up Indexed params
            let ff_params = ForceFieldParamsIndexed::new(
                params_general,
                mol.mol_specific_params,
                mol.atoms,
                adjacency_list,
                &mut hydrogen_md_type,
            )?;

            if mol.static_ {
            } else {
            }
        }

        // This assumes nonbonded interactions only with external atoms; this is fine for
        // rigid protein models, and is how this is currently structured.
        let bonds_static = Vec::new();
        let adj_list_static = Vec::new();

        println!("\nBuilding FF params indexed static for docking...");
        let ff_params_static = ForceFieldParamsIndexed::new(
            ff_params_protein,
            None,
            &atoms_static_,
            &bonds_static,
            &adj_list_static,
            &mut hydrogen_md_type,
        )?;

        // Convert from AtomGeneric to AtomDynamics

        // We are using this approach instead of `.into`, so we can use the atom_posits from
        // the positioned ligand. (its atom coords are relative; we need absolute)
        let mut atoms_dy = Vec::with_capacity(atoms_dy_.len());
        for (i, atom) in atoms_dy_.iter().enumerate() {
            atoms_dy.push(AtomDynamics::new(
                &atom,
                atom_posits,
                &ff_params_dynamic,
                i,
            )?);
        }

        let mut atoms_static = Vec::with_capacity(atoms_static_.len());
        let atom_posits_static: Vec<_> = atoms_static_.iter().map(|a| a.posit).collect();

        for (i, atom) in atoms_static_.iter().enumerate() {
            atoms_static.push(AtomDynamics::new(
                &atom,
                &atom_posits_static,
                &ff_params_static,
                i,
            )?);
        }

        // let cell = SimBox::new_padded(&atoms_dy);
        let cell = SimBox::new_fixed_size(&atoms_dy);

        let mut result = Self {
            atoms: atoms_dy,
            adjacency_list: adjacency_list.to_vec(),
            atoms_static,
            cell,
            pairs_excluded_12_13: HashSet::new(),
            pairs_14_scaled: HashSet::new(),
            force_field_params: ff_params_dynamic,
            temp_target,
            hydrogen_md_type,
            ..Default::default()
        };

        result.barostat.pressure_target = pressure_target;

        result.water = make_water_mols(
            &result.cell,
            result.temp_target,
            &result.atoms,
            &result.atoms_static,
        );
        result.water_pme_sites_forces = vec![[Vec3::new_zero(); 3]; result.water.len()];

        result.setup_nonbonded_exclusion_scale_flags();

        result.init_neighbors();

        // Initializes the FFT planner[s], among other things.
        result.regen_pme();

        // Set up our LJ cache.
        for i in 0..result.atoms.len() {
            for &j in &result.neighbors_nb.dy_dy[i] {
                if j < i {
                    // Prevents duplication of the pair in the other order.
                    continue;
                }

                non_bonded::setup_lj_cache(
                    &result.atoms[i],
                    &result.atoms[j],
                    LjTableIndices::DynDyn((i, j)),
                    &mut result.lj_tables,
                );
            }
        }

        for (i_lig, a_lig) in result.atoms.iter_mut().enumerate() {
            // Dynamic, static
            for (i_static, a_static) in result.atoms_static.iter().enumerate() {
                non_bonded::setup_lj_cache(
                    a_lig,
                    a_static,
                    LjTableIndices::DynStatic((i_lig, i_static)),
                    &mut result.lj_tables,
                );
            }

            // Dynamic, water
            if !result.water.is_empty() {
                // Each water is identical, so we only need to do this once per lig, and static atom.
                non_bonded::setup_lj_cache(
                    a_lig,
                    &result.water[0].o,
                    LjTableIndices::DynWater(i_lig),
                    &mut result.lj_tables,
                );
            }
        }

        // Static, water
        if !result.water.is_empty() {
            for (i_static, a_static) in result.atoms_static.iter().enumerate() {
                non_bonded::setup_lj_cache(
                    a_static,
                    &result.water[0].o,
                    LjTableIndices::StaticWater(i_static),
                    &mut result.lj_tables,
                );
            }
        }

        Ok(result)
    }

    /// Reset acceleration and virial pair. Do this each step after the first half-step and drift, and
    /// shaking the fixed hydrogens.
    /// We must reset the virial pair prior to accumulating it, which we do when calculating non-bonded
    /// forces. Also reset forces on water.
    fn reset_accels(&mut self) {
        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
        }
        for mol in &mut self.water {
            mol.o.accel = Vec3::new_zero();
            mol.m.accel = Vec3::new_zero();
            mol.h0.accel = Vec3::new_zero();
            mol.h1.accel = Vec3::new_zero();
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

    /// One **Velocity-Verlet** step (leap-frog style) of length `dt` is in picoseconds (10^-12),
    /// with typical values of 0.001, or 0.002ps (1 or 2fs).
    /// This method orchestrates the dynamics at each time step.
    pub fn step(&mut self, dev: &ComputationDevice, dt: f64) {
        let dt_half = 0.5 * dt;

        // First half-kick (v += a dt/2) and drift (x += v dt)
        // todo: Do we want traditional verlet instead of velocity verlet (VV)?
        // Note: We do not apply the accel unit conversion, nor mass division here; they're already
        // included in this values from the previous step.
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half; // Half-kick

            a.posit += a.vel * dt; // Drift
            a.posit = self.cell.wrap(a.posit);

            // todo: What is this? Implement it, or remove it?
            // todo: Should this take water displacements into account?
            // track the largest squared displacement to know when to rebuild the list
            self.neighbors_nb.max_displacement_sq = self
                .neighbors_nb
                .max_displacement_sq
                .max((a.vel * dt).magnitude_squared());
        }

        // todo: Consider applying the thermostat between the first half-kick and drift.
        // todo: e.g. half-kick, then shake H and settle velocity water (?), then thermostat, then drift. (?) ,
        // todo then settle positions?

        self.water_vv_first_half_and_drift(dt, dt_half);

        // The order we perform these steps is important.
        if let HydrogenMdType::Fixed(_) = &self.hydrogen_md_type {
            self.shake_hydrogens();
        }

        self.reset_accels();

        self.apply_all_forces(dev);

        let start = Instant::now();
        // todo: YOu need to update potential energy from LR PME as well.
        self.handle_spme_recip(dev);
        if self.step_count == 0 {
            let elapsed = start.elapsed();
            println!("SPME recip time: {:?} μs", elapsed.as_micros());
        }

        // Forces (bonded and nonbonded, to dynamic and water atoms) have been applied; perform other
        // steps required for integration; second half-kick, RATTLE for hydrogens; SETTLE for water. -----

        // Second half-kick using the forces calculated this step, and update accelerations using the atom's mass;
        // Between the accel reset and this step, the accelerations have been missing those factors; this is an optimization to
        // do it once at the end.
        for a in &mut self.atoms {
            // We divide by mass here, once accelerations have been computed in parts above; this
            // is an optimization to prevent dividing each accel component by it.
            // This is the step where we A: convert force to accel, and B: Convert units from the param
            // units to the ones we use in dynamics.
            a.accel *= ACCEL_CONVERSION / a.mass;
            a.vel += a.accel * dt_half;
        }

        // self.water_vv_second_half(&mut self.forces_on_water, dt_half);
        self.water_vv_second_half(dt_half);

        if let HydrogenMdType::Fixed(_) = &self.hydrogen_md_type {
            self.rattle_hydrogens();
        }

        // I believe we must run barostat prior to thermostat, in our current configuration.
        self.apply_barostat_berendsen(dt);
        self.apply_thermostat_csvr(dt, self.temp_target);

        self.time += dt;
        self.step_count += 1;

        // todo: Ratio for this too?
        self.build_neighbors_if_needed();

        // We keeping the cell centered on the dynamics atoms. Note that we don't change the dimensions,
        // as these are under management by the barostat.
        if self.step_count % CENTER_SIMBOX_RATIO == 0 {
            self.cell.recenter(&self.atoms);

            // todo: Will this interfere with carrying over state from the previous step?
            self.regen_pme();
        }

        if self.step_count % SNAPSHOT_RATIO == 0 {
            self.take_snapshot();
        }
    }

    fn handle_spme_recip(&mut self, dev: &ComputationDevice) {
        const K_COUL: f64 = 1.; // todo: ChatGPT really wants this, but I don't think I need it.

        let (pos_all, q_all, map) = self.gather_pme_particles_wrapped();

        let (mut f_recip, e_recip) = match &mut self.pme_recip {
            Some(pme_recip) => {
                match dev {
                    ComputationDevice::Cpu => pme_recip.forces(&pos_all, &q_all),
                    #[cfg(feature = "cuda")]
                    ComputationDevice::Gpu((stream, module)) => {
                        // self.pme_recip.forces_gpu(stream, module, &pos_all, &q_all)
                        // todo: GPU isn't improving this, but it should be
                        pme_recip.forces(&pos_all, &q_all)
                    }
                }
            }
            None => {
                panic!("No PME recip available; not computing SPME recip.");
            }
        };

        self.potential_energy += e_recip;

        // println!("F RECIP: {:?}", &f_recip[0..20]);

        // todo: QC this.
        // Scale to Amber force units
        for f in f_recip.iter_mut() {
            *f *= K_COUL;
        }

        let mut w_recip = 0.0;
        for (k, tag) in map.iter().enumerate() {
            match *tag {
                PMEIndex::Dyn(i) => {
                    self.atoms[i].accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]); // tin-foil virial
                }
                PMEIndex::WatO(i) => {
                    self.water[i].o.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::WatM(i) => {
                    self.water[i].m.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::WatH0(i) => {
                    self.water[i].h0.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::WatH1(i) => {
                    self.water[i].h1.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::Static(_) => { /* contributes to field, no accel update */ }
            }
        }
        self.barostat.virial_pair_kcal += w_recip;

        // 1–4 Coulomb scaling correction
        for &(i, j) in &self.pairs_14_scaled {
            let diff = self
                .cell
                .min_image(self.atoms[i].posit - self.atoms[j].posit);
            let r = diff.magnitude();
            let dir = diff / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            let Some(pme_recip) = &mut self.pme_recip else {
                panic!("Missing PME recip; code error");
            };
            let df = ewald_comp_force(dir, r, qi, qj, pme_recip.alpha)
                * (SCALE_COUL_14 - 1.0) // todo: Cache this.
                * K_COUL;

            self.atoms[i].accel += df;
            self.atoms[j].accel -= df;
            self.barostat.virial_pair_kcal += (dir * r).dot(df); // r·F
        }
    }

    // todo: GPT helper. QC, and simplify as required.
    /// Gather all particles that contribute to PME (dyn, water sites, statics).
    /// Returns positions wrapped to the primary box, their charges, and a map telling
    /// us which original DOF each entry corresponds to.
    fn gather_pme_particles_wrapped(&self) -> (Vec<Vec3>, Vec<f64>, Vec<PMEIndex>) {
        let n_dyn = self.atoms.len();
        let n_wat = self.water.len();
        let n_st = self.atoms_static.len();

        // Capacity hint: dyn + 4*water + statics
        let mut pos = Vec::with_capacity(n_dyn + 4 * n_wat + n_st);
        let mut q = Vec::with_capacity(pos.capacity());
        let mut map = Vec::with_capacity(pos.capacity());

        // Dynamic atoms
        for (i, a) in self.atoms.iter().enumerate() {
            pos.push(self.cell.wrap(a.posit)); // [0,L) per axis
            q.push(a.partial_charge); // already scaled to Amber units
            map.push(PMEIndex::Dyn(i));
        }

        // Water sites (OPC: O usually has 0 charge; include anyway—cost is negligible)
        for (i, w) in self.water.iter().enumerate() {
            pos.push(self.cell.wrap(w.o.posit));
            q.push(w.o.partial_charge);
            map.push(PMEIndex::WatO(i));
            pos.push(self.cell.wrap(w.m.posit));
            q.push(w.m.partial_charge);
            map.push(PMEIndex::WatM(i));
            pos.push(self.cell.wrap(w.h0.posit));
            q.push(w.h0.partial_charge);
            map.push(PMEIndex::WatH0(i));
            pos.push(self.cell.wrap(w.h1.posit));
            q.push(w.h1.partial_charge);
            map.push(PMEIndex::WatH1(i));
        }

        // Static atoms (contribute to field but you won't update accel)
        for (i, a) in self.atoms_static.iter().enumerate() {
            pos.push(self.cell.wrap(a.posit));
            q.push(a.partial_charge);
            map.push(PMEIndex::Static(i));
        }

        // Optional sanity check (debug only): near-neutral total charge
        #[cfg(debug_assertions)]
        {
            let qsum: f64 = q.iter().sum();
            if qsum.abs() > 1e-6 {
                eprintln!(
                    "[PME] Warning: net charge = {qsum:.6e} (PME assumes neutral or a uniform background)"
                );
            }
        }

        (pos, q, map)
    }

    /// Run this at init, and whenever you update the sim box.
    pub fn regen_pme(&mut self) {
        let [lx, ly, lz] = self.cell.extent.to_arr();
        self.pme_recip = Some(PmeRecip::new(
            (SPME_N, SPME_N, SPME_N),
            (lx, ly, lz),
            EWALD_ALPHA,
        ));
    }

    /// Note: This is currently only for the dynamic atoms; does not take water kinetic energy into account.
    fn current_kinetic_energy(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 0.5 * a.mass * a.vel.magnitude_squared())
            .sum()
    }

    pub fn take_snapshot(&mut self) {
        let mut water_o_posits = Vec::with_capacity(self.water.len());
        let mut water_h0_posits = Vec::with_capacity(self.water.len());
        let mut water_h1_posits = Vec::with_capacity(self.water.len());

        for water in &self.water {
            water_o_posits.push(water.o.posit);
            water_h0_posits.push(water.h0.posit);
            water_h1_posits.push(water.h1.posit);
        }

        self.snapshots.push(Snapshot {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            // todo: Calculate and store kinetic energy elsewhere, A/R.
            energy_kinetic: self.current_kinetic_energy() as f32,
            energy_potential: self.potential_energy as f32,
        })
    }
}

#[inline]
/// Mutable aliasing helpers.
pub fn split2_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
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
pub fn split4_mut<T>(
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
