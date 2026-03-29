//! Related to storing snapshots (also known as trajectories) of MD runs.

use std::{
    collections::{HashMap, HashSet},
    io::{self, ErrorKind},
    path::Path,
    time::Instant,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, ChargeType, MmCif, Mol2, MolType,
    dcd::{DcdFrame, DcdTrajectory, DcdUnitCell},
    gromacs::{GromacsFrame, OutputControl, output::write_trr},
    md_params::{ForceFieldParams, LjParams, MassParams},
    xtc::write_xtc,
};
#[cfg(feature = "cuda")]
use cudarc::{driver::CudaContext, nvrtc::Ptx};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use na_seq::Element;

use crate::{
    AtomDynamics,
    COMPUTATION_TIME_RATIO,
    ComputationDevice,
    FfMolType,
    KCAL_TO_NATIVE,
    MdConfig,
    MdState,
    MolDynamics,
    ParamError,
    SPME_RATIO,
    SimBoxInit,
    Solvent,
    WaterInitTemplate,
    barostat::SimBox,
    merge_params,
    neighbors::NeighborsNb,
    non_bonded::{CHARGE_UNIT_SCALER, LjTables},
    param_inference::update_small_mol_params,
    params::{FfParamSet, ForceFieldParamsIndexed},
    // snapshot_mdt::save_mdt,
    solvent::init::{make_water_mols, pack_custom_solvent},
    util::{ComputationTime, build_adjacency_list},
};
#[cfg(feature = "cuda")]
use crate::{
    PTX,
    gpu_interface::{ForcesPositsGpu, PerNeighborGpu},
};

// Append to any snapshot-saving files every this number of snapshots. E.g.
// DCD, TRR, XTC. We want this to be such that we don't experience too much memory use.
// // todo:  Update A/R. Likely higher. Lower now just to test.
pub(crate) const TRAJ_FILE_SAVE_INTERVAL: usize = 100;
// pub(crate) const TRAJ_FILE_SAVE_INTERVAL: usize = 3_000;

const TRAJ_OUT_PATH: &str = "./md_out";

#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
/// For saving snapshots.
pub struct SnapshotHandlers {
    pub memory: Option<usize>,
    pub dcd: Option<usize>,
    /// This includes detailed data for saving positions, velocities, forces etc separately
    /// to TRR files, and saving energy data to EDR files. Also can write XTC,
    pub gromacs: OutputControl,
}

impl Default for SnapshotHandlers {
    fn default() -> Self {
        Self {
            memory: Some(10),
            dcd: None,
            gromacs: OutputControl {
                // Not the GROMACS default, but to use this in our workflows that parse TRR,
                // we need something here.
                nstxout: Some(10),
                ..Default::default()
            },
        }
    }
}

/// Pressure, temperature, energy, etc. Could also be described as thermodynamic properties.
#[derive(Clone, Debug)]
pub struct SnapshotEnergyData {
    pub energy_kinetic: f32,
    pub energy_potential: f32,
    /// Used to track which molecule each atom is associated with in our flattened structures.
    /// This is the potential energy between every pair of molecules.
    pub energy_potential_between_mols: Vec<f32>,
    /// Energy from non-bonded interactions only. A simple proxy for molecule-molecule potential energy,
    /// although includes energy within the molecule as well.
    pub energy_potential_nonbonded: f32,
    /// E.g. energy in covalent bonds, as modelled as oscillators.
    pub energy_potential_bonded: f32,
    /// Optionally added as a post-processing step.
    pub hydrogen_bonds: Vec<HydrogenBond>,
    /// Instantaneous temperature in Kelvin.
    pub temperature: f32,
    /// Instantaneous pressure in Bar.
    pub pressure: f32,
    /// Instantaneous ∂H/∂λ in kcal/mol, for alchemical free energy calculations.
    ///
    /// Non-zero only when `MdState::alch_mol_idx` is set.  For linear decoupling
    /// this equals the negative of the solute–solvent interaction energy at the
    /// current configuration.  Average this over a λ window's trajectory and pass
    /// the result to `alchemical::collect_window` / `alchemical::free_energy_ti`.
    pub dh_dl: f32,
}

/// This stores the positions and velocities of all atoms in the system, and the total energy.
/// It represents the output of the simulation. A set of these can be used to play it back over time.
/// We save load and save this to disk in the __ format.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub time: f64,
    pub atom_posits: Vec<Vec3>,
    pub atom_velocities: Option<Vec<Vec3>>,
    pub energy_data: Option<SnapshotEnergyData>,
    // /// Posits and velocities by mol: Outer index is the molecule index, corresponding to molecules
    // /// in `MdState`
    // // todo: Experimenting with storing snaps as per-mol. This may replace the flat per-atom approach,
    // // todo: but we're leaving per-atom fields in for now. This may effectively double the non-solvent
    // // todo size of the snapshot.
    // pub atom_posits_by_mol: Vec<Vec<Vec3>>,
    pub water_o_posits: Vec<Vec3>,
    pub water_h0_posits: Vec<Vec3>,
    pub water_h1_posits: Vec<Vec3>,
    /// Single velocity per solvent molecule, as it's rigid.
    pub water_velocities: Option<Vec<Vec3>>,
}

impl Snapshot {
    /// Initialize with position data only. We construct these with positions only, then augment
    /// with velocity and energy data as required.
    pub fn new(state: &MdState) -> Self {
        let mut water_o_posits = Vec::with_capacity(state.water.len());
        let mut water_h0_posits = Vec::with_capacity(state.water.len());
        let mut water_h1_posits = Vec::with_capacity(state.water.len());

        for water in &state.water {
            water_o_posits.push(water.o.posit);
            water_h0_posits.push(water.h0.posit);
            water_h1_posits.push(water.h1.posit);
        }

        Self {
            atom_posits: state.atoms.iter().map(|a| a.posit).collect(),
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            ..Default::default()
        }
    }

    pub fn update_with_velocities(&mut self, state: &MdState) {
        self.atom_posits = state.atoms.iter().map(|a| a.vel).collect();
        self.water_velocities = Some(state.water.iter().map(|w| w.o.vel).collect());
    }

    pub fn update_with_energy(&mut self, state: &MdState, pressure: f32, temperature: f32) {
        self.atom_posits = state.atoms.iter().map(|a| a.vel).collect();
        self.water_velocities = Some(state.water.iter().map(|w| w.o.vel).collect());

        let energy_potential_between_mols = state
            .potential_energy_between_mols
            .iter()
            .map(|v| *v as f32)
            .collect();

        self.energy_data = Some(SnapshotEnergyData {
            energy_kinetic: state.kinetic_energy as f32,
            energy_potential: state.potential_energy as f32,
            energy_potential_between_mols,
            energy_potential_nonbonded: state.potential_energy_nonbonded as f32,
            energy_potential_bonded: state.potential_energy_bonded as f32,
            hydrogen_bonds: Vec::new(), // Populated later A/R.
            temperature,
            pressure,
            dh_dl: state.compute_dh_dl() as f32,
        });
    }

    /// Unflatten positions and velocities into a per-molecule basis. `mol_start_indices` may be
    /// taken directly from `MdState`. Inner: (Posit, Vel). Does not unflatten the solvent, which is placed
    /// after all non-solvent molecules in the flat arrays.
    pub fn unflatten(&self, mol_start_indices: &[usize]) -> io::Result<Vec<Vec<(Vec3, Vec3)>>> {
        let n_atoms = self.atom_posits.len();
        let mut per_mol = Vec::with_capacity(mol_start_indices.len());

        for (i, &start) in mol_start_indices.iter().enumerate() {
            let end = if i + 1 < mol_start_indices.len() {
                mol_start_indices[i + 1]
            } else {
                n_atoms
            };

            if end > self.atom_posits.len() {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Snapshot atom position out of range. posit: {end} Len: {}",
                        self.atom_posits.len()
                    ),
                ));
            }

            let atoms = self.atom_posits[start..end]
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let v = self
                        .atom_velocities
                        .as_deref()
                        .and_then(|vels| vels.get(start + i))
                        .copied()
                        .unwrap_or_default();
                    (p, v)
                })
                .collect();

            per_mol.push(atoms);
        }

        Ok(per_mol)
    }

    /// The element indices must match the atom posits.
    pub fn populate_hydrogen_bonds(&mut self, _atoms: &[AtomDynamics]) {
        // let result = create_hydrogen_bonds(&atoms, &self.atom_posits, &self.water_o_posits, &self.bonds);

        // self.hydrogen_bonds = result;
    }

    pub fn to_dcd(&self, cell: &SimBox, write_water: bool) -> DcdFrame {
        let mut atom_posits = self.atom_posits.clone();

        if write_water {
            for pos in &self.water_o_posits {
                atom_posits.push(*pos);
            }
            for pos in &self.water_h0_posits {
                atom_posits.push(*pos);
            }
            for pos in &self.water_h1_posits {
                atom_posits.push(*pos);
            }
        }

        DcdFrame {
            time: self.time,
            atom_posits,
            unit_cell: DcdUnitCell {
                bounds_low: cell.bounds_low,
                bounds_high: cell.bounds_high,
            },
        }
    }

    /// Note: Most of our fields are not available in the DCD format, so we leave them empty, using
    /// the Default impl.
    pub fn from_dcd(dcd: &DcdTrajectory) -> Vec<Self> {
        let mut result = Vec::with_capacity(dcd.frames.len());

        for frame in &dcd.frames {
            result.push(Snapshot {
                time: frame.time,
                atom_posits: frame.atom_posits.clone(),
                ..Default::default()
            })
        }

        result
    }
}

/// Used for visualizing hydrogen bonds on a given snapshot.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HBondAtomType {
    Standard,
    WaterO,
    WaterH0,
    WaterH1,
}

/// Used for visualizing hydrogen bonds on a given snapshot.
#[derive(Clone, Debug)]
pub struct HydrogenBond {
    pub donor: (HBondAtomType, usize),
    pub acceptor: (HBondAtomType, usize),
    pub hydrogen: (HBondAtomType, usize),
}

impl Snapshot {
    pub fn make_mol2(&self, atoms_: &[AtomGeneric], bonds: &[BondGeneric]) -> io::Result<Mol2> {
        if atoms_.len() != self.atom_posits.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Atom position mismatch",
            ));
        }

        let mut atoms = atoms_.to_vec();
        for (i, atom) in atoms.iter_mut().enumerate() {
            atom.posit = self.atom_posits[i].into();
        }

        Ok(Mol2 {
            ident: "MD run".to_string(),
            metadata: HashMap::new(),
            atoms,
            bonds: bonds.to_vec(),
            mol_type: MolType::Small,
            charge_type: ChargeType::User,
            pharmacophore_features: Vec::new(),
            comment: None,
        })
    }

    pub fn make_mmcif(&self, atoms_: &[AtomGeneric], _bonds: &[BondGeneric]) -> io::Result<MmCif> {
        if atoms_.len() != self.atom_posits.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Atom position mismatch",
            ));
        }

        let mut atoms = atoms_.to_vec();
        for (i, atom) in atoms.iter_mut().enumerate() {
            atom.posit = self.atom_posits[i].into();
        }

        Ok(MmCif {
            ident: "MD run".to_string(),
            metadata: HashMap::new(),
            atoms,
            chains: Vec::new(),
            residues: Vec::new(),
            secondary_structure: Vec::new(),
            experimental_method: None,
        })
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

        // We combine all molecule general and specific params into this set, then
        // create Indexed params from it.
        let mut params = ForceFieldParams::default();

        // Used for updating indices for tracking purposes.
        let mut atom_ct_prior_to_this_mol = 0;

        let mut mol_start_indices = Vec::new();

        // Pre-pack custom solvent molecules so their atoms, bonds, and parameters are included
        // in ForceFieldParamsIndexed (built below from the combined atoms list).
        // We extract the declared positions from the regular `mols` to use as exclusion zones.
        let custom_packed: Vec<MolDynamics> = match (&cfg.solvent, &cfg.sim_box) {
            (Solvent::Custom((mols_solvent, _)), SimBoxInit::Fixed((low, high))) => {
                let existing: Vec<Vec3F64> = mols
                    .iter()
                    .flat_map(|m| -> Vec<Vec3F64> {
                        if let Some(ap) = &m.atom_posits {
                            ap.clone()
                        } else {
                            m.atoms.iter().map(|a| a.posit).collect()
                        }
                    })
                    .collect();

                pack_custom_solvent(*low, *high, &existing, mols_solvent)
            }
            (Solvent::Custom(_), SimBoxInit::Pad(_)) => {
                return Err(ParamError {
                    descrip: "Custom solvent with a Pad sim box is not yet supported; \
                     skipping custom solvent packing."
                        .to_string(),
                });
            }
            _ => Vec::new(),
        };

        // Build a combined slice: caller-supplied mols first, then packed custom solvents.
        let combined_mols: Vec<MolDynamics>;
        let all_mols: &[MolDynamics] = if custom_packed.is_empty() {
            mols
        } else {
            combined_mols = mols.iter().cloned().chain(custom_packed).collect();
            &combined_mols
        };

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

            // let mut p: Vec<Vec3> = Vec::new(); // to store the ref.
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

        let force_field_params = ForceFieldParamsIndexed::new(
            &params,
            &atoms_md,
            &adjacency_list,
            cfg.hydrogen_constraint,
        )?;

        let mut mass_accel_factor = Vec::with_capacity(atoms_md.len());

        // Assign mass, LJ params, etc.
        for (i, atom) in atoms_md.iter_mut().enumerate() {
            atom.assign_data_from_params(&force_field_params, i);
            mass_accel_factor.push(KCAL_TO_NATIVE / atom.mass);
        }

        let cell = SimBox::new(&atoms_md, &cfg.sim_box);

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
        result.check_for_overlaps_oob()?;

        result.cell.recenter(&result.atoms);

        // Set up our LJ cache. Do this prior to building neighbors for the first time,
        // as that also sets up the GPU-struct LJ data.
        result.lj_tables = LjTables::new(&result.atoms);

        result.neighbors_nb = NeighborsNb::new(result.cfg.neighbor_skin, result.cfg.coulomb_cutoff);

        result.barostat.pressure_target = cfg.pressure_target.unwrap_or_default() as f64;

        // Custom solvent molecules were pre-packed and added to `all_mols` before the atom-
        // processing loop above, so their atoms are already in `result.atoms` and their
        // parameters are already included in `force_field_params`.  Water (below) will avoid
        // them automatically because `make_water_mols` checks against `result.atoms`.

        result.water = {
            let count = match &cfg.solvent {
                Solvent::None | Solvent::WaterOpc => None,
                Solvent::WaterOpcSpecifyMolCount(c) => Some(*c),
                Solvent::Custom((_, c)) => Some(*c),
            };

            let water_template_override: Option<WaterInitTemplate> = cfg
                .water_template_path
                .as_deref()
                .and_then(|path_str| {
                    match WaterInitTemplate::load(std::path::Path::new(path_str)) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            eprintln!(
                                "Warning: could not load water template from {path_str:?}: {e}. Using default."
                            );
                            None
                        }
                    }
                });

            make_water_mols(
                &result.cell,
                &result.atoms,
                count,
                water_template_override.as_ref(),
                cfg.skip_water_pbc_filter,
            )
        };

        // Add counter-ions to neutralize any net charge.
        // Joung–Cheatham parameters tuned for OPC water (Amber frcmod.ionsjc_opc).
        // sigma = 2 * R_MIN_HALF / 2^(1/6)
        if n_ions > 0 && !result.water.is_empty() {
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

            let stride = (result.water.len() / n_ions).max(1);
            let mut water_to_remove: Vec<usize> = (0..n_ions)
                .map(|i| (i * stride).min(result.water.len() - 1))
                .collect();

            for (k, &w_idx) in water_to_remove.iter().enumerate() {
                let atom_idx = result.atoms.len() + k;
                let posit = result.water[w_idx].o.posit;

                result.atoms.push(AtomDynamics {
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
                result.force_field_params.mass.insert(
                    atom_idx,
                    MassParams {
                        atom_type: ff_type.to_string(),
                        mass,
                        comment: None,
                    },
                );
                result.force_field_params.lennard_jones.insert(
                    atom_idx,
                    LjParams {
                        atom_type: ff_type.to_string(),
                        sigma,
                        eps,
                    },
                );
                result.adjacency_list.push(Vec::new());
                result.mass_accel_factor.push(KCAL_TO_NATIVE / mass);
                result.mol_start_indices.push(atom_idx);
            }
            // Expand per-mol energy tracking for the new ion "molecules".
            let n_mols = result.mol_start_indices.len();
            result
                .potential_energy_between_mols
                .resize(n_mols.pow(2), 0.0);

            // Remove displaced water molecules (in reverse index order).
            water_to_remove.sort_unstable_by(|a, b| b.cmp(a));
            water_to_remove.dedup();
            for idx in water_to_remove {
                result.water.remove(idx);
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

        Ok(result)
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
        self.potential_energy_between_mols = vec![0.; self.mol_start_indices.len().pow(2)]
    }

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
            if self.step_count.is_multiple_of(SPME_RATIO) {
                // Compute SPME recip forces as usual, and cache for use in steps where we don't.

                // Note: This relies on SPME_RATIO being divisible by COMPUTATION_TIME_RATIO.
                // It will produce inaccurate results otherwise.
                if log_time {
                    start = Instant::now();
                }

                let data = self.handle_spme_recip(dev);

                if SPME_RATIO != 1 {
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

    /// Run this each step: For each enabled snapshot handler, store to memory, or save to
    /// disk as required.
    pub(crate) fn handle_snapshots(&mut self, pressure: f32) {
        let i = self.step_count;

        // Compute temperature a maximum of once in this fn.
        let mut temperature = None;

        if let Some(ratio) = self.cfg.snapshot_handlers.memory
            && i.is_multiple_of(ratio)
        {
            if temperature.is_none() {
                temperature = Some(self.measure_temperature() as f32);
            }

            let ss = {
                let mut v = Snapshot::new(self);
                v.update_with_velocities(self);
                v.update_with_energy(self, pressure, temperature.unwrap());

                v
            };

            self.snapshots.push(ss);
        }

        if let Some(ratio) = self.cfg.snapshot_handlers.dcd
            && i.is_multiple_of(ratio)
        {
            // DCD: No energy or velocity, for now. todo: Check teh spec.
            self.snapshot_queue_for_dcd.push(Snapshot::new(self));
        }

        let oc = &self.cfg.snapshot_handlers.gromacs;
        if let Some(ratio) = oc.nstxout
            && i.is_multiple_of(ratio as usize)
        {
            let ss = {
                let mut v = Snapshot::new(self);

                if let Some(ratio_v) = oc.nstvout
                    && i.is_multiple_of(ratio_v as usize)
                {
                    v.update_with_velocities(self);
                }

                // We are ignoring `nstcalcenergy`.
                if let Some(ratio_e) = oc.nstenergy
                    && i.is_multiple_of(ratio_e as usize)
                {
                    if temperature.is_none() {
                        temperature = Some(self.measure_temperature() as f32);
                    }

                    v.update_with_energy(self, pressure as f32, temperature.unwrap());
                }

                // todo: Handle force writing (`nstfout`).

                v
            };

            self.snapshot_queue_for_trr.push(ss);
        }

        if let Some(ratio) = oc.nstxout_compressed
            && i.is_multiple_of(ratio as usize)
        {
            self.snapshot_queue_for_xtc.push(Snapshot::new(self));
        }

        self.handle_ss_file_writes();
    }

    /// Peridically offloads the in-memory snapshot queues for various file-handlers onto disk.
    /// Clear the queues. Appends to DCD and TRR files.
    fn handle_ss_file_writes(&mut self) {
        if self.step_count == 0 {
            // Choose the lowest run index N for which no trajectory files exist yet, so
            // that each fresh MD run writes to its own set of files (traj_N.dcd etc.)
            // while periodic saves within the same run keep appending to those files.
            let out = Path::new(TRAJ_OUT_PATH);
            self.run_index = (0..)
                .find(|&n| {
                    !out.join(format!("traj_{n}.dcd")).exists()
                        && !out.join(format!("traj_{n}.trr")).exists()
                        && !out.join(format!("traj_{n}.xtc")).exists()
                })
                .unwrap_or(0);

            return;
        }

        // Clear queues as required.
        if !self.step_count.is_multiple_of(TRAJ_FILE_SAVE_INTERVAL) {
            return;
        }

        let n = self.run_index;

        if !self.snapshot_queue_for_dcd.is_empty() {
            let frames: Vec<_> = self
                .snapshot_queue_for_dcd
                .iter()
                .map(|ss| ss.to_dcd(&self.cell, false))
                .collect();

            let dcd = DcdTrajectory { frames };

            let path = Path::new(TRAJ_OUT_PATH).join(format!("traj_{n}.dcd"));
            if let Err(e) = dcd.save(&path) {
                eprintln!("Error writing DCD: {e:?}");
            }

            self.snapshot_queue_for_dcd.clear();
        }

        if !self.snapshot_queue_for_trr.is_empty() {
            let path = Path::new(TRAJ_OUT_PATH).join(format!("traj_{n}.trr"));
            let frames = ss_to_gromacs_frames(&self.snapshot_queue_for_trr);
            if let Err(e) = write_trr(&path, &frames) {
                eprintln!("Error writing TRR: {e:?}");
            }

            self.snapshot_queue_for_trr.clear();
        }

        // todo: Make sure this fails gracefully if python3 or mdtraj isn't available.
        if !self.snapshot_queue_for_xtc.is_empty() {
            let frames: Vec<_> = self
                .snapshot_queue_for_xtc
                .iter()
                .map(|ss| ss.to_dcd(&self.cell, false))
                .collect();

            let path = Path::new(TRAJ_OUT_PATH).join(format!("traj_{n}.xtc"));
            if let Err(e) = write_xtc(&path, &frames) {
                eprintln!("Error writing XTC: {e:?}");
            }

            self.snapshot_queue_for_xtc.clear();
        }
    }
}

impl MdState {
    /// Compute the instantaneous ∂H/∂λ in kcal/mol for the current alchemical molecule.
    ///
    /// For linear coupling/decoupling:
    ///   H(λ) = H_solvent + (1−λ)·U_alch_interact + H_solute_bonded
    ///   ∂H/∂λ = −U_alch_interact
    ///
    /// where U_alch_interact is the non-bonded interaction energy of molecule
    /// `alch_mol_idx` with all other molecules, taken from
    /// `potential_energy_between_mols` (updated each step by `apply_nonbonded_forces`).
    ///
    /// Returns 0.0 when `alch_mol_idx` is `None`.
    ///
    /// # Physical correctness
    /// For true thermodynamic integration at intermediate λ values, the non-bonded
    /// forces on the alchemical molecule must be scaled by `(1 − λ)` in
    /// `apply_nonbonded_forces`.  Without that scaling every window samples the
    /// fully-coupled ensemble and TI is equivalent to a single-point FEP estimate.
    pub fn compute_dh_dl(&self) -> f64 {
        let m = match self.alch_mol_idx {
            Some(m) => m,
            None => return 0.0,
        };

        let n = self.mol_start_indices.len();
        // potential_energy_between_mols is a symmetric N×N matrix:
        // both [m*n + j] and [j*n + m] hold the interaction energy for pair (m, j).
        // Sum row m (j ≠ m) to get the total solute–solvent interaction energy.
        let u_interact: f64 = (0..n)
            .filter(|&j| j != m)
            .map(|j| self.potential_energy_between_mols[m * n + j])
            .sum();

        -u_interact
    }
} // impl MdState (compute_dh_dl)

/// Convert GROMACS trajectory frames into `Snapshot` values.
/// This converts positions in nm and velocities in nm/ps to Å, and Å/ps
///
/// `solute_atom_count` is the number of non-water atoms (computed before solvation).
/// Atoms beyond that index are OPC water molecules, laid out as groups of 4:
/// OW, HW1, HW2, MW (virtual site). MW positions are discarded since `Snapshot`
/// has no field for them and the virtual site carries no mass.
pub fn gromacs_frames_to_ss(frames: &[GromacsFrame], solute_atom_count: usize) -> Vec<Snapshot> {
    // OPC water has 4 sites per molecule (OW, HW1, HW2, MW virtual site).
    const OPC_SITES_PER_MOL: usize = 4;

    frames
        .iter()
        .map(|frame| {
            let n = frame.atom_posits.len();
            let solute_end = solute_atom_count.min(n);

            let atom_posits: Vec<Vec3> = frame.atom_posits[..solute_end]
                .iter()
                .map(|p| {
                    Vec3::new(
                        (p.x * 10.0) as f32,
                        (p.y * 10.0) as f32,
                        (p.z * 10.0) as f32,
                    )
                })
                .collect();

            let water_block = &frame.atom_posits[solute_end..];
            let n_water_mols = water_block.len() / OPC_SITES_PER_MOL;

            let mut water_o_posits = Vec::with_capacity(n_water_mols);
            let mut water_h0_posits = Vec::with_capacity(n_water_mols);
            let mut water_h1_posits = Vec::with_capacity(n_water_mols);

            for i in 0..n_water_mols {
                let base = i * OPC_SITES_PER_MOL;
                let to_vec3 = |p: &lin_alg::f64::Vec3| {
                    Vec3::new(
                        (p.x * 10.0) as f32,
                        (p.y * 10.0) as f32,
                        (p.z * 10.0) as f32,
                    )
                };

                water_o_posits.push(to_vec3(&water_block[base]));
                water_h0_posits.push(to_vec3(&water_block[base + 1]));
                water_h1_posits.push(to_vec3(&water_block[base + 2]));
                // base + 3 is the MW virtual site — no Snapshot field for it.
            }

            Snapshot {
                time: frame.time,
                atom_posits,
                water_o_posits,
                water_h0_posits,
                water_h1_posits,
                ..Snapshot::default()
            }
        })
        .collect()
}

/// Convert `Snapshot` values into GROMACS trajectory frames.
///
/// This is the inverse of `gromacs_frames_to_ss`:
/// - Positions are converted from Å → nm (÷ 10).
/// - Velocities are converted from Å/ps → nm/ps (÷ 10), when present.
/// - Solute atoms come first, followed by water molecules laid out as
///   [OW, HW1, HW2] per molecule.  The OPC MW virtual site is omitted
///   because it was discarded on load and its position is unknown.
/// - Water velocities (one COM velocity per molecule) are replicated to
///   all three sites (OW, HW1, HW2) when available.
pub fn ss_to_gromacs_frames(ss: &[Snapshot]) -> Vec<GromacsFrame> {
    let to_nm = |p: &Vec3| -> lin_alg::f64::Vec3 {
        lin_alg::f64::Vec3 {
            x: p.x as f64 / 10.0,
            y: p.y as f64 / 10.0,
            z: p.z as f64 / 10.0,
        }
    };

    ss.iter()
        .map(|snap| {
            // Solute atoms (Å → nm).
            let mut atom_posits: Vec<lin_alg::f64::Vec3> =
                snap.atom_posits.iter().map(to_nm).collect();

            // Water sites: OW, HW1, HW2 per molecule (no MW virtual site).
            let n_water = snap.water_o_posits.len();
            for i in 0..n_water {
                atom_posits.push(to_nm(&snap.water_o_posits[i]));
                atom_posits.push(to_nm(&snap.water_h0_posits[i]));
                atom_posits.push(to_nm(&snap.water_h1_posits[i]));
            }

            // Velocities: solute then water, all Å/ps → nm/ps.
            let atom_velocities = if let Some(vels) = &snap.atom_velocities {
                let mut all_vels: Vec<lin_alg::f64::Vec3> = vels.iter().map(to_nm).collect();

                if let Some(water_vels) = &snap.water_velocities {
                    for wv in water_vels {
                        let v = to_nm(wv);
                        all_vels.push(v); // OW
                        all_vels.push(v); // HW1
                        all_vels.push(v); // HW2
                    }
                }

                all_vels
            } else {
                Vec::new()
            };

            GromacsFrame {
                time: snap.time,
                atom_posits,
                atom_velocities,
                energy: None,
            }
        })
        .collect()
}
