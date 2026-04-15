//! Related to storing snapshots (also known as trajectories) of MD runs.

use std::{
    collections::HashMap,
    fs,
    io::{self, ErrorKind},
    path::Path,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, ChargeType, MmCif, Mol2, MolType,
    dcd::{DcdFrame, DcdTrajectory, DcdUnitCell},
    gromacs,
    gromacs::{GromacsFrame, GromacsOutput, OutputControl, output::write_trr},
    xtc::write_xtc,
};
#[cfg(feature = "cuda")]
use cudarc::{driver::CudaContext, nvrtc::Ptx};
use lin_alg::f32::Vec3;

use crate::{AtomDynamics, MdState, barostat::SimBox, solvent::MASS_WATER_MOL};
#[cfg(feature = "cuda")]
use crate::{
    PTX,
    gpu_interface::{ForcesPositsGpu, PerNeighborGpu},
};

// Append to any snapshot-saving files every this number of snapshots. E.g.
// DCD, TRR, XTC. We want this to be such that we don't experience too much memory use.
const TRAJ_FILE_SAVE_INTERVAL: usize = 2_000;

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
    /// This uses GROMACS default for XTC, TRR velocity and force, and energy logging.
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
/// All energies are in kcal / mol
#[derive(Clone, Debug)]
pub struct SnapshotEnergyData {
    /// kcal / mol
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
    pub dh_dl: Option<f32>,
    /// Simulation box volume in **Å³**.
    pub volume: f32,
    // System density in **kg/m³**.
    /// System density in **amu/Å³3**.
    pub density: f32,
}

impl From<gromacs::OutputEnergy> for SnapshotEnergyData {
    fn from(e: gromacs::OutputEnergy) -> Self {
        // kJ/mol → kcal/mol
        const KJ_TO_KCAL: f32 = 1.0 / 4.184;
        // nm³ → Å³ (1 nm = 10 Å, so 1 nm³ = 1000 Å³)
        const NM3_TO_ANG3: f32 = 1_000.0;
        // kg/m³ → amu/Å³ (1 kg = 1/1.66054e-27 amu; 1 m³ = 1e30 Å³)
        const KG_M3_TO_AMU_ANG3: f32 = 6.02214076e-4;

        Self {
            energy_kinetic: e.kinetic_energy.unwrap_or_default() * KJ_TO_KCAL,
            energy_potential: e.potential_energy.unwrap_or_default() * KJ_TO_KCAL,
            energy_potential_between_mols: Vec::new(),
            energy_potential_nonbonded: 0.,
            energy_potential_bonded: 0.,
            hydrogen_bonds: Vec::new(),
            // K and bar need no conversion.
            temperature: e.temperature.unwrap_or_default(),
            pressure: e.pressure.unwrap_or_default(),
            dh_dl: None,
            volume: e.volume.unwrap_or_default() * NM3_TO_ANG3,
            density: e.density.unwrap_or_default() * KG_M3_TO_AMU_ANG3,
        }
    }
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
    pub water_o_posits: Vec<Vec3>,
    pub water_h0_posits: Vec<Vec3>,
    pub water_h1_posits: Vec<Vec3>,
    /// Single velocity per solvent molecule, as it's rigid.
    pub water_velocities: Option<Vec<Vec3>>,
    /// Force acting on each atom.
    pub force: Option<Vec<Vec3>>,
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
            time: state.time,
            atom_posits: state.atoms.iter().map(|a| a.posit).collect(),
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            ..Default::default()
        }
    }

    pub fn update_with_velocities(&mut self, state: &MdState) {
        self.atom_velocities = Some(state.atoms.iter().map(|a| a.vel).collect());
        self.water_velocities = Some(state.water.iter().map(|w| w.o.vel).collect());
    }

    pub fn update_with_energy(&mut self, state: &MdState, pressure: f32, temperature: f32) {
        self.atom_velocities = Some(state.atoms.iter().map(|a| a.vel).collect());
        self.water_velocities = Some(state.water.iter().map(|w| w.o.vel).collect());

        let energy_potential_between_mols = state
            .potential_energy_between_mols
            .iter()
            .map(|v| *v as f32)
            .collect();

        let mut mass = 0.;
        for atom in &state.atoms {
            mass += atom.mass as f64;
        }
        mass += MASS_WATER_MOL as f64 * state.water.len() as f64;

        let volume = state.cell.volume();
        let density = mass as f32 / volume;

        self.energy_data = Some(SnapshotEnergyData {
            energy_kinetic: state.kinetic_energy as f32,
            energy_potential: state.potential_energy as f32,
            energy_potential_between_mols,
            energy_potential_nonbonded: state.potential_energy_nonbonded as f32,
            energy_potential_bonded: state.potential_energy_bonded as f32,
            hydrogen_bonds: Vec::new(), // Populated later A/R.
            temperature,
            pressure,
            dh_dl: Some(state.compute_dh_dl() as f32),
            volume,
            density,
        });
    }

    /// Unflatten positions and velocities on a per-molecule basis. `mol_start_indices` may be
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
            for i in 0..self.water_o_posits.len() {
                atom_posits.push(self.water_o_posits[i]);
                atom_posits.push(self.water_h0_posits[i]);
                atom_posits.push(self.water_h1_posits[i]);
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

impl From<GromacsFrame> for Snapshot {
    fn from(frame: GromacsFrame) -> Self {
        // nm → Å
        let atom_posits = frame
            .atom_posits
            .iter()
            .map(|p| Vec3 {
                x: (p.x * 10.0) as f32,
                y: (p.y * 10.0) as f32,
                z: (p.z * 10.0) as f32,
            })
            .collect();

        // nm/ps → Å/ps; absent if the velocity block was empty.
        let atom_velocities = if frame.atom_velocities.is_empty() {
            None
        } else {
            Some(
                frame
                    .atom_velocities
                    .iter()
                    .map(|v| Vec3 {
                        x: (v.x * 10.0) as f32,
                        y: (v.y * 10.0) as f32,
                        z: (v.z * 10.0) as f32,
                    })
                    .collect(),
            )
        };

        Self {
            time: frame.time,
            atom_posits,
            atom_velocities,
            energy_data: frame.energy.map(SnapshotEnergyData::from),
            water_o_posits: Vec::new(),
            water_h0_posits: Vec::new(),
            water_h1_posits: Vec::new(),
            water_velocities: None,
            // kJ/(mol·nm) → kcal/(mol·Å): divide by 4.184 and by 10.
            force: if frame.atom_forces.is_empty() {
                None
            } else {
                const KJ_NM_TO_KCAL_ANG: f32 = 1.0 / 41.84;
                Some(
                    frame
                        .atom_forces
                        .iter()
                        .map(|f| Vec3 {
                            x: (f.x as f32) * KJ_NM_TO_KCAL_ANG,
                            y: (f.y as f32) * KJ_NM_TO_KCAL_ANG,
                            z: (f.z as f32) * KJ_NM_TO_KCAL_ANG,
                        })
                        .collect(),
                )
            },
        }
    }
}

impl From<DcdFrame> for Snapshot {
    fn from(frame: DcdFrame) -> Self {
        // DcdFrame.time is in fs; Snapshot.time is in ps.
        Self {
            time: frame.time / 1_000.0,
            atom_posits: frame.atom_posits,
            atom_velocities: None,
            energy_data: None,
            water_o_posits: Vec::new(),
            water_h0_posits: Vec::new(),
            water_h1_posits: Vec::new(),
            water_velocities: None,
            force: None,
        }
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
                v.force = Some(self.atoms.iter().map(|a| a.force).collect());

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

        // Check whether any GROMACS TRR output is due this step: positions,
        // velocities, energy, or forces. A snapshot is created whenever *any*
        // of these intervals fires, not only when nstxout fires.
        let write_posit = oc.nstxout.is_some_and(|r| i.is_multiple_of(r as usize));
        let write_vel = oc.nstvout.is_some_and(|r| i.is_multiple_of(r as usize));
        let write_en = oc.nstenergy.is_some_and(|r| i.is_multiple_of(r as usize));
        let write_f = oc.nstfout.is_some_and(|r| i.is_multiple_of(r as usize));

        if write_posit || write_vel || write_en || write_f {
            let ss = {
                let mut s = Snapshot::new(self);

                if write_vel {
                    s.update_with_velocities(self);
                }

                // We are ignoring `nstcalcenergy`.
                if write_en {
                    if temperature.is_none() {
                        temperature = Some(self.measure_temperature() as f32);
                    }

                    s.update_with_energy(self, pressure, temperature.unwrap());
                }

                if write_f {
                    s.force = Some(self.atoms.iter().map(|a| a.force).collect());
                }

                s
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
        if !self.step_count.is_multiple_of(TRAJ_FILE_SAVE_INTERVAL) {
            return;
        }
        self.flush_snapshot_queues();
    }

    /// Flush any remaining snapshots in the DCD/TRR/XTC queues to disk.
    /// Call this at the end of a simulation run to ensure the last frames
    /// (those accumulated since the most recent `TRAJ_FILE_SAVE_INTERVAL`
    /// write) are not lost.
    pub fn flush_snapshot_queues(&mut self) {
        // On the first call, choose the lowest run index N for which no trajectory files
        // exist yet, so that each fresh MD run writes to its own set of files (traj_N.*).
        if self.run_index.is_none() {
            let out = Path::new(TRAJ_OUT_PATH);
            self.run_index = (0..).find(|&n| {
                !out.join(format!("traj_{n}.dcd")).exists()
                    && !out.join(format!("traj_{n}.trr")).exists()
                    && !out.join(format!("traj_{n}.xtc")).exists()
            });
        }

        let n = self.run_index.unwrap_or(0);

        if let Err(e) = fs::create_dir_all(TRAJ_OUT_PATH) {
            eprintln!("Error creating output directory '{TRAJ_OUT_PATH}': {e:?}");
            return;
        }

        if !self.snapshot_queue_for_dcd.is_empty() {
            let frames: Vec<_> = self
                .snapshot_queue_for_dcd
                .iter()
                .map(|ss| ss.to_dcd(&self.cell, true))
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

        // todo: Make sure this fails gracefully if mdtraj isn't available.
        if !self.snapshot_queue_for_xtc.is_empty() {
            let frames: Vec<_> = self
                .snapshot_queue_for_xtc
                .iter()
                .map(|ss| ss.to_dcd(&self.cell, true))
                .collect();

            let path = Path::new(TRAJ_OUT_PATH).join(format!("traj_{n}.xtc"));
            if let Err(e) = write_xtc(&path, &frames) {
                eprintln!("Error writing XTC: {e:?}");
            }

            self.snapshot_queue_for_xtc.clear();
        }
    }
}

/// Convert GROMACS trajectory frames into `Snapshot` values.
/// This converts positions in nm and velocities in nm/ps to Å, and Å/ps
///
/// `solute_atom_count` is the number of non-water atoms (computed before solvation).
/// Atoms beyond that index are OPC water molecules, laid out as groups of 4:
/// OW, HW1, HW2, MW (virtual site). MW positions are discarded since `Snapshot`
/// has no field for them and the virtual site carries no mass.
pub fn gromacs_frames_to_ss(out: &GromacsOutput) -> Vec<Snapshot> {
    // OPC water has 4 sites per molecule (OW, HW1, HW2, MW virtual site).
    const OPC_SITES_PER_MOL: usize = 4;
    const NM_TO_ANGSTROM: f64 = 10.;

    out.trajectory
        .iter()
        .map(|frame| {
            let n = frame.atom_posits.len();
            let solute_end = out.solute_atom_count.min(n);

            let atom_posits: Vec<Vec3> = frame.atom_posits[..solute_end]
                .iter()
                .map(|p| {
                    Vec3::new(
                        (p.x * NM_TO_ANGSTROM) as f32,
                        (p.y * NM_TO_ANGSTROM) as f32,
                        (p.z * NM_TO_ANGSTROM) as f32,
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
                        (p.x * NM_TO_ANGSTROM) as f32,
                        (p.y * NM_TO_ANGSTROM) as f32,
                        (p.z * NM_TO_ANGSTROM) as f32,
                    )
                };

                water_o_posits.push(to_vec3(&water_block[base]));
                water_h0_posits.push(to_vec3(&water_block[base + 1]));
                water_h1_posits.push(to_vec3(&water_block[base + 2]));
                // base + 3 is the MW virtual site — no Snapshot field for it.
            }

            let energy_data = frame
                .energy
                .as_ref()
                .map(|f| SnapshotEnergyData::from(f.clone()));

            Snapshot {
                time: frame.time,
                atom_posits,
                water_o_posits,
                water_h0_posits,
                water_h1_posits,
                energy_data,
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

            // Forces: kcal/(mol·Å) → kJ/(mol·nm): multiply by 41.84.
            let atom_forces = if let Some(forces) = &snap.force {
                let to_kj_nm = |f: &Vec3| -> lin_alg::f64::Vec3 {
                    lin_alg::f64::Vec3 {
                        x: f.x as f64 * 41.84,
                        y: f.y as f64 * 41.84,
                        z: f.z as f64 * 41.84,
                    }
                };
                forces.iter().map(to_kj_nm).collect()
            } else {
                Vec::new()
            };

            GromacsFrame {
                time: snap.time,
                atom_posits,
                atom_velocities,
                atom_forces,
                energy: None,
            }
        })
        .collect()
}
