//! Related to storing snapshots (also known as trajectories) of MD runs.

use std::{
    collections::HashMap,
    io::{self, ErrorKind},
    path::PathBuf,
};

use crate::AtomDynamics;
use crate::ambient::SimBox;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::dcd::DcdUnitCell;
use bio_files::{
    AtomGeneric, BondGeneric, ChargeType, MmCif, Mol2, MolType,
    dcd::{DcdFrame, DcdTrajectory},
};
use lin_alg::f32::Vec3;

// // Append to any snapshot-saving files every this number of snapshots.
// // todo:  Update A/R. Likely higher.
// pub(crate) const FILE_SAVE_INTERVAL: usize = 100;

#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Clone, PartialEq, Debug, Default)]
pub enum SaveType {
    #[default]
    Memory,
    Dcd(PathBuf),
}

#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotHandler {
    pub save_type: SaveType,
    /// Save every this many steps.
    pub ratio: usize,
}

impl Default for SnapshotHandler {
    fn default() -> Self {
        Self {
            save_type: SaveType::Memory,
            ratio: 1,
        }
    }
}

/// This stores the positions and velocities of all atoms in the system, and the total energy.
/// It represents the output of the simulation. A set of these can be used to play it back over time.
/// We save load and save this to disk in the __ format.
#[derive(Clone, Debug, Default)]
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
}

impl Snapshot {
    /// The element indices must match the atom posits.
    pub fn populate_hydrogen_bonds(&mut self, atoms: &[AtomDynamics]) {
        // let result = create_hydrogen_bonds(&atoms, &self.atom_posits, &self.water_o_posits, &self.bonds);

        // self.hydrogen_bonds = result;
    }

    pub fn to_dcd(&self, cell: &SimBox, write_water: bool) -> DcdFrame {
        let mut atom_posits = self.atom_posits.clone();

        if write_water {
            for pos in &self.water_o_posits {
                atom_posits.push(pos.clone());
            }
            for pos in &self.water_h0_posits {
                atom_posits.push(pos.clone());
            }
            for pos in &self.water_h1_posits {
                atom_posits.push(pos.clone());
            }
        }

        DcdFrame {
            time: self.time,
            atom_posits,
            unit_cell: DcdUnitCell {
                bounds_low: cell.bounds_low.into(),
                bounds_high: cell.bounds_high.into(),
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

macro_rules! parse_le {
    ($bytes:expr, $t:ty, $range:expr) => {{ <$t>::from_le_bytes($bytes[$range].try_into().unwrap()) }};
}

macro_rules! copy_le {
    ($dest:expr, $src:expr, $range:expr) => {{ $dest[$range].copy_from_slice(&$src.to_le_bytes()) }};
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
            comment: None,
        })
    }

    pub fn make_mmcif(&self, atoms_: &[AtomGeneric], bonds: &[BondGeneric]) -> io::Result<MmCif> {
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
