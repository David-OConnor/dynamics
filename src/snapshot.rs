//! Related to storing snapshots (also known as trajectories) of MD runs.

use std::{
    collections::HashMap,
    io::{self, ErrorKind},
    path::PathBuf,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, BondGeneric, ChargeType, MmCif, Mol2, MolType};
use lin_alg::f32::Vec3;

use crate::AtomDynamics;

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
    /// E.g. for saving to file. Saves all items as 32-bit floating point.
    pub fn to_bytes(&self) -> Vec<u8> {
        const SIZE: usize = 12 * 7; // Assumes 32
        let mut result = Vec::with_capacity(SIZE);

        let mut i = 0;

        copy_le!(result, self.time, i..i + 4);
        i += 4;
        copy_le!(result, self.atom_posits.len() as u32, i..i + 4);
        i += 4;
        copy_le!(result, self.water_o_posits.len() as u32, i..i + 4);
        i += 4;

        for pos in &self.atom_posits {
            copy_le!(result, pos.x, i..i + 4);
            i += 4;
            copy_le!(result, pos.y, i..i + 4);
            i += 4;
            copy_le!(result, pos.z, i..i + 4);
            i += 4;
        }

        for v in &self.atom_velocities {
            copy_le!(result, v.x, i..i + 4);
            i += 4;
            copy_le!(result, v.y, i..i + 4);
            i += 4;
            copy_le!(result, v.z, i..i + 4);
            i += 4;
        }

        for pos in &self.water_o_posits {
            copy_le!(result, pos.x, i..i + 4);
            i += 4;
            copy_le!(result, pos.y, i..i + 4);
            i += 4;
            copy_le!(result, pos.z, i..i + 4);
            i += 4;
        }

        for pos in &self.water_h0_posits {
            copy_le!(result, pos.x, i..i + 4);
            i += 4;
            copy_le!(result, pos.y, i..i + 4);
            i += 4;
            copy_le!(result, pos.z, i..i + 4);
            i += 4;
        }

        for pos in &self.water_h1_posits {
            copy_le!(result, pos.x, i..i + 4);
            i += 4;
            copy_le!(result, pos.y, i..i + 4);
            i += 4;
            copy_le!(result, pos.z, i..i + 4);
            i += 4;
        }

        for v in &self.water_velocities {
            copy_le!(result, v.x, i..i + 4);
            i += 4;
            copy_le!(result, v.y, i..i + 4);
            i += 4;
            copy_le!(result, v.z, i..i + 4);
            i += 4;
        }

        copy_le!(result, self.energy_kinetic, i..i + 4);
        i += 4;
        copy_le!(result, self.energy_potential, i..i + 4);
        i += 4;
        copy_le!(result, self.temperature, i..i + 4);
        i += 4;
        copy_le!(result, self.pressure, i..i + 4);
        i += 4;

        copy_le!(result, self.energy_potential_nonbonded, i..i + 4);
        i += 4;

        copy_le!(result, self.energy_potential_bonded, i..i + 4);
        // i += 4;

        result
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut i = 0usize;

        let time_f32 = parse_le!(bytes, f32, i..i + 4);
        i += 4;
        let n_atoms = parse_le!(bytes, u32, i..i + 4) as usize;
        i += 4;
        let n_waters = parse_le!(bytes, u32, i..i + 4) as usize;
        i += 4;

        let mut read_vec3s = |count: usize| {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                let x = parse_le!(bytes, f32, i..i + 4);
                i += 4;
                let y = parse_le!(bytes, f32, i..i + 4);
                i += 4;
                let z = parse_le!(bytes, f32, i..i + 4);
                i += 4;
                v.push(Vec3 { x, y, z });
            }
            v
        };

        let atom_posits = read_vec3s(n_atoms);
        let atom_velocities = read_vec3s(n_atoms);
        let water_o_posits = read_vec3s(n_waters);
        let water_h0_posits = read_vec3s(n_waters);
        let water_h1_posits = read_vec3s(n_waters);
        let water_velocities = read_vec3s(n_waters);

        let energy_kinetic = parse_le!(bytes, f32, i..i + 4);
        i += 4;
        let energy_potential = parse_le!(bytes, f32, i..i + 4);
        i += 4;
        // todo: Omitting H bonds and mol potential for now.
        let temperature = parse_le!(bytes, f32, i..i + 4);
        i += 4;
        let pressure = parse_le!(bytes, f32, i..i + 4);
        i += 4;

        let energy_potential_nonbonded = parse_le!(bytes, f32, i..i + 4);
        i += 4;

        let energy_potential_bonded = parse_le!(bytes, f32, i..i + 4);
        // i += 4;

        Ok(Self {
            time: time_f32 as f64,
            atom_posits,
            atom_velocities,
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            water_velocities,
            energy_kinetic,
            energy_potential,
            energy_potential_nonbonded,
            energy_potential_bonded,
            energy_potential_between_mols: Vec::new(),
            hydrogen_bonds: Vec::new(),
            temperature,
            pressure,
        })
    }

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
