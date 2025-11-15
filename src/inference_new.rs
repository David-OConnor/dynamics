//! Uses reasoning similar to AnteChamber's to estimate force field parameters.
//! [Reference source code](https://github.com/Amber-MD/AmberClassic/tree/main/src/antechamber)
//!
//! Reference Antechamber config files e.g. Also review the Antechamber config files in the Amber install dir: ATOMTYPE_BCC.DEF in $AMBERHOME/dat/antechamber
//!
//! Description of its parts: https://ambermd.org/antechamber/ac.html#am1bcc
//! Of interest:
//! -[Atomtype](https://github.com/Amber-MD/AmberClassic/blob/main/src/antechamber/atomtype.c)
//! -[Bondtype](https://github.com/Amber-MD/AmberClassic/blob/main/src/antechamber/bondtype.c)
//! -[Parmchk](https://github.com/Amber-MD/AmberClassic/blob/main/src/antechamber/parmchk2.c)

use std::path::Path;
use std::io;
use bio_files::{AtomGeneric, BondGeneric, amber_typedef::{DefFile, AtomTypeDef}};
use na_seq::Element;
use crate::util::build_adjacency_list;

/// See note: Only loading ones we need for small organic molecules.
const DEF_ABCG2: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_ABCG2.DEF");
// const DEF_AMBER: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_AMBER.DEF");
// const DEF_BCC: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_BCC.DEF");
// const DEF_GAS: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GAS.DEF");
// const DEF_GFF: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GFF.DEF");
const DEF_GFF2: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GFF2.DEF");
// const DEF_SYBYL: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_SYBYL.DEF");


#[derive(Debug)]
struct AtomEnv {
    degree: u8,
    attached_h: u8,
}

fn build_env(atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> Vec<AtomEnv> {
    atoms
        .iter()
        .enumerate()
        .map(|(idx, atom)| {
            let neighbors = &adj[idx];

            let degree = neighbors.len() as u8;

            let attached_h = neighbors
                .iter()
                .filter(|&&j| {
                    atoms[j].element == Element::Hydrogen
                })
                .count() as u8;

            AtomEnv { degree, attached_h }
        })
        .collect()
}

/// Note: We've commented out all but the ones we need for small organic molecules.
pub struct Defs {
    /// Newer AM1-BCC charge model. Use this with Gaff2 for small organic molecules?
    pub abcg2: DefFile,
    // /// Protein, DNA, RNA?
    // pub amber: DefFile,
    // /// AM1-BCC for BCC corrections? For small organic molecules?
    // pub bcc: DefFile,
    // /// Gas-phase ESP/RESP setups? Niche.
    // pub gas: DefFile,
    // /// Legacy Gaff?
    // pub gff: DefFile,
    /// Gaff2, for small organic molecules
    pub gff2: DefFile,
    // /// Used in TRIPOS/DOCK and old QSAR tools?
    // pub sybyl: DefFile,
}

impl Defs {
    fn new() -> io::Result<Self> {
        let abcg2 = DefFile::load(Path::new(DEF_ABCG2))?;
        // let amber = DefFile::load(Path::new(DEF_AMBER))?;
        // let bcc = DefFile::load(Path::new(DEF_BCC))?;
        // let gas = DefFile::load(Path::new(DEF_GAS))?;
        // let gff = DefFile::load(Path::new(DEF_GFF))?;
        let gff2 = DefFile::load(Path::new(DEF_GFF2))?;
        // let sybyl = DefFile::load(Path::new(DEF_SYBYL))?;
        
        Ok(Self {
            abcg2,
            // amber,
            // bcc,
            // gas,
            // gff,
            gff2,
            // sybyl,
        })
    }
}

/// Find Amber force field types for atoms in a small organic molecule.
/// todo: Partial charge here, or elsewhere?
/// todo: Load
pub fn find_ff_types(atoms: &[AtomGeneric], bonds: &[BondGeneric], defs: &Defs) -> Vec<String> {
    let adj = build_adjacency_list(atoms, bonds).unwrap();
    let env = build_env(atoms, &adj);

    let mut result = Vec::with_capacity(atoms.len());

    'atom_loop: for (i, atom) in atoms.iter().enumerate() {
        let env_i = &env[i];

        // The order of rules in the DEF file matters; we keep it.
        // todo: Which def file[s] should we use?
        for def in &defs.gff2.atomtypes {
            if matches_def(def, atom, env_i) {
                result.push(def.name.clone());
                continue 'atom_loop;
            }
        }

        // No match – use a dummy/untyped atom name (Amber often uses "du").
        result.push("du".to_string());
    }

    result
}

fn matches_def(def: &AtomTypeDef, atom: &AtomGeneric, env: &AtomEnv) -> bool {
    // f4: atomic number / element
    if let Some(e) = def.element {
        if atom.element != e {
            return false;
        }
    }

    // f5: number of directly attached atoms
    if let Some(n) = def.attached_atoms {
        if env.degree != n {
            return false;
        }
    }

    // f6: number of directly attached hydrogens
    if let Some(nh) = def.attached_h {
        if env.attached_h != nh {
            return false;
        }
    }

    // f7: electron-withdrawing count – skip for now
    if def.ew_count.is_some() {
        return false;
    }

    // f8: atomic property string – this is the whole apcheck() machinery
    if def.atomic_property.is_some() {
        return false;
    }

    // f9: special chemical environment – this is jatspecial()
    if def.env.is_some() {
        return false;
    }

    true
}