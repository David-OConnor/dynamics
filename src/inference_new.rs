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

use std::{io, path::Path, time::Instant};

use bio_files::{
    AtomGeneric, BondGeneric, BondType,
    amber_typedef::{AmberDef, AtomTypeDef},
};
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

/// Note: We've commented out all but the ones we need for small organic molecules.
pub struct AmberDefSet {
    /// Newer AM1-BCC charge model. Use this with Gaff2 for small organic molecules?
    pub abcg2: AmberDef,
    // /// Protein, DNA, RNA?
    // pub amber: AmberDef,
    // /// AM1-BCC for BCC corrections? For small organic molecules?
    // pub bcc: AmberDef,
    // /// Gas-phase ESP/RESP setups? Niche.
    // pub gas: AmberDef,
    // /// Legacy Gaff?
    // pub gff: AmberDef,
    /// Gaff2, for small organic molecules
    pub gff2: AmberDef,
    // /// Used in TRIPOS/DOCK and old QSAR tools?
    // pub sybyl: AmberDef,
}

impl AmberDefSet {
    pub fn new() -> io::Result<Self> {
        let abcg2 = AmberDef::new(DEF_ABCG2)?;

        // let amber = AmberDef::new(DEF_AMBER)?;
        // let bcc = AmberDef::new(DEF_BCC)?;
        // let gas = AmberDef::new(DEF_GAS)?;
        // let gff = AmberDef::new(DEF_GFF)?;

        let gff2 = AmberDef::new(DEF_GFF2)?;
        // let sybyl = AmberDef::new(DEF_SYBYL));

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

/// Describes an atom with information about the atoms it's bonded to, for
/// the purposes of assigning it's FF type.
#[derive(Debug)]
struct AtomEnvData {
    degree: u8,
    num_attached_h: u8,
    ring_sizes: Vec<u8>,
    is_aromatic: bool,
}

/// Part of the ring detection pipeline.
fn dfs_find_rings(
    start: usize,
    current: usize,
    depth: u8,
    max_size: u8,
    visited: &mut [bool],
    path: &mut Vec<usize>,
    adj: &[Vec<usize>],
    ring_sizes: &mut [Vec<u8>],
) {
    if depth >= max_size {
        return;
    }

    for &nbr in &adj[current] {
        if nbr == start && depth + 1 >= 3 {
            let size = depth + 1;
            for &idx in path.iter() {
                if !ring_sizes[idx].contains(&size) {
                    ring_sizes[idx].push(size);
                }
            }
        } else if !visited[nbr] {
            visited[nbr] = true;
            path.push(nbr);
            dfs_find_rings(
                start,
                nbr,
                depth + 1,
                max_size,
                visited,
                path,
                adj,
                ring_sizes,
            );
            path.pop();
            visited[nbr] = false;
        }
    }
}

/// This is used to identify rings the atom is part of; this is important for
/// assigning the correct atom type, e.g. "c6" if the atom is part of a 6-member ring,
/// instead of "c3".
fn detect_rings(adj: &[Vec<usize>], max_size: u8) -> Vec<Vec<u8>> {
    let n = adj.len();
    let mut ring_sizes = vec![Vec::<u8>::new(); n];

    for start in 0..n {
        let mut visited = vec![false; n];
        visited[start] = true;
        let mut path = vec![start];
        dfs_find_rings(
            start,
            start,
            0,
            max_size,
            &mut visited,
            &mut path,
            adj,
            &mut ring_sizes,
        );
    }

    ring_sizes
}

// todo: Should we use bonds here too, which have BondType::Aromatic?
fn is_aromatic_atom(
    idx: usize,
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    ring_sizes: &[Vec<u8>],
) -> bool {
    let rs = &ring_sizes[idx];

    if !rs.iter().any(|&s| (5..=7).contains(&s)) {
        return false;
    }

    let neighbors = &adj[idx];

    if neighbors.len() != 3 {
        return false;
    }

    let shared_ring_neighbors = neighbors
        .iter()
        .filter(|&&j| {
            atoms[j].element != Element::Hydrogen && ring_sizes[j].iter().any(|s| rs.contains(s))
        })
        .count();

    shared_ring_neighbors >= 2
}

fn build_env(atoms: &[AtomGeneric], bonds: &[BondGeneric], adj: &[Vec<usize>]) -> Vec<AtomEnvData> {
    let ring_sizes = detect_rings(adj, 8);

    let mut aromatic_bond_counts = vec![0u8; atoms.len()];

    for bond in bonds {
        if bond.bond_type != BondType::Aromatic {
            continue;
        }

        // Get index instead of SN.
        let i = bond.atom_0_sn as usize - 1;
        let j = bond.atom_1_sn as usize - 1;

        // Only count if both atoms are in a "reasonable" ring (5–7).
        let in_ring_i = ring_sizes[i].iter().any(|s| (5..=7).contains(s));
        let in_ring_j = ring_sizes[j].iter().any(|s| (5..=7).contains(s));

        if in_ring_i && in_ring_j {
            aromatic_bond_counts[i] = aromatic_bond_counts[i].saturating_add(1);
            aromatic_bond_counts[j] = aromatic_bond_counts[j].saturating_add(1);
        }
    }

    atoms
        .iter()
        .enumerate()
        .map(|(idx, _atom)| {
            let neighbors = &adj[idx];

            let degree = neighbors.len() as u8;

            let num_attached_h = neighbors
                .iter()
                .filter(|&&j| atoms[j].element == Element::Hydrogen)
                .count() as u8;

            let is_in_ring_5_7 = ring_sizes[idx].iter().any(|s| (5..=7).contains(s));
            let is_aromatic = is_in_ring_5_7 && aromatic_bond_counts[idx] >= 2;

            AtomEnvData {
                degree,
                num_attached_h,
                ring_sizes: ring_sizes[idx].clone(),
                is_aromatic,
            }
        })
        .collect()
}

fn parse_ring_size(s: &str) -> Option<u8> {
    let s = s.trim();
    let pos = s.find("RG")?;
    let start = pos + 2;
    let mut digits = String::new();

    for ch in s[start..].chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            break;
        }
    }

    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn matches_def(
    def: &AtomTypeDef,
    idx: usize,
    atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
    adj: &[Vec<usize>],
) -> bool {
    let atom = &atoms[idx];
    let env = &env_all[idx];

    if let Some(e) = def.element {
        if atom.element != e {
            return false;
        }
    }

    if let Some(n) = def.attached_atoms {
        if env.degree != n {
            return false;
        }
    }

    if let Some(nh) = def.attached_h {
        if env.num_attached_h != nh {
            return false;
        }
    }

    if def.electron_withdrawal_count.is_some() {
        return false;
    }

    if let Some(ref prop) = def.atomic_property {
        if let Some(ring_size) = parse_ring_size(prop) {
            if !env.ring_sizes.contains(&ring_size) {
                return false;
            }
        }

        if prop.contains("AR") && !env.is_aromatic {
            return false;
        }
    }

    if let Some(ref env_str) = def.chem_env {
        // "&" means "no special environment"
        if env_str != "&" {
            if atom.element == Element::Hydrogen {
                if !hydrogen_env_matches(env_str, idx, atoms, env_all, adj) {
                    return false;
                }
            } else {
                // For now skip non-H env patterns we don't understand.
                return false;
            }
        }
    }

    true
}

/// Used for assigning atom types on Hydrogens.
fn single_heavy_neighbor(idx: usize, atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> Option<usize> {
    let neighbors = &adj[idx];

    let mut heavy = neighbors
        .iter()
        .copied()
        .filter(|&j| atoms[j].element != Element::Hydrogen);

    let first = heavy.next()?;
    if heavy.next().is_some() {
        return None;
    }
    Some(first)
}

fn hydrogen_env_matches(
    env_str: &str,
    idx: usize,
    atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
    adj: &[Vec<usize>],
) -> bool {
    let env_str = env_str.trim();

    if env_str == "&" {
        return true;
    }

    if !env_str.starts_with('(') || !env_str.ends_with(')') {
        return false;
    }

    let inner = env_str[1..env_str.len() - 1].trim();

    let nb = match single_heavy_neighbor(idx, atoms, adj) {
        Some(i) => i,
        None => return false,
    };

    let nb_atom = &atoms[nb];
    let nb_env = &env_all[nb];

    match inner {
        "O" => nb_atom.element == Element::Oxygen,
        "N" => nb_atom.element == Element::Nitrogen,
        "S" => nb_atom.element == Element::Sulfur,
        "P" => nb_atom.element == Element::Phosphorus,

        inner if inner.starts_with('C') => {
            // (C4), (C3), (C) ...
            let digits = &inner[1..].trim();
            if digits.is_empty() {
                nb_atom.element == Element::Carbon
            } else if let Ok(target_deg) = digits.parse::<u8>() {
                nb_atom.element == Element::Carbon && nb_env.degree == target_deg
            } else {
                false
            }
        }

        _ => false,
    }
}

/// Find Amber force field types for atoms in a small organic molecule. Usese reasoning similar to
/// Amber's Antechamber program to assign these based on atom type, and neighboring atoms and bonds.
/// For example, if in a ring, the nature of the ring, the elements of neighbors etc.
/// todo: Partial charge here, or elsewhere?
/// todo: Load
pub fn find_ff_types(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    defs: &AmberDefSet,
) -> Vec<String> {
    let start = Instant::now();

    println!("Inferring FF types...");
    let adj = build_adjacency_list(atoms, bonds).unwrap();
    let env = build_env(atoms, bonds, &adj);

    let mut result = Vec::with_capacity(atoms.len());

    'atom_loop: for (i, atom) in atoms.iter().enumerate() {
        let env_i = &env[i];

        // The order of rules in the DEF file matters; we keep it.
        for def in &defs.gff2.atomtypes {
            if matches_def(def, i, atoms, &env, &adj) {
                result.push(def.name.clone());
                continue 'atom_loop;
            }
        }

        // No match – use a dummy/untyped atom name (Amber often uses "du").
        result.push("du".to_string());
    }

    let elapsed = start.elapsed().as_millis();
    println!("Complete in {elapsed} ms");

    result
}
