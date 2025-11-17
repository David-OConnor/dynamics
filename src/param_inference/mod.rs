//! Uses reasoning similar to AnteChamber's to estimate force field parameters using DEF files.
//! Specifically, we use DEF_GFF2 for Gaff2 force field names, and DEF_ABCG2 for FRCMOD
//! bonded paraemter (generally dihedral) overrides.
//! [Reference source code](https://github.com/Amber-MD/AmberClassic/tree/main/src/antechamber)
//!
//! Reference Antechamber config files e.g. Also review the Antechamber config files in the Amber install dir: ATOMTYPE_BCC.DEF in $AMBERHOME/dat/antechamber
//!
//! Description of its parts: https://ambermd.org/antechamber/ac.html#am1bcc
//! Of interest:
//! -[Atomtype](https://github.com/Amber-MD/AmberClassic/blob/main/src/antechamber/atomtype.c)
//! -[Bondtype](https://github.com/Amber-MD/AmberClassic/blob/main/src/antechamber/bondtype.c)
//! -[Parmchk](https://github.com/Amber-MD/AmberClassic/blob/main/src/antechamber/parmchk2.c)

mod chem_env;

use std::{io, path::Path, time::Instant};

use bio_files::{
    AtomGeneric, BondGeneric, BondType,
    amber_typedef::{AmberDef, AtomTypeDef},
};
use chem_env::*;
use na_seq::Element;

use crate::util::build_adjacency_list;

/// See note: Only loading ones we need for small organic molecules.
const DEF_ABCG2: &str = include_str!("../../param_data/antechamber_defs/ATOMTYPE_ABCG2.DEF");
// const DEF_AMBER: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_AMBER.DEF");
// const DEF_BCC: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_BCC.DEF");
// const DEF_GAS: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GAS.DEF");
// const DEF_GFF: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GFF.DEF");
const DEF_GFF2: &str = include_str!("../../param_data/antechamber_defs/ATOMTYPE_GFF2.DEF");
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
    num_double_bonds: u8,
    num_triple_bonds: u8,
    has_double_to_hetero: bool,
    // todo: Do we need a struct that accurately represents the f9 chem env data? Is this it?
    // e.g. something that has a `From<String>` method that parses `(XD4[sb',db])	&`(C(N4))`, etc (f9 col)
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
    let mut num_double_bonds = vec![0u8; atoms.len()];
    let mut num_triple_bonds = vec![0u8; atoms.len()];
    let mut has_double_to_hetero = vec![false; atoms.len()];

    for bond in bonds {
        let i = bond.atom_0_sn as usize - 1;
        let j = bond.atom_1_sn as usize - 1;

        match bond.bond_type {
            BondType::Aromatic => {
                let in_ring_i = ring_sizes[i].iter().any(|s| (5..=7).contains(s));
                let in_ring_j = ring_sizes[j].iter().any(|s| (5..=7).contains(s));

                if in_ring_i && in_ring_j {
                    aromatic_bond_counts[i] = aromatic_bond_counts[i].saturating_add(1);
                    aromatic_bond_counts[j] = aromatic_bond_counts[j].saturating_add(1);
                }
            }
            BondType::Double => {
                num_double_bonds[i] = num_double_bonds[i].saturating_add(1);
                num_double_bonds[j] = num_double_bonds[j].saturating_add(1);

                let ei = atoms[i].element;
                let ej = atoms[j].element;

                let i_hetero = matches!(
                    ei,
                    Element::Oxygen | Element::Nitrogen | Element::Sulfur | Element::Phosphorus
                );
                let j_hetero = matches!(
                    ej,
                    Element::Oxygen | Element::Nitrogen | Element::Sulfur | Element::Phosphorus
                );

                if i_hetero && !j_hetero {
                    has_double_to_hetero[j] = true;
                } else if j_hetero && !i_hetero {
                    has_double_to_hetero[i] = true;
                }
            }
            BondType::Triple => {
                num_triple_bonds[i] = num_triple_bonds[i].saturating_add(1);
                num_triple_bonds[j] = num_triple_bonds[j].saturating_add(1);
            }
            _ => {}
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
                num_double_bonds: num_double_bonds[idx],
                num_triple_bonds: num_triple_bonds[idx],
                has_double_to_hetero: has_double_to_hetero[idx],
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

/// Used when determining which type of hydrogen to match, based on withdrawal count.
fn is_elec_withdrawing_element(el: Element) -> bool {
    use Element::*;
    matches!(
        el,
        Oxygen | Nitrogen | Sulfur | Phosphorus | Fluorine | Chlorine | Bromine | Iodine
    )
}

/// Electron withdrawal. Used to determine H type.
fn ew_count_for_heavy(idx: usize, atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> u8 {
    let mut count = 0u8;

    for &j in &adj[idx] {
        let nb = &atoms[j];

        if nb.element == Element::Hydrogen {
            continue;
        }

        if is_elec_withdrawing_element(nb.element) {
            count = count.saturating_add(1);
        }
    }

    // h1/h2/h3 only go up to 3; clamp to keep behavior sane.
    count.min(3)
}

/// Electron withdrawal. Used to determine H type.
fn ew_count_for_atom(idx: usize, atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> u8 {
    if atoms[idx].element == Element::Hydrogen {
        if let Some(nb) = single_heavy_neighbor(idx, atoms, adj) {
            ew_count_for_heavy(nb, atoms, adj)
        } else {
            0
        }
    } else {
        ew_count_for_heavy(idx, atoms, adj)
    }
}

fn non_h_env_matches(
    env_str: &str,
    idx: usize,
    _atoms: &[AtomGeneric],
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

    let inner = &env_str[1..env_str.len() - 1].trim();
    let parts: Vec<&str> = inner
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();

    if parts.is_empty() {
        return false;
    }

    // We *only* support XX[AR...] patterns here.
    // Examples:
    //   (XX[AR1],XX[AR1],XX[AR1])  -- cp
    //   (XX[AR1.AR2.AR3])         -- nu/nv/etc
    let all_xx = parts
        .iter()
        .all(|p| p.starts_with("XX[") && p.ends_with(']'));
    if !all_xx {
        // Unknown env syntax (C3[DB], C3(XA1), N2[DB], etc) – we treat as "doesn't match"
        return false;
    }

    let mut required_aromatic_neighbors = 0usize;

    for part in &parts {
        let inside = &part[3..part.len() - 1]; // between "XX[" and "]"
        let tags: Vec<&str> = inside.split('.').map(str::trim).collect();

        if tags.iter().any(|t| t.starts_with("AR")) {
            required_aromatic_neighbors += 1;
        } else {
            // XX[…] without AR – we don't know how to interpret it, fail
            return false;
        }
    }

    if required_aromatic_neighbors == 0 {
        return false;
    }

    let aromatic_neighbors = adj[idx].iter().filter(|&&j| env_all[j].is_aromatic).count();

    aromatic_neighbors >= required_aromatic_neighbors
}

fn atomic_property_matches(
    prop: &str,
    idx: usize,
    _atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
) -> bool {
    let prop = prop.trim();
    let env = &env_all[idx];

    // RGn ring size (e.g. [RG3], [RG4])
    if let Some(ring_size) = parse_ring_size(prop) {
        if !env.ring_sizes.contains(&ring_size) {
            return false;
        }
    }

    // Any AR* tag → must be aromatic
    if prop.contains("AR") && !env.is_aromatic {
        return false;
    }

    // Gently handle [DB] / [TB] when they appear as plain tokens inside [...]
    if let (Some(l), Some(r)) = (prop.find('['), prop.rfind(']')) {
        if r > l + 1 {
            let inside = &prop[l + 1..r];
            let tokens: Vec<&str> = inside.split(',').map(str::trim).collect();

            let mut needs_db = false;
            let mut needs_tb = false;

            for t in tokens {
                match t {
                    "DB" => needs_db = true,
                    "TB" => needs_tb = true,
                    _ => {}
                }
            }

            if needs_db && env.num_double_bonds == 0 {
                return false;
            }

            if needs_tb && env.num_triple_bonds == 0 {
                return false;
            }
        }
    }

    // We intentionally ignore DL, sb, numeric qualifiers (2DL, 1DB, 3sb, …) here.
    true
}

/// Checks each atom against a FF def to determine if it's a match. If it is, we assign the FF
/// type from the def.
///
fn matches_def(
    def: &AtomTypeDef,
    idx: usize,
    atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
) -> bool {
    let atom = &atoms[idx];
    let env = &env_all[idx];

    // ---- cd: fully heuristic, ignore DEF f5/f6 details ----
    if def.name == "cd" {
        return is_cd_like(idx, atoms, env, bonds, adj);
    }

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

    // --- SPECIAL: nh: sp3 N should be directly attached to an aromatic carbon ---
    if def.name == "nh" {
        let has_aromatic_c_neighbor = adj[idx]
            .iter()
            .any(|&j| atoms[j].element == Element::Carbon && env_all[j].is_aromatic);
        if !has_aromatic_c_neighbor {
            return false;
        }
    }

    // never use cp for now
    if def.name == "cp" {
        return false;
    }

    // --- SPECIAL: ca (pure aromatic C, benzene-like) ---
    if def.name == "ca" {
        if !env.is_aromatic {
            return false;
        }
        if !env.ring_sizes.contains(&6) {
            return false;
        }
    }

    // --- SPECIAL: carbonyl carbon ---
    if def.name == "c" {
        if !is_carbonyl_carbon(idx, atoms, bonds) {
            return false;
        }
        return true;
    }

    // --- SPECIAL: keep nb exocyclic ---
    if def.name == "nb" {
        if !env.ring_sizes.is_empty() {
            return false;
        }
    }

    // --- SPECIAL: ring sp2 N in conjugated system ---
    if def.name == "nc" {
        if env.ring_sizes.is_empty() {
            return false;
        }
        if env.degree != 2 {
            return false;
        }

        let mut has_db = false;
        for b in bonds {
            let i = b.atom_0_sn as usize - 1;
            let j = b.atom_1_sn as usize - 1;
            if i == idx || j == idx {
                if matches!(b.bond_type, BondType::Double) {
                    let other = if i == idx { j } else { i };
                    if atoms[other].element != Element::Hydrogen {
                        has_db = true;
                    }
                }
            }
        }
        if !has_db {
            return false;
        }
    }

    if let Some(ew_needed) = def.electron_withdrawal_count {
        let ew_here = ew_count_for_atom(idx, atoms, adj);
        if ew_here != ew_needed {
            return false;
        }
    }

    if let Some(ref prop) = def.atomic_property {
        // don't re-interpret properties for c / cd; we already handle them
        if def.name != "c" && def.name != "cd" {
            if let Some(ring_size) = parse_ring_size(prop) {
                if !env.ring_sizes.contains(&ring_size) {
                    return false;
                }
            }

            if prop.contains("AR") {
                let mut aromatic = env.is_aromatic;
                if !aromatic {
                    let mut has_db_like = false;
                    for b in bonds {
                        let i = b.atom_0_sn as usize - 1;
                        let j = b.atom_1_sn as usize - 1;
                        if i == idx || j == idx {
                            match b.bond_type {
                                BondType::Double | BondType::Aromatic => {
                                    has_db_like = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    if !env.ring_sizes.is_empty() && has_db_like {
                        aromatic = true;
                    }
                }
                if !aromatic {
                    return false;
                }
            }

            if prop.contains("[DB]") || prop.contains("[TB]") {
                let mut has_db = false;
                let mut has_tb = false;
                for b in bonds {
                    let i = b.atom_0_sn as usize - 1;
                    let j = b.atom_1_sn as usize - 1;
                    if i == idx || j == idx {
                        match b.bond_type {
                            BondType::Double => has_db = true,
                            BondType::Triple => has_tb = true,
                            _ => {}
                        }
                    }
                }

                if prop.contains("[DB]") && !has_db {
                    return false;
                }
                if prop.contains("[TB]") && !has_tb {
                    return false;
                }
            }
        }
    }

    if let Some(ref env_str) = def.chem_env {
        if env_str != "&" {
            if atom.element == Element::Hydrogen {
                if !hydrogen_env_matches(env_str, idx, atoms, env_all, adj) {
                    return false;
                }
            } else {
                let pattern: ChemEnvPattern = env_str.as_str().into();
                if !pattern.matches(idx, atoms, env_all, bonds, adj) {
                    return false;
                }
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

/// This converts SP2 carbons to the correct types after the main algorithm infers
/// types from DEF files. E.g. "cc", "cd", "c" etc.
/// Adjust field names / bond-order enum to match your BondGeneric definition.
fn postprocess_sp2_carbons(atoms: &[AtomGeneric], bonds: &[BondGeneric], types: &mut [String]) {
    // Neighbour list with bond orders.
    let mut nb: Vec<Vec<(usize, BondType)>> = vec![Vec::new(); atoms.len()];

    for bond in bonds {
        let i = bond.atom_0_sn as usize - 1;
        let j = bond.atom_1_sn as usize - 1;
        let order = bond.bond_type;

        nb[i].push((j, order));
        nb[j].push((i, order));
    }

    // Pass 1: identify carbonyl carbons: C=O / C=N / C=S / C=P.
    // Only refine generic sp2 carbons (`c2`).
    for i in 0..atoms.len() {
        if atoms[i].element != Element::Carbon {
            continue;
        }

        if types[i].as_str() != "c2" {
            continue;
        }

        let mut double_to_hetero = false;

        for &(j, order) in &nb[i] {
            if matches!(order, BondType::Double) {
                match atoms[j].element {
                    Element::Oxygen | Element::Nitrogen | Element::Sulfur | Element::Phosphorus => {
                        double_to_hetero = true;
                        break;
                    }
                    _ => {}
                }
            }
        }

        if double_to_hetero {
            types[i] = "c".to_string();
        }
    }

    // Pass 2: cc / cd for remaining sp² carbons (still `c2`).
    for i in 0..atoms.len() {
        if atoms[i].element != Element::Carbon {
            continue;
        }

        // Only refine generic sp² (`c2`); keep `c3` as truly sp³.
        if types[i].as_str() != "c2" {
            continue;
        }

        let mut double_to_carbon = 0u8;
        let mut single_to_aromatic = false;
        let mut single_to_carbonyl = false;

        for &(j, order) in &nb[i] {
            match order {
                BondType::Double if atoms[j].element == Element::Carbon => {
                    double_to_carbon += 1;
                }
                BondType::Single => {
                    if types[j] == "c" {
                        single_to_carbonyl = true;
                    }
                    if types[j] == "ca" || types[j] == "cp" {
                        single_to_aromatic = true;
                    }
                }
                _ => {}
            }
        }

        if single_to_carbonyl {
            // sp² C directly attached to a carbonyl carbon → cd
            types[i] = "cd".to_string();
        } else if double_to_carbon > 0 || single_to_aromatic {
            // Conjugated sp² C (C=C or vinyl attached to aromatic) → cc
            types[i] = "cc".to_string();
        }
    }
}

/// Run this towards the end of the pipeline to correctly mark "nh" etc, instead of
/// "n3".
fn postprocess_nh_from_aromatic_neighbors(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    // Build simple adjacency without caring about bond order.
    let mut nb: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];

    for bond in bonds {
        let i = bond.atom_0_sn as usize - 1;
        let j = bond.atom_1_sn as usize - 1;
        nb[i].push(j);
        nb[j].push(i);
    }

    for i in 0..atoms.len() {
        if atoms[i].element != Element::Nitrogen {
            continue;
        }

        // Only retag nitrogens that are currently generic sp3 `n3`.
        if types[i].as_str() != "n3" {
            continue;
        }

        // Look for at least one aromatic carbon neighbor (ca/cp).
        let has_aromatic_neighbor = nb[i]
            .iter()
            .any(|&j| matches!(types[j].as_str(), "ca" | "cp"));

        if has_aromatic_neighbor {
            types[i] = "nh".to_string();
        }
    }
}

fn postprocess_sulfonyl_s(atoms: &[AtomGeneric], bonds: &[BondGeneric], types: &mut [String]) {
    // We only need adjacency + degrees; reuse the same helper.
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        // Only consider sulfur currently typed as generic s6.
        if atom.element != Element::Sulfur {
            continue;
        }

        if types[i].as_str() != "s6" {
            continue;
        }

        // Sulfonyl S here has 4 neighbors.
        let degree = adj[i].len();
        if degree != 4 {
            continue;
        }

        // Count "double-bond-like" oxygens: O with only one neighbor (the S).
        let double_like_o = adj[i]
            .iter()
            .filter(|&&j| atoms[j].element == Element::Oxygen && adj[j].len() == 1)
            .count();

        // If S has at least two such O neighbors, treat it as sulfonyl S (`sy`).
        if double_like_o >= 2 {
            types[i] = "sy".to_string();
        }
    }
}

fn postprocess_ns_from_env(atoms: &[AtomGeneric], bonds: &[BondGeneric], types: &mut [String]) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Element::Nitrogen {
            continue;
        }

        // Only refine Ns currently typed as generic `n7`.
        if types[i].as_str() != "n7" {
            continue;
        }

        let neighbors = &adj[i];

        // ns expects 3 neighbors total.
        if neighbors.len() != 3 {
            continue;
        }

        // Exactly one hydrogen neighbor.
        let num_h = neighbors
            .iter()
            .filter(|&&j| atoms[j].element == Element::Hydrogen)
            .count();

        if num_h != 1 {
            continue;
        }

        let mut has_aromatic_c = false;
        let mut has_carbonyl_c = false;

        for &j in neighbors {
            if atoms[j].element != Element::Carbon {
                continue;
            }

            match types[j].as_str() {
                "ca" | "cp" => has_aromatic_c = true,
                "c" => has_carbonyl_c = true,
                _ => {}
            }
        }

        if has_aromatic_c && has_carbonyl_c {
            types[i] = "ns".to_string();
        }
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

    println!("\n\nDebug dump of dev env types: ");

    for def in &defs.gff2.atomtypes {
        if let Some(env) = &def.chem_env {
            let pattern: ChemEnvPattern = env.as_str().into();
            println!("Env pattern for {}:{:?}", def.name, pattern);
        }
    }

    let adj = build_adjacency_list(atoms, bonds).unwrap();
    let env = build_env(atoms, bonds, &adj);

    let result = assign_types(&defs.gff2.atomtypes, atoms, &env, bonds, &adj);

    // You can optionally re-enable postprocess_sp2_carbons here once you like
    // postprocess_sp2_carbons(atoms, bonds, &mut result);

    let elapsed = start.elapsed().as_micros();
    println!("Complete in {elapsed} μs");

    result
}

/// A specific override. Hopefully we don't need too many of these!
fn is_cd_like(
    idx: usize,
    atoms: &[AtomGeneric],
    env: &AtomEnvData,
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
) -> bool {
    let atom = &atoms[idx];

    // Only consider carbons
    if atom.element != Element::Carbon {
        return false;
    }

    // Must be at least roughly sp2 (at least one double bond)
    if env.num_double_bonds == 0 {
        return false;
    }

    // Look for a double bond to a heavy atom, but NOT to oxygen (carbonyls are "c").
    let mut has_db_to_heavy = false;
    let mut has_db_to_oxygen = false;
    let mut db_partner: Option<usize> = None;

    for b in bonds {
        let i = b.atom_0_sn as usize - 1;
        let j = b.atom_1_sn as usize - 1;

        if i == idx || j == idx {
            if matches!(b.bond_type, BondType::Double) {
                let other = if i == idx { j } else { i };

                if atoms[other].element != Element::Hydrogen {
                    has_db_to_heavy = true;
                    if atoms[other].element == Element::Oxygen {
                        has_db_to_oxygen = true;
                    }

                    if atoms[other].element == Element::Carbon {
                        db_partner.get_or_insert(other);
                    }
                }
            }
        }
    }

    if !has_db_to_heavy {
        return false;
    }

    // carbonyl C=O is handled as "c"
    if has_db_to_oxygen {
        return false;
    }

    // Neighbour analysis
    let mut hetero_count: u8 = 0;
    let mut has_s_neighbor = false;
    let mut has_o_single_neighbor = false;
    let mut has_aromatic_c_neighbor = false;
    let mut has_carbonyl_neighbor = false;

    for &nb in &adj[idx] {
        let el = atoms[nb].element;

        if el != Element::Carbon && el != Element::Hydrogen {
            hetero_count = hetero_count.saturating_add(1);
        }

        if el == Element::Sulfur {
            has_s_neighbor = true;
        }

        // O single-bonded directly to this carbon (e.g. C–O in an enol/ether)
        if el == Element::Oxygen {
            for b in bonds {
                let i = b.atom_0_sn as usize - 1;
                let j = b.atom_1_sn as usize - 1;

                if ((i == idx && j == nb) || (i == nb && j == idx))
                    && matches!(b.bond_type, BondType::Single)
                {
                    has_o_single_neighbor = true;
                    break;
                }
            }
        }

        if el == Element::Carbon {
            // Is this neighbour carbon aromatic?
            for b in bonds {
                let i = b.atom_0_sn as usize - 1;
                let j = b.atom_1_sn as usize - 1;

                if (i == nb || j == nb) && matches!(b.bond_type, BondType::Aromatic) {
                    has_aromatic_c_neighbor = true;
                }
            }

            // Is this neighbour a carbonyl carbon (C with a double bond to O/N/S/P)?
            if is_carbonyl_carbon(nb, atoms, bonds) {
                has_carbonyl_neighbor = true;
            }
        }
    }

    // Does our C=C partner sit next to a carbonyl carbon?
    let partner_has_carbonyl = db_partner.map_or(false, |p| {
        for &nb in &adj[p] {
            if atoms[nb].element == Element::Carbon && is_carbonyl_carbon(nb, atoms, bonds) {
                return true;
            }
        }
        false
    });

    // 1) Strongly activated: S neighbour or ≥2 hetero neighbours → cd
    if has_s_neighbor || hetero_count >= 2 {
        return true;
    }

    // 2) O-substituted vinyl attached to an aromatic ring (O9T-style case),
    //    but only when the C=C partner is *not* itself next to a carbonyl (avoid CPB atom 11).
    if hetero_count == 1
        && has_o_single_neighbor
        && has_aromatic_c_neighbor
        && !partner_has_carbonyl
    {
        return true;
    }

    // 3) Enone bridge: sp2 carbon that has a carbonyl-carbon neighbour and
    //    at least one other hetero-substituted carbon neighbour.
    if has_carbonyl_neighbor {
        let mut het_sub_neighbors: u8 = 0;

        for &nb in &adj[idx] {
            let el = atoms[nb].element;

            if el != Element::Carbon && el != Element::Hydrogen {
                // direct hetero neighbour
                het_sub_neighbors = het_sub_neighbors.saturating_add(1);
                continue;
            }

            if el == Element::Carbon {
                // neighbour is a carbonyl carbon?
                if is_carbonyl_carbon(nb, atoms, bonds) {
                    het_sub_neighbors = het_sub_neighbors.saturating_add(1);
                    continue;
                }

                // neighbour carbon with a single-bonded oxygen (like C2 in CPB)
                'inner: for b in bonds {
                    let i = b.atom_0_sn as usize - 1;
                    let j = b.atom_1_sn as usize - 1;

                    if i == nb || j == nb {
                        let other = if i == nb { j } else { i };
                        if atoms[other].element == Element::Oxygen
                            && matches!(b.bond_type, BondType::Single)
                        {
                            het_sub_neighbors = het_sub_neighbors.saturating_add(1);
                            break 'inner;
                        }
                    }
                }
            }
        }

        if het_sub_neighbors >= 2 {
            // e.g. CPB atom 12: neighbour to carbonyl C and to an O-substituted C
            return true;
        }
    }

    // Everything else: not cd; leave it as cc/c2/etc.
    false
}

fn assign_types(
    defs: &[AtomTypeDef],
    atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
) -> Vec<String> {
    let mut types = Vec::with_capacity(atoms.len());

    for idx in 0..atoms.len() {
        let mut ty = "DU".to_owned();

        for def in defs {
            // Don't take cd directly from the DEF; we’ll assign it heuristically
            if def.name == "cd" {
                continue;
            }

            if matches_def(def, idx, atoms, env_all, bonds, adj) {
                ty = def.name.clone();
                break;
            }
        }

        // --- cd refinement: override certain sp2 carbons as cd ---
        if is_cd_like(idx, atoms, &env_all[idx], bonds, adj) {
            if ty == "ce" || ty == "c2" || ty == "cc" || ty == "ca" {
                ty = "cd".to_owned();
            }
        }

        types.push(ty);
    }

    types
}
