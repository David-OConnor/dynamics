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
pub(crate) mod frcmod;
mod frcmod_missing_params;
mod parmchk_parse;
mod post_process;

use std::{io, time::Instant};

use bio_files::{
    AtomGeneric, BondGeneric, BondType,
    amber_typedef::{AmberDef, AtomTypeDef},
    md_params::ForceFieldParams,
};
use chem_env::*;
pub use frcmod::assign_missing_params;
use na_seq::Element;
use post_process::*;

use crate::{partial_charge_inference::infer_charge, util::build_adjacency_list};

/// See note: Only loading ones we need for small organic molecules.
const DEF_ABCG2: &str = include_str!("../../param_data/antechamber_defs/ATOMTYPE_ABCG2.DEF");
// const DEF_AMBER: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_AMBER.DEF");
// const DEF_BCC: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_BCC.DEF");
// const DEF_GAS: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GAS.DEF");
// const DEF_GFF: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_GFF.DEF");
const DEF_GFF2: &str = include_str!("../../param_data/antechamber_defs/ATOMTYPE_GFF2.DEF");
// const DEF_SYBYL: &str = include_str!("../param_data/antechamber_defs/ATOMTYPE_SYBYL.DEF");

// const BCCPARM: &str = include_str!("../../param_data/antechamber_defs/BCCPARM.DAT");

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
    // /// For partial charges, e.g. BCCPARM.DAT
    // bcc_parm: i8, // todo
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
            // bcc_parm: 0,
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
//
// // todo: Should we use bonds here too, which have BondType::Aromatic?
// fn is_aromatic_atom(
//     idx: usize,
//     atoms: &[AtomGeneric],
//     adj: &[Vec<usize>],
//     ring_sizes: &[Vec<u8>],
// ) -> bool {
//     let rs = &ring_sizes[idx];
//
//     if !rs.iter().any(|&s| (5..=7).contains(&s)) {
//         return false;
//     }
//
//     let neighbors = &adj[idx];
//
//     if neighbors.len() != 3 {
//         return false;
//     }
//
//     let shared_ring_neighbors = neighbors
//         .iter()
//         .filter(|&&j| {
//             atoms[j].element != Element::Hydrogen && ring_sizes[j].iter().any(|s| rs.contains(s))
//         })
//         .count();
//
//     shared_ring_neighbors >= 2
// }

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
        Oxygen | Nitrogen | Sulfur | Fluorine | Chlorine | Bromine | Iodine
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
    if let Some(ring_size) = parse_ring_size(prop)
        && !env.ring_sizes.contains(&ring_size)
    {
        return false;
    }

    // Any AR* tag → must be aromatic
    if prop.contains("AR") && !env.is_aromatic {
        return false;
    }

    // Gently handle [DB] / [TB] when they appear as plain tokens inside [...]
    if let (Some(l), Some(r)) = (prop.find('['), prop.rfind(']'))
        && r > l + 1
    {
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
        return false;
    }

    if let Some(e) = def.element
        && atom.element != e
    {
        return false;
    }

    if let Some(n) = def.attached_atoms
        && env.degree != n
    {
        return false;
    }

    if let Some(nh) = def.attached_h
        && env.num_attached_h != nh
    {
        return false;
    }

    // --- SPECIAL: ce should not grab plain non-conjugated alkenes ---
    if def.name == "ce" {
        // Require at least one double bond at this atom
        if env.num_double_bonds == 0 {
            return false;
        }

        // And *either* be in a ring *or* be the carbon side of a C=O / C=N / C=S / C=P
        if env.ring_sizes.is_empty() && !env.has_double_to_hetero {
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
    if def.name == "nb" && !env.ring_sizes.is_empty() {
        return false;
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
            if i == idx || j == idx && matches!(b.bond_type, BondType::Double) {
                let other = if i == idx { j } else { i };
                if atoms[other].element != Element::Hydrogen {
                    has_db = true;
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
            if let Some(ring_size) = parse_ring_size(prop)
                && !env.ring_sizes.contains(&ring_size)
            {
                return false;
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

    if let Some(ref env_str) = def.chem_env
        && env_str != "&"
    {
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

    // println!("\n\nDebug dump of dev env types: ");
    // for def in &defs.gff2.atomtypes {
    //     if let Some(env) = &def.chem_env {
    //         let pattern: ChemEnvPattern = env.as_str().into();
    //         println!("Env pattern for {}:{:?}", def.name, pattern);
    //     }
    // }

    let adj = build_adjacency_list(atoms, bonds).unwrap();
    let env = build_env(atoms, bonds, &adj);

    let mut result = assign_types(&defs.gff2.atomtypes, atoms, &env, bonds, &adj);

    postprocess_carbonyl_c(atoms, bonds, &mut result);

    postprocess_ring_n_types(atoms, bonds, &adj, &env, &mut result);
    postprocess_nd_sp2_hetero(atoms, bonds, &adj, &env, &mut result);
    postprocess_ne_to_n2(atoms, bonds, &mut result);

    postprocess_na_ring_bridge(atoms, &adj, &env, &mut result);

    postprocess_nb_aromatic(atoms, &adj, &env, &mut result);
    postprocess_p5(atoms, bonds, &mut result);
    postprocess_nu_to_n7(atoms, bonds, &mut result);

    postprocess_nb_to_na_ring_with_h(atoms, &env, &mut result);
    postprocess_n7_to_nu(atoms, &adj, &mut result);
    postprocess_c2_to_c_three_oxygens(atoms, &adj, &mut result);
    postprocess_na_to_n3(atoms, &adj, &env, &mut result);
    postprocess_sy_to_s6(atoms, bonds, &mut result);
    postprocess_sy_to_s6_if_nonaryl_sulfonamide(atoms, bonds, &mut result);
    postprocess_py_to_p5_by_o_count(atoms, &adj, &mut result);

    postprocess_cz_demote_ring_nd(atoms, bonds, &mut result);
    postprocess_cc_to_cd_ring_hetero(atoms, &adj, &env, &mut result);
    postprocess_s6_to_sy(atoms, bonds, &mut result);
    postprocess_s6_to_sy_if_attached_to_nh2_only(atoms, bonds, &adj, &mut result);

    postprocess_n8_to_nv_guanidinium(atoms, bonds, &mut result);
    postprocess_nv_to_n8_non_guanidinium(atoms, bonds, &mut result);
    postprocess_n3_to_na_bridge_nd(atoms, &adj, &env, &mut result);

    postprocess_cz_to_c2_guanidinium_mixed_n(atoms, bonds, &mut result);
    postprocess_tris_n_c_to_c2(atoms, bonds, &mut result);
    postprocess_nd_to_nc_ring_no_n_neighbor(atoms, bonds, &mut result);
    postprocess_cz_to_cd_if_has_explicit_multibond(atoms, bonds, &mut result);
    postprocess_nd_to_nc_if_double_to_cd(atoms, bonds, &mut result);
    postprocess_nd_to_nc_only_for_c_s_motifs(atoms, bonds, &mut result);
    postprocess_n7_to_nu_if_exocyclic(atoms, bonds, &mut result);
    postprocess_n3_to_n_if_attached_to_acyl_carbon(atoms, bonds, &mut result);
    postprocess_n3_to_nh_if_conjugated(atoms, bonds, &adj, &mut result);
    postprocess_cz_to_ca_if_ring_no_n_neighbors(atoms, bonds, &mut result);
    postprocess_cz_to_ca_if_has_aromatic_bond(atoms, bonds, &mut result);

    postprocess_c2_to_cf_if_conjugated_to_carbonyl(atoms, bonds, &adj, &mut result);
    postprocess_c2_to_ce_if_conjugated_to_carbonyl(atoms, bonds, &adj, &mut result);

    postprocess_cc_to_ca_if_has_aromatic_bond(atoms, bonds, &mut result);
    postprocess_cd_to_ca_if_has_aromatic_bond(atoms, bonds, &mut result);
    //
    postprocess_h_to_hx_alpha_carbon(atoms, &adj, &mut result);
    postprocess_n7_to_n6_if_small_ring(atoms, bonds, &adj, &mut result);
    postprocess_sy_to_s6_if_nonaryl_sulfonyl(atoms, bonds, &mut result);
    postprocess_s6_to_sy_if_primary_sulfonamide(atoms, &adj, &mut result);

    postprocess_n3_to_na_if_attached_to_alkenyl_c(atoms, bonds, &adj, &mut result);
    postprocess_c2_to_ce_if_vinylic_attached_to_aromatic(atoms, bonds, &adj, &mut result);
    postprocess_n1_to_n2_unless_sp_like(atoms, bonds, &adj, &mut result);

    let elapsed = start.elapsed().as_micros();
    println!("Complete in {elapsed} μs");

    result
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
        let mut ty = "du".to_owned();

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

        types.push(ty);
    }

    types
}

/// A high level interface for param inference: Updates FF type, partial charge for
/// each atom in a small organic molecule, and provides molecule-specfiic (FRCMOD) overrides.
///
/// Provide Adjacency list if cached; if not, we'll build it.
pub fn update_small_mol_params(
    atoms: &mut [AtomGeneric],
    bonds: &[BondGeneric],
    adjacency_list: Option<&[Vec<usize>]>,
    gaff2: &ForceFieldParams,
) -> io::Result<ForceFieldParams> {
    // todo: Move this elsewhere; you no longer need geostd.
    let defs = AmberDefSet::new()?;
    let ff_types = find_ff_types(atoms, bonds, &defs);

    for (i, atom) in atoms.iter_mut().enumerate() {
        atom.force_field_type = Some(ff_types[i].clone());
    }

    let charge = infer_charge(atoms, bonds).map_err(|e| io::Error::other(e))?;

    for (i, atom) in atoms.iter_mut().enumerate() {
        atom.partial_charge = Some(charge[i]);
    }

    let adj_list = match adjacency_list {
        Some(a) => a,
        None => &build_adjacency_list(atoms, bonds)
            .map_err(|_| io::Error::other("Problem building adjacency list"))?,
    };

    let params = assign_missing_params(atoms, adj_list, gaff2)?;

    Ok(params)
}
