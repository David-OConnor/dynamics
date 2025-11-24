//! Individual case handling that we are not able to infer from DEF alone,
//! or if it's simpler to implement here.
//!
//! Warning: We are currently mixing up cc and cd atom types in many places.
//!
//! todo: COmbine these, so you only loop through atoms once.

use bio_files::{AtomGeneric, BondGeneric, BondType};
use na_seq::Element::*;

use crate::{
    param_inference::{AtomEnvData, chem_env::is_carbonyl_carbon},
    util::build_adjacency_list,
};

pub(in crate::param_inference) fn postprocess_ne_to_n2(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Nitrogen {
            continue;
        }

        if types[i].as_str() != "ne" {
            continue;
        }

        let mut o_neighbors = 0u8;

        for &j in &adj[i] {
            if atoms[j].element == Oxygen {
                o_neighbors = o_neighbors.saturating_add(1);
            }
        }

        // Treat "ne" only as nitro-like: N with 2+ O neighbours.
        if o_neighbors < 2 {
            types[i] = "n2".to_string();
        }
    }
}

pub(in crate::param_inference) fn postprocess_p5(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Phosphorus {
            continue;
        }

        let degree = adj[i].len();
        if degree < 4 {
            continue;
        }

        let o_neighbors = adj[i]
            .iter()
            .filter(|&&j| atoms[j].element == Oxygen)
            .count();

        if o_neighbors >= 3 {
            types[i] = "p5".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_nu_to_n7(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};

    for n_idx in 0..atoms.len() {
        if atoms[n_idx].element != Nitrogen {
            continue;
        }

        if types[n_idx].as_str() != "nu" {
            continue;
        }

        let neighbors = &adj[n_idx];
        let degree = neighbors.len();

        // neutral 3-coordinate N with at least one H
        if degree != 3 {
            continue;
        }

        let h_count = neighbors
            .iter()
            .filter(|&&j| atoms[j].element == Hydrogen)
            .count();

        if h_count == 0 {
            continue;
        }

        // skip any N with double/triple bonds
        let mut has_multiple_bond = false;
        for b in bonds {
            let a0 = b.atom_0_sn as usize - 1;
            let a1 = b.atom_1_sn as usize - 1;

            if a0 == n_idx || a1 == n_idx {
                if !matches!(b.bond_type, BondType::Single | BondType::Aromatic) {
                    has_multiple_bond = true;
                    break;
                }
            }
        }

        if has_multiple_bond {
            continue;
        }

        // Guard: skip urea/guanidinium-like nitrogens attached to a tri-N carbon
        // with at least one C=N double bond to N.
        let mut urea_like = false;

        'outer: for &c_idx in neighbors {
            if atoms[c_idx].element != Carbon {
                continue;
            }

            // Count N neighbours of this carbon
            let mut n_neigh = 0u8;
            for &c_nb in &adj[c_idx] {
                if atoms[c_nb].element == Nitrogen {
                    n_neigh = n_neigh.saturating_add(1);
                }
            }

            if n_neigh < 2 {
                continue;
            }

            // Check for a C=N double bond from this carbon to any N
            let mut has_double_to_n = false;
            for b in bonds {
                let a0 = b.atom_0_sn as usize - 1;
                let a1 = b.atom_1_sn as usize - 1;

                if a0 == c_idx || a1 == c_idx {
                    if matches!(b.bond_type, BondType::Double) {
                        let other = if a0 == c_idx { a1 } else { a0 };
                        if atoms[other].element == Nitrogen {
                            has_double_to_n = true;
                            break;
                        }
                    }
                }
            }

            if n_neigh >= 2 && has_double_to_n {
                urea_like = true;
                break 'outer;
            }
        }

        if urea_like {
            continue;
        }

        // Looks like an n7-style neutral amide/amine N
        types[n_idx] = "n7".to_owned();
    }
}



pub(in crate::param_inference) fn postprocess_ring_n_types(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Nitrogen {
            continue;
        }

        let env = &env_all[i];

        // --- n3 -> na: tertiary N in a 5/6-membered ring with ≥2 ring sp2 neighbours ---
        if types[i].as_str() == "n3"
            && env.degree == 3
            && env.num_attached_h == 0
            && env.ring_sizes.iter().any(|&s| s == 5 || s == 6)
        {
            let mut ring_sp2_neighbors = 0u8;

            for &nb in &adj[i] {
                if atoms[nb].element == Hydrogen {
                    continue;
                }

                let nb_env = &env_all[nb];
                let nb_ty = types[nb].as_str();

                let is_sp2_like = nb_env.is_aromatic
                    || nb_env.num_double_bonds > 0
                    || matches!(nb_ty, "ca" | "cc" | "cd" | "ce" | "cf" | "cg");

                if !nb_env.ring_sizes.is_empty() && is_sp2_like {
                    ring_sp2_neighbors = ring_sp2_neighbors.saturating_add(1);
                }
            }

            // need at least two ring sp2 neighbours to call it "na"
            if ring_sp2_neighbors >= 2 {
                types[i] = "na".to_owned();
            }
        }

        // --- refine ring sp2 nitrogens: nc vs nd, starting from nc / n1 ---
        let ty = types[i].as_str();
        if ty != "nc" && ty != "n1" {
            continue;
        }

        if env.ring_sizes.is_empty() || env.num_double_bonds == 0 {
            continue;
        }

        // Find the non-H atom we are double-bonded to
        let mut db_partner: Option<usize> = None;
        for b in bonds {
            let a0 = b.atom_0_sn as usize - 1;
            let a1 = b.atom_1_sn as usize - 1;

            if (a0 == i || a1 == i) && matches!(b.bond_type, BondType::Double) {
                let other = if a0 == i { a1 } else { a0 };
                if atoms[other].element != Hydrogen {
                    db_partner = Some(other);
                    break;
                }
            }
        }

        let p = match db_partner {
            Some(p) => p,
            None => continue,
        };

        // Count hetero neighbors (including us) on the double-bond partner
        let mut partner_hetero_neighbors = 0u8;
        for &nb in &adj[p] {
            match atoms[nb].element {
                Carbon | Hydrogen => {}
                _ => {
                    partner_hetero_neighbors = partner_hetero_neighbors.saturating_add(1);
                }
            }
        }

        // Partner strongly hetero-substituted → nd; otherwise nc
        if partner_hetero_neighbors >= 2 {
            types[i] = "nd".to_owned();
        } else {
            types[i] = "nc".to_owned();
        }
    }
}


pub(in crate::param_inference) fn postprocess_carbonyl_c(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use Carbon;

    for (idx, atom) in atoms.iter().enumerate() {
        if atom.element != Carbon {
            continue;
        }

        if is_carbonyl_carbon(idx, atoms, bonds) {
            types[idx] = "c".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_nd_sp2_hetero(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    for idx in 0..atoms.len() {
        if atoms[idx].element != Nitrogen {
            continue;
        }

        if types[idx].as_str() != "nc" {
            continue;
        }

        let env = &env_all[idx];

        // ring N in 5- or 6-membered ring
        if !env.ring_sizes.iter().any(|&s| s == 5 || s == 6) {
            continue;
        }

        // sp2-like: degree 2 and exactly one double bond
        if env.degree != 2 || env.num_double_bonds != 1 {
            continue;
        }

        // must have a single-bonded hetero neighbour (O/N/S)
        let mut has_single_hetero = false;

        'outer: for &nb in &adj[idx] {
            let el = atoms[nb].element;
            // Only treat O / S single-bonded neighbours as the “extra hetero”
            if !matches!(el, Oxygen | Sulfur) {
                continue;
            }

            for b in bonds {
                let i = b.atom_0_sn as usize - 1;
                let j = b.atom_1_sn as usize - 1;

                if (i == idx && j == nb) || (j == idx && i == nb) {
                    if matches!(b.bond_type, BondType::Single) {
                        has_single_hetero = true;
                    }
                    break;
                }
            }

            if has_single_hetero {
                break 'outer;
            }
        }

        if has_single_hetero {
            types[idx] = "nd".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_na_ring_bridge(
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        // Only refine generic sp2-like nitrogens
        if types[i].as_str() != "n2" {
            continue;
        }

        let env = &env_all[i];

        // Ring N in a 5- or 6-membered ring
        if !env.ring_sizes.iter().any(|&s| s == 5 || s == 6) {
            continue;
        }

        // 3-coordinate: two heavy neighbours + one H
        if env.degree != 3 || env.num_attached_h != 1 {
            continue;
        }

        let mut has_n_neighbor = false;
        let mut has_sp2_c_neighbor = false;

        for &nb in &adj[i] {
            match atoms[nb].element {
                Hydrogen => {}
                Nitrogen => {
                    has_n_neighbor = true;
                }
                Carbon => {
                    if matches!(types[nb].as_str(), "c" | "ca" | "cc" | "cd") {
                        has_sp2_c_neighbor = true;
                    }
                }
                _ => {}
            }
        }

        if has_n_neighbor && has_sp2_c_neighbor {
            types[i] = "na".to_owned();
        }
    }
}

pub fn postprocess_nb_aromatic(
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        let env = &env_all[i];

        // Only consider N in 5- or 6-membered rings
        if !env.ring_sizes.iter().any(|&s| s == 5 || s == 6) {
            continue;
        }

        // Candidate starting types that can become nb
        let ty = types[i].as_str();
        if !matches!(ty, "n2" | "nu" | "n7") {
            continue;
        }

        let neighbors = &adj[i];

        let mut ring_heavy_neighbors = 0u8;
        let mut conjugated_neighbors = 0u8;

        for &nb in neighbors {
            if atoms[nb].element == Hydrogen {
                continue;
            }

            let nb_env = &env_all[nb];

            if !nb_env.ring_sizes.is_empty() {
                ring_heavy_neighbors = ring_heavy_neighbors.saturating_add(1);
            }

            if nb_env.num_double_bonds > 0 || nb_env.is_aromatic {
                conjugated_neighbors = conjugated_neighbors.saturating_add(1);
            }
        }

        // Need at least two ring heavy neighbors and at least one conjugated neighbor
        if ring_heavy_neighbors < 2 || conjugated_neighbors == 0 {
            continue;
        }

        types[i] = "nb".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_nb_to_na_ring_with_h(
    atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        if types[i].as_str() != "nb" {
            continue;
        }

        let env = &env_all[i];

        // nb -> na only for ring nitrogens with exactly one attached H
        if !env.ring_sizes.iter().any(|&s| s == 5 || s == 6) {
            continue;
        }

        if env.degree != 3 {
            continue;
        }

        if env.num_attached_h != 1 {
            continue;
        }

        types[i] = "na".to_owned();
    }
}
pub(in crate::param_inference) fn postprocess_n7_to_nu(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        if types[i].as_str() != "n7" {
            continue;
        }

        let nbrs = &adj_list[i];

        let h_count = nbrs
            .iter()
            .filter(|&&j| atoms[j].element == Hydrogen)
            .count();

        if h_count == 0 {
            continue;
        }

        let c_neighbors: Vec<usize> = nbrs
            .iter()
            .copied()
            .filter(|&j| atoms[j].element == Carbon)
            .collect();

        if c_neighbors.is_empty() {
            continue;
        }

        let mut has_tri_n_carbon = false;

        for &c_idx in &c_neighbors {
            let n_neigh = adj_list[c_idx]
                .iter()
                .filter(|&&k| atoms[k].element == Nitrogen)
                .count();

            if n_neigh >= 2 {
                has_tri_n_carbon = true;
                break;
            }
        }

        if !has_tri_n_carbon {
            continue;
        }

        types[i] = "nu".to_owned();
    }
}


pub(in crate::param_inference) fn postprocess_c2_to_c_three_oxygens(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    types: &mut [String],
) {
    for i in 0..atoms.len() {
        if types[i] != "c2" {
            continue;
        }

        let nbrs = &adj_list[i];

        // C with exactly three O neighbors → treat as 'c'
        if nbrs.len() == 3 && nbrs.iter().all(|&j| atoms[j].element == Oxygen) {
            types[i] = "c".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_sy_to_s6(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    types: &mut [String],
) {
    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Sulfur {
            continue;
        }

        // Only refine things currently labelled as generic sulfur
        if types[i].as_str() != "sy" {
            continue;
        }

        let nbrs = &adj_list[i];
        let o_neighbors = nbrs
            .iter()
            .filter(|&&j| atoms[j].element == Oxygen)
            .count();

        // Highly oxidised sulfur: at least two O neighbours and at least three total neighbours
        if o_neighbors >= 2 && nbrs.len() >= 3 {
            types[i] = "s6".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_na_to_n3(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        if types[i].as_str() != "na" {
            continue;
        }

        let env = &env_all[i];

        // Only consider 3-coordinate nitrogens
        if env.degree != 3 {
            continue;
        }

        // Require “purely aliphatic” environment:
        //  - all non-H neighbours must be carbon
        //  - those carbons must *not* be sp²/aromatic (no double/triple bonds, not aromatic)
        let mut all_aliphatic = true;

        for &nb in &adj_list[i] {
            let nb_atom = &atoms[nb];

            match nb_atom.element {
                Hydrogen => {
                    // fine
                }
                Carbon => {
                    let nb_env = &env_all[nb];

                    if nb_env.num_double_bonds > 0
                        || nb_env.num_triple_bonds > 0
                        || nb_env.is_aromatic
                    {
                        all_aliphatic = false;
                        break;
                    }
                }
                _ => {
                    // Any hetero neighbor (O, N, S, etc.) means this is
                    // conjugated / hetero-substituted, so keep it as `na`.
                    all_aliphatic = false;
                    break;
                }
            }
        }

        if all_aliphatic {
            types[i] = "n3".to_owned();
        }
    }
}

//
// pub(in crate::param_inference) fn postprocess_p5_to_py(
//     atoms: &[AtomGeneric],
//     adj_list: &[Vec<usize>],
//     types: &mut [String],
// ) {
//     for i in 0..types.len() {
//         if types[i] != "p5" {
//             continue;
//         }
//
//         let mut o_like = 0;
//         let mut carbonyl_o = 0;
//
//         for &nbr_idx in &adj_list[i] {
//             let t = types[nbr_idx].as_str();
//             if t.starts_with('o') {
//                 o_like += 1;
//             }
//             if t == "o" {
//                 carbonyl_o += 1;
//             }
//         }
//
//         // P with at least two O-like neighbours and at least one `o` → py
//         if o_like >= 2 && carbonyl_o >= 1 {
//             types[i] = "py".to_owned();
//         }
//     }
// }

pub(in crate::param_inference) fn postprocess_py_to_p5_by_o_count(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    types: &mut [String],
) {
    for i in 0..atoms.len() {
        if atoms[i].element != Phosphorus {
            continue;
        }

        if types[i].as_str() != "py" {
            continue;
        }

        let o_neighbors = adj_list[i]
            .iter()
            .filter(|&&j| atoms[j].element == Oxygen)
            .count();

        // Treat P with fewer than 3 O neighbours as p5, not py.
        if o_neighbors < 3 {
            types[i] = "p5".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_cz_demote_ring_nd(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Carbon {
            continue;
        }

        if types[i].as_str() != "cz" {
            continue;
        }

        // Collect nitrogen neighbours
        let n_neighbors: Vec<_> = adj[i]
            .iter()
            .copied()
            .filter(|&j| atoms[j].element == Nitrogen)
            .collect();

        // Only touch the "tri-nitrogen" guanidinium-like centers
        if n_neighbors.len() != 3 {
            continue;
        }

        // If *any* N neighbour is an nd (ring N), keep this as cc, not cz
        let has_ring_nd = n_neighbors.iter().any(|&j| types[j].as_str() == "nd");

        if has_ring_nd {
            types[i] = "cc".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_cc_to_cd_ring_hetero(
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen};

    for i in 0..atoms.len() {
        if atoms[i].element != Carbon {
            continue;
        }

        if types[i].as_str() != "cc" {
            continue;
        }

        let env = &env_all[i];

        // ring sp2 carbon (5- or 6-membered ring, at least one double bond)
        if !env.ring_sizes.iter().any(|&s| s == 5 || s == 6) {
            continue;
        }

        if env.num_double_bonds == 0 {
            continue;
        }

        let mut has_ring_hetero_n = false;

        for &nb in &adj[i] {
            if atoms[nb].element != Nitrogen {
                continue;
            }

            // only treat clearly heteroaromatic / conjugated ring N as triggers
            match types[nb].as_str() {
                "na" | "nb" | "nu" | "n7" => {
                    if !env_all[nb].ring_sizes.is_empty() {
                        has_ring_hetero_n = true;
                        break;
                    }
                }
                _ => {}
            }
        }

        if has_ring_hetero_n {
            types[i] = "cd".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_s6_to_sy(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Sulfur {
            continue;
        }

        // Only refine things we currently think are s6
        if types[i].as_str() != "s6" {
            continue;
        }

        let nbrs = &adj[i];
        if nbrs.len() != 4 {
            continue;
        }

        let mut o_double = 0u8;
        let mut n_idx: Option<usize> = None;
        let mut c_idx: Option<usize> = None;

        for &j in nbrs {
            match atoms[j].element {
                Oxygen => {
                    // Count O neighbours that are double-bonded to S
                    let mut is_double = false;
                    for b in bonds {
                        let a0 = b.atom_0_sn as usize - 1;
                        let a1 = b.atom_1_sn as usize - 1;

                        if (a0 == i && a1 == j) || (a0 == j && a1 == i) {
                            if matches!(b.bond_type, BondType::Double) {
                                is_double = true;
                            }
                            break;
                        }
                    }
                    if is_double {
                        o_double = o_double.saturating_add(1);
                    }
                }
                Nitrogen => {
                    n_idx = Some(j);
                }
                Carbon => {
                    if c_idx.is_none() {
                        c_idx = Some(j);
                    }
                }
                _ => {}
            }
        }

        // We only care about S(=O)2 with two non-oxygen neighbours
        if o_double != 2 {
            continue;
        }

        let (n_idx, c_idx) = match (n_idx, c_idx) {
            (Some(n), Some(c)) => (n, c),
            _ => continue,
        };

        // Sulfonamide-like pattern:
        //   S(=O)2-N7-aryl → sy
        let n_ty = types[n_idx].as_str();
        let c_ty = types[c_idx].as_str();

        let c_is_sp2_aromatic = matches!(c_ty, "ca" | "cc" | "cd" | "ce" | "cf" | "cg");

        if n_ty == "n7" && c_is_sp2_aromatic {
            types[i] = "sy".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_n8_to_nv_guanidinium(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (n_idx, atom) in atoms.iter().enumerate() {
        if atom.element != Nitrogen {
            continue;
        }

        if types[n_idx].as_str() != "n8" {
            continue;
        }

        let nbrs = &adj[n_idx];

        let mut promote = false;
        for &nb in nbrs {
            if atoms[nb].element != Carbon {
                continue;
            }

            if is_terminal_guanidinium_nh2(n_idx, nb, atoms, bonds, &adj, types) {
                promote = true;
                break;
            }
        }

        if promote {
            types[n_idx] = "nv".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_n3_to_na_bridge_nd(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen};

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        if types[i].as_str() != "n3" {
            continue;
        }

        let env = &env_all[i];

        if env.degree != 3 || env.num_attached_h != 0 {
            continue;
        }

        let mut has_nd_neighbor = false;
        let mut has_ring_sp2_c = false;

        for &nb in &adj_list[i] {
            match atoms[nb].element {
                Nitrogen => {
                    if types[nb].as_str() == "nd" {
                        has_nd_neighbor = true;
                    }
                }
                Carbon => {
                    let c_ty = types[nb].as_str();
                    if matches!(c_ty, "ca" | "cc" | "cd" | "ce" | "cf" | "cg")
                        && !env_all[nb].ring_sizes.is_empty()
                    {
                        has_ring_sp2_c = true;
                    }
                }
                _ => {}
            }
        }

        if has_nd_neighbor && has_ring_sp2_c {
            types[i] = "na".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_nv_to_n8_non_guanidinium(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (n_idx, atom) in atoms.iter().enumerate() {
        if atom.element != Nitrogen {
            continue;
        }

        if types[n_idx].as_str() != "nv" {
            continue;
        }

        let nbrs = &adj[n_idx];

        let mut in_valid_guanidinium = false;
        for &nb in nbrs {
            if atoms[nb].element != Carbon {
                continue;
            }

            if is_terminal_guanidinium_nh2(n_idx, nb, atoms, bonds, &adj, types) {
                in_valid_guanidinium = true;
                break;
            }
        }

        // If this nv is not attached to a proper guanidinium center, demote it.
        if !in_valid_guanidinium {
            types[n_idx] = "n8".to_owned();
        }
    }
}


pub(in crate::param_inference) fn postprocess_cz_to_c2_guanidinium_mixed_n(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen, Hydrogen};

    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    for (i, atom) in atoms.iter().enumerate() {
        // Only consider carbon atoms
        if atom.element != Carbon {
            continue;
        }

        let nbrs = &adj[i];

        // Guanidinium centre: exactly three neighbours, all nitrogens.
        if nbrs.len() != 3 {
            continue;
        }

        if !nbrs.iter().all(|&j| atoms[j].element == Nitrogen) {
            continue;
        }

        // Count attached hydrogens for each nitrogen neighbour.
        let mut h_counts = [0u8; 3];

        for (k, &n_idx) in nbrs.iter().enumerate() {
            let mut h_count = 0u8;
            for &nn in &adj[n_idx] {
                if atoms[nn].element == Hydrogen {
                    h_count = h_count.saturating_add(1);
                }
            }
            h_counts[k] = h_count;
        }

        // Ignore ordering of neighbours.
        h_counts.sort_unstable();

        // 00L’s guanidinium has one N with two H, two N with one H: [1, 1, 2].
        if h_counts == [1, 1, 2] {
            types[i] = "c2".to_owned();
        }
    }
}

fn is_terminal_guanidinium_nh2(
    n_idx: usize,
    c_idx: usize,
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &[String],
) -> bool {
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};

    // Centre must be carbon
    if atoms[c_idx].element != Carbon {
        return false;
    }

    // N–C must be a single bond
    let mut is_single_nc = false;
    for b in bonds {
        let a0 = b.atom_0_sn as usize - 1;
        let a1 = b.atom_1_sn as usize - 1;

        if (a0 == n_idx && a1 == c_idx) || (a1 == n_idx && a0 == c_idx) {
            if matches!(b.bond_type, BondType::Single) {
                is_single_nc = true;
            }
            break;
        }
    }
    if !is_single_nc {
        return false;
    }

    // This N must be an NH₂: at least two attached hydrogens
    let mut h_count = 0u8;
    for &nb in &adj[n_idx] {
        if atoms[nb].element == Hydrogen {
            h_count = h_count.saturating_add(1);
        }
    }
    if h_count < 2 {
        return false;
    }

    // The carbon must have at least two nitrogen neighbours
    let mut n_nbrs: Vec<usize> = Vec::new();
    for &nb in &adj[c_idx] {
        if atoms[nb].element == Nitrogen {
            n_nbrs.push(nb);
        }
    }
    if n_nbrs.len() < 2 {
        return false;
    }

    // And at least one N neighbour must be double-bonded and typed n2 (C=N)
    let mut has_double_n2 = false;
    'outer: for &n_j in &n_nbrs {
        for b in bonds {
            let a0 = b.atom_0_sn as usize - 1;
            let a1 = b.atom_1_sn as usize - 1;

            if (a0 == c_idx && a1 == n_j) || (a1 == c_idx && a0 == n_j) {
                if matches!(b.bond_type, BondType::Double) && types[n_j].as_str() == "n2" {
                    has_double_n2 = true;
                }
                break;
            }
        }
        if has_double_n2 {
            break 'outer;
        }
    }

    has_double_n2
}
