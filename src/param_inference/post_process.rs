//! Individual case handling that we are not able to infer from DEF alone,
//! or if it's simpler to implement here.
//!
//! Warning: We are currently mixing up cc and cd atom types in many places.

// /// Run this towards the end of the pipeline to correctly mark "nh" etc, instead of
// /// "n3".
// fn postprocess_nh_from_aromatic_neighbors(
//     atoms: &[AtomGeneric],
//     bonds: &[BondGeneric],
//     types: &mut [String],
// ) {
//     // Build simple adjacency without caring about bond order.
//     let mut nb: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];
//
//     for bond in bonds {
//         let i = bond.atom_0_sn as usize - 1;
//         let j = bond.atom_1_sn as usize - 1;
//         nb[i].push(j);
//         nb[j].push(i);
//     }
//
//     for i in 0..atoms.len() {
//         if atoms[i].element != Element::Nitrogen {
//             continue;
//         }
//
//         // Only retag nitrogens that are currently generic sp3 `n3`.
//         if types[i].as_str() != "n3" {
//             continue;
//         }
//
//         // Look for at least one aromatic carbon neighbor (ca/cp).
//         let has_aromatic_neighbor = nb[i]
//             .iter()
//             .any(|&j| matches!(types[j].as_str(), "ca" | "cp"));
//
//         if has_aromatic_neighbor {
//             types[i] = "nh".to_string();
//         }
//     }
// }
//
// fn postprocess_sulfonyl_s(atoms: &[AtomGeneric], bonds: &[BondGeneric], types: &mut [String]) {
//     // We only need adjacency + degrees; reuse the same helper.
//     let adj = match build_adjacency_list(atoms, bonds) {
//         Ok(a) => a,
//         Err(_) => return,
//     };
//
//     for (i, atom) in atoms.iter().enumerate() {
//         // Only consider sulfur currently typed as generic s6.
//         if atom.element != Element::Sulfur {
//             continue;
//         }
//
//         if types[i].as_str() != "s6" {
//             continue;
//         }
//
//         // Sulfonyl S here has 4 neighbors.
//         let degree = adj[i].len();
//         if degree != 4 {
//             continue;
//         }
//
//         // Count "double-bond-like" oxygens: O with only one neighbor (the S).
//         let double_like_o = adj[i]
//             .iter()
//             .filter(|&&j| atoms[j].element == Element::Oxygen && adj[j].len() == 1)
//             .count();
//
//         // If S has at least two such O neighbors, treat it as sulfonyl S (`sy`).
//         if double_like_o >= 2 {
//             types[i] = "sy".to_string();
//         }
//     }
// }
//
// fn postprocess_ns_from_env(atoms: &[AtomGeneric], bonds: &[BondGeneric], types: &mut [String]) {
//     let adj = match build_adjacency_list(atoms, bonds) {
//         Ok(a) => a,
//         Err(_) => return,
//     };
//
//     for (i, atom) in atoms.iter().enumerate() {
//         if atom.element != Element::Nitrogen {
//             continue;
//         }
//
//         // Only refine Ns currently typed as generic `n7`.
//         if types[i].as_str() != "n7" {
//             continue;
//         }
//
//         let neighbors = &adj[i];
//
//         // ns expects 3 neighbors total.
//         if neighbors.len() != 3 {
//             continue;
//         }
//
//         // Exactly one hydrogen neighbor.
//         let num_h = neighbors
//             .iter()
//             .filter(|&&j| atoms[j].element == Element::Hydrogen)
//             .count();
//
//         if num_h != 1 {
//             continue;
//         }
//
//         let mut has_aromatic_c = false;
//         let mut has_carbonyl_c = false;
//
//         for &j in neighbors {
//             if atoms[j].element != Element::Carbon {
//                 continue;
//             }
//
//             match types[j].as_str() {
//                 "ca" | "cp" => has_aromatic_c = true,
//                 "c" => has_carbonyl_c = true,
//                 _ => {}
//             }
//         }
//
//         if has_aromatic_c && has_carbonyl_c {
//             types[i] = "ns".to_string();
//         }
//     }
// }

// /// This converts SP2 carbons to the correct types after the main algorithm infers
// /// types from DEF files. E.g. "cc", "cd", "c" etc.
// /// Adjust field names / bond-order enum to match your BondGeneric definition.
// fn postprocess_sp2_carbons(atoms: &[AtomGeneric], bonds: &[BondGeneric], types: &mut [String]) {
//     // Neighbour list with bond orders.
//     let mut nb: Vec<Vec<(usize, BondType)>> = vec![Vec::new(); atoms.len()];
//
//     for bond in bonds {
//         let i = bond.atom_0_sn as usize - 1;
//         let j = bond.atom_1_sn as usize - 1;
//         let order = bond.bond_type;
//
//         nb[i].push((j, order));
//         nb[j].push((i, order));
//     }
//
//     // Pass 1: identify carbonyl carbons: C=O / C=N / C=S / C=P.
//     // Only refine generic sp2 carbons (`c2`).
//     for i in 0..atoms.len() {
//         if atoms[i].element != Element::Carbon {
//             continue;
//         }
//
//         if types[i].as_str() != "c2" {
//             continue;
//         }
//
//         let mut double_to_hetero = false;
//
//         for &(j, order) in &nb[i] {
//             if matches!(order, BondType::Double) {
//                 match atoms[j].element {
//                     Element::Oxygen | Element::Nitrogen | Element::Sulfur | Element::Phosphorus => {
//                         double_to_hetero = true;
//                         break;
//                     }
//                     _ => {}
//                 }
//             }
//         }
//
//         if double_to_hetero {
//             types[i] = "c".to_string();
//         }
//     }
//
//     // Pass 2: cc / cd for remaining sp² carbons (still `c2`).
//     for i in 0..atoms.len() {
//         if atoms[i].element != Element::Carbon {
//             continue;
//         }
//
//         // Only refine generic sp² (`c2`); keep `c3` as truly sp³.
//         if types[i].as_str() != "c2" {
//             continue;
//         }
//
//         let mut double_to_carbon = 0u8;
//         let mut single_to_aromatic = false;
//         let mut single_to_carbonyl = false;
//
//         for &(j, order) in &nb[i] {
//             match order {
//                 BondType::Double if atoms[j].element == Element::Carbon => {
//                     double_to_carbon += 1;
//                 }
//                 BondType::Single => {
//                     if types[j] == "c" {
//                         single_to_carbonyl = true;
//                     }
//                     if types[j] == "ca" || types[j] == "cp" {
//                         single_to_aromatic = true;
//                     }
//                 }
//                 _ => {}
//             }
//         }
//
//         if single_to_carbonyl {
//             // sp² C directly attached to a carbonyl carbon → cd
//             types[i] = "cd".to_string();
//         } else if double_to_carbon > 0 || single_to_aromatic {
//             // Conjugated sp² C (C=C or vinyl attached to aromatic) → cc
//             types[i] = "cc".to_string();
//         }
//     }
// }

use bio_files::{AtomGeneric, BondGeneric, BondType};
use na_seq::Element;

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
        if atom.element != Element::Nitrogen {
            continue;
        }

        if types[i].as_str() != "ne" {
            continue;
        }

        let mut o_neighbors = 0u8;

        for &j in &adj[i] {
            if atoms[j].element == Element::Oxygen {
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
        if atom.element != Element::Phosphorus {
            continue;
        }

        let degree = adj[i].len();
        if degree < 4 {
            continue;
        }

        let o_neighbors = adj[i]
            .iter()
            .filter(|&&j| atoms[j].element == Element::Oxygen)
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

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Element::Nitrogen {
            continue;
        }

        if types[i].as_str() != "nu" {
            continue;
        }

        let neighbors = &adj[i];
        let degree = neighbors.len();

        // target: neutral 3-coordinate N with at least one H
        if degree != 3 {
            continue;
        }

        let h_count = neighbors
            .iter()
            .filter(|&&j| atoms[j].element == Element::Hydrogen)
            .count();

        if h_count == 0 {
            continue;
        }

        // skip any N with double/triple bonds
        let mut has_multiple_bond = false;
        for b in bonds {
            let a0 = b.atom_0_sn as usize - 1;
            let a1 = b.atom_1_sn as usize - 1;

            if a0 == i || a1 == i {
                if !matches!(b.bond_type, BondType::Single | BondType::Aromatic) {
                    has_multiple_bond = true;
                    break;
                }
            }
        }

        if has_multiple_bond {
            continue;
        }

        // looks like an n7-style N
        types[i] = "n7".to_owned();
    }
}

/// "cc" and "cd" types are SP2 Carbons (2 bonds each). cc-cc and cd-cd are single bonds;
/// cc=cd is a double bond.cc is a “generic conjugated sp² carbon”
/// cd is a more “more activated sp² carbon,” usually:
/// adjacently conjugated to a strong EWG (C=O, N, S, etc),
/// or in a bridged / strongly polarized conjugated system,
/// or in “β to carbonyl” / enone-like motifs, where that carbon is more electron-deficient.
pub(in crate::param_inference) fn postprocess_cc_cd(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let n = atoms.len();

    // Graph over indices where type is cc/cd and element is carbon.
    // Each edge is (neighbor_index, is_double_bond).
    let mut graph: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];

    for b in bonds {
        let i = b.atom_0_sn as usize - 1;
        let j = b.atom_1_sn as usize - 1;

        let ti = types.get(i).map(String::as_str);
        let tj = types.get(j).map(String::as_str);

        let both_group = matches!(ti, Some("cc") | Some("cd"))
            && matches!(tj, Some("cc") | Some("cd"))
            && atoms[i].element == Element::Carbon
            && atoms[j].element == Element::Carbon;

        if !both_group {
            continue;
        }

        // Only treat explicit single/double bonds; ignore aromatic etc.
        let is_double = match b.bond_type {
            BondType::Double => true,
            BondType::Single => false,
            _ => continue,
        };

        graph[i].push((j, is_double));
        graph[j].push((i, is_double));
    }

    // color[i] = None (unvisited) or Some(false = cc, true = cd).
    let mut color: Vec<Option<bool>> = vec![None; n];

    for start in 0..n {
        if graph[start].is_empty() {
            continue;
        }
        if !matches!(types[start].as_str(), "cc" | "cd") {
            continue;
        }
        if color[start].is_some() {
            continue;
        }

        color[start] = Some(false); // root as cc
        let mut stack = vec![start];

        while let Some(u) = stack.pop() {
            let cu = color[u].unwrap();

            for &(v, is_double) in &graph[u] {
                let expected = if is_double { !cu } else { cu };

                match color[v] {
                    None => {
                        color[v] = Some(expected);
                        stack.push(v);
                    }
                    Some(cv) => {
                        // If inconsistent (odd cycle), leave the existing color.
                        if cv != expected {
                            // You could log or debug-print here if desired.
                        }
                    }
                }
            }
        }
    }

    for i in 0..n {
        if let Some(c) = color[i] {
            types[i] = if c { "cd".to_owned() } else { "cc".to_owned() };
        }
    }
}

pub(in crate::param_inference) fn postprocess_ring_n_types(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use Element::*;

    for (idx, atom) in atoms.iter().enumerate() {
        if atom.element != Nitrogen {
            continue;
        }

        let env = &env_all[idx];

        // --- n3 -> na: ring tertiary N with no H, conjugated to unsaturated / ring neighbors ---
        if types[idx] == "n3"
            && !env.ring_sizes.is_empty()
            && env.degree == 3
            && env.num_attached_h == 0
        {
            let mut neighbor_unsat = false;

            for &nb in &adj[idx] {
                let nb_env = &env_all[nb];

                if !nb_env.ring_sizes.is_empty()
                    || nb_env.num_double_bonds > 0
                    || nb_env.num_triple_bonds > 0
                {
                    neighbor_unsat = true;
                    break;
                }
            }

            if neighbor_unsat {
                types[idx] = "na".to_owned();
            }
        }

        // --- refine ring sp2 nitrogens: nc vs nd, starting from nc / n1 ---
        let ty = types[idx].as_str();
        if ty != "nc" && ty != "n1" {
            continue;
        }

        if env.ring_sizes.is_empty() || env.num_double_bonds == 0 {
            continue;
        }

        // Find the non-H atom we are double-bonded to
        let mut db_partner: Option<usize> = None;
        for b in bonds {
            let i = b.atom_0_sn as usize - 1;
            let j = b.atom_1_sn as usize - 1;

            if (i == idx || j == idx) && matches!(b.bond_type, BondType::Double) {
                let other = if i == idx { j } else { i };
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

        // Partner strongly hetero-substituted → nd
        if partner_hetero_neighbors >= 2 {
            types[idx] = "nd".to_owned();
        } else {
            types[idx] = "nc".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_carbonyl_c(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use Element::Carbon;

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
    use na_seq::Element::{Nitrogen, Oxygen, Sulfur};

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
    _bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};

    for idx in 0..atoms.len() {
        if atoms[idx].element != Nitrogen {
            continue;
        }

        // Only refine generic sp2-like nitrogens
        if types[idx].as_str() != "n2" {
            continue;
        }

        let env = &env_all[idx];

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

        for &nb in &adj[idx] {
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
            types[idx] = "na".to_owned();
        }
    }
}

pub fn postprocess_nb_aromatic(
    atoms: &[AtomGeneric],
    _bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use na_seq::Element::Nitrogen;

    for idx in 0..atoms.len() {
        if atoms[idx].element != Nitrogen {
            continue;
        }

        let env = &env_all[idx];

        // Only consider N in 5- or 6-membered rings
        if !env.ring_sizes.iter().any(|&s| s == 5 || s == 6) {
            continue;
        }

        // Candidate starting types that can become nb
        let ty = types[idx].as_str();
        if !matches!(ty, "n2" | "nu" | "n7") {
            continue;
        }

        let neighbors = &adj[idx];

        let mut ring_heavy_neighbors = 0u8;
        let mut conjugated_neighbors = 0u8;

        for &nb in neighbors {
            if atoms[nb].element == Element::Hydrogen {
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

        types[idx] = "nb".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_nb_to_na_ring_with_h(
    atoms: &[AtomGeneric],
    _bonds: &[BondGeneric],
    env_all: &[AtomEnvData],
    types: &mut [String],
) {
    use na_seq::Element::Nitrogen;

    for idx in 0..atoms.len() {
        if atoms[idx].element != Nitrogen {
            continue;
        }

        if types[idx].as_str() != "nb" {
            continue;
        }

        let env = &env_all[idx];

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

        types[idx] = "na".to_owned();
    }
}
