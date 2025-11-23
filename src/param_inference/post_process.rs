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

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Nitrogen {
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

    let mut graph: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];

    for b in bonds {
        let i = b.atom_0_sn as usize - 1;
        let j = b.atom_1_sn as usize - 1;

        let ti = types.get(i).map(String::as_str);
        let tj = types.get(j).map(String::as_str);

        let both_group = matches!(ti, Some("cc") | Some("cd"))
            && matches!(tj, Some("cc") | Some("cd"))
            && atoms[i].element == Carbon
            && atoms[j].element == Carbon;

        if !both_group {
            continue;
        }

        let is_double = match b.bond_type {
            BondType::Double => true,
            BondType::Single => false,
            _ => continue,
        };

        graph[i].push((j, is_double));
        graph[j].push((i, is_double));
    }

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

        // Seed from original GAFF label when available
        // let root_is_cd = matches!(atoms[start].force_field_type.as_deref(), Some("cd"));
        // color[start] = Some(root_is_cd);
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
                        if cv != expected {
                            // inconsistent; keep existing color
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
    for i in 0..atoms.len() {
        // Only consider atoms currently labelled n7
        if types[i].as_str() != "n7" {
            continue;
        }

        let mut h_count = 0;
        let mut c_count = 0;
        let mut aromatic_c_count = 0;

        for &nbr_idx in &adj_list[i] {
            let nbr = &atoms[nbr_idx];

            match nbr.element {
                Hydrogen => h_count += 1,
                Carbon => {
                    c_count += 1;

                    if let Some(ff) = &nbr.force_field_type {
                        // Treat aromatic / generic sp2 / carbonyl carbons
                        if matches!(
                            ff.as_str(),
                            "ca" | "cc" | "cd" | "ce" | "cf" | "c6" | "c" | "c2"
                        ) {
                            aromatic_c_count += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        // Stricter rule:
        // - exactly one H
        // - at least two C neighbors
        // - at least one of those C is aromatic/sp2-like
        if h_count == 1 && c_count >= 2 && aromatic_c_count >= 1 {
            types[i] = "nu".to_owned();
        }
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
    adj_list: &[Vec<usize>], // kept for signature compatibility; now unused
    types: &mut [String],
) {
    use na_seq::Element::Sulfur;

    for (i, atom) in atoms.iter().enumerate() {
        if atom.element != Sulfur {
            continue;
        }

        // We only care about things we currently think are "sy"
        if types[i].as_str() != "sy" {
            continue;
        }
    }
}

pub(in crate::param_inference) fn postprocess_na_to_n3(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    types: &mut [String],
) {
    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }

        if types[i] != "na" {
            continue;
        }

        // Three single bonds → treat as an sp3 amine, not aromatic N
        if adj_list[i].len() == 3 {
            types[i] = "n3".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_p5_to_py(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    types: &mut [String],
) {
    for i in 0..types.len() {
        if types[i] != "p5" {
            continue;
        }

        let mut o_like = 0;
        let mut carbonyl_o = 0;

        for &nbr_idx in &adj_list[i] {
            let t = types[nbr_idx].as_str();
            if t.starts_with('o') {
                o_like += 1;
            }
            if t == "o" {
                carbonyl_o += 1;
            }
        }

        // P with at least two O-like neighbours and at least one `o` → py
        if o_like >= 2 && carbonyl_o >= 1 {
            types[i] = "py".to_owned();
        }
    }
}

//
// pub(in crate::param_inference) fn postprocess_restore_misc_from_training(
//     atoms: &[AtomGeneric],
//     types: &mut [String],
// ) {
//     // todo: Uhoh. I believe this whole function is a farce. And any post processing
//     // todo that uses atom FF type.
//     for i in 0..atoms.len() {
//         if let Some(ff) = atoms[i].force_field_type.as_deref() {
//             if ff == "ca"
//                 || ff == "c1"
//                 || ff == "c2"
//                 || ff == "cc"
//                 || ff == "cd"
//                 || ff == "cf"
//                 || ff == "cg"
//                 || ff == "ch"
//                 || ff == "cp"
//                 || ff == "cq"
//                 || ff == "cu"
//                 || ff == "cs"
//                 || ff == "cv"
//                 || ff == "cz"
//                 || ff == "h4"
//                 || ff == "ha"
//                 || ff == "hb"
//                 || ff == "hc"
//                 || ff == "hx"
//                 || ff == "n"
//                 || ff == "n1"
//                 || ff == "n2"
//                 || ff == "n3"
//                 || ff == "n4"
//                 || ff == "n5"
//                 || ff == "n6"
//                 || ff == "n7"
//                 || ff == "na"
//                 || ff == "nb"
//                 || ff == "nc"
//                 || ff == "nd"
//                 || ff == "ne"
//                 || ff == "nf"
//                 || ff == "nh"
//                 || ff == "nm"
//                 || ff == "np"
//                 || ff == "nq"
//                 || ff == "nr"
//                 || ff == "ns"
//                 || ff == "nv"
//                 || ff == "nq"
//                 || ff == "nt"
//                 || ff == "p3"
//                 || ff == "p4"
//                 || ff == "p5"
//                 || ff == "p6"
//                 || ff == "p7"
//                 || ff == "pe"
//                 || ff == "py"
//                 || ff == "s4"
//                 || ff == "sy"
//                 || ff == "oh"
//             {
//                 types[i] = ff.to_owned();
//             }
//         }
//     }
// }

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

// pub(in crate::param_inference) fn postprocess_n3_to_n_from_training(
//     atoms: &[AtomGeneric],
//     types: &mut [String],
// ) {
//     for (i, atom) in atoms.iter().enumerate() {
//         if atom.element != Nitrogen {
//             continue;
//         }
//
//         let Some(orig_ff) = atom.force_field_type.as_deref() else {
//             continue;
//         };
//
//         if orig_ff == "n" && types[i].as_str() == "n3" {
//             types[i] = "n".to_owned();
//         }
//     }
// }
