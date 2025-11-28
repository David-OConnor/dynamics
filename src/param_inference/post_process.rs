//! Individual case handling that we are not able to infer from DEF alone,
//! or if it's simpler to implement here.
//!
//! Warning: this module is a mess
//!
//! todo: COmbine these, so you only loop through atoms once.

use bio_files::{
    AtomGeneric, BondGeneric,
    BondType::{self, *},
};
use na_seq::Element::*;
use std::collections::VecDeque;

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
                if !matches!(b.bond_type, Single | Aromatic) {
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
                    if matches!(b.bond_type, Double) {
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

            if (a0 == i || a1 == i) && matches!(b.bond_type, Double) {
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

        // Count *other* hetero neighbors (excluding ourselves) on the double-bond partner.
        // Only O/N count here; do NOT let thio-substitution (S) force nd.
        let mut partner_on_hetero = 0u8;

        // If this N is attached to an aromatic 6-member ring atom (fused aromatic),
        // keep it as nc even if the C=N partner is hetero-substituted.
        let mut fused_aryl_neighbor = false;
        for &nb in &adj[i] {
            if atoms[nb].element == Hydrogen {
                continue;
            }
            let nb_env = &env_all[nb];
            if nb_env.is_aromatic && nb_env.ring_sizes.iter().any(|&s| s == 6) {
                fused_aryl_neighbor = true;
                break;
            }
        }

        // Count *other* hetero neighbors (excluding ourselves) on the double-bond partner.
        // Only O/N count here; do NOT let thio-substitution (S) force nd.
        let mut partner_hetero = 0u8;
        for &nb in &adj[p] {
            if nb == i {
                continue;
            }
            match atoms[nb].element {
                Oxygen | Nitrogen => partner_hetero = partner_hetero.saturating_add(1),
                _ => {}
            }
        }

        if partner_hetero >= 1 && !fused_aryl_neighbor {
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
                    if matches!(b.bond_type, Single) {
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

            if n_neigh >= 3 {
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

        if env.degree != 3 {
            continue;
        }

        let mut all_aliphatic = true;

        for &nb in &adj_list[i] {
            let nb_atom = &atoms[nb];

            match nb_atom.element {
                Hydrogen => {}
                Carbon => {
                    let nb_env = &env_all[nb];

                    let nb_ty = types[nb].as_str();
                    let nb_is_sp2_like = matches!(
                        nb_ty,
                        "c2" | "c" | "ca" | "c6" | "c5" | "cc" | "cd" | "ce" | "cf" | "cg"
                    );

                    if nb_is_sp2_like
                        || nb_env.num_double_bonds > 0
                        || nb_env.num_triple_bonds > 0
                        || nb_env.is_aromatic
                    {
                        all_aliphatic = false;
                        break;
                    }
                }
                _ => {
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
pub(in crate::param_inference) fn postprocess_sy_to_s6(
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

        if types[i].as_str() != "sy" {
            continue;
        }

        let o_neighbors = adj[i]
            .iter()
            .filter(|&&j| atoms[j].element == Oxygen)
            .count();

        if o_neighbors >= 3 {
            types[i] = "s6".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_sy_to_s6_if_nonaryl_sulfonamide(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen, Oxygen, Sulfur};

    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    let off = bond_offset(atoms.len(), bonds);

    for s in 0..atoms.len() {
        if atoms[s].element != Sulfur || types[s].as_str() != "sy" {
            continue;
        }

        let nbrs = &adj[s];
        if nbrs.len() != 4 {
            continue;
        }

        let mut o_double = 0u8;
        let mut non_o: Vec<usize> = Vec::with_capacity(2);

        for &j in nbrs {
            if atoms[j].element == Oxygen {
                if bond_ty(s, j, bonds, off) == Some(Double) {
                    o_double = o_double.saturating_add(1);
                }
            } else {
                non_o.push(j);
            }
        }

        if o_double != 2 || non_o.len() != 2 {
            continue;
        }

        let (n_idx, c_idx) = match (atoms[non_o[0]].element, atoms[non_o[1]].element) {
            (Nitrogen, Carbon) => (non_o[0], non_o[1]),
            (Carbon, Nitrogen) => (non_o[1], non_o[0]),
            _ => continue,
        };

        let mut c_is_aromatic = false;
        for b in bonds {
            if !matches!(b.bond_type, Aromatic) {
                continue;
            }
            let a0 = (b.atom_0_sn as usize).saturating_sub(off);
            let a1 = (b.atom_1_sn as usize).saturating_sub(off);
            if a0 == c_idx || a1 == c_idx {
                c_is_aromatic = true;
                break;
            }
        }

        if c_is_aromatic {
            continue;
        }

        types[s] = "s6".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_sy_to_s6_if_nonaryl_sulfonyl(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use BondType::Double;
    use na_seq::Element::{Carbon, Nitrogen, Oxygen, Sulfur};

    let off = bond_offset(atoms.len(), bonds);

    let mut adj = vec![Vec::new(); atoms.len()];
    for b in bonds {
        let i0 = (b.atom_0_sn as usize).saturating_sub(off);
        let i1 = (b.atom_1_sn as usize).saturating_sub(off);
        if i0 < atoms.len() && i1 < atoms.len() {
            adj[i0].push(i1);
            adj[i1].push(i0);
        }
    }

    fn is_aromatic_atom(i: usize, adj: &[Vec<usize>], bonds: &[BondGeneric], off: usize) -> bool {
        adj[i]
            .iter()
            .any(|&j| bond_ty(i, j, bonds, off) == Some(BondType::Aromatic))
    }

    for s in 0..atoms.len() {
        if atoms[s].element != Sulfur || types[s].as_str() != "sy" {
            continue;
        }
        if adj[s].len() != 4 {
            continue;
        }

        let mut o_double = 0u8;
        let mut others = Vec::with_capacity(2);

        for &j in &adj[s] {
            if atoms[j].element == Oxygen {
                if bond_ty(s, j, bonds, off) == Some(Double) {
                    o_double = o_double.saturating_add(1);
                }
            } else {
                others.push(j);
            }
        }

        if o_double != 2 || others.len() != 2 {
            continue;
        }

        let e0 = atoms[others[0]].element;
        let e1 = atoms[others[1]].element;

        if !matches!(
            (e0, e1),
            (Carbon, Carbon) | (Carbon, Nitrogen) | (Nitrogen, Carbon)
        ) {
            continue;
        }

        let mut any_aryl_c = false;
        for &j in &others {
            if atoms[j].element == Carbon && is_aromatic_atom(j, &adj, bonds, off) {
                any_aryl_c = true;
                break;
            }
        }
        if any_aryl_c {
            continue;
        }

        types[s] = "s6".to_owned();
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

        if types[i].as_str() != "s6" {
            continue;
        }

        let nbrs = &adj[i];
        if nbrs.len() != 4 {
            continue;
        }

        let mut o_double = 0u8;
        let mut non_o: Vec<usize> = Vec::with_capacity(2);

        for &j in nbrs {
            match atoms[j].element {
                Oxygen => {
                    let mut is_double = false;

                    for b in bonds {
                        let a0 = b.atom_0_sn as usize - 1;
                        let a1 = b.atom_1_sn as usize - 1;

                        if (a0 == i && a1 == j) || (a0 == j && a1 == i) {
                            if matches!(b.bond_type, Double) {
                                is_double = true;
                            }
                            break;
                        }
                    }

                    if is_double {
                        o_double = o_double.saturating_add(1);
                    }
                }
                _ => non_o.push(j),
            }
        }

        if o_double != 2 || non_o.len() != 2 {
            continue;
        }

        let a = non_o[0];
        let b = non_o[1];

        // sulfone: S(=O)2 with two carbon neighbours
        if atoms[a].element == Carbon && atoms[b].element == Carbon {
            types[i] = "sy".to_owned();
            continue;
        }

        // sulfonamide-like: S(=O)2 - N7 - aryl
        let (n_idx, c_idx) = match (atoms[a].element, atoms[b].element) {
            (Nitrogen, Carbon) => (a, b),
            (Carbon, Nitrogen) => (b, a),
            _ => continue,
        };

        let n_ty = types[n_idx].as_str();
        let c_ty = types[c_idx].as_str();
        let c_is_sp2_aromatic = matches!(c_ty, "ca" | "cc" | "cd" | "ce" | "cf" | "cg");

        if n_ty == "n7" && c_is_sp2_aromatic {
            types[i] = "sy".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_s6_to_sy_if_attached_to_nh2_only(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    use BondType::{Double, Single};
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen, Oxygen, Sulfur};

    let off = bond_offset(atoms.len(), bonds);

    for s in 0..atoms.len() {
        if atoms[s].element != Sulfur || types[s].as_str() != "s6" {
            continue;
        }

        let mut o_double = 0usize;
        let mut has_c_single = false;
        let mut nh2_neighbor: Option<usize> = None;

        for &j in &adj[s] {
            match (atoms[j].element, bond_ty(s, j, bonds, off)) {
                (Oxygen, Some(Double)) => o_double += 1,
                (Carbon, Some(Single)) => has_c_single = true,
                (Nitrogen, Some(Single)) => nh2_neighbor = Some(j),
                _ => {}
            }
        }

        if o_double != 2 || !has_c_single {
            continue;
        }

        let Some(n) = nh2_neighbor else {
            continue;
        };

        let h_deg = adj[n]
            .iter()
            .filter(|&&k| atoms[k].element == Hydrogen)
            .count();
        let heavy_deg = adj[n].len() - h_deg;

        if h_deg == 2 && heavy_deg == 1 {
            types[s] = "sy".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_s6_to_sy_if_primary_sulfonamide(
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    for s in 0..atoms.len() {
        if atoms[s].element != Sulfur || types[s].as_str() != "s6" {
            continue;
        }

        let mut o_neighbors = 0usize;
        let mut c_neighbors = 0usize;
        let mut n_nh2 = false;

        for &j in &adj[s] {
            match atoms[j].element {
                Oxygen => o_neighbors += 1,
                Carbon => c_neighbors += 1,
                Nitrogen => {
                    let h_deg = adj[j]
                        .iter()
                        .filter(|&&k| atoms[k].element == Hydrogen)
                        .count();
                    let heavy_deg = adj[j].len() - h_deg;

                    if h_deg == 2 && heavy_deg == 1 {
                        n_nh2 = true;
                    }
                }
                _ => {}
            }
        }

        if o_neighbors == 2 && c_neighbors == 1 && n_nh2 {
            types[s] = "sy".to_owned();
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
    use na_seq::Element::{Carbon, Nitrogen};

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

        // Keep nv if it's attached to a carbon that has any C=N double bond (amidine/guanidine-like).
        let mut attached_to_c_n_double = false;
        for &c_idx in &adj[n_idx] {
            if atoms[c_idx].element != Carbon {
                continue;
            }

            for b in bonds {
                if !matches!(b.bond_type, Double) {
                    continue;
                }

                let a0 = b.atom_0_sn as usize - 1;
                let a1 = b.atom_1_sn as usize - 1;

                if a0 == c_idx && atoms[a1].element == Nitrogen {
                    attached_to_c_n_double = true;
                    break;
                }
                if a1 == c_idx && atoms[a0].element == Nitrogen {
                    attached_to_c_n_double = true;
                    break;
                }
            }

            if attached_to_c_n_double {
                break;
            }
        }

        if attached_to_c_n_double {
            continue; // valid nv, don't demote
        }

        types[n_idx] = "n8".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_cz_to_c2_guanidinium_mixed_n(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};

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
            if matches!(b.bond_type, Single) {
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
                if matches!(b.bond_type, Double) && types[n_j].as_str() == "n2" {
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

pub(in crate::param_inference) fn postprocess_nd_to_nc_ring_no_n_neighbor(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Hydrogen, Nitrogen};
    use std::collections::VecDeque;

    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    fn is_in_ring(idx: usize, adj: &[Vec<usize>]) -> bool {
        if adj[idx].len() < 2 {
            return false;
        }

        for &nei in &adj[idx] {
            let mut visited = vec![false; adj.len()];
            visited[idx] = true;

            let mut q = VecDeque::new();
            for &start in &adj[idx] {
                if start != nei {
                    visited[start] = true;
                    q.push_back(start);
                }
            }

            while let Some(cur) = q.pop_front() {
                if cur == nei {
                    return true;
                }
                for &nxt in &adj[cur] {
                    if !visited[nxt] {
                        visited[nxt] = true;
                        q.push_back(nxt);
                    }
                }
            }
        }

        false
    }

    fn heavy_degree(i: usize, atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> usize {
        adj[i]
            .iter()
            .filter(|&&j| atoms[j].element != Hydrogen)
            .count()
    }

    fn bond_order(a: usize, b: usize, bonds: &[BondGeneric]) -> Option<u8> {
        for bo in bonds {
            let i0 = (bo.atom_0_sn as usize).saturating_sub(1);
            let i1 = (bo.atom_1_sn as usize).saturating_sub(1);
            if (i0 == a && i1 == b) || (i0 == b && i1 == a) {
                return match bo.bond_type {
                    Single => Some(1),
                    Double => Some(2),
                    Triple => Some(3),
                    _ => None, // ar, am, etc.
                };
            }
        }
        None
    }

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }
        if types[i].as_str() != "nd" {
            continue;
        }
        if !is_in_ring(i, &adj) {
            continue;
        }

        let has_n_neighbor = adj[i].iter().any(|&j| atoms[j].element == Nitrogen);
        if has_n_neighbor {
            continue;
        }

        // Only demote if the nd-N is double-bonded to an *unsubstituted* ring carbon
        let mut dbl_to: Option<usize> = None;
        for &j in &adj[i] {
            if bond_order(i, j, bonds) == Some(2) {
                if dbl_to.is_some() {
                    dbl_to = None; // multiple double bonds: bail
                    break;
                }
                dbl_to = Some(j);
            }
        }

        let Some(j) = dbl_to else { continue };

        if atoms[j].element != na_seq::Element::Carbon {
            continue;
        }

        // If that carbon is substituted by a heavy atom (degree > 2), keep nd.
        if heavy_degree(j, atoms, &adj) != 2 {
            continue;
        }

        types[i] = "nc".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_cz_to_cd_if_has_explicit_multibond(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::Carbon;

    for i in 0..atoms.len() {
        if atoms[i].element != Carbon {
            continue;
        }
        if types[i].as_str() != "cz" {
            continue;
        }

        let mut has_aromatic = false;
        let mut has_multibond = false;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            if a0 != i && a1 != i {
                continue;
            }

            match b.bond_type {
                Aromatic => has_aromatic = true,
                Double | Triple => has_multibond = true,
                _ => {}
            }
        }

        if !has_aromatic && has_multibond {
            types[i] = "cd".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_n7_to_nu_if_exocyclic(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Hydrogen, Nitrogen};

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }
        if types[i].as_str() != "n7" {
            continue;
        }

        let mut has_aromatic_bond = false;
        let mut has_multibond = false;
        let mut heavy_neighbors = 0usize;
        let mut has_h = false;
        let mut attached_to_aromatic_atom = false;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            let j = if a0 == i {
                a1
            } else if a1 == i {
                a0
            } else {
                continue;
            };

            match b.bond_type {
                Aromatic => has_aromatic_bond = true,
                Double | Triple => has_multibond = true,
                _ => {}
            }

            if atoms[j].element == Hydrogen {
                has_h = true;
            } else {
                heavy_neighbors += 1;

                for b2 in bonds {
                    let b20 = (b2.atom_0_sn as usize).saturating_sub(1);
                    let b21 = (b2.atom_1_sn as usize).saturating_sub(1);
                    if (b20 == j || b21 == j) && b2.bond_type == Aromatic {
                        attached_to_aromatic_atom = true;
                        break;
                    }
                }
            }
        }

        if !has_aromatic_bond
            && attached_to_aromatic_atom
            && !has_multibond
            && heavy_neighbors == 2
            && has_h
        {
            types[i] = "nu".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_n3_to_n_if_attached_to_acyl_carbon(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen, Oxygen};

    fn is_carbonyl_like_carbon(j: usize, atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> bool {
        if atoms[j].element != Carbon {
            return false;
        }

        let mut o_neighbors = 0usize;
        let mut has_c_o_double = false;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            let k = if a0 == j {
                a1
            } else if a1 == j {
                a0
            } else {
                continue;
            };

            if atoms[k].element == Oxygen {
                o_neighbors += 1;
                if b.bond_type == Double {
                    has_c_o_double = true;
                }
            }
        }

        has_c_o_double || o_neighbors >= 2
    }

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen {
            continue;
        }
        if types[i].as_str() != "n3" {
            continue;
        }

        let mut attached_to_acyl = false;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            let j = if a0 == i {
                a1
            } else if a1 == i {
                a0
            } else {
                continue;
            };

            if is_carbonyl_like_carbon(j, atoms, bonds) {
                attached_to_acyl = true;
                break;
            }
        }

        if attached_to_acyl {
            types[i] = "n".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_nd_to_nc_if_double_to_cd(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    postprocess_nd_to_nc_only_for_c_s_motifs(atoms, bonds, types);
}

pub(in crate::param_inference) fn postprocess_nd_to_nc_only_for_c_s_motifs(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen, Sulfur};

    for i in 0..atoms.len() {
        if atoms[i].element != Nitrogen || types[i].as_str() != "nd" {
            continue;
        }

        let mut dbl_c: Option<usize> = None;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            let j = if a0 == i {
                a1
            } else if a1 == i {
                a0
            } else {
                continue;
            };

            if b.bond_type == Double && atoms[j].element == Carbon {
                dbl_c = Some(j);
                break;
            }
        }

        let Some(cj) = dbl_c else { continue };
        if types[cj].as_str() != "cd" {
            continue;
        }

        let mut c_has_s_single_or_ar = false;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            let k = if a0 == cj {
                a1
            } else if a1 == cj {
                a0
            } else {
                continue;
            };

            if (b.bond_type == Single || b.bond_type == Aromatic) && atoms[k].element == Sulfur {
                c_has_s_single_or_ar = true;
                break;
            }
        }

        if c_has_s_single_or_ar {
            types[i] = "nc".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_n3_to_nh_if_conjugated(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    let off = bond_offset(atoms.len(), bonds);

    fn is_in_ring(idx: usize, adj: &[Vec<usize>]) -> bool {
        if adj[idx].len() < 2 {
            return false;
        }

        use std::collections::VecDeque;

        for &nei in &adj[idx] {
            let mut visited = vec![false; adj.len()];
            visited[idx] = true;

            let mut q = VecDeque::new();
            for &start in &adj[idx] {
                if start != nei {
                    visited[start] = true;
                    q.push_back(start);
                }
            }

            while let Some(cur) = q.pop_front() {
                if cur == nei {
                    return true;
                }
                for &nxt in &adj[cur] {
                    if !visited[nxt] {
                        visited[nxt] = true;
                        q.push_back(nxt);
                    }
                }
            }
        }

        false
    }

    fn path_exists_excluding(
        blocked: usize,
        start: usize,
        goal: usize,
        adj: &[Vec<usize>],
    ) -> bool {
        if start == goal {
            return true;
        }

        use std::collections::VecDeque;

        let mut visited = vec![false; adj.len()];
        visited[blocked] = true;
        visited[start] = true;

        let mut q = VecDeque::new();
        q.push_back(start);

        while let Some(cur) = q.pop_front() {
            for &nxt in &adj[cur] {
                if visited[nxt] {
                    continue;
                }
                if nxt == goal {
                    return true;
                }
                visited[nxt] = true;
                q.push_back(nxt);
            }
        }

        false
    }

    fn ring_neighbors(n: usize, atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> Vec<usize> {
        let heavy: Vec<usize> = adj[n]
            .iter()
            .copied()
            .filter(|&j| atoms[j].element != na_seq::Element::Hydrogen)
            .collect();

        let mut in_ring_edge = vec![false; heavy.len()];
        for i in 0..heavy.len() {
            for j in (i + 1)..heavy.len() {
                if path_exists_excluding(n, heavy[i], heavy[j], adj) {
                    in_ring_edge[i] = true;
                    in_ring_edge[j] = true;
                }
            }
        }

        let mut r = Vec::new();
        for (k, &h) in heavy.iter().enumerate() {
            if in_ring_edge[k] {
                r.push(h);
            }
        }
        r
    }

    fn has_internal_unsat_toward(
        blocked_n: usize,
        a: usize,
        toward: usize,
        bonds: &[BondGeneric],
        off: usize,
        adj: &[Vec<usize>],
    ) -> bool {
        for bo in bonds {
            if !matches!(bo.bond_type, Double | Triple | Aromatic) {
                continue;
            }

            let a0 = (bo.atom_0_sn as usize).saturating_sub(off);
            let a1 = (bo.atom_1_sn as usize).saturating_sub(off);

            let other = if a0 == a {
                a1
            } else if a1 == a {
                a0
            } else {
                continue;
            };

            if other == blocked_n {
                continue;
            }

            if path_exists_excluding(blocked_n, other, toward, adj) {
                return true;
            }
        }
        false
    }

    fn looks_aromatic_like_n(
        n: usize,
        atoms: &[AtomGeneric],
        bonds: &[BondGeneric],
        adj: &[Vec<usize>],
        off: usize,
    ) -> bool {
        if !is_in_ring(n, adj) {
            return false;
        }

        let rn = ring_neighbors(n, atoms, adj);
        if rn.len() < 2 {
            return false;
        }

        let mut hits = 0usize;
        for &a in &rn {
            let mut ok = false;
            for &b in &rn {
                if a == b {
                    continue;
                }
                if has_internal_unsat_toward(n, a, b, bonds, off, adj) {
                    ok = true;
                    break;
                }
            }
            if ok {
                hits += 1;
            }
        }

        hits >= 2
    }

    fn carbon_has_double_to_hetero(
        c: usize,
        bonds: &[BondGeneric],
        atoms: &[AtomGeneric],
        adj: &[Vec<usize>],
        off: usize,
    ) -> bool {
        for bo in bonds {
            if !matches!(bo.bond_type, Double | Triple) {
                continue;
            }

            let a0 = (bo.atom_0_sn as usize).saturating_sub(off);
            let a1 = (bo.atom_1_sn as usize).saturating_sub(off);

            let other = if a0 == c {
                a1
            } else if a1 == c {
                a0
            } else {
                continue;
            };

            match atoms[other].element {
                Oxygen | Sulfur => return true,
                Nitrogen => {
                    if !is_in_ring(c, adj) {
                        return true;
                    }
                }
                Carbon | Hydrogen => {}
                _ => {
                    if !is_in_ring(c, adj) {
                        return true;
                    }
                }
            }
        }
        false
    }

    for n in 0..atoms.len() {
        if types[n].as_str() != "n3" {
            continue;
        }

        let heavy_deg = adj[n]
            .iter()
            .filter(|&&j| atoms[j].element != na_seq::Element::Hydrogen)
            .count();
        if heavy_deg != 3 {
            continue;
        }

        if looks_aromatic_like_n(n, atoms, bonds, adj, off) {
            types[n] = "na".to_owned();
            continue;
        }

        for &c in &adj[n] {
            if atoms[c].element != Carbon {
                continue;
            }

            if bond_ty(n, c, bonds, off) != Some(Single) {
                continue;
            }

            if carbon_has_double_to_hetero(c, bonds, atoms, adj, off) {
                types[n] = "nh".to_owned();
                break;
            }
        }
    }

    for n in 0..atoms.len() {
        if types[n].as_str() != "nh" {
            continue;
        }
        if looks_aromatic_like_n(n, atoms, bonds, adj, off) {
            types[n] = "na".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_tris_n_c_to_c2(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Nitrogen};

    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    fn is_in_ring(idx: usize, adj: &[Vec<usize>]) -> bool {
        use std::collections::VecDeque;

        if adj[idx].len() < 2 {
            return false;
        }

        for &nei in &adj[idx] {
            let mut visited = vec![false; adj.len()];
            visited[idx] = true;

            let mut q = VecDeque::new();
            for &start in &adj[idx] {
                if start != nei {
                    visited[start] = true;
                    q.push_back(start);
                }
            }

            while let Some(cur) = q.pop_front() {
                if cur == nei {
                    return true;
                }
                for &nxt in &adj[cur] {
                    if !visited[nxt] {
                        visited[nxt] = true;
                        q.push_back(nxt);
                    }
                }
            }
        }

        false
    }

    for c_idx in 0..atoms.len() {
        if atoms[c_idx].element != Carbon {
            continue;
        }

        if is_in_ring(c_idx, &adj) {
            continue;
        }

        let nbrs = &adj[c_idx];
        if nbrs.len() != 3 || !nbrs.iter().all(|&j| atoms[j].element == Nitrogen) {
            continue;
        }

        let mut has_double_to_n = false;
        for b in bonds {
            if !matches!(b.bond_type, Double) {
                continue;
            }

            let a0 = b.atom_0_sn as usize - 1;
            let a1 = b.atom_1_sn as usize - 1;

            if a0 == c_idx || a1 == c_idx {
                let other = if a0 == c_idx { a1 } else { a0 };
                if atoms[other].element == Nitrogen {
                    has_double_to_n = true;
                    break;
                }
            }
        }

        if has_double_to_n {
            types[c_idx] = "c2".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_cz_to_ca_if_ring_no_n_neighbors(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::{Carbon, Hydrogen, Nitrogen};
    use std::collections::VecDeque;

    let adj = match build_adjacency_list(atoms, bonds) {
        Ok(a) => a,
        Err(_) => return,
    };

    fn is_in_ring(idx: usize, adj: &[Vec<usize>]) -> bool {
        if adj[idx].len() < 2 {
            return false;
        }

        for &nei in &adj[idx] {
            let mut visited = vec![false; adj.len()];
            visited[idx] = true;

            let mut q = VecDeque::new();
            for &start in &adj[idx] {
                if start != nei {
                    visited[start] = true;
                    q.push_back(start);
                }
            }

            while let Some(cur) = q.pop_front() {
                if cur == nei {
                    return true;
                }
                for &nxt in &adj[cur] {
                    if !visited[nxt] {
                        visited[nxt] = true;
                        q.push_back(nxt);
                    }
                }
            }
        }

        false
    }

    for i in 0..atoms.len() {
        if atoms[i].element != Carbon {
            continue;
        }
        if types[i].as_str() != "cz" {
            continue;
        }

        let n_neighbors = adj[i]
            .iter()
            .filter(|&&j| atoms[j].element == Nitrogen)
            .count();

        if n_neighbors != 0 {
            continue;
        }

        if is_in_ring(i, &adj) {
            types[i] = "ca".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_cz_to_ca_if_has_aromatic_bond(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    use na_seq::Element::Carbon;

    for i in 0..atoms.len() {
        if atoms[i].element != Carbon {
            continue;
        }
        if types[i].as_str() != "cz" {
            continue;
        }

        let mut has_aromatic_bond = false;

        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(1);
            let a1 = (b.atom_1_sn as usize).saturating_sub(1);

            if a0 != i && a1 != i {
                continue;
            }

            if matches!(b.bond_type, Aromatic) {
                has_aromatic_bond = true;
                break;
            }
        }

        if has_aromatic_bond {
            types[i] = "ca".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_c2_to_cf_if_conjugated_to_carbonyl(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    use BondType::{Aromatic, Double, Single};
    use na_seq::Element::{Carbon, Oxygen};

    let off = bond_offset(atoms.len(), bonds);

    fn is_aromatic_atom(i: usize, adj: &[Vec<usize>], bonds: &[BondGeneric], off: usize) -> bool {
        adj[i]
            .iter()
            .any(|&j| bond_ty(i, j, bonds, off) == Some(Aromatic))
    }

    for c in 0..atoms.len() {
        if atoms[c].element != Carbon || types[c].as_str() != "c2" {
            continue;
        }

        // Find the sp2 partner: c = x (double) where x is carbon
        let mut dbl_c: Option<usize> = None;
        for &j in &adj[c] {
            if atoms[j].element == Carbon && bond_ty(c, j, bonds, off) == Some(Double) {
                dbl_c = Some(j);
                break;
            }
        }
        let Some(x) = dbl_c else {
            continue;
        };

        // x must be alpha to a carbonyl: x - k (single) and k is C(=O)
        let mut has_alpha_carbonyl = false;
        for &k in &adj[x] {
            if k == c {
                continue;
            }
            if bond_ty(x, k, bonds, off) == Some(Single) && is_carbonyl_c(k, atoms, adj, bonds, off)
            {
                has_alpha_carbonyl = true;
                break;
            }
        }
        if !has_alpha_carbonyl {
            continue;
        }

        // c should be attached to (or very near) an aromatic system (prevents overfiring on simple enones)
        let mut aryl_like = false;
        for &j in &adj[c] {
            if j == x {
                continue;
            }
            if atoms[j].element == Carbon {
                if is_aromatic_atom(j, adj, bonds, off) {
                    aryl_like = true;
                    break;
                }
                if adj[j].iter().any(|&k| is_aromatic_atom(k, adj, bonds, off)) {
                    aryl_like = true;
                    break;
                }
            }
        }
        if !aryl_like {
            continue;
        }

        types[c] = "cf".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_c2_to_ce_if_conjugated_to_carbonyl(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    let off = bond_offset(atoms.len(), bonds);

    fn has_any_aromatic_bond(
        i: usize,
        adj: &[Vec<usize>],
        bonds: &[BondGeneric],
        off: usize,
    ) -> bool {
        adj[i]
            .iter()
            .any(|&j| bond_ty(i, j, bonds, off) == Some(BondType::Aromatic))
    }

    for c in 0..atoms.len() {
        if types[c].as_str() != "c2" || atoms[c].element != Carbon {
            continue;
        }
        if has_any_aromatic_bond(c, adj, bonds, off) {
            continue;
        }

        let mut has_single_to_carbonyl_c = false;
        let mut has_cc_double = false;

        for &j in &adj[c] {
            if atoms[j].element == Hydrogen {
                continue;
            }
            let Some(bt) = bond_ty(c, j, bonds, off) else {
                continue;
            };

            if bt == BondType::Single && is_carbonyl_c(j, atoms, adj, bonds, off) {
                has_single_to_carbonyl_c = true;
            }
            if bt == BondType::Double && atoms[j].element == Carbon {
                has_cc_double = true;
            }
        }

        if has_single_to_carbonyl_c && has_cc_double {
            types[c] = "ce".to_owned();
        }
    }
}

fn is_carbonyl_c(
    i: usize,
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    bonds: &[BondGeneric],
    off: usize,
) -> bool {
    if atoms[i].element != Carbon {
        return false;
    }
    adj[i]
        .iter()
        .any(|&j| atoms[j].element == Oxygen && bond_ty(i, j, bonds, off) == Some(BondType::Double))
}

fn has_oxygen_neighbor(i: usize, atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> bool {
    adj[i].iter().any(|&j| atoms[j].element == Oxygen)
}

fn is_carboxyl_like_c(
    i: usize,
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    result: &[String],
) -> bool {
    if atoms[i].element != Carbon {
        return false;
    }

    let o_count = adj[i]
        .iter()
        .filter(|&&j| atoms[j].element == Oxygen)
        .count();
    if o_count < 2 {
        return false;
    }

    // Optional: keep this if you only want to consider already-carbonyl-typed carbons.
    result[i].as_str() == "c"
}
pub fn postprocess_h_to_hx_alpha_carbon(
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    result: &mut [String],
) {
    for ca in 0..atoms.len() {
        if atoms[ca].element != Carbon || result[ca].as_str() != "c3" {
            continue;
        }

        let mut has_nh3_like = false;
        let mut has_carboxyl_c = false;
        let mut has_other_c = false;

        for &nb in &adj[ca] {
            match atoms[nb].element {
                Nitrogen => {
                    let nh = adj[nb]
                        .iter()
                        .filter(|&&j| atoms[j].element == Hydrogen)
                        .count();
                    if nh >= 3 || result[nb].as_str() == "nz" {
                        has_nh3_like = true;
                    }
                }
                Carbon => {
                    if is_carboxyl_like_c(nb, atoms, adj, result)
                        || (result[nb].as_str() == "c" && has_oxygen_neighbor(nb, atoms, adj))
                    {
                        has_carboxyl_c = true;
                    } else {
                        has_other_c = true;
                    }
                }
                _ => {}
            }
        }

        if !(has_nh3_like && has_carboxyl_c && has_other_c) {
            continue;
        }

        for &h in &adj[ca] {
            if atoms[h].element == Hydrogen && result[h].as_str() != "hx" {
                result[h] = "hx".to_owned();
            }
        }
    }
}

pub(in crate::param_inference) fn postprocess_n7_to_n6_if_small_ring(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    let off = bond_offset(atoms.len(), bonds);

    fn smallest_cycle_len_through(idx: usize, adj: &[Vec<usize>]) -> Option<usize> {
        if adj[idx].len() < 2 {
            return None;
        }

        let mut best: Option<usize> = None;

        for (a_i, &a) in adj[idx].iter().enumerate() {
            for &b in &adj[idx][(a_i + 1)..] {
                let mut visited = vec![false; adj.len()];
                visited[idx] = true;

                let mut q = VecDeque::new();
                q.push_back((a, 0usize));
                visited[a] = true;

                while let Some((cur, d)) = q.pop_front() {
                    if cur == b {
                        let cycle_len = d + 2;
                        best = Some(best.map_or(cycle_len, |x| x.min(cycle_len)));
                        break;
                    }
                    for &nxt in &adj[cur] {
                        if !visited[nxt] {
                            visited[nxt] = true;
                            q.push_back((nxt, d + 1));
                        }
                    }
                }
            }
        }

        best
    }

    for n in 0..atoms.len() {
        if types[n].as_str() != "n7" {
            continue;
        }
        if atoms[n].element != Nitrogen {
            continue;
        }

        let h_deg = adj[n]
            .iter()
            .filter(|&&j| atoms[j].element == Hydrogen)
            .count();
        let heavy_deg = adj[n].len() - h_deg;

        if h_deg != 1 || heavy_deg != 2 {
            continue;
        }

        let Some(cycle_len) = smallest_cycle_len_through(n, adj) else {
            continue;
        };
        if cycle_len > 4 {
            continue;
        }

        let mut all_single = true;
        for &j in &adj[n] {
            if atoms[j].element == Hydrogen {
                continue;
            }
            if bond_ty(n, j, bonds, off) != Some(Single) {
                all_single = false;
                break;
            }
        }
        if !all_single {
            continue;
        }

        types[n] = "n6".to_owned();
    }
}

pub(in crate::param_inference) fn postprocess_ce_to_cf_if_amidine_like(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    adj: &[Vec<usize>],
    types: &mut [String],
) {
    fn bond_offset(atoms_len: usize, bonds: &[BondGeneric]) -> usize {
        let mut max_sn = 0usize;
        for b in bonds {
            max_sn = max_sn.max(b.atom_0_sn as usize);
            max_sn = max_sn.max(b.atom_1_sn as usize);
        }
        if max_sn == atoms_len { 1 } else { 0 }
    }

    let off = bond_offset(atoms.len(), bonds);

    fn bond_ty(a: usize, b: usize, bonds: &[BondGeneric], off: usize) -> Option<BondType> {
        for bo in bonds {
            let i0 = (bo.atom_0_sn as usize).saturating_sub(off);
            let i1 = (bo.atom_1_sn as usize).saturating_sub(off);
            if (i0 == a && i1 == b) || (i0 == b && i1 == a) {
                return Some(bo.bond_type);
            }
        }
        None
    }

    fn is_aromatic_atom(i: usize, adj: &[Vec<usize>], bonds: &[BondGeneric], off: usize) -> bool {
        adj[i]
            .iter()
            .any(|&j| bond_ty(i, j, bonds, off) == Some(BondType::Aromatic))
    }

    for c in 0..atoms.len() {
        if types[c].as_str() != "ce" {
            continue;
        }
        if atoms[c].element != Carbon {
            continue;
        }

        // Guard 1: never rewrite an actually-aromatic atom (prevents 010-style regressions)
        if is_aromatic_atom(c, adj, bonds, off) {
            continue;
        }

        let heavy: Vec<usize> = adj[c]
            .iter()
            .copied()
            .filter(|&j| atoms[j].element != Hydrogen)
            .collect();
        if heavy.len() != 3 {
            continue;
        }

        // Guard 2: amidine-like center must have exactly two N heavy neighbors
        let n_count = heavy
            .iter()
            .filter(|&&j| atoms[j].element == Nitrogen)
            .count();
        if n_count != 2 {
            continue;
        }

        let mut has_double_to_n = false;
        let mut has_single_to_n = false;
        let mut bonded_to_aromatic_carbon = false;

        let mut ok = true;

        for &j in &heavy {
            let bt = match bond_ty(c, j, bonds, off) {
                Some(bt) => bt,
                None => {
                    ok = false;
                    break;
                }
            };

            match (atoms[j].element, bt) {
                (Nitrogen, Double) => has_double_to_n = true,
                (Nitrogen, Single) => has_single_to_n = true,
                (Carbon, Single) => {
                    // Topology-based aromatic check (NOT type-based)
                    if is_aromatic_atom(j, adj, bonds, off) {
                        bonded_to_aromatic_carbon = true;
                    }
                }
                (_, Single) => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }

        if ok && has_double_to_n && has_single_to_n && !bonded_to_aromatic_carbon {
            types[c] = "cf".to_owned();
        }
    }
}

fn bond_ty(a: usize, b: usize, bonds: &[BondGeneric], off: usize) -> Option<BondType> {
    for bo in bonds {
        let i0 = (bo.atom_0_sn as usize).saturating_sub(off);
        let i1 = (bo.atom_1_sn as usize).saturating_sub(off);
        if (i0 == a && i1 == b) || (i0 == b && i1 == a) {
            return Some(bo.bond_type);
        }
    }
    None
}

fn bond_offset(atoms_len: usize, bonds: &[BondGeneric]) -> usize {
    let mut max_sn = 0usize;
    for b in bonds {
        max_sn = max_sn.max(b.atom_0_sn as usize);
        max_sn = max_sn.max(b.atom_1_sn as usize);
    }
    if max_sn == atoms_len { 1 } else { 0 }
}

pub(in crate::param_inference) fn postprocess_cc_to_ca_if_has_aromatic_bond(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let off = bond_offset(atoms.len(), bonds);

    for i in 0..atoms.len() {
        if atoms[i].element != Carbon {
            continue;
        }
        if types[i].as_str() != "cc" {
            continue;
        }

        let mut has_aromatic = false;
        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(off);
            let a1 = (b.atom_1_sn as usize).saturating_sub(off);

            if (a0 == i || a1 == i) && matches!(b.bond_type, Aromatic) {
                has_aromatic = true;
                break;
            }
        }

        if has_aromatic {
            types[i] = "ca".to_owned();
        }
    }
}

pub(in crate::param_inference) fn postprocess_cd_to_ca_if_has_aromatic_bond(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    types: &mut [String],
) {
    let off = bond_offset(atoms.len(), bonds);

    for i in 0..atoms.len() {
        if atoms[i].element != Carbon {
            continue;
        }
        if types[i].as_str() != "cd" {
            continue;
        }

        let mut has_aromatic = false;
        for b in bonds {
            let a0 = (b.atom_0_sn as usize).saturating_sub(off);
            let a1 = (b.atom_1_sn as usize).saturating_sub(off);

            if (a0 == i || a1 == i) && matches!(b.bond_type, BondType::Aromatic) {
                has_aromatic = true;
                break;
            }
        }

        if has_aromatic {
            types[i] = "ca".to_owned();
        }
    }
}
