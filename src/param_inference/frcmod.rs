//! This module computes per-molecule overrides to GAFF2; this is what Amber's Parmchk or Parmchk2
//! accomplishes. THis is generally Dihedrals and impropers, but can be any bonded parameter
//! this molecule contains, but is missing from Gaff2.dat.
//!
//! This searches Gaff2 for the closest match based on ff types, and returns it.
//! // todo: Possibly with a corrective factor?
//!
//! todo: Split this up A/R for organizatio.

// todo: QC this. You're still getting some incorrect dihedrals, and impropers.

use std::{collections::HashMap, f32::consts::PI, io, io::ErrorKind};

use bio_files::{
    AtomGeneric,
    md_params::{AngleBendingParams, BondStretchingParams, DihedralParams, ForceFieldParams},
};

use crate::param_inference::{
    frcmod_missing_params::MissingParams,
    parmchk_parse::{AtCor, Corr, ParmChk, load_atcor},
};

const WILDCARD_PENALTY: f32 = 50.0;

/// Identify all combinations of FF type in this set of atoms that requires bonded parameters.
/// For each, check if this parameter is set in our universal small organic molecule param set (Amber's GAFF2).
/// If not present, apply the closest universal param to it based on force field types.
/// todo: With correction factors?
pub fn assign_missing_params(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    gaff2: &ForceFieldParams,
) -> io::Result<ForceFieldParams> {
    let mut result = ForceFieldParams::default();

    let params_missing = MissingParams::new(atoms, adj_list, gaff2)?;

    // println!("\n\n Dihe missing: ");
    // for d in &params_missing.dihedral {
    //     println!("-Di: {:?}", d);
    // }

    // println!("\n\n Improper missing: ");
    // for d in &params_missing.improper {
    //     println!("-Imp: {:?}", d);
    // }

    let atcor = load_atcor()?;
    let parmchk = ParmChk::new()?;

    // println!("\n\n Atcor loaded:");
    // for v in &atcor {
    //     println!("- {v:?}");
    // }
    //
    // println!("\n\n Parm loaded:");
    // for v in &parmchk.parms {
    //     println!("parm: {v:?}");
    // }
    // println!("\n\n Parm Corr:");
    // for v in &parmchk.corr {
    //     println!("corr: {v:?}");
    // }
    // println!("\n\n Parm Aliases:");
    // for v in &parmchk.aliases {
    //     println!("Ali-: {v:?}");
    // }
    // println!("\n\n Equa to park:");
    // for v in &parmchk.equa_to_parm {
    //     println!("Eq-: {v:?}");
    // }

    // todo: We currently get wonky values from Equa to parm ,which doesn't seem to make sense,
    // todo and only afffect a few specific cases. Hard-code them here.
    // todo: If you keep this in, don't build it each time.

    // todo: This currently has data from both atcor and parmchk equa.
    let mut eq_map = HashMap::new();
    // PARMCHK eqa:
    eq_map.insert("cs".to_owned(), "c");
    eq_map.insert("o2".to_owned(), "o");
    eq_map.insert("nj".to_owned(), "n");
    eq_map.insert("sq".to_owned(), "ss");
    eq_map.insert("c6".to_owned(), "c3");
    // ATCOR:
    eq_map.insert("ns".to_owned(), "n");
    eq_map.insert("n7".to_owned(), "n3");
    eq_map.insert("n8".to_owned(), "n3");
    eq_map.insert("n9".to_owned(), "n3");

    // todo: Add these bonds and valence angles once dihedrals and impropers work.
    // --- bonds -------------------------------------------------------------
    for (t0, t1) in &params_missing.bond {
        let key = (t0.to_owned(), t1.to_owned());

        if let Some(params) = find_bond_alts(gaff2, &parmchk, &atcor, &eq_map, &key) {
            result.bond.insert(key, params);
        } else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Unable to find a bondparam for {t0} {t1}"),
            ));
        }
    }

    // --- Valence angles -------------------------------------------------------------
    for (t0, t1, t2) in &params_missing.angle {
        let key = (t0.to_owned(), t1.to_owned(), t2.to_owned());

        if let Some(params) = find_angle_alts(gaff2, &parmchk, &atcor, &eq_map, &key) {
            result.angle.insert(key, params);
        } else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Unable to find a valence angle param for {t0} {t1} {t2}"),
            ));
        }
    }

    // --- dihedrals ---------------------------------------------------------
    for (t0, t1, t2, t3) in &params_missing.dihedral {
        let key = (t0.to_owned(), t1.to_owned(), t2.to_owned(), t3.to_owned());

        if let Some(params) = find_dihedral_alts(gaff2, &parmchk, &atcor, &eq_map, &key, true) {
            result.dihedral.insert(key, params);
        } else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Unable to find a dihedral param for {t0} {t1} {t2} {t3}"),
            ));
        }
    }

    // --- impropers ------------------
    for (t0, t1, t2, t3) in &params_missing.improper {
        let key = (t0.to_owned(), t1.to_owned(), t2.to_owned(), t3.to_owned());

        if let Some(params) = find_dihedral_alts(gaff2, &parmchk, &atcor, &eq_map, &key, false) {
            result.improper.insert(key, params);
        }
        // We are allowed to be missing Impropers, unlike other parameters.
    }

    Ok(result)
}

// todo: DRY among bond, valence, dihedral logic, and within each.
fn find_bond_alts(
    gaff2: &ForceFieldParams,
    parmchk: &ParmChk,
    atcor: &AtCor,
    eq_map: &HashMap<String, &str>,
    key: &(String, String),
) -> Option<BondStretchingParams> {
    // Apply Atcor.
    // If a given param is in Atcor, use that to map to its canonical type. For now, we just
    // pick the first one; that seems to work for these like "n7"
    let key0 = match atcor.get(&key.0) {
        Some(v) => v.1[0].clone(),
        None => key.0.clone(),
    };
    let key1 = match atcor.get(&key.1) {
        Some(v) => v.1[0].clone(),
        None => key.1.clone(),
    };

    // Apply eq values from PARMCHK; this is our ad hoc approach, as the parsing isn't giving
    // us useful information here.
    let key0 = match eq_map.get(&key0) {
        Some(v) => *v,
        None => &key0,
    };
    let key1 = match eq_map.get(&key1) {
        Some(v) => *v,
        None => &key1,
    };

    let corr_0_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key0).collect();
    let corr_1_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key1).collect();

    // Now that we've changed the names, try to get the param without further substitutions.
    if let Some(param) = gaff2.get_bond(&(key0.to_owned(), key1.to_owned()), false) {
        return Some(param.clone());
    }

    // todo: Later, don't return immediately on a match; evaluate options, and compare or score.
    // First attempt: Change a single candidate:
    for c in &corr_0_candidates {
        if let Some(param) = gaff2.get_bond(&(c.other.to_owned(), key1.to_owned()), false) {
            return Some(param.clone());
        }
    }
    for c in &corr_1_candidates {
        if let Some(param) = gaff2.get_bond(&(key0.to_owned(), c.other.to_owned()), false) {
            return Some(param.to_owned());
        }
    }

    // Second attempt: Try swapping both items.
    for c0 in &corr_0_candidates {
        let c0 = match eq_map.get(&c0.other) {
            Some(v) => *v,
            None => &c0.other,
        };
        // todo: DRY again
        for c1 in &corr_1_candidates {
            let c1 = match eq_map.get(&c1.other) {
                Some(v) => *v,
                None => &c1.other,
            };

            if let Some(param) = gaff2.get_bond(&(c0.to_owned(), c1.to_owned()), false) {
                return Some(param.to_owned());
            }
            // try its equivalents. We currently use aliases and/or equa
        }
    }

    None
}

// todo: DRY among bond, valence, dihedral logic, and within each.
fn find_angle_alts(
    gaff2: &ForceFieldParams,
    parmchk: &ParmChk,
    atcor: &AtCor,
    eq_map: &HashMap<String, &str>,
    key: &(String, String, String),
) -> Option<AngleBendingParams> {
    // Apply Atcor.
    // If a given param is in Atcor, use that to map to its canonical type. For now, we just
    // pick the first one; that seems to work for these like "n7"
    let key0 = match atcor.get(&key.0) {
        Some(v) => v.1[0].clone(),
        None => key.0.clone(),
    };
    let key1 = match atcor.get(&key.1) {
        Some(v) => v.1[0].clone(),
        None => key.1.clone(),
    };
    let key2 = match atcor.get(&key.2) {
        Some(v) => v.1[0].clone(),
        None => key.2.clone(),
    };

    // Apply eq values from PARMCHK; this is our ad hoc approach, as the parsing isn't giving
    // us useful information here.
    let key0 = match eq_map.get(&key0) {
        Some(v) => *v,
        None => &key0,
    };
    let key1 = match eq_map.get(&key1) {
        Some(v) => *v,
        None => &key1,
    };
    let key2 = match eq_map.get(&key2) {
        Some(v) => *v,
        None => &key2,
    };

    let corr_0_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key0).collect();
    let corr_1_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key1).collect();
    let corr_2_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key2).collect();

    // Now that we've changed the names, try to get the param without further substitutions.
    if let Some(param) =
        gaff2.get_valence_angle(&(key0.to_owned(), key1.to_owned(), key2.to_owned()), false)
    {
        return Some(param.clone());
    }

    // todo: Later, don't return immediately on a match; evaluate options, and compare or score.
    // First attempt: Change a single candidate:
    for c in &corr_0_candidates {
        if let Some(param) = gaff2.get_valence_angle(
            &(c.other.to_owned(), key1.to_owned(), key2.to_owned()),
            false,
        ) {
            return Some(param.clone());
        }
    }
    for c in &corr_1_candidates {
        if let Some(param) = gaff2.get_valence_angle(
            &(key0.to_owned(), c.other.to_owned(), key2.to_owned()),
            false,
        ) {
            return Some(param.to_owned());
        }
    }
    for c in &corr_2_candidates {
        if let Some(param) = gaff2.get_valence_angle(
            &(key0.to_owned(), key1.to_owned(), c.other.to_owned()),
            false,
        ) {
            return Some(param.to_owned());
        }
    }

    // Second attempt: Try swapping both ends.
    for c0 in &corr_0_candidates {
        let c0 = match eq_map.get(&c0.other) {
            Some(v) => *v,
            None => &c0.other,
        };
        // todo: DRY again
        for c1 in &corr_1_candidates {
            let c1 = match eq_map.get(&c1.other) {
                Some(v) => *v,
                None => &c1.other,
            };

            if let Some(param) =
                gaff2.get_valence_angle(&(c0.to_owned(), key1.to_owned(), c1.to_owned()), false)
            {
                return Some(param.to_owned());
            }
            // try its equivalents. We currently use aliases and/or equa
        }
    }

    None
}

/// For proper and improper dihedrals.
// todo: DRY among bond, valence, dihedral logic, and within each.
fn find_dihedral_alts(
    gaff2: &ForceFieldParams,
    parmchk: &ParmChk,
    atcor: &AtCor,
    eq_map: &HashMap<String, &str>,
    key: &(String, String, String, String),
    proper: bool,
) -> Option<Vec<DihedralParams>> {
    // Apply Atcor.
    // If a given param is in Atcor, use that to map to its canonical type. For now, we just
    // pick the first one; that seems to work for these like "n7"
    // let key0 = match atcor.get(&key.0) {
    //     Some(v) => v.1[0].clone(),
    //     None => key.0.clone(),
    // };
    // let key1 = match atcor.get(&key.1) {
    //     Some(v) => v.1[0].clone(),
    //     None => key.1.clone(),
    // };
    // let key2 = match atcor.get(&key.2) {
    //     Some(v) => v.1[0].clone(),
    //     None => key.2.clone(),
    // };
    // let key3 = match atcor.get(&key.3) {
    //     Some(v) => v.1[0].clone(),
    //     None => key.3.clone(),
    // };
    let key0 = key.0.clone();
    let key1 = key.1.clone();
    let key2 = key.2.clone();
    let key3 = key.3.clone();

    // Apply eq values from PARMCHK; this is our ad hoc approach, as the parsing isn't giving
    // us useful information here.
    let key0 = match eq_map.get(&key0) {
        Some(v) => *v,
        None => &key0,
    };
    let key1 = match eq_map.get(&key1) {
        Some(v) => *v,
        None => &key1,
    };
    let key2 = match eq_map.get(&key2) {
        Some(v) => *v,
        None => &key2,
    };
    let key3 = match eq_map.get(&key3) {
        Some(v) => *v,
        None => &key3,
    };

    let corr_0_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key0).collect();
    let corr_1_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key1).collect();
    let corr_2_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key2).collect();
    let corr_3_candidates: Vec<_> = parmchk.corr.iter().filter(|c| c.base == key3).collect();

    // todo: You likely need some weighting system in this matching.

    // Now that we've changed the names, try to get the param without further substitutions.
    // First with no wildcards:
    if let Some(param) = gaff2.get_dihedral(
        &(
            key0.to_owned(),
            key1.to_owned(),
            key2.to_owned(),
            key3.to_owned(),
        ),
        proper,
        false,
    ) {
        return Some(param.clone());
    }

    // Compute best non-wildcard and best wildcard candidate, then pick the global best.
    let best_no_wc = dihedral_inner(
        key0,
        key1,
        key2,
        key3,
        &corr_0_candidates,
        &corr_1_candidates,
        &corr_2_candidates,
        &corr_3_candidates,
        gaff2,
        parmchk,
        eq_map,
        proper,
        false,
    );

    let best_wc = dihedral_inner(
        key0,
        key1,
        key2,
        key3,
        &corr_0_candidates,
        &corr_1_candidates,
        &corr_2_candidates,
        &corr_3_candidates,
        gaff2,
        parmchk,
        eq_map,
        proper,
        true,
    );

    // todo: Figure out if/how you use this. GEostd FRCMOD seem to use this, but where?
    let improper_default = DihedralParams {
        atom_types: (
            "X".to_string(),
            "X".to_string(),
            "X".to_string(),
            "X".to_string(),
        ),
        divider: 1,
        barrier_height: 1.1,
        phase: PI,
        periodicity: 2,
        comment: Some("Using the default value".to_owned()),
    };

    let best = match (best_no_wc, best_wc) {
        (Some(b), None) => Some(b),
        (None, Some(b)) => Some(b),
        (Some(b1), Some(b2)) => {
            // choose lower-penalty of the two
            if b1.0 <= b2.0 { Some(b1) } else { Some(b2) }
        }
        (None, None) => None,
    };

    if proper {
        // For proper dihedrals, *never* synthesize a default – just use GAFF2 matches.
        return best.map(|(_, params)| params);
    }

    // For impropers:

    if let Some((_, p)) = best {
        // We found a real GAFF2 / wildcard match → use it.
        return Some(p);
    }

    let key_for_default = (key.0.as_str(), key.1.as_str(), key.2.as_str(), key.3.as_str());

    // First: special-case carbonyl-like impropers (e.g. cd-n-c-o, n-ns-c-o)
    if is_carbonyl_improper_candidate(key_for_default) {
        let improper_carbonyl = DihedralParams {
            atom_types: ("X".into(), "X".into(), "c".into(), "o".into()),
            divider: 1,
            barrier_height: 10.5,
            phase: PI,
            periodicity: 2,
            comment: Some("Using general improper torsional angle  X- X- c- o".to_owned()),
        };
        return Some(vec![improper_carbonyl]);
    }

    // Otherwise: the generic 1.1 planar-ish default
    if is_default_improper_candidate(key_for_default) {
        let improper_default = DihedralParams {
            atom_types: ("X".into(), "X".into(), "X".into(), "X".into()),
            divider: 1,
            barrier_height: 1.1,
            phase: PI,
            periodicity: 2,
            comment: Some("Using the default value".to_owned()),
        };

        return Some(vec![improper_default]);
    }

    // Don't create an FRCMOD entry at all for random sp3/H junk.
    None
}

fn is_carbonyl_improper_candidate(key: (&str, &str, &str, &str)) -> bool {
    let (a, b, c, d) = key;
    let ts = [a, b, c, d];

    // Require at least one carbonyl carbon "c" and one carbonyl oxygen "o"
    let has_c = ts.iter().any(|t| *t == "c");
    let has_o = ts.iter().any(|t| *t == "o");

    if !(has_c && has_o) {
        return false;
    }

    // Reuse your existing "planar-ish" heuristics so we don't fire on nonsense.
    let heavy_count = ts.iter().filter(|t| !is_hydrogen_ff_type(t)).count();
    if heavy_count < 3 {
        return false;
    }

    let sp2_like_count = ts.iter().filter(|t| is_sp2_like_ff_type(t)).count();
    if sp2_like_count < 2 {
        return false;
    }

    true
}

/// Helper to abstract over wildcards, and apply a scoring system.
fn dihedral_inner(
    key0: &str,
    key1: &str,
    key2: &str,
    key3: &str,
    corr_0_candidates: &[&Corr],
    corr_1_candidates: &[&Corr],
    corr_2_candidates: &[&Corr],
    corr_3_candidates: &[&Corr],
    gaff2: &ForceFieldParams,
    parmchk: &ParmChk,
    eq_map: &HashMap<String, &str>,
    proper: bool,
    wildcard_allowed: bool,
) -> Option<(f32, Vec<DihedralParams>)> {
    let mut best: Option<(f32, Vec<DihedralParams>)> = None;

    let mut consider = |a: &str, b: &str, c: &str, d: &str| {
        if let Some(params) = gaff2.get_dihedral(
            &(a.to_owned(), b.to_owned(), c.to_owned(), d.to_owned()),
            proper,
            wildcard_allowed,
        ) {
            // Look at the GAFF2 pattern we actually matched, not the query.
            let (gaff_a, gaff_b, gaff_c, gaff_d) = &params[0].atom_types;

            let wildcard_pen = if proper {
                // Proper dihedrals: penalize wildcards mainly at ends.
                (gaff_a == "X") as u8 as f32 * WILDCARD_PENALTY
                    + (gaff_d == "X") as u8 as f32 * WILDCARD_PENALTY
            } else {
                // Impropers: patterns like X-X-ca-ha are *expected*.
                // Only penalize the fully generic X-X-X-X case.
                if gaff_a == "X" && gaff_b == "X" && gaff_c == "X" && gaff_d == "X" {
                    WILDCARD_PENALTY
                } else {
                    0.0
                }
            };

            let penalty = if proper {
                // same as before for proper dihedrals
                parmchk.dihe_outer_penalty(key0, a)
                    + parmchk.dihe_inner_penalty(key1, b)
                    + parmchk.dihe_inner_penalty(key2, c)
                    + parmchk.dihe_outer_penalty(key3, d)
                    + wildcard_pen
            } else {
                // impropers: central atom match matters most
                // (we assume key1 is the central site; that's how GAFF/Amber usually order impropers)
                parmchk.improper_penalty(key1, b) + wildcard_pen
            };

            match &mut best {
                None => best = Some((penalty, params.clone())),
                Some((best_p, best_params)) => {
                    if penalty < *best_p {
                        *best_p = penalty;
                        *best_params = params.clone();
                    }
                }
            }
        }
    };

    // If wildcards are allowed, let the raw key participate too, so
    // patterns like X-c3-c3-X can compete against substituted ones.
    if wildcard_allowed {
        consider(key0, key1, key2, key3);
    }

    // Change a single candidate:
    for c in corr_0_candidates {
        consider(&c.other, key1, key2, key3);
    }
    for c in corr_1_candidates {
        consider(key0, &c.other, key2, key3);
    }
    for c in corr_2_candidates {
        consider(key0, key1, &c.other, key3);
    }
    for c in corr_3_candidates {
        consider(key0, key1, key2, &c.other);
    }

    // (Note: What we allow wildcards here is from observing some FRCMOD files; it may not be correct)
    // Second attempt. Proper dihedrals: Try swapping both ends
    for c0 in corr_0_candidates {
        let c0 = match eq_map.get(&c0.other) {
            Some(v) => *v,
            None => &c0.other,
        };

        for c1 in corr_1_candidates {
            let c1 = match eq_map.get(&c1.other) {
                Some(v) => *v,
                None => &c1.other,
            };

            consider(c0, key1, key2, c1);
        }
    }

    best
}

// /// We use this check to determien whether to include a default imporoper value, or leave it off.
// /// (Assuming no GAFF2 matches)
// fn should_use_improper_default(key0: &str, key1: &str, key2: &str, key3: &str) -> bool {
//     // We treat key2 as the central atom for impropers, consistent with GAFF patterns like X-X-ca-ha.
//     let center = key2;
//
//     if !matches!(
//         center,
//         "ca" | "ce"
//             | "cf"
//             | "c2"
//             | "c"
//             | "cc"
//             | "cd"
//             | "c6"
//             | "na"
//             | "nb"
//             | "nc"
//             | "nd"
//             | "ne"
//             | "n2"
//     ) {
//         return false;
//     }
//
//     // Optional extra guard: require that not *all* substituents are hydrogens.
//     let types = [key0, key1, key2, key3];
//     let non_h_count = types.iter().filter(|t| !t.starts_with('h')).count();
//
//     non_h_count >= 3
// }

fn is_sp2_like_ff_type(t: &str) -> bool {
    matches!(
        t,
        // carbons
        "c"  | "c2" | "ca" | "cc" | "cd" | "ce" | "c6" |
        // nitrogens
        "n1" | "n2" | "na" | "nb" | "nc" | "nd" | "ne" | "ns" |
        // oxygens/sulfur that are almost always planar-ish centers in GAFF2 impropers
        "o"  | "os" | "oh" | "op" | "oq" | "s"  | "so" | "sx" | "sy"
    )
}

fn is_hydrogen_ff_type(t: &str) -> bool {
    matches!(
        t,
        "h1" | "h2" | "h3" | "ha" | "hc" | "hn" | "ho" | "hs" | "hp"
    )
}

/// Decide if we should ever apply the synthetic 1.1 default improper
/// for this 4-type key. This is *only* for impropers (`proper == false`).
fn is_default_improper_candidate(key: (&str, &str, &str, &str)) -> bool {
    let (a, b, c, d) = key;

    // Treat the third type as the central atom, consistent with GAFF patterns
    // like X -X -ca-ha (center = ca).
    let center = c;

    // Only consider synthetic defaults if the central atom is clearly sp2-like
    // (aromatic, carbonyl, sp2 N, etc.).
    if !is_sp2_like_ff_type(center) {
        return false;
    }

    // Neighbours are the other three atoms.
    let neighbors = [a, b, d];

    // Require at least 2 heavy (non-H) neighbours around the center.
    let heavy_neighbors = neighbors.iter().filter(|t| !is_hydrogen_ff_type(t)).count();

    heavy_neighbors >= 2
}
