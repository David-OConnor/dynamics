//! This module computes per-molecule overrides to GAFF2; this is what Amber's Parmchk or Parmchk2
//! accomplishes. THis is generally Dihedrals and impropers, but can be any bonded parameter
//! this molecule contains, but is missing from Gaff2.dat.
//!
//! This searches Gaff2 for the closest match based on ff types, and returns it.
//! // todo: Possibly with a corrective factor?
//!
//! todo: Split this up A/R for organizatio.

use std::{collections::HashMap, io, io::ErrorKind};

use bio_files::{
    AtomGeneric,
    md_params::{AngleBendingParams, BondStretchingParams, DihedralParams, ForceFieldParams},
};
use candle_nn::rnn::Direction;

use crate::param_inference::{
    frcmod_missing_params::MissingParams,
    parmchk_parse::{AtCor, Corr, ParmChk, load_atcor},
};

const WILDCARD_PENALTY: f32 = 2.0;

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

    println!("\n\n Dihe missing: ");
    for d in &params_missing.dihedral {
        println!("-Di: {:?}", d);
    }

    println!("\n\n Improper missing: ");
    for d in &params_missing.improper {
        println!("-Imp: {:?}", d);
    }

    let atcor = load_atcor()?;
    let parmchk = ParmChk::new()?;

    println!("\n\n Atcor loaded:");
    for v in &atcor {
        println!("- {v:?}");
    }

    println!("\n\n Parm loaded:");
    for v in &parmchk.parms {
        println!("parm: {v:?}");
    }
    println!("\n\n Parm Corr:");
    for v in &parmchk.corr {
        println!("corr: {v:?}");
    }
    println!("\n\n Parm Aliases:");
    for v in &parmchk.aliases {
        println!("Ali-: {v:?}");
    }
    println!("\n\n Equa to park:");
    for v in &parmchk.equa_to_parm {
        println!("Eq-: {v:?}");
    }

    // todo: We currently get wonky values from Equa to parm ,which doesn't seem to make sense,
    // todo and only afffect a few specific cases. Hard-code them here.
    // todo: If you keep this in, don't build it each time.
    let mut eq_map = HashMap::new();
    eq_map.insert("cs".to_owned(), "c");
    eq_map.insert("o2".to_owned(), "o");
    eq_map.insert("nj".to_owned(), "n");
    eq_map.insert("sq".to_owned(), "ss");
    eq_map.insert("c6".to_owned(), "c3");

    // todo: Add these bonds and valence angles once dihedrals and impropers work.
    // --- bonds -------------------------------------------------------------
    for (t0, t1) in &params_missing.bond {
        let key = (t0.to_owned(), t1.to_owned());

        if let Some(params) = find_bond_alts(gaff2, &parmchk, &atcor, &eq_map, &key) {
            println!("Found Bond alts for {key:?}: {params:?}"); // todo temp

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
            println!("Found Angle alts for {key:?}: {params:?}"); // todo temp

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
            println!("Found Dihedral alts for {key:?}: {params:?}"); // todo temp

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
            println!("Found improper for {key:?}: {params:?}"); // todo temp

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
    if let Some(param) = gaff2.get_bond(
        &(
            key0.to_owned(),
            key1.to_owned(),
        ),
        false,
    ) {
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
    if let Some(param) = gaff2.get_valence_angle(
        &(
            key0.to_owned(),
            key1.to_owned(),
            key2.to_owned(),
        ),
        false,
    ) {
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
    let key3 = match atcor.get(&key.3) {
        Some(v) => v.1[0].clone(),
        None => key.3.clone(),
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

    let params_no_wc = dihedral_inner(
        key0,
        key1,
        key2,
        key3,
        &corr_0_candidates,
        &corr_1_candidates,
        &corr_2_candidates,
        &corr_3_candidates,
        gaff2,
        eq_map,
        proper,
        false,
    );

    match params_no_wc {
        Some(params) => {
            Some(params)
        }
        None => {
            dihedral_inner(
                key0,
                key1,
                key2,
                key3,
                &corr_0_candidates,
                &corr_1_candidates,
                &corr_2_candidates,
                &corr_3_candidates,
                gaff2,
                eq_map,
                proper,
                true,
            )
        }
    }
}

/// Helper to abstract over wildcard
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
    eq_map: &HashMap<String, &str>,
    proper: bool,
    wildcard_allowed: bool,
) -> Option<Vec<DihedralParams>> {
    // todo: Later, don't return immediately on a match; evaluate options, and compare or score.

    // let mut results =

    // Change a single candidate:
    for c in corr_0_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(
                c.other.to_owned(),
                key1.to_owned(),
                key2.to_owned(),
                key3.to_owned(),
            ),
            proper,
            wildcard_allowed,
        ) {
            return Some(param.clone());
        }
    }
    for c in corr_1_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(
                key0.to_owned(),
                c.other.to_owned(),
                key2.to_owned(),
                key3.to_owned(),
            ),
            proper,
            wildcard_allowed,
        ) {
            return Some(param.to_owned());
        }
    }
    for c in corr_2_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(
                key0.to_owned(),
                key1.to_owned(),
                c.other.to_owned(),
                key3.to_owned(),
            ),
            proper,
            wildcard_allowed,
        ) {
            return Some(param.to_owned());
        }
    }
    // todo: This whole fn is DRY.
    for c in corr_3_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(
                key0.to_owned(),
                key1.to_owned(),
                key2.to_owned(),
                c.other.to_owned(),
            ),
            proper,
            wildcard_allowed,
        ) {
            return Some(param.to_owned());
        }
    }

    // Second attempt: Try swapping both ends.
    for c0 in corr_0_candidates {
        let c0 = match eq_map.get(&c0.other) {
            Some(v) => *v,
            None => &c0.other,
        };
        // todo: DRY again
        for c1 in corr_1_candidates {
            let c1 = match eq_map.get(&c1.other) {
                Some(v) => *v,
                None => &c1.other,
            };

            if let Some(param) = gaff2.get_dihedral(
                &(
                    c0.to_owned(),
                    key1.to_owned(),
                    key2.to_owned(),
                    c1.to_owned(),
                ),
                proper,
                wildcard_allowed,
            ) {
                return Some(param.to_owned());
            }
        }
    }

    None
}
