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
    parmchk_parse::{ParmChk, load_atcor},
};
use crate::param_inference::parmchk_parse::AtCor;

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

    // todo: Add these bonds and valence angles once dihedrals and impropers work.
    // --- bonds -------------------------------------------------------------
    // for (t0, t1) in &params_missing.bond {
    //     let key = (t0.to_owned(), t1.to_owned());
    //
    //     if let Some(params) = find_bond_alts(gaff2, &parmchk, &atcor, &key, true) {
    //         println!("Found Bond alts for {key:?}: {params:?}"); // todo temp
    //
    //         result.bond.insert(key, params);
    //     } else {
    //         return Err(io::Error::new(
    //             ErrorKind::InvalidData,
    //             format!("Unable to find a bondparam for {t0} {t1}"),
    //         ));
    //     }
    // }
    //
    // // --- Valence angles -------------------------------------------------------------
    // for (t0, t1, t2) in &params_missing.angle {
    //     let key = (t0.to_owned(), t1.to_owned(), t2.to_owned());
    //
    //     if let Some(params) = find_angle_alts(gaff2, &parmchk, &atcor, &key, true) {
    //         println!("Found Angle alts for {key:?}: {params:?}"); // todo temp
    //
    //         result.angle.insert(key, params);
    //     } else {
    //         return Err(io::Error::new(
    //             ErrorKind::InvalidData,
    //             format!("Unable to find a valence angle param for {t0} {t1} {t2}"),
    //         ));
    //     }
    // }

    // --- dihedrals ---------------------------------------------------------
    for (t0, t1, t2, t3) in &params_missing.dihedral {
        let key = (t0.to_owned(), t1.to_owned(), t2.to_owned(), t3.to_owned());

        if let Some(params) = find_dihedral_alts(gaff2, &parmchk, &atcor, &key, true) {
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

        if let Some(params) = find_dihedral_alts(gaff2, &parmchk, &atcor, &key, false) {
            println!("Found improper for {key:?}: {params:?}"); // todo temp

            result.dihedral.insert(key, params);
        }
        // We are allowed to be missing Impropers, unlike other parameters.
    }

    Ok(result)
}

fn find_dihedral_alts(
    gaff2: &ForceFieldParams,
    parmchk: &ParmChk,
    atcor: &AtCor,
    key: &(String, String, String, String),
    proper: bool,
) -> Option<Vec<DihedralParams>> {
    // Apply Atcor.
    // If a given param is in Atcor, use that to map to its canonical type. For now, we just
    // pick the first one; that seems to work for these like "n7"
    let key0 = match atcor.get(&key.0) {
        Some(v) => v.1[0].clone(),
        None => key.0.clone()
    };
    let key1 = match atcor.get(&key.1) {
        Some(v) => v.1[0].clone(),
        None => key.0.clone()
    };
    let key2 = match atcor.get(&key.2) {
        Some(v) => v.1[0].clone(),
        None => key.0.clone()
    };
    let key3 = match atcor.get(&key.3) {
        Some(v) => v.1[0].clone(),
        None => key.0.clone()
    };

    // Apply eq values from PARMCHK.
    // todo: We currently get wonky values from Equa to parm ,which doesn't seem to make sense,
    // todo and only afffect a few specific cases. Hard-code them here.
    // todo: If you keep this in, don't build it each time.
    let mut eq_map = HashMap::new();
    eq_map.insert("cs".to_owned(), "c");
    eq_map.insert("o2".to_owned(), "o");
    eq_map.insert("nj".to_owned(), "n");
    eq_map.insert("sq".to_owned(), "ss");
    eq_map.insert("c6".to_owned(), "c3");

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

    println!("\n\nKey 0: {key0} Corr 0 candidates: {corr_0_candidates:?}");
    println!("\nKey 1: {key1} Corr 1 candidates: {corr_1_candidates:?}");
    println!("\nKey 2: {key2} Corr 2 candidates: {corr_2_candidates:?}");
    println!("\nKey 3: {key3} Corr 3 candidates: {corr_3_candidates:?}");

    // todo: Later, don't return immediately on a match; evaluate options, and compare or score.
    // First attempt: Change a single candidate:
    for c in &corr_0_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(c.other.to_owned(), key1.to_owned(), key2.to_owned(), key3.to_owned()),
            proper,
        ) {
            return Some(param.clone());
        }
    }
    for c in &corr_1_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(key0.to_owned(), c.other.to_owned(), key2.to_owned(), key3.to_owned()),
            proper,
        ) {
            return Some(param.to_owned());
        }
    }
    for c in &corr_2_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(key0.to_owned(), key1.to_owned(), c.other.to_owned(), key3.to_owned()),
            proper,
        ) {
            return Some(param.to_owned());
        }
    }
    // todo: This whole fn is DRY.
    for c in &corr_3_candidates {
        if let Some(param) = gaff2.get_dihedral(
            &(key0.to_owned(), key1.to_owned(), key2.to_owned(), c.other.to_owned()),
            proper,
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

            if let Some(param) = gaff2.get_dihedral(
                &(c0.to_owned(), key1.to_owned(), key2.to_owned(), c1.to_owned()),
                proper,
            ) {
                return Some(param.to_owned());
            }
            // try its equivalents. We currently use aliases and/or equa
        }
    }

    None
}
