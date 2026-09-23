//! Estimate missing GAFF2 bonded parameters using Amber's directed EQUA/CORR
//! replacements, weights, and positional penalty definitions from PARMCHK.DAT.
//! Reference: `parmchk2.c`, AmberClassic 8e55e97ada48b96eefaec2e6a3fa849018aaeea5.
//!
//! Search the existing parameter tables instead of a Cartesian product of atom
//! substitutions. This includes changes at every position, preserves all Fourier
//! terms, and resolves equal scores deterministically. Equivalent specific terms
//! precede generic terms, which precede corresponding-type estimates.
//!
//! This remains an estimator, not a complete parmchk2 port: empirical bond/angle
//! fitting (PARM_BLBA_GAFF2.DAT) is not implemented. An unavailable required term
//! returns an error. Comments record the source and penalty of each estimate.

use std::{f32::consts::PI, io};

use bio_files::{
    AtomGeneric,
    md_params::{DihedralParams, ForceFieldParams},
};

use super::{
    frcmod_missing_params::MissingParams,
    parmchk_parse::{PARAMETERS, Position},
    topology::invalid,
};

#[derive(Clone, Copy)]
enum Term {
    Bond,
    Angle,
    Proper,
    Improper,
}

/// Amber's search stages: exact, equivalent specific, original generic,
/// equivalent generic, corresponding specific, corresponding generic.
fn score(from: &[&str], to: &[&str], term: Term) -> Option<(u8, f32, usize)> {
    let p = &*PARAMETERS;
    let mut equivalent = true;
    let mut changed = 0;
    let mut penalty = 0.;
    let mut wildcard = false;
    for (i, (&a, &b)) in from.iter().zip(to).enumerate() {
        if b == "X" {
            if matches!(term, Term::Bond | Term::Angle)
                || (matches!(term, Term::Proper) && i != 0 && i != 3)
            {
                return None;
            }
            wildcard = true;
            if matches!(term, Term::Improper) {
                penalty += p.weight(if i == 2 { "WEIGHT_X3" } else { "WEIGHT_X" });
            }
            continue;
        }
        let position = match term {
            Term::Bond => Position::Bond,
            Term::Angle if i == 1 => Position::AngleCenter,
            Term::Angle => Position::AngleOuter,
            Term::Proper if i == 1 || i == 2 => Position::TorsionCenter,
            Term::Proper => Position::TorsionOuter,
            Term::Improper if i == 2 => Position::ImproperCenter,
            Term::Improper => Position::ImproperOuter,
        };
        let (eq, value) = p.penalty(a, b, position)?;
        equivalent &= eq;
        penalty += value;
        changed += usize::from(a != b);
    }
    let stage = match (equivalent, wildcard, changed == 0) {
        (true, false, true) => 0,
        (true, false, false) => 1,
        (true, true, true) => 2,
        (true, true, false) => 3,
        (false, false, _) => 4,
        (false, true, _) => 5,
    };
    if !equivalent {
        penalty += p.group_penalty(to);
        if matches!(term, Term::Proper) {
            penalty += p.pair_penalty([from[1], from[2]], [to[1], to[2]]);
        }
    }
    Some((stage, penalty, changed))
}

const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 3],
    [0, 3, 1],
    [1, 0, 3],
    [1, 3, 0],
    [3, 0, 1],
    [3, 1, 0],
];
fn orientations<'a>(types: &[&'a str], term: Term) -> Vec<Vec<&'a str>> {
    if matches!(term, Term::Improper) {
        PERMUTATIONS
            .iter()
            .map(|p| vec![types[p[0]], types[p[1]], types[2], types[p[2]]])
            .collect()
    } else {
        vec![types.to_vec(), types.iter().rev().copied().collect()]
    }
}

#[derive(PartialEq, PartialOrd)]
struct CandidateRank<'a> {
    stage: u8,
    penalty: f32,
    replacement_order: Vec<usize>,
    key: Vec<&'a str>,
}

fn best<'a, T>(
    from: &[&str],
    table: impl Iterator<Item = (Vec<&'a str>, &'a T)>,
    term: Term,
) -> Option<(&'a T, String)> {
    // Lexical final tie break makes results independent of HashMap iteration.
    let mut best: Option<(CandidateRank<'_>, &T)> = None;
    for (key, value) in table {
        for target in orientations(&key, term) {
            let Some((stage, penalty, _)) = score(from, &target, term) else {
                continue;
            };
            // Equal-score candidates follow PARMCHK's replacement order. In
            // particular n6 lists n7 before n3; collapsing both to n3 changes
            // the torsion. Amber enumerates central atoms first for EQUA propers.
            let order: Vec<_> = if matches!(term, Term::Proper) && stage == 1 {
                [1, 2, 0, 3]
                    .iter()
                    .map(|&i| PARAMETERS.replacement_order(from[i], target[i]))
                    .collect()
            } else {
                from.iter()
                    .zip(&target)
                    .map(|(&a, &b)| PARAMETERS.replacement_order(a, b))
                    .collect()
            };
            let rank = CandidateRank {
                stage,
                penalty,
                replacement_order: order,
                key: key.clone(),
            };
            if best.as_ref().is_none_or(|(old, _)| rank < *old) {
                best = Some((rank, value));
            }
        }
    }
    best.map(|(CandidateRank { penalty, key, .. }, value)| {
        (
            value,
            format!(
                "Estimated from {}; PARMCHK penalty {penalty:.3}",
                key.join("-")
            ),
        )
    })
}

/// Identify and estimate missing bonded parameters. Atom types must already be
/// assigned, and adjacency indices must refer to the supplied atom slice.
/// Does not modify the universal parameter set or infer nonbonded parameters.
pub fn assign_missing_params(
    atoms: &[AtomGeneric],
    adj: &[Vec<usize>],
    gaff2: &ForceFieldParams,
) -> io::Result<ForceFieldParams> {
    let missing = MissingParams::new(atoms, adj, gaff2)?;
    let mut result = ForceFieldParams::default();
    for key in missing.bond {
        let from = [key.0.as_str(), key.1.as_str()];
        let (source, comment) = best(
            &from,
            gaff2
                .bond
                .iter()
                .map(|(k, v)| (vec![k.0.as_str(), k.1.as_str()], v)),
            Term::Bond,
        )
        .ok_or_else(|| invalid(format!("No bond parameter for {key:?}")))?;
        let mut value = source.clone();
        value.atom_types = key.clone();
        value.comment = Some(comment);
        result.bond.insert(key, value);
    }
    for key in missing.angle {
        let from = [key.0.as_str(), key.1.as_str(), key.2.as_str()];
        let (source, comment) = best(
            &from,
            gaff2
                .angle
                .iter()
                .map(|(k, v)| (vec![k.0.as_str(), k.1.as_str(), k.2.as_str()], v)),
            Term::Angle,
        )
        .ok_or_else(|| invalid(format!("No angle parameter for {key:?}")))?;
        let mut value = source.clone();
        value.atom_types = key.clone();
        value.comment = Some(comment);
        result.angle.insert(key, value);
    }
    for (keys, table, term) in [
        (missing.dihedral, &gaff2.dihedral, Term::Proper),
        (missing.improper, &gaff2.improper, Term::Improper),
    ] {
        for key in keys {
            let from = [
                key.0.as_str(),
                key.1.as_str(),
                key.2.as_str(),
                key.3.as_str(),
            ];
            let hit = best(
                &from,
                table.iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| {
                    (
                        vec![k.0.as_str(), k.1.as_str(), k.2.as_str(), k.3.as_str()],
                        v,
                    )
                }),
                term,
            );
            let values = if let Some((source, comment)) = hit {
                source
                    .iter()
                    .map(|source| {
                        let mut v = source.clone();
                        v.atom_types = key.clone();
                        v.comment = Some(comment.clone());
                        v
                    })
                    .collect()
            } else if matches!(term, Term::Improper) {
                // parmchk2's last resort, only for flagged three-coordinate hubs.
                vec![DihedralParams {
                    atom_types: key.clone(),
                    divider: 1,
                    barrier_height: 1.1,
                    phase: PI,
                    periodicity: 2,
                    comment: Some(
                        "Estimated: Amber default improper; no matching parameter".into(),
                    ),
                }]
            } else {
                return Err(invalid(format!("No proper torsion parameter for {key:?}")));
            };
            if matches!(term, Term::Proper) {
                result.dihedral.insert(key, values);
            } else {
                result.improper.insert(key, values);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn improper_center_is_third_and_never_reversed() {
        let from = ["hc", "c3", "c", "o"];
        assert!(score(&from, &["X", "X", "c", "o"], Term::Improper).is_some());
        assert!(score(&from, &["o", "c", "X", "X"], Term::Improper).is_none());
        for permutation in orientations(&from, Term::Improper) {
            assert_eq!(permutation[2], "c");
        }
    }
    #[test]
    fn equivalent_specific_precedes_generic_and_corresponding() {
        let from = ["hc", "c3", "n7", "hn"];
        let eq = score(&from, &["hc", "c3", "n3", "hn"], Term::Proper).unwrap();
        let generic = score(&from, &["X", "c3", "n7", "X"], Term::Proper).unwrap();
        assert!(eq < generic);
    }
    #[test]
    fn infers_missing_params_for_os_c3_nz_hn_chain() {
        let params = crate::params::FfParamSet::new_amber().unwrap();
        let types = ["os", "c3", "nz", "hn", "hn", "hn"];
        let atoms = types
            .iter()
            .enumerate()
            .map(|(i, ty)| AtomGeneric {
                serial_number: i as u32 + 1,
                force_field_type: Some((*ty).into()),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let adj = vec![
            vec![1],
            vec![0, 2],
            vec![1, 3, 4, 5],
            vec![2],
            vec![2],
            vec![2],
        ];
        let inferred =
            assign_missing_params(&atoms, &adj, params.small_mol.as_ref().unwrap()).unwrap();
        assert!(
            inferred
                .angle
                .contains_key(&("nz".into(), "c3".into(), "os".into()))
        );
        assert!(
            inferred
                .dihedral
                .keys()
                .any(|k| k.1 == "nz" && k.2 == "c3" || k.1 == "c3" && k.2 == "nz")
        );
        assert!(inferred.improper.is_empty());
    }
}
