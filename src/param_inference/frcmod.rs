//! This module computes per-molecule overrides to GAFF2; this is what Amber's Parmchk or Parmchk2
//! accomplishes. THis is generally Dihedrals and impropers, but can be any bonded parameter
//! this molecule contains, but is missing from Gaff2.dat.
//!
//! This searches Gaff2 for the closest match based on ff types, and returns it.
//! // todo: Possibly with a corrective factor?

use std::{
    collections::{HashMap, HashSet},
    io,
    io::ErrorKind,
    str::FromStr,
};

use bio_files::{
    AtomGeneric,
    md_params::{AngleBendingParams, BondStretchingParams, DihedralParams, ForceFieldParams},
};
use itertools::Itertools;

// todo: Document, here, what each of these do.
const PARMCHK: &str = include_str!("../../param_data/antechamber_defs/PARMCHK.DAT");
const ATCOR: &str = include_str!("../../param_data/antechamber_defs/ATCOR.DAT");

/// Atom types
#[derive(Default)]
pub struct MissingParams {
    pub bond: Vec<(String, String)>,
    pub angle: Vec<(String, String, String)>,
    pub dihedral: Vec<(String, String, String, String)>,
    pub improper: Vec<(String, String, String, String)>,
}

/// Find proper and improper dihedral angles that this molecule has, but are not included in gaff2.dat.
/// Overrides are required for these. `params` passed should be from Gaff2.
///
/// Returns (dihedral, improper). Force field combinations present in the molecule, but not
/// gaff2.dat.
///
/// Note: Repetition here with that in `prep.rs`, but this is simplified. See that for reference.
/// todo: Missing valance and bond params too A/R
pub(super) fn find_missing_dihedrals(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    gaff_params: &ForceFieldParams,
) -> io::Result<MissingParams> {
    let mut result = MissingParams::default();

    for (i0, neighbors) in adj_list.iter().enumerate() {
        for &i1 in neighbors {
            if i0 >= i1 {
                continue; // Only add each bond once.
            }

            if atoms[i0].force_field_type.is_none() || atoms[i1].force_field_type.is_none() {
                eprintln!(
                    "Error finding missing bond params for param inference: Missing FF type."
                );
                return Err(io::Error::new(ErrorKind::Other, "Missing FF type"));
            }

            let type_0 = atoms[i0].force_field_type.as_ref().unwrap();
            let type_1 = atoms[i1].force_field_type.as_ref().unwrap();

            let key = (type_0.clone(), type_1.clone());

            if gaff_params.get_bond(&key).is_none() {
                result.bond.push(key);
            }
        }
    }

    // Valence angles: Every connection between 3 atoms bonded linearly.
    for (ctr, neighbors) in adj_list.iter().enumerate() {
        if neighbors.len() < 2 {
            continue;
        }
        for (&n0, &n1) in neighbors.iter().tuple_combinations() {
            if atoms[n0].force_field_type.is_none()
                || atoms[ctr].force_field_type.is_none()
                || atoms[n1].force_field_type.is_none()
            {
                eprintln!(
                    "Error finding missing valence angles for param inference: Missing FF type."
                );
                return Err(io::Error::new(ErrorKind::Other, "Missing FF type"));
            }

            let type_n0 = atoms[n0].force_field_type.as_ref().unwrap();
            let type_ctr = atoms[ctr].force_field_type.as_ref().unwrap();
            let type_n1 = atoms[n1].force_field_type.as_ref().unwrap();

            let key = (type_n0.clone(), type_ctr.clone(), type_n1.clone());

            if gaff_params.get_valence_angle(&key).is_none() {
                result.angle.push(key);
            }
        }
    }

    // Proper and improper dihedral angles.
    let mut seen = HashSet::<(usize, usize, usize, usize)>::new();

    // Proper dihedrals: Atoms 1-2-3-4 bonded linearly
    for (i1, nbr_j) in adj_list.iter().enumerate() {
        for &i2 in nbr_j {
            if i1 >= i2 {
                continue;
            } // handle each j-k bond once

            for &i0 in adj_list[i1].iter().filter(|&&x| x != i2) {
                for &i3 in adj_list[i2].iter().filter(|&&x| x != i1) {
                    if i0 == i3 {
                        continue;
                    }

                    // Canonicalise so (i1, i2) is always (min, max)
                    let idx_key = if i1 < i2 {
                        (i0, i1, i2, i3)
                    } else {
                        (i3, i2, i1, i0)
                    };
                    if !seen.insert(idx_key) {
                        continue;
                    }

                    if atoms[i0].force_field_type.is_none()
                        || atoms[i1].force_field_type.is_none()
                        || atoms[i2].force_field_type.is_none()
                        || atoms[i3].force_field_type.is_none()
                    {
                        eprintln!(
                            "Error finding missing dihedrals for param inference: Missing FF type."
                        );
                        return Err(io::Error::new(ErrorKind::Other, "Missing FF type"));
                    }

                    let type_0 = atoms[i0].force_field_type.as_ref().unwrap();
                    let type_1 = atoms[i1].force_field_type.as_ref().unwrap();
                    let type_2 = atoms[i2].force_field_type.as_ref().unwrap();
                    let type_3 = atoms[i3].force_field_type.as_ref().unwrap();

                    let key = (
                        type_0.clone(),
                        type_1.clone(),
                        type_2.clone(),
                        type_3.clone(),
                    );

                    if gaff_params.get_dihedral(&key, true).is_none() {
                        result.dihedral.push(key);
                    }
                }
            }
        }
    }

    // Improper dihedrals 2-1-3-4. Atom 3 is the hub, with the other 3 atoms bonded to it.
    // The order of the others in the angle calculation affects the sign of the result.
    // Generally only for planar configs.
    //
    // Note: The sattelites are expected to be in alphabetical order, re their FF types.
    // So, for the hub of "ca" with sattelites of "ca", "ca", and "os", the correct combination
    // to look for in the params is "ca-ca-ca-os"
    for (ctr, satellites) in adj_list.iter().enumerate() {
        if satellites.len() < 3 {
            continue;
        }

        // Unique unordered triples of neighbours
        for a in 0..satellites.len() - 2 {
            for b in a + 1..satellites.len() - 1 {
                for d in b + 1..satellites.len() {
                    let (sat0, sat1, sat2) = (satellites[a], satellites[b], satellites[d]);

                    let idx_key = (sat0, sat1, ctr, sat2); // order is fixed → no swap
                    if !seen.insert(idx_key) {
                        continue;
                    }

                    if atoms[sat0].force_field_type.is_none()
                        || atoms[sat1].force_field_type.is_none()
                        || atoms[ctr].force_field_type.is_none()
                        || atoms[sat2].force_field_type.is_none()
                    {
                        eprintln!(
                            "Error finding missing improper dihedrals for param inference: Missing FF type."
                        );
                        return Err(io::Error::new(ErrorKind::Other, "Missing FF type"));
                    }

                    let type_0 = atoms[sat0].force_field_type.as_ref().unwrap();
                    let type_1 = atoms[sat1].force_field_type.as_ref().unwrap();
                    let type_ctr = atoms[ctr].force_field_type.as_ref().unwrap();
                    let type_2 = atoms[sat2].force_field_type.as_ref().unwrap();

                    // Sort satellites alphabetically; required to ensure we don't miss combinations.
                    let mut sat_types = [type_0.clone(), type_1.clone(), type_2.clone()];
                    sat_types.sort();

                    let key = (
                        sat_types[0].clone(),
                        sat_types[1].clone(),
                        type_ctr.clone(),
                        sat_types[2].clone(),
                    );

                    // todo: Re this note: it may be tough to determine which impropers we need.
                    // In the case of improper, unlike all other param types, we are allowed to
                    // have missing values. Impropers areonly, by Amber convention, for planar
                    // hub and spoke setups, so non-planar ones will be omitted. These may occur,
                    // for example, at ring intersections.
                    if gaff_params.get_dihedral(&key, false).is_none() {
                        result.improper.push(key);
                    }
                }
            }
        }
    }

    Ok(result)
}

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

    let params_missing = find_missing_dihedrals(atoms, adj_list, gaff2)?;

    let atcor = load_atcor()?;
    let parmchk = ParmChk::new()?;

    println!("\n Atcor loaded:");
    for v in &atcor {
        println!("- {v:?}");
    }

    println!("\n Parm loaded:");
    for v in &parmchk.parms {
        println!("parm: {v:?}");
    }
    for v in &parmchk.corr {
        println!("corr: {v:?}");
    }

    // Build simple equivalence map: type -> [self, equiv1, equiv2, ...]
    let mut equiv_map: HashMap<String, Vec<String>> = HashMap::new();
    for (t, (_num, eqs)) in &atcor {
        let mut v = Vec::with_capacity(1 + eqs.len());
        v.push(t.clone());
        v.extend(eqs.iter().cloned());
        equiv_map.insert(t.clone(), v);
    }

    let candidates_for = |t: &str, equiv_map: &HashMap<String, Vec<String>>| -> Vec<String> {
        equiv_map
            .get(t)
            .cloned()
            .unwrap_or_else(|| vec![t.to_owned()])
    };

    // --- bonds -------------------------------------------------------------

    for (t1, t2) in &params_missing.bond {
        if let Some(src) = lookup_bond_in_ff(gaff2, t1, t2) {
            insert_bond_override(&mut result, (t1.clone(), t2.clone()), src);
            continue;
        }

        let c1 = candidates_for(t1, &equiv_map);
        let c2 = candidates_for(t2, &equiv_map);

        let mut best: Option<(f32, (String, String))> = None;

        for a1 in &c1 {
            for a2 in &c2 {
                if let Some(_src) = lookup_bond_in_ff(gaff2, a1, a2) {
                    let penalty = parmchk.bond_penalty(t1, a1) + parmchk.bond_penalty(t2, a2);

                    if best.as_ref().map_or(true, |(p, _)| penalty < *p) {
                        best = Some((penalty, (a1.clone(), a2.clone())));
                    }
                }
            }
        }

        if let Some((_p, best_key)) = best {
            if let Some(src) = lookup_bond_in_ff(gaff2, &best_key.0, &best_key.1) {
                insert_bond_override(&mut result, (t1.clone(), t2.clone()), src);
            }
        }
    }

    // --- valence angles ----------------------------------------------------

    for (t1, t2, t3) in &params_missing.angle {
        if let Some(src) = lookup_angle_in_ff(gaff2, t1, t2, t3) {
            insert_angle_override(&mut result, (t1.clone(), t2.clone(), t3.clone()), src);
            continue;
        }

        let c1 = candidates_for(t1, &equiv_map);
        let c2 = candidates_for(t2, &equiv_map);
        let c3 = candidates_for(t3, &equiv_map);

        let mut best: Option<(f32, (String, String, String))> = None;

        for a1 in &c1 {
            for a2 in &c2 {
                for a3 in &c3 {
                    if let Some(_src) = lookup_angle_in_ff(gaff2, a1, a2, a3) {
                        let penalty = parmchk.angle_outer_penalty(t1, a1)
                            + parmchk.angle_center_penalty(t2, a2)
                            + parmchk.angle_outer_penalty(t3, a3);

                        if best.as_ref().map_or(true, |(p, _)| penalty < *p) {
                            best = Some((penalty, (a1.clone(), a2.clone(), a3.clone())));
                        }
                    }
                }
            }
        }

        if let Some((_p, best_key)) = best {
            if let Some(src) = lookup_angle_in_ff(gaff2, &best_key.0, &best_key.1, &best_key.2) {
                insert_angle_override(&mut result, (t1.clone(), t2.clone(), t3.clone()), src);
            }
        }
    }

    // --- dihedrals ---------------------------------------------------------

    for (t1, t2, t3, t4) in &params_missing.dihedral {
        // 1) direct / wildcard match in GAFF2
        if let Some(src_params) = lookup_dihedral_in_ff(gaff2, t1, t2, t3, t4) {
            insert_dihedral_override(
                &mut result,
                (t1.clone(), t2.clone(), t3.clone(), t4.clone()),
                src_params,
            );
            continue;
        }

        // 2) try substitutions guided by ATCOR + PARMCHK
        let c1 = candidates_for(t1, &equiv_map);
        let c2 = candidates_for(t2, &equiv_map);
        let c3 = candidates_for(t3, &equiv_map);
        let c4 = candidates_for(t4, &equiv_map);

        let mut best: Option<(f32, (String, String, String, String))> = None;

        for a1 in &c1 {
            for a2 in &c2 {
                for a3 in &c3 {
                    for a4 in &c4 {
                        if let Some(_src) = lookup_dihedral_in_ff(gaff2, a1, a2, a3, a4) {
                            let penalty = parmchk.dihe_outer_penalty(t1, a1)
                                + parmchk.dihe_inner_penalty(t2, a2)
                                + parmchk.dihe_inner_penalty(t3, a3)
                                + parmchk.dihe_outer_penalty(t4, a4);

                            if best.as_ref().map_or(true, |(p, _)| penalty < *p) {
                                best = Some((
                                    penalty,
                                    (a1.clone(), a2.clone(), a3.clone(), a4.clone()),
                                ));
                            }
                        }
                    }
                }
            }
        }

        if let Some((_penalty, best_key)) = best {
            if let Some(src_params) =
                lookup_dihedral_in_ff(gaff2, &best_key.0, &best_key.1, &best_key.2, &best_key.3)
            {
                insert_dihedral_override(
                    &mut result,
                    (t1.clone(), t2.clone(), t3.clone(), t4.clone()),
                    src_params,
                );
            }
        } else {
            // nothing found; leave missing or optionally add dummy ATTN, need revision entries
        }
    }

    // --- impropers (same idea, simpler centre-atom focus) ------------------

    for (c, a, b, d) in &params_missing.improper {
        if let Some(src_params) = lookup_improper_in_ff(gaff2, c, a, b, d) {
            insert_improper_override(
                &mut result,
                (c.clone(), a.clone(), b.clone(), d.clone()),
                src_params,
            );
            continue;
        }

        let center_candidates = candidates_for(c, &equiv_map);
        let a_candidates = candidates_for(a, &equiv_map);
        let b_candidates = candidates_for(b, &equiv_map);
        let d_candidates = candidates_for(d, &equiv_map);

        let mut best: Option<(f32, (String, String, String, String))> = None;

        for cc in &center_candidates {
            for aa in &a_candidates {
                for bb in &b_candidates {
                    for dd in &d_candidates {
                        if let Some(_src) = lookup_improper_in_ff(gaff2, cc, aa, bb, dd) {
                            let penalty = parmchk.improper_penalty(c, cc)
                                + parmchk.improper_penalty(a, aa)
                                + parmchk.improper_penalty(b, bb)
                                + parmchk.improper_penalty(d, dd);

                            if best.as_ref().map_or(true, |(p, _)| penalty < *p) {
                                best = Some((
                                    penalty,
                                    (cc.clone(), aa.clone(), bb.clone(), dd.clone()),
                                ));
                            }
                        }
                    }
                }
            }
        }

        if let Some((_p, best_key)) = best {
            if let Some(src_params) =
                lookup_improper_in_ff(gaff2, &best_key.0, &best_key.1, &best_key.2, &best_key.3)
            {
                insert_improper_override(
                    &mut result,
                    (c.clone(), a.clone(), b.clone(), d.clone()),
                    src_params,
                );
            }
        }
    }

    Ok(result)
}

// ---- FF lookups / inserts --------------------------------------------------

fn type_matches(pattern: &str, actual: &str) -> bool {
    pattern == "X" || pattern == actual
}

fn dihedral_key_matches(
    key: &(String, String, String, String),
    t1: &str,
    t2: &str,
    t3: &str,
    t4: &str,
) -> bool {
    // forward
    (type_matches(&key.0, t1)
        && type_matches(&key.1, t2)
        && type_matches(&key.2, t3)
        && type_matches(&key.3, t4))
        // reversed
        || (type_matches(&key.3, t1)
        && type_matches(&key.2, t2)
        && type_matches(&key.1, t3)
        && type_matches(&key.0, t4))
}

fn lookup_dihedral_in_ff<'a>(
    ff: &'a ForceFieldParams,
    t1: &str,
    t2: &str,
    t3: &str,
    t4: &str,
) -> Option<&'a Vec<DihedralParams>> {
    ff.dihedral
        .iter()
        .find(|(k, _)| dihedral_key_matches(k, t1, t2, t3, t4))
        .map(|(_, v)| v)
}

fn insert_dihedral_override(
    dst: &mut ForceFieldParams,
    key: (String, String, String, String),
    src: &Vec<DihedralParams>,
) {
    dst.dihedral.insert(key, src.clone());
}

// impropers: assume centre is first entry in the key; adjust to your convention
fn improper_key_matches(
    key: &(String, String, String, String),
    c: &str,
    a: &str,
    b: &str,
    d: &str,
) -> bool {
    if !type_matches(&key.2, c) {
        return false;
    }

    let mut key_side = vec![key.0.as_str(), key.1.as_str(), key.3.as_str()];
    let mut tgt_side = vec![a, b, d];

    key_side.sort_unstable();
    tgt_side.sort_unstable();

    key_side
        .iter()
        .zip(tgt_side.iter())
        .all(|(k, t)| type_matches(k, t))
}

fn lookup_improper_in_ff<'a>(
    ff: &'a ForceFieldParams,
    c: &str,
    a: &str,
    b: &str,
    d: &str,
) -> Option<&'a Vec<DihedralParams>> {
    ff.improper
        .iter()
        .find(|(k, _)| improper_key_matches(k, c, a, b, d))
        .map(|(_, v)| v)
}

fn insert_improper_override(
    dst: &mut ForceFieldParams,
    key: (String, String, String, String),
    src: &Vec<DihedralParams>,
) {
    dst.improper.insert(key, src.clone());
}

// ---- PARMCHK parsing / similarity -----------------------------------------

/// From `PARM` and `EQUA` lines in `PARMCHK.DAT`.
/// Example:
/// PARM 	C	1		0	12.01	0	6
/// EQUA 	c
#[derive(Debug)]
struct Parm {
    atom_type: String, // e.g. "C", "f" etc.
    improper_flag: bool,
    group_id: u8,
    mass: f32,
    equivalent_flag: bool,
    atomic_num: u8,
    equa: String,
}

impl Parm {
    /// Parse from a pair of lines.
    pub fn parse(parm: &str, equa: &str) -> io::Result<Self> {
        let cols_parm: Vec<&str> = parm.split_whitespace().collect();
        if cols_parm.len() < 7 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Invalid PARMCHK PARM data; not enough columns: {cols_parm:?}"),
            ));
        }

        let group_id = cols_parm[3]
            .parse()
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
        let atomic_num = cols_parm[6]
            .parse()
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        let cols_equa: Vec<&str> = equa.split_whitespace().collect();
        if cols_equa.len() < 2 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid PARMCHK EQUA data; not enough columns.",
            ));
        }

        Ok(Self {
            atom_type: cols_parm[1].to_owned(),
            improper_flag: cols_parm[2] == "1",
            group_id,
            mass: parse_float(cols_parm[4])?,
            equivalent_flag: cols_parm[5] == "1",
            atomic_num,
            equa: cols_equa[1].to_owned(),
        })
    }
}

/// From `CORR` lines in `PARMCHK.DAT`. Note that `ATCOR.DAT` also prefixes lines with `CORR`,
/// but these have a different format.
/// Example:
/// CORR    c2   4.9  21.9   4.3   2.9   2.9   2.3  96.7  -1.0  36.1
#[derive(Debug)]
struct Corr {
    base: String,
    other: String,
    vals: [f32; 9],
}

impl Corr {
    pub fn parse(line: &str, base: &str) -> io::Result<Self> {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 11 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Invalid PARMCHK COR data; not enough columns: {cols:?}"),
            ));
        }

        Ok(Self {
            base: base.to_owned(),
            other: cols[1].to_owned(),
            vals: [
                parse_float(cols[2])?,
                parse_float(cols[3])?,
                parse_float(cols[4])?,
                parse_float(cols[5])?,
                parse_float(cols[6])?,
                parse_float(cols[7])?,
                parse_float(cols[8])?,
                parse_float(cols[9])?,
                parse_float(cols[10])?,
            ],
        })
    }
}

/// Data from PARMCHK.DAT. This is used to map force field types from arbitrary small molecules
/// to those in GAFF2.dat. (Mabye broader as well?) We use this to determine the closest GAFF2 dihedrals (etc)
/// to mape to each missing one.
#[derive(Debug)]
struct ParmChk {
    parms: Vec<Parm>,
    corr: Vec<Corr>,
    corr_map: HashMap<(String, String), [f32; 9]>,
    aliases: HashMap<String, String>, // PARM type -> canonical type for CORR
}

impl ParmChk {
    pub fn new() -> io::Result<Self> {
        let mut parms = Vec::new();
        let mut corr = Vec::new();
        let mut corr_map = HashMap::new();
        let mut aliases = HashMap::new();

        let lines: Vec<&str> = PARMCHK.lines().collect();
        let mut current_parm: Option<String> = None;

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            if line.starts_with("PARM") {
                let cols: Vec<&str> = line.split_whitespace().collect();
                if cols.len() >= 2 {
                    current_parm = Some(cols[1].to_owned());
                }

                // only treat as a PARM/EQUA pair if the next line is EQUA
                if i + 1 < lines.len() && lines[i + 1].trim_start().starts_with("EQUA") {
                    let equa_line = lines[i + 1];
                    parms.push(Parm::parse(lines[i], equa_line)?);
                    i += 2;
                    continue;
                } else {
                    i += 1;
                    continue;
                }
            } else if line.starts_with("CORR") {
                let cols: Vec<&str> = line.split_whitespace().collect();

                if let Some(base) = &current_parm {
                    if cols.len() <= 2 {
                        // short form: "CORR c3" -> base uses corr of c3
                        let target = cols.get(1).ok_or_else(|| {
                            io::Error::new(
                                ErrorKind::InvalidData,
                                format!("Invalid short PARMCHK CORR line: {cols:?}"),
                            )
                        })?;
                        aliases.insert(base.clone(), (*target).to_string());
                        i += 1;
                        continue;
                    }

                    // full numeric CORR line
                    let c = Corr::parse(lines[i], base)?;
                    corr_map.insert((c.base.clone(), c.other.clone()), c.vals);
                    corr_map.insert((c.other.clone(), c.base.clone()), c.vals);
                    corr.push(c);
                }

                i += 1;
                continue;
            }

            i += 1;
        }

        Ok(Self {
            parms,
            corr,
            corr_map,
            aliases,
        })
    }

    fn canonical(&self, t: &str) -> String {
        let mut cur = t;
        let mut depth = 0;

        // follow alias chain like Be -> c3 -> ...
        loop {
            if let Some(next) = self.aliases.get(cur) {
                cur = next;
                depth += 1;
                if depth > 10 {
                    break;
                }
            } else {
                break;
            }
        }

        cur.to_string()
    }

    fn corr_value(&self, a: &str, b: &str, idx: usize) -> f32 {
        if a == b {
            return 0.0;
        }

        let ca = self.canonical(a);
        let cb = self.canonical(b);

        self.corr_map
            .get(&(ca.clone(), cb.clone()))
            .map(|v| v[idx])
            .unwrap_or(1.0e6)
    }

    fn bond_penalty(&self, from: &str, to: &str) -> f32 {
        // heuristic: use column 0 as "bond similarity"
        self.corr_value(from, to, 0)
    }

    fn angle_center_penalty(&self, from: &str, to: &str) -> f32 {
        // heuristic: use column 2 for angle-center similarity
        self.corr_value(from, to, 2)
    }

    fn angle_outer_penalty(&self, from: &str, to: &str) -> f32 {
        // heuristic: use column 3 for angle-outer similarity
        self.corr_value(from, to, 3)
    }

    // index choices are a best guess; tweak if needed after inspecting PARMCHK.DAT docs
    fn dihe_inner_penalty(&self, from: &str, to: &str) -> f32 {
        self.corr_value(from, to, 4)
    }

    fn dihe_outer_penalty(&self, from: &str, to: &str) -> f32 {
        self.corr_value(from, to, 5)
    }

    fn improper_penalty(&self, from: &str, to: &str) -> f32 {
        self.corr_value(from, to, 6)
    }
}

// ---- bonds ---------------------------------------------------------------

fn bond_key_matches(key: &(String, String), t1: &str, t2: &str) -> bool {
    // symmetric: t1-t2 == t2-t1, with wildcard X
    (type_matches(&key.0, t1) && type_matches(&key.1, t2))
        || (type_matches(&key.0, t2) && type_matches(&key.1, t1))
}

fn lookup_bond_in_ff<'a>(
    ff: &'a ForceFieldParams,
    t1: &str,
    t2: &str,
) -> Option<&'a BondStretchingParams> {
    ff.bond
        .iter()
        .find(|(k, _)| bond_key_matches(k, t1, t2))
        .map(|(_, v)| v)
}

fn insert_bond_override(
    dst: &mut ForceFieldParams,
    key: (String, String),
    src: &BondStretchingParams,
) {
    dst.bond.insert(key, src.clone());
}

// ---- angles --------------------------------------------------------------

fn angle_key_matches(key: &(String, String, String), t1: &str, t2: &str, t3: &str) -> bool {
    // center is t2; ends can swap, wildcards allowed
    type_matches(&key.1, t2)
        && ((type_matches(&key.0, t1) && type_matches(&key.2, t3))
            || (type_matches(&key.0, t3) && type_matches(&key.2, t1)))
}

fn lookup_angle_in_ff<'a>(
    ff: &'a ForceFieldParams,
    t1: &str,
    t2: &str,
    t3: &str,
) -> Option<&'a AngleBendingParams> {
    ff.angle
        .iter()
        .find(|(k, _)| angle_key_matches(k, t1, t2, t3))
        .map(|(_, v)| v)
}

fn insert_angle_override(
    dst: &mut ForceFieldParams,
    key: (String, String, String),
    src: &AngleBendingParams,
) {
    dst.angle.insert(key, src.clone());
}

/// Load data from Amber's ATCOR.dat, included in the application binary.
/// todo: Adjust this A/R once you understand it better and attempt to use it.
fn load_atcor() -> io::Result<HashMap<String, (u8, Vec<String>)>> {
    let mut result = HashMap::new();

    for line in ATCOR.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();

        if cols.len() < 4 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid Atcor data; not enough columns.",
            ));
        }

        let num: u8 = cols[2].parse().unwrap();
        let mut remaining = Vec::new();

        for col in &cols[3..] {
            remaining.push(col.to_string());
        }

        result.insert(cols[1].to_owned(), (num, remaining));
    }

    Ok(result)
}

/// Reduces repetition
fn parse_float(v: &str) -> io::Result<f32> {
    v.parse()
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))
}
