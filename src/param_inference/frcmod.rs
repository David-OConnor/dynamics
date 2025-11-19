//! This module computes per-molecule overrides to GAFF2; this is what Amber's Parmchk or Parmchk2
//! accomplishes. THis is generally Dihedrals and impropers, but can be any bonded parameter
//! this molecule contains, but is missing from Gaff2.dat.
//!
//! This searches Gaff2 for the closest match based on ff types, and returns it.
//! // todo: Possibly with a corrective factor?

use std::{collections::HashSet, io};

use bio_files::{AtomGeneric, BondGeneric, md_params::ForceFieldParams};

pub(super) const MAX_DIHEDRAL_TERMS: usize = 3;
pub(super) const DIHEDRAL_FEATS: usize = 3;

// todo: Document, here, what each of these do.
const PARMCHK: &str = include_str!("../../param_data/antechamber_defs/PARMCHK.DAT");
const ATCOR: &str = include_str!("../../param_data/antechamber_defs/ATCOR.DAT");

/// Atom types
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
/// We may integrate this into our FRCMOD inference pipeline.
/// todo: Missing valance and bond params too A/R
// todo: Not sure where this will go
pub(super) fn find_missing_dihedrals(
    atoms: &[AtomGeneric],
    adj_list: &[Vec<usize>],
    gaff_params: &ForceFieldParams,
) -> io::Result<MissingParams> {
    // todo: This is a copy+paste+modify from FFParamsIndexed::new() for now.
    // Proper and improper dihedral angles.
    let mut seen = HashSet::<(usize, usize, usize, usize)>::new();

    let mut dihedral = Vec::new();
    let mut improper = Vec::new();

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
                        return Err(io::Error::new(io::ErrorKind::Other, "Missing FF type"));
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
                        dihedral.push(key);
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
                        return Err(io::Error::new(io::ErrorKind::Other, "Missing FF type"));
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
                        improper.push(key);
                    }
                }
            }
        }
    }

    // todo
    let bond = Vec::new();
    let angle = Vec::new();

    Ok(MissingParams {
        bond,
        angle,
        dihedral,
        improper,
    })
}

pub fn find_missing_params(
    atoms: &[AtomGeneric],
    // bonds: &[BondGeneric],
    adj_list: &[Vec<usize>],
    gaff2: &ForceFieldParams,
) -> io::Result<ForceFieldParams> {
    let mut result = ForceFieldParams::default();
    // todo: Find missing valence and bond too.
    let params_missing = find_missing_dihedrals(atoms, adj_list, gaff2)?;

    for dihe in &params_missing.dihedral {}

    for improper in &params_missing.dihedral {}

    Ok(result)
}

// pub fn update_with_frcmod(
//     atoms: &mut [AtomGeneric],
//     bonds: &[BondGeneric],
//     adj_list: &[Vec<usize>],
//     params: &ForceFieldParams,
// ) -> io::Result<()> {
//     // todo: Find missing valence and bond too.
//     let params_missing = find_missing_dihedrals(atoms, adj_list, params)?;
//
//     for dihe in &params_missing.0 {}
//
//     Ok(())
// }
