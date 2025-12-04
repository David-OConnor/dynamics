//! We use this module to identify which bonded parameters are present in a small organic molecule,
//! but are absent from GAFF2.dat.

use std::{collections::HashSet, io, io::ErrorKind};

use bio_files::{AtomGeneric, md_params::ForceFieldParams};
use itertools::Itertools;

/// These values are atom forcefield types.
#[derive(Default)]
pub struct MissingParams {
    pub bond: Vec<(String, String)>,
    pub angle: Vec<(String, String, String)>,
    pub dihedral: Vec<(String, String, String, String)>,
    pub improper: Vec<(String, String, String, String)>,
}

impl MissingParams {
    /// Find proper and improper dihedral angles that this molecule has, but are not included in gaff2.dat.
    /// Overrides are required for these. `params` passed should be from Gaff2.
    ///
    /// Returns (dihedral, improper). Force field combinations present in the molecule, but not
    /// gaff2.dat.
    ///
    /// Note: Repetition here with that in `prep.rs`, but this is simplified. See that for reference.
    /// todo: Missing valance and bond params too A/R
    pub(in crate::param_inference) fn new(
        atoms: &[AtomGeneric],
        adj_list: &[Vec<usize>],
        gaff_params: &ForceFieldParams,
    ) -> io::Result<Self> {
        let mut result = Self::default();

        for (i0, neighbors) in adj_list.iter().enumerate() {
            for &i1 in neighbors {
                if i0 >= i1 {
                    continue; // Only add each bond once.
                }

                if atoms[i0].force_field_type.is_none() || atoms[i1].force_field_type.is_none() {
                    eprintln!(
                        "Error finding missing bond params for param inference: Missing FF type {} - {}",
                        atoms[i0].serial_number, atoms[i1].serial_number
                    );
                    return Err(io::Error::other(format!(
                        "Missing FF type: {} - {}",
                        atoms[i0].serial_number, atoms[i1].serial_number
                    )));
                }

                let type_0 = atoms[i0].force_field_type.as_ref().unwrap();
                let type_1 = atoms[i1].force_field_type.as_ref().unwrap();

                let key = (type_0.clone(), type_1.clone());

                if gaff_params.get_bond(&key, true).is_none() {
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
                    return Err(io::Error::other("Missing FF type"));
                }

                let type_n0 = atoms[n0].force_field_type.as_ref().unwrap();
                let type_ctr = atoms[ctr].force_field_type.as_ref().unwrap();
                let type_n1 = atoms[n1].force_field_type.as_ref().unwrap();

                let key = (type_n0.clone(), type_ctr.clone(), type_n1.clone());

                if gaff_params.get_valence_angle(&key, true).is_none() {
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
                            return Err(io::Error::other("Missing FF type"));
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

                        if gaff_params.get_dihedral(&key, true, true).is_none() {
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
                            return Err(io::Error::other("Missing FF type"));
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
                        if gaff_params.get_dihedral(&key, false, true).is_none() {
                            result.improper.push(key);
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}
