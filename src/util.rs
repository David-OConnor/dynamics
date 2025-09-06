//! Misc utility functions.

use bio_files::{AtomGeneric, BondGeneric};

use crate::ParamError;

/// Build a list of indices that relate atoms that are connected by covalent bonds.
/// For each outer atom index, the inner values are indices of the atom it's bonded to.
///
/// Note: If you store bonds with atom indices directly, you may wish to build this in a faster
/// way and cache it, vice this serial-number lookup.
pub fn build_adjacency_list(
    bonds: &Vec<BondGeneric>,
    atoms: &[AtomGeneric],
) -> Result<Vec<Vec<usize>>, ParamError> {
    let mut result = vec![Vec::new(); atoms.len()];

    // For each bond, record its atoms as neighbors of each other
    for bond in bonds {
        let mut atom_0 = None;
        let mut atom_1 = None;

        let mut found = false;
        for (i, atom) in atoms.iter().enumerate() {
            if atom.serial_number == bond.atom_0_sn {
                atom_0 = Some(i);
            }
            if atom.serial_number == bond.atom_1_sn {
                atom_1 = Some(i);
            }
            if atom_0.is_some() && atom_1.is_some() {
                result[atom_0.unwrap()].push(atom_1.unwrap());
                result[atom_1.unwrap()].push(atom_0.unwrap());

                found = true;
                break;
            }
        }

        if !found {
            return Err(ParamError::new(
                "Invalid bond to atom mapping when building adjacency list.",
            ));
        }
    }

    Ok(result)
}
