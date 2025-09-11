//! Misc utility functions.

use std::{
    fs::File,
    io,
    io::{Read, Write},
    path::Path,
};

use bio_files::{AtomGeneric, BondGeneric};

use crate::{ParamError, snapshot::Snapshot};

/// Build a list of indices that relate atoms that are connected by covalent bonds.
/// For each outer atom index, the inner values are indices of the atom it's bonded to.
///
/// Note: If you store bonds with atom indices directly, you may wish to build this in a faster
/// way and cache it, vice this serial-number lookup.
pub(crate) fn build_adjacency_list(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
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

pub fn save_snapshots(snapshots: &[Snapshot], path: &Path) -> io::Result<()> {
    let mut file = File::create(path)?;

    let mut result = Vec::new();
    let mut i = 0;

    // todo: Add a header if/when required.

    for snap in snapshots {
        let snap_ser = snap.to_bytes();
        result[i..i + snap_ser.len()].copy_from_slice(&snap_ser);
    }

    file.write_all(&result)?;

    Ok(())
}

pub fn load_snapshots(path: &Path) -> io::Result<Vec<Snapshot>> {
    let mut f = File::open(path)?;
    let mut out = Vec::new();

    loop {
        let mut len_buf = [0u8; 4];
        match f.read_exact(&mut len_buf) {
            Ok(()) => {
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut buf = vec![0u8; len];
                f.read_exact(&mut buf)?;
                let s = Snapshot::from_bytes(&buf)?;
                out.push(s);
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }

    Ok(out)
}
