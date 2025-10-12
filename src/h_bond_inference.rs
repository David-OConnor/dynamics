//! todo: This is a C+P from Daedalus, with args modified, e.g. Atom type, f32 vs f64.
// todo to not use its Atom type.

use std::f32::consts::TAU;

use na_seq::Element::{Fluorine, Hydrogen, Nitrogen, Oxygen, Sulfur};
use crate::AtomDynamics;
use crate::snapshot::HydrogenBond;
use crate::water_opc::WaterMol;

// Note: Chimera shows H bonds as ranging generally from 2.8 to 3.3.
// Note: These values all depend on which is the donor. Your code doesn't take this into account.
const H_BOND_O_O_DIST: f32 = 2.7;
const H_BOND_N_N_DIST: f32 = 3.05;
const H_BOND_O_N_DIST: f32 = 2.9;

const H_BOND_N_F_DIST: f32 = 2.75;
const H_BOND_N_S_DIST: f32 = 3.35;
const H_BOND_S_O_DIST: f32 = 3.35;
const H_BOND_S_F_DIST: f32 = 3.2; // rarate

const H_BOND_DIST_THRESH: f32 = 0.3;
const H_BOND_DIST_GRID: f32 = 3.6;

const H_BOND_ANGLE_THRESH: f32 = TAU / 3.;

/// Helper
fn h_bond_candidate_el(atom: &AtomDynamics) -> bool {
    matches!(atom.element, Nitrogen | Oxygen | Sulfur | Fluorine)
}

fn hydrogen_bond_inner(
    bonds: &mut Vec<HydrogenBond>,
    donor_heavy: &AtomDynamics,
    donor_h: &AtomDynamics,
    acc_candidate: &AtomDynamics,
    donor_heavy_i: usize,
    donor_h_i: usize,
    acc_i: usize,
    relaxed_dist_thresh: bool,
) {
    let d_e = donor_heavy.element; // Cleans up the verbose code below.
    let a_e = acc_candidate.element;
    // todo: Take into account typical lenghs of donor and receptor; here your order isn't used.
    let dist_thresh = if d_e == Oxygen && a_e == Oxygen {
        H_BOND_O_O_DIST
    } else if d_e == Nitrogen && a_e == Nitrogen {
        H_BOND_N_N_DIST
    } else if (d_e == Oxygen && a_e == Nitrogen) || (d_e == Nitrogen && a_e == Oxygen) {
        H_BOND_O_N_DIST
    } else if (d_e == Fluorine && a_e == Nitrogen) || (d_e == Nitrogen && a_e == Fluorine) {
        H_BOND_N_F_DIST
    } else {
        H_BOND_N_S_DIST // Good enough for other combos involving S and F, for now.
    };

    let modifier = if relaxed_dist_thresh {
        H_BOND_DIST_THRESH * 2.
    } else {
        H_BOND_DIST_THRESH
    };

    let dist_thresh_min = dist_thresh - modifier;
    let dist_thresh_max = dist_thresh + modifier;

    let dist = (acc_candidate.posit - donor_heavy.posit).magnitude();
    if dist < dist_thresh_min || dist > dist_thresh_max {
        return;
    }

    let angle = {
        let donor_h = donor_h.posit - donor_heavy.posit;
        let donor_acceptor = donor_heavy.posit - acc_candidate.posit;

        donor_acceptor
            .to_normalized()
            .dot(donor_h.to_normalized())
            .acos()
    };

    if angle > H_BOND_ANGLE_THRESH {
        bonds.push(HydrogenBond {
            donor: donor_heavy_i,
            acceptor: acc_i,
            hydrogen: donor_h_i,
        });
    }
}

/// Create hydrogen bonds between all atoms in a group. See `create_hydrogen_bonds_one_way` for the more
/// flexible fn it calls.
pub(crate) fn create_hydrogen_bonds(atoms: &[AtomDynamics], waters: &[WaterMol], bonds: &[Bond]) -> Vec<HydrogenBond> {
    let indices: Vec<_> = (0..atoms.len()).collect();
    create_hydrogen_bonds_one_way(atoms, &indices, bonds, atoms, &indices, false)
}

/// Infer hydrogen bonds from a list of atoms. This takes into account bond distance between suitable
/// atom types (generally N and O; sometimes S and F), and geometry regarding the hydrogen covalently
/// bonded to the donor.
///
/// Separates donor from acceptor inputs, for use in cases like bonds between targets and ligands.
/// We indlude indices, in the case where atoms are subsets of molecules; this allows bonds indices
/// to be preserved.
fn create_hydrogen_bonds_one_way(
    atoms_donor: &[AtomDynamics],
    atoms_donor_i: &[usize],
    bonds_donor: &[Bond],
    atoms_acc: &[AtomDynamics],
    atoms_acc_i: &[usize],
    relaxed_dist_thresh: bool,
) -> Vec<HydrogenBond> {
    let mut result = Vec::new();

    // Bonds between donor and H.
    let potential_donor_bonds: Vec<&Bond> = bonds_donor
        .iter()
        .filter(|b| {
            // Maps from a global index, to a local atom from a subset.
            let atom_0 = find_atom(atoms_donor, atoms_donor_i, b.atom_0);
            let atom_1 = find_atom(atoms_donor, atoms_donor_i, b.atom_1);

            let (Some(atom_0), Some(atom_1)) = (atom_0, atom_1) else {
                eprintln!(
                    "Error! Can't find atoms from indices when making H bonds (Donor finding)"
                );
                return false;
            };

            let cfg_0_valid = h_bond_candidate_el(atom_0) && atom_1.element == Hydrogen;
            let cfg_1_valid = h_bond_candidate_el(atom_1) && atom_0.element == Hydrogen;

            cfg_0_valid || cfg_1_valid
        })
        .collect();

    let potential_acceptors: Vec<(usize, &AtomDynamics)> = atoms_acc
        .iter()
        .enumerate()
        .filter(|(_, a)| h_bond_candidate_el(a))
        .map(|(i, a)| (atoms_acc_i[i], a))
        .collect();

    for donor_bond in potential_donor_bonds {
        let donor_0 = find_atom(atoms_donor, atoms_donor_i, donor_bond.atom_0);
        let donor_1 = find_atom(atoms_donor, atoms_donor_i, donor_bond.atom_1);

        let (Some(donor_0), Some(donor_1)) = (donor_0, donor_1) else {
            eprintln!("Error! Can't find atoms from indices when making H bonds");
            continue;
        };

        let (donor_heavy, donor_h, donor_heavy_i, donor_h_i) = if donor_0.element == Hydrogen {
            (donor_1, donor_0, donor_bond.atom_1, donor_bond.atom_0)
        } else {
            (donor_0, donor_1, donor_bond.atom_0, donor_bond.atom_1)
        };

        for (acc_i, acc_candidate) in &potential_acceptors {
            hydrogen_bond_inner(
                &mut result,
                donor_heavy,
                donor_h,
                acc_candidate,
                donor_heavy_i,
                donor_h_i,
                *acc_i,
                relaxed_dist_thresh,
            );
        }
    }

    result
}

/// A helper fn. Maps from a global index, to a local atom from a subset.
fn find_atom<'a>(atoms: &'a [AtomDynamics], indices: &[usize], i_to_find: usize) -> Option<&'a AtomDynamics> {
    for (i_set, atom) in atoms.iter().enumerate() {
        if indices[i_set] == i_to_find {
            return Some(atom);
        }
    }

    None
}