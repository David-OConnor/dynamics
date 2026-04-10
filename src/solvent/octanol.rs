//! Octanol solvent helpers.
//!
//! This defines the bonded topology and partial charges for 1-octanol, and provides a loader
//! that stamps pre-equilibrated `.gro` coordinates/velocities onto that molecule template.

use std::collections::BTreeMap;

use bio_files::{AtomGeneric, BondGeneric, BondType, gromacs::gro::Gro};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::Element::{Carbon, Hydrogen, Oxygen};

use crate::{FfMolType, MolDynamics, ParamError, util::build_adjacency_list};

#[allow(clippy::doc_lazy_continuation)]
/// Using PubChem data as a reference. Partial charges are computed using ORCA. We use this input:
/// `! HF 6-31G* Opt TightSCF TightOpt RESP`
///
/// This may be less accurate than more modern basis sets and MBIS charges, but may perform
/// better with Amber MD force fields.
pub fn make_octanol() -> MolDynamics {
    // We set FF type and partial charge on each atom explicitly so MD uses this vetted octanol
    // charge model directly instead of re-inferring it from geometry at runtime.
    #[rustfmt::skip]
    let atoms = [
        (Oxygen,   Vec3::new( 4.9442, -0.3976, -0.0463), "oh", -0.730420), //  1
        (Carbon,   Vec3::new( 0.0273, -0.3598,  0.0738), "c3",  0.154597), //  2
        (Carbon,   Vec3::new(-1.2646,  0.4624,  0.0583), "c3", -0.078612), //  3
        (Carbon,   Vec3::new( 1.2942,  0.4958,  0.0044), "c3", -0.034888), //  4
        (Carbon,   Vec3::new(-2.5319, -0.3950,  0.0532), "c3", -0.011862), //  5
        (Carbon,   Vec3::new( 2.5500, -0.3747, -0.0005), "c3", -0.092695), //  6
        (Carbon,   Vec3::new(-3.7874,  0.4741, -0.0356), "c3",  0.208880), //  7
        (Carbon,   Vec3::new( 3.8175,  0.4679, -0.0278), "c3",  0.388562), //  8
        (Carbon,   Vec3::new(-5.0492, -0.3730, -0.0796), "c3", -0.327900), //  9
        (Hydrogen, Vec3::new( 0.0503, -0.9725,  0.9835), "hc", -0.031046), // 10
        (Hydrogen, Vec3::new( 0.0176, -1.0547, -0.7753), "hc", -0.034188), // 11
        (Hydrogen, Vec3::new(-1.2784,  1.1276,  0.9306), "hc",  0.010158), // 12
        (Hydrogen, Vec3::new(-1.2632,  1.1063, -0.8301), "hc",  0.004831), // 13
        (Hydrogen, Vec3::new( 1.3198,  1.1783,  0.8623), "hc", -0.008392), // 14
        (Hydrogen, Vec3::new( 1.2714,  1.1141, -0.9010), "hc",  0.001771), // 15
        (Hydrogen, Vec3::new(-2.5656, -1.0062,  0.9629), "hc",  0.001025), // 16
        (Hydrogen, Vec3::new(-2.5025, -1.0845, -0.7992), "hc",  0.001729), // 17
        (Hydrogen, Vec3::new( 2.5379, -1.0481, -0.8666), "hc",  0.009347), // 18
        (Hydrogen, Vec3::new( 2.5665, -1.0240,  0.8838), "hc",  0.029857), // 19
        (Hydrogen, Vec3::new(-3.8347,  1.1483,  0.8274), "hc", -0.034454), // 20
        (Hydrogen, Vec3::new(-3.7454,  1.0998, -0.9347), "hc", -0.033270), // 21
        (Hydrogen, Vec3::new( 3.8553,  1.1097, -0.9137), "h1", -0.055477), // 22
        (Hydrogen, Vec3::new( 3.8880,  1.1005,  0.8622), "h1",  0.018941), // 23
        (Hydrogen, Vec3::new(-5.1386, -0.9902,  0.8201), "hc",  0.072069), // 24
        (Hydrogen, Vec3::new(-5.0475, -1.0346, -0.9517), "hc",  0.071193), // 25
        (Hydrogen, Vec3::new(-5.9342,  0.2680, -0.1411), "hc",  0.075105), // 26
        (Hydrogen, Vec3::new( 4.8901, -0.9332, -0.8561), "ho",  0.425140), // 27
    ];

    #[rustfmt::skip]
    let bonds = [
        (1, 8),
        (1, 27),
        (2, 3),
        (2, 4),
        (2, 10),
        (2, 11),
        (3, 5),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 14),
        (4, 15),
        (5, 7),
        (5, 16),
        (5, 17),
        (6, 8),
        (6, 18),
        (6, 19),
        (7, 9),
        (7, 20),
        (7, 21),
        (8, 22),
        (8, 23),
        (9, 24),
        (9, 25),
        (9, 26),
    ];

    let atoms: Vec<_> = atoms
        .into_iter()
        .enumerate()
        .map(|(i, (element, posit, ff_name, q))| AtomGeneric {
            serial_number: i as u32 + 1,
            posit,
            element,
            type_in_res_general: Some(ff_name.to_string()),
            force_field_type: Some(ff_name.to_string()),
            partial_charge: Some(q),
            hetero: true,
            ..Default::default()
        })
        .collect();

    let bonds: Vec<_> = bonds
        .into_iter()
        .map(|(atom_0_sn, atom_1_sn)| BondGeneric {
            bond_type: BondType::Single,
            atom_0_sn,
            atom_1_sn,
        })
        .collect();

    MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        adjacency_list: build_adjacency_list(&atoms, &bonds).ok(),
        atoms,
        bonds,
        ..Default::default()
    }
}

pub(crate) fn octanol_mols_from_gro(gro: &Gro) -> Result<Vec<MolDynamics>, ParamError> {
    const NM_TO_ANGSTROM: f64 = 10.0;

    let template = make_octanol();
    let template_atom_types: Vec<_> = template
        .atoms
        .iter()
        .map(|atom| {
            atom.force_field_type
                .as_deref()
                .unwrap_or_default()
                .to_string()
        })
        .collect();

    let mut octanol_by_mol_id: BTreeMap<u32, Vec<&bio_files::gromacs::gro::AtomGro>> =
        BTreeMap::new();

    for atom in &gro.atoms {
        if atom.mol_name == "octan" {
            octanol_by_mol_id.entry(atom.mol_id).or_default().push(atom);
        }
    }

    let mut mols = Vec::with_capacity(octanol_by_mol_id.len());

    for (mol_id, gro_atoms) in octanol_by_mol_id {
        if gro_atoms.len() != template.atoms.len() {
            return Err(ParamError::new(&format!(
                "Octanol molecule {mol_id} has {} atoms; expected {}.",
                gro_atoms.len(),
                template.atoms.len(),
            )));
        }

        let mut mol = template.clone();
        let mut atom_posits = Vec::with_capacity(gro_atoms.len());
        let mut atom_velocities = Vec::with_capacity(gro_atoms.len());

        for (i, (atom, gro_atom)) in mol.atoms.iter_mut().zip(gro_atoms.iter()).enumerate() {
            if gro_atom.atom_type != template_atom_types[i] {
                return Err(ParamError::new(&format!(
                    "Octanol molecule {mol_id} atom order does not match the template."
                )));
            }

            let Some(vel) = gro_atom.velocity else {
                return Err(ParamError::new(&format!(
                    "Missing velocity on octanol atom {} in GRO template.",
                    gro_atom.serial_number
                )));
            };

            let posit = gro_atom.posit * NM_TO_ANGSTROM;
            atom.posit = posit;
            atom_posits.push(posit);

            let vel: Vec3F32 = vel.into();
            atom_velocities.push(vel * NM_TO_ANGSTROM as f32);
        }

        mol.atom_posits = Some(atom_posits);
        mol.atom_init_velocities = Some(atom_velocities);
        mols.push(mol);
    }

    Ok(mols)
}
