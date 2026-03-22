//! This module contains general-purpose tests.

use std::path::Path;

use bio_files::Mol2;

use crate::{
    param_inference::{AmberDefSet, find_ff_types},
    partial_charge_inference::files::{GEOSTD_PATH, find_mol2_paths},
};

mod forces;
mod md_sim;
mod spme;
mod therm_baro;

#[test]
fn test_ff_types_geostd() {
    for (i, path) in find_mol2_paths(Path::new(GEOSTD_PATH))
        .unwrap()
        .iter()
        .enumerate()
    {
        let mol = Mol2::load(&path).unwrap();
        println!("\nTesting FF types on mol: {:?}  #: {}", mol.ident, i);

        let ff_types_expected: Vec<_> = mol
            .atoms
            .iter()
            .map(|a| a.force_field_type.as_ref().unwrap())
            .collect();

        let defs = AmberDefSet::new().unwrap();
        let ff_types_actual = find_ff_types(&mol.atoms, &mol.bonds, &defs);

        for i in 0..mol.atoms.len() {
            if (ff_types_expected[i] == "cc" && ff_types_actual[i] == "cd")
                || (ff_types_expected[i] == "cd" && ff_types_actual[i] == "cc")
                || (ff_types_expected[i] == "nd" && ff_types_actual[i] == "nc")
                || (ff_types_expected[i] == "nc" && ff_types_actual[i] == "nd")
            {
                println!("cc/cd exception");
                continue;
            }

            if ff_types_expected[i] == "c2" && ff_types_actual[i] == "cc" {
                println!("c2/cc exception");
                continue;
            }

            if ff_types_expected[i] == "cp" && ff_types_actual[i] == "ca" {
                println!("cp/ca exception");
                continue;
            }

            if ff_types_expected[i] == "cf" && ff_types_actual[i] == "ce" {
                println!("cf/ce exception");
                continue;
            }

            if ff_types_expected[i] == "nv" && ff_types_actual[i] == "n8" {
                println!("nv/n8 exception");
                continue;
            }

            if ff_types_expected[i].to_lowercase() == "du" || mol.ident == "SME" {
                continue;
            }

            assert_eq!(*ff_types_expected[i], *ff_types_actual[i]);
        }
    }
}
