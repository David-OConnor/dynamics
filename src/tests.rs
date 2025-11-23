// use crate::param_inference::{find_ff_types, AmberDefSet};
// use crate::partial_charge_inference::files::{find_mol2_paths, GEOSTD_PATH};
use super::*;
use crate::{
    param_inference::{AmberDefSet, find_ff_types},
    partial_charge_inference::files::{GEOSTD_PATH, find_mol2_paths},
};
// use crate::*;

fn setup_test_pair(dist: f32) -> [AtomGeneric; 2] {
    let atom_0 = AtomGeneric {
        force_field_type: Some("ca".to_string()),
        element: Element::Carbon,
        partial_charge: Some(1.),
        ..Default::default()
    };

    let atom_1 = AtomGeneric {
        posit: Vec3::new(dist, 0.0, 0.0).into(),
        force_field_type: Some("ca".to_string()),
        element: Element::Carbon,
        partial_charge: Some(1.),
        ..Default::default()
    };

    [atom_0, atom_1]
}

/// Test that forces are approximately correct for a pair at various distances.
#[test]
fn test_forces_on_pair() {
    let dists = [2., 3., 5., 8., 10., 12.];

    let mut results: Vec<i8> = Vec::with_capacity(dists.len());

    for dist in dists {
        let atoms = setup_test_pair(dist);

        // todo

        // let mol = MolDynamics {
        //     ff_mol_type: FfMolType::SmallOrganic,
        //     atoms: atoms.to_vec(),
        //     atom_posits: None,
        //     bonds: Vec::new(),
        //     adjacency_list: None,
        //     static_: false,
        //     bonded_only: false,
        //     mol_specific_params: None,
        //     // ..Default::default()
        // };

        // Uncomment as required for validating individual processes.
        let cfg = MdConfig {
            overrides: MdOverrides {
                allow_missing_dihedral_params: true,
                skip_water: true,
                thermo_disabled: true,
                baro_disabled: true,
                ..Default::default()
            },
            max_init_relaxation_iters: None,
            ..Default::default()
        };

        let param_set = FfParamSet::new_amber().unwrap();

        // todo: Test different devices.
        let dev = ComputationDevice::Cpu;

        println!("Initializing MD state...");
        // let mut md = MdState::new(&dev, &cfg, &[mol], &param_set).unwrap();

        // md.step(&dev, 0.001);

        // todo: Need a way to get per-type forces from the sim.

        results.push(0);
    }
}

// todo: This would be a good place to run a sample of the geostd set to validate
// todo: FF types, partial charges, and FRCMOD overrides.

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
            // todo tmep
            println!("Testing atom {}", mol.atoms[i]);


            if ff_types_expected[i].to_lowercase() == "du" || mol.ident == "SME" {
                continue
            }

            assert_eq!(*ff_types_expected[i], *ff_types_actual[i]);
        }
    }
}
