use super::*;

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

        let mol = MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: atoms.to_vec(),
            atom_posits: None,
            bonds: Vec::new(),
            adjacency_list: None,
            static_: false,
            bonded_only: false,
            mol_specific_params: None,
        };

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
        let mut md = MdState::new(&dev, &cfg, &[mol], &param_set).unwrap();

        md.step(&dev, 0.001);

        // todo: Need a way to get per-type forces from the sim.

        results.push(0);
    }
}
