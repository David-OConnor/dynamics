//! Regression coverage for the common backend interface and molecule-local resolution.

use bio_files::{
    AtomGeneric, BondGeneric, BondType,
    md_params::{BondStretchingParams, ForceFieldParams, LjParams, MassParams},
};
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::{
    ComputationDevice, ForceFieldFamily, MdConfig, MdState, MolDynamics, Scale14, Solvent,
    WaterModel, gromacs,
    params::{
        FfParamSet, NonBondedRules, ParameterizationOptions, Scale14ByMolType, TableForceField,
    },
};

fn table_set() -> FfParamSet {
    let mut parameters = ForceFieldParams::default();
    for (name, mass) in [("site_a", 12.), ("site_b", 1.)] {
        parameters.mass.insert(
            name.into(),
            MassParams {
                atom_type: name.into(),
                mass,
                comment: None,
            },
        );
        parameters.lennard_jones.insert(
            name.into(),
            LjParams {
                atom_type: name.into(),
                sigma: 3.,
                eps: 0.1,
            },
        );
    }
    parameters.bond.insert(
        ("site_a".into(), "site_b".into()),
        BondStretchingParams {
            atom_types: ("site_a".into(), "site_b".into()),
            k_b: 100.,
            r_0: 1.1,
            comment: None,
        },
    );
    FfParamSet::custom(
        TableForceField {
            name: "test classical tables".into(),
            parameters,
        },
        NonBondedRules {
            scale_14: Scale14ByMolType::uniform(Scale14 {
                lj: 0.7,
                coulomb: 0.6,
            }),
            ..Default::default()
        },
        WaterModel::OPC,
    )
}

fn two_atoms(offset: f64) -> MolDynamics {
    MolDynamics {
        atoms: vec![
            AtomGeneric {
                serial_number: 31,
                element: Element::Carbon,
                posit: Vec3::new(offset, 0., 0.),
                force_field_type: Some("site_a".into()),
                partial_charge: Some(0.1),
                ..Default::default()
            },
            AtomGeneric {
                serial_number: 73,
                element: Element::Hydrogen,
                posit: Vec3::new(offset + 1.1, 0., 0.),
                force_field_type: Some("site_b".into()),
                partial_charge: Some(-0.1),
                ..Default::default()
            },
        ],
        bonds: vec![BondGeneric {
            atom_0_sn: 31,
            atom_1_sn: 73,
            bond_type: BondType::Single,
        }],
        ..Default::default()
    }
}

fn methane() -> MolDynamics {
    let mut atoms = vec![AtomGeneric {
        serial_number: 100,
        element: Element::Carbon,
        ..Default::default()
    }];
    let mut bonds = Vec::new();
    for (i, (x, y, z)) in [(1., 1., 1.), (1., -1., -1.), (-1., 1., -1.), (-1., -1., 1.)]
        .into_iter()
        .enumerate()
    {
        let serial = 200 + i as u32 * 17;
        atoms.push(AtomGeneric {
            serial_number: serial,
            element: Element::Hydrogen,
            posit: Vec3::new(x, y, z) * 0.63,
            ..Default::default()
        });
        bonds.push(BondGeneric {
            atom_0_sn: 100,
            atom_1_sn: serial,
            bond_type: BondType::Single,
        });
    }
    MolDynamics {
        atoms,
        bonds,
        ..Default::default()
    }
}

#[test]
fn every_builtin_loads_embedded_data() {
    for family in [
        ForceFieldFamily::Amber,
        ForceFieldFamily::Charmm36,
        ForceFieldFamily::Opls,
    ] {
        let set = FfParamSet::from_family(family).unwrap();
        assert_eq!(set.family, family);
        assert!(!set.backend.name().is_empty());
    }
}

#[test]
fn parameter_resolution_does_not_require_charges() {
    let set = table_set();
    let mut input = two_atoms(0.);
    for atom in &mut input.atoms {
        atom.partial_charge = None;
    }
    let typed = set
        .assign_atom_types(&input, &ParameterizationOptions::local_only())
        .unwrap();
    assert_eq!(typed.atom_types(), &["site_a", "site_b"]);
    let parameters = typed.assign_parameters(&set.nonbonded_rules).unwrap();
    assert_eq!(parameters.explicit.bonds.len(), 1);
    assert!(typed.assign_partial_charges().is_err());
    assert!(input.atoms.iter().all(|atom| atom.partial_charge.is_none()));
}

#[test]
fn amber_stages_are_independent_and_charge_indices_are_serial_safe() {
    let set = FfParamSet::new_amber().unwrap();
    let input = methane();
    let typed = set
        .assign_atom_types(&input, &ParameterizationOptions::local_only())
        .unwrap();
    assert_eq!(typed.atom_types()[0], "c3");
    assert_eq!(
        typed
            .assign_parameters(&set.nonbonded_rules)
            .unwrap()
            .explicit
            .bonds
            .len(),
        4
    );
    assert!(
        input
            .atoms
            .iter()
            .all(|a| a.partial_charge.is_none() && a.force_field_type.is_none())
    );
    let charges = typed.assign_partial_charges().unwrap();
    assert!(charges.method.contains("GeoStd"));
    assert!(charges.values.iter().all(|q| q.is_finite()));
    let mut contiguous = input.clone();
    for (i, atom) in contiguous.atoms.iter_mut().enumerate() {
        atom.serial_number = i as u32 + 1;
    }
    for (i, bond) in contiguous.bonds.iter_mut().enumerate() {
        bond.atom_0_sn = 1;
        bond.atom_1_sn = i as u32 + 2;
    }
    let other = set
        .assign_atom_types(&contiguous, &ParameterizationOptions::local_only())
        .unwrap()
        .assign_partial_charges()
        .unwrap();
    assert_eq!(charges.values, other.values);
}

#[test]
fn retyping_does_not_force_charge_recalculation() {
    let set = FfParamSet::new_amber().unwrap();
    let mut input = methane();
    for (i, atom) in input.atoms.iter_mut().enumerate() {
        atom.force_field_type = Some(if i == 0 { "c3" } else { "hc" }.into());
        atom.partial_charge = Some(0.123);
    }
    let options = ParameterizationOptions {
        reassign_atom_types: true,
        ..ParameterizationOptions::local_only()
    };
    let typed = set.assign_atom_types(&input, &options).unwrap();
    assert_eq!(
        typed.assign_partial_charges().unwrap().values,
        vec![0.123; 5]
    );
}

#[test]
fn opls_ligand_uses_embedded_types_and_charges_offline() {
    let set = FfParamSet::new_opls().unwrap();
    let input = methane();
    let assigned = set
        .assign(&input, &ParameterizationOptions::local_only())
        .unwrap();
    assert!(
        assigned
            .atom_types
            .iter()
            .all(|name| name.starts_with("opls_"))
    );
    assert!(assigned.partial_charges.method.contains("OPLS"));
    assert!(assigned.partial_charges.values.iter().sum::<f32>().abs() < 0.005);
    assert_eq!(assigned.parameters.explicit.bonds.len(), 4);
}

#[test]
fn supplied_charge_policy_is_independent_of_the_typing_backend() {
    let mut input = methane();
    for atom in &mut input.atoms {
        atom.partial_charge = Some(0.);
    }
    let options = ParameterizationOptions {
        charge_policy: crate::params::ChargePolicy::Supplied,
        ..ParameterizationOptions::local_only()
    };
    for set in [
        FfParamSet::new_amber().unwrap(),
        FfParamSet::new_opls().unwrap(),
    ] {
        let assignment = set.assign(&input, &options).unwrap();
        assert_eq!(
            assignment.partial_charges.values,
            vec![0.; input.atoms.len()]
        );
        assert_eq!(assignment.partial_charges.method, "caller-supplied charges");
        assert_eq!(assignment.parameters.explicit.bonds.len(), 4);
    }
}

#[test]
fn strict_table_backend_rejects_missing_lj_without_amber_fallback() {
    let mut set = table_set();
    let table = (&mut *set.backend as &mut dyn std::any::Any)
        .downcast_mut::<TableForceField>()
        .unwrap();
    table.parameters.lennard_jones.remove("site_b");
    assert!(
        set.parameterize(&two_atoms(0.))
            .unwrap_err()
            .descrip
            .contains("Missing LJ")
    );
}

#[test]
fn custom_backend_native_and_export_keep_overrides_local() {
    let set = table_set();
    let first = two_atoms(0.);
    let mut second = two_atoms(5.);
    let mut overrides = ForceFieldParams::default();
    overrides.bond.insert(
        ("site_a".into(), "site_b".into()),
        BondStretchingParams {
            atom_types: ("site_a".into(), "site_b".into()),
            k_b: 250.,
            r_0: 1.1,
            comment: None,
        },
    );
    second.mol_specific_params = Some(overrides);
    let cfg = MdConfig {
        solvent: Solvent::None,
        max_init_relaxation_iters: None,
        hydrogen_constraint: crate::HydrogenConstraint::Flexible,
        ..MdConfig::for_family(ForceFieldFamily::Custom)
    };
    let (state, _) = MdState::new(
        &ComputationDevice::Cpu,
        &cfg,
        &[first.clone(), second.clone()],
        &set,
    )
    .unwrap();
    assert_eq!(state.force_field_params.bond_stretching[&(0, 1)].k_b, 100.);
    assert_eq!(state.force_field_params.bond_stretching[&(2, 3)].k_b, 250.);
    let prepared = vec![
        ("ONE".into(), set.parameterize(&first).unwrap(), 1),
        ("TWO".into(), set.parameterize(&second).unwrap(), 1),
    ];
    let text = gromacs::make_topology(&prepared, &set, None, false).unwrap();
    assert!(text.contains("[ bonds ]"));
    let defaults = gromacs::topology_config(&set, None).defaults();
    assert!((defaults.1 - 0.7).abs() < 1e-6 && (defaults.2 - 0.6).abs() < 1e-6);
}

#[test]
fn stale_adjacency_cache_is_rejected_before_assignment() {
    let set = table_set();
    let mut input = two_atoms(0.);
    input.adjacency_list = Some(vec![vec![], vec![]]);
    assert!(
        set.parameterize(&input)
            .unwrap_err()
            .descrip
            .contains("Adjacency cache")
    );
}
