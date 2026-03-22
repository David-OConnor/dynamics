//! This module contains general-purpose tests.

use std::path::Path;

use bio_files::{AtomGeneric, BondGeneric, BondType, Mol2};
use lin_alg::f32::Vec3;
use na_seq::Element;

use crate::{
    ComputationDevice, FfMolType, MdConfig, MdOverrides, MdState, MolDynamics, SimBoxInit,
    param_inference::{AmberDefSet, find_ff_types},
    params::FfParamSet,
    partial_charge_inference::files::{GEOSTD_PATH, find_mol2_paths},
};

mod spme;
mod therm_baro;

/// Build two test "ca" carbons positioned symmetrically around (30, 30, 30) in a 60 Å box.
/// The atoms are given distinct serial numbers so that bonds can refer to them.
fn setup_test_pair(dist: f32) -> [AtomGeneric; 2] {
    let c = 30.0_f32; // centre of the 60 Å box

    let atom_0 = AtomGeneric {
        serial_number: 1,
        posit: Vec3::new(c - dist / 2., c, c).into(),
        force_field_type: Some("ca".to_string()),
        element: Element::Carbon,
        partial_charge: Some(1.),
        ..Default::default()
    };

    let atom_1 = AtomGeneric {
        serial_number: 2,
        posit: Vec3::new(c + dist / 2., c, c).into(),
        force_field_type: Some("ca".to_string()),
        element: Element::Carbon,
        partial_charge: Some(1.),
        ..Default::default()
    };

    [atom_0, atom_1]
}

/// Holistic force test: LJ, bond-stretching, Coulomb short-range, and SPME long-range,
/// exercised together through the normal simulation pathway.
///
/// Layout (box 60 Å cube):
///   • mol_a — atoms 0 (left) and 1 (right), bonded; provides bonded forces.
///   • mol_b — atom 2, isolated 25 Å away in y; ensures nb_pairs > 0 so the
///     integrator step runs, and exercises non-bonded (LJ + Coulomb + SPME) forces.
///
/// The bonded pair (0–1) has 1-2 exclusions so non-bonded forces between them come
/// only from SPME reciprocal (long-range); atom 2 provides the short-range non-bonded
/// target.  All four force types are therefore active in a single step.
#[test]
fn test_forces_on_pair() {
    let dists = [2., 3., 5., 8., 10., 12.];

    // Load once outside the loop — it reads AMBER parameter files.
    let param_set = FfParamSet::new_amber().unwrap();
    let dev = ComputationDevice::Cpu;

    // 60 Å box: large enough for SPME (> 2*(cutoff+skin) = 32 Å on each side).
    // Atoms are placed near the centre so they are well inside the box.
    let cfg = MdConfig {
        sim_box: SimBoxInit::Fixed((Vec3::new(0., 0., 0.), Vec3::new(60., 60., 60.))),
        overrides: MdOverrides {
            skip_solvent: true,
            thermo_disabled: true,
            baro_disabled: true,
            ..Default::default()
        },
        max_init_relaxation_iters: None,
        ..Default::default()
    };

    for dist in dists {
        let [atom_0, atom_1] = setup_test_pair(dist);

        // mol_a: bonded pair — contributes bonded forces.
        let mol_a = MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: vec![atom_0, atom_1],
            bonds: vec![BondGeneric {
                atom_0_sn: 1,
                atom_1_sn: 2,
                bond_type: BondType::Aromatic,
            }],
            ..Default::default()
        };

        // mol_b: isolated "ca" atom — within the 12 Å cutoff so it appears in
        // nb_pairs and the integrator step proceeds.  The non-bonded forces it
        // exerts on atoms 0/1 are much smaller than the bonded restoring force.
        let c = 30.0_f32;
        let mol_b = MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: vec![AtomGeneric {
                serial_number: 3,
                posit: Vec3::new(c, c - 8., c).into(), // 8 Å away in −y from centre
                force_field_type: Some("ca".to_string()),
                element: Element::Carbon,
                partial_charge: Some(1.),
                ..Default::default()
            }],
            ..Default::default()
        };

        println!("Initializing MD state for dist={dist:.1} Å…");
        let mut md = MdState::new(&dev, &cfg, &[mol_a, mol_b], &param_set).unwrap();

        // One step: computes all forces (bonded + LJ + Coulomb SR + SPME).
        md.step(&dev, 0.001, None);

        let f0 = md.atoms[0].force;
        let f1 = md.atoms[1].force;
        let f2 = md.atoms[2].force;

        println!(
            "  dist={dist:.1}Å  F[0]=({:.3},{:.3},{:.3})  \
             F[1]=({:.3},{:.3},{:.3})  F[2]=({:.3},{:.3},{:.3}) kcal/(mol·Å)",
            f0.x, f0.y, f0.z, f1.x, f1.y, f1.z, f2.x, f2.y, f2.z,
        );

        // All force components must be finite.
        assert!(
            f0.x.is_finite() && f0.y.is_finite() && f0.z.is_finite(),
            "dist={dist}: atom 0 force non-finite: {f0:?}"
        );
        assert!(
            f1.x.is_finite() && f1.y.is_finite() && f1.z.is_finite(),
            "dist={dist}: atom 1 force non-finite: {f1:?}"
        );
        assert!(
            f2.x.is_finite() && f2.y.is_finite() && f2.z.is_finite(),
            "dist={dist}: atom 2 force non-finite: {f2:?}"
        );

        // Forces must be nonzero.
        assert!(f0.magnitude() > 0.0, "dist={dist}: atom 0 force is zero");
        assert!(f1.magnitude() > 0.0, "dist={dist}: atom 1 force is zero");
        assert!(f2.magnitude() > 0.0, "dist={dist}: atom 2 force is zero");

        // Momentum conservation: sum of all forces ≈ 0.
        // Exact equality holds for bonded forces; SPME has small image corrections (~2%).
        let f_sum = f0 + f1 + f2;
        let f_max = f0.magnitude().max(f1.magnitude()).max(f2.magnitude());
        assert!(
            f_sum.magnitude() < 0.05 * f_max,
            "dist={dist}: momentum not conserved: |ΣF|={:.4e}, max|F|={f_max:.4e}",
            f_sum.magnitude()
        );

        // Bonded restoring force: all test distances are beyond the ca–ca equilibrium
        // (~1.4 Å), so the bond pulls atom 0 toward atom 1 (+x) and vice versa.
        // Non-bonded forces from atom 2 are ~25 Å away and negligible in comparison.
        assert!(
            f0.x > 0.0,
            "dist={dist}: F[0].x should be +ve (bond restoring), got {:.4}",
            f0.x
        );
        assert!(
            f1.x < 0.0,
            "dist={dist}: F[1].x should be −ve (bond restoring), got {:.4}",
            f1.x
        );
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
