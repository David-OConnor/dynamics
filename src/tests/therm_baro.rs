//! Tests for temperature, pressure, and kinetic energy measurement.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use lin_alg::f32::Vec3;

use crate::{
    AtomDynamics, MdState, NATIVE_TO_KCAL, Solvent,
    barostat::BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED,
    solvent::{H_MASS, MASS_WATER_MOL, O_MASS, WaterMol},
    thermostat::GAS_CONST_R,
};

fn assert_close(got: f64, expected: f64, rel_tol: f64, label: &str) {
    if expected.abs() < 1e-12 {
        assert!(got.abs() < 1e-9, "{label}: expected ≈0, got {got:.6e}");
        return;
    }
    let rel = ((got - expected) / expected).abs();
    assert!(
        rel < rel_tol,
        "{label}: got {got:.6e}, expected {expected:.6e}, rel_err={rel:.4} (tol={rel_tol})"
    );
}

// ── kinetic energy ────────────────────────────────────────────────────────────

#[test]
fn test_measure_kinetic_energy_atoms() {
    let mut md = MdState::default();
    md.atoms = vec![
        AtomDynamics {
            mass: 12.0,
            vel: Vec3::new(3.0, 0.0, 0.0),
            ..Default::default()
        },
        AtomDynamics {
            mass: 1.0,
            vel: Vec3::new(0.0, 2.0, 0.0),
            ..Default::default()
        },
    ];
    let expected = 0.5 * (12.0 * 9.0 + 1.0 * 4.0) * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy();
    assert_close(got, expected, 1e-6, "KE from two atoms");
}

#[test]
fn test_measure_kinetic_energy_excludes_static() {
    let mut md = MdState::default();
    md.atoms = vec![
        AtomDynamics {
            mass: 12.0,
            vel: Vec3::new(1.0, 0.0, 0.0),
            static_: false,
            ..Default::default()
        },
        AtomDynamics {
            mass: 12.0,
            vel: Vec3::new(10.0, 10.0, 10.0),
            static_: true,
            ..Default::default()
        },
    ];
    let expected = 0.5 * 12.0 * 1.0 * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy();
    assert_close(got, expected, 1e-6, "KE excludes static atom");
}

#[test]
fn test_measure_kinetic_energy_includes_water() {
    let mut md = MdState::default();
    let v_o = Vec3::new(2.0, 0.0, 0.0);
    let v_h = Vec3::new(0.0, 1.0, 0.0);

    md.water = vec![WaterMol {
        o: AtomDynamics {
            mass: O_MASS,
            vel: v_o,
            ..Default::default()
        },
        h0: AtomDynamics {
            mass: H_MASS,
            vel: v_h,
            ..Default::default()
        },
        h1: AtomDynamics {
            mass: H_MASS,
            vel: v_h,
            ..Default::default()
        },
        m: AtomDynamics::default(),
    }];

    let expected = 0.5
        * (O_MASS as f64 * 4.0 + H_MASS as f64 * 1.0 + H_MASS as f64 * 1.0)
        * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy();
    assert_close(got, expected, 1e-5, "KE from water molecule");
}

/// Rigid-body translation: all water sites share the same velocity, so
/// translational KE = 0.5 * M_water * v_com².
#[test]
fn test_measure_kinetic_energy_translational_rigid_translation() {
    let mut md = MdState::default();
    let v_com = Vec3::new(3.0, 0.0, 0.0);

    md.water = vec![WaterMol {
        o: AtomDynamics {
            mass: O_MASS,
            vel: v_com,
            ..Default::default()
        },
        h0: AtomDynamics {
            mass: H_MASS,
            vel: v_com,
            ..Default::default()
        },
        h1: AtomDynamics {
            mass: H_MASS,
            vel: v_com,
            ..Default::default()
        },
        m: AtomDynamics::default(),
    }];

    let expected = 0.5 * MASS_WATER_MOL as f64 * 9.0 * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy_translational();
    assert_close(got, expected, 1e-5, "translational KE — rigid translation");
}

/// Pure internal motion (COM stationary): full KE > 0 but translational KE ≈ 0.
#[test]
fn test_measure_kinetic_energy_translational_excludes_rotation() {
    let mut md = MdState::default();
    let v_h = Vec3::new(0.0, 1.0, 0.0);
    let v_o = Vec3::new(0.0, -2.0 * H_MASS / O_MASS, 0.0); // COM vel = 0

    md.water = vec![WaterMol {
        o: AtomDynamics {
            mass: O_MASS,
            vel: v_o,
            ..Default::default()
        },
        h0: AtomDynamics {
            mass: H_MASS,
            vel: v_h,
            ..Default::default()
        },
        h1: AtomDynamics {
            mass: H_MASS,
            vel: v_h,
            ..Default::default()
        },
        m: AtomDynamics::default(),
    }];

    let ke_full = md.measure_kinetic_energy();
    let ke_trans = md.measure_kinetic_energy_translational();

    assert!(ke_full > 1e-10, "full KE must be nonzero");
    assert!(
        ke_trans < 1e-5 * ke_full,
        "translational KE should be ~0 for pure internal motion; \
         got ke_trans={ke_trans:.4e}, ke_full={ke_full:.4e}"
    );
}

// ── temperature ───────────────────────────────────────────────────────────────

#[test]
fn test_measure_temperature_roundtrip_300k() {
    let mut md = MdState::default();
    let dof = 9;
    let t_target = 300.0;
    md.kinetic_energy = 0.5 * dof as f64 * GAS_CONST_R * t_target;
    md.thermo_dof = dof;
    let got = md.measure_temperature();
    assert_close(got, t_target, 1e-9, "round-trip temperature at 300 K");
}

// ── pressure: integration tests ───────────────────────────────────────────────
//
// These tests run a real MdState::new + step and read Snapshot::pressure.
// They are designed to catch broken virial accumulation, wrong unit conversions,
// and other pipeline bugs that formula-only tests cannot detect.

/// Ideal-gas baseline: no forces → W = 0 → P = 2·KE / (3·V) · BAR_CONST.
/// Tests the KE → pressure path end-to-end including the snapshot write.
/// Will catch bugs in NATIVE_TO_KCAL conversion, the KE loop, or BAR_CONST.
#[test]
fn test_pressure_no_forces() {
    use bio_files::{AtomGeneric, BondGeneric, BondType};
    use na_seq::Element;

    use crate::{
        ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MolDynamics, SimBoxInit,
        params::FfParamSet,
    };

    let param_set = FfParamSet::new_amber().unwrap();
    let dev = ComputationDevice::Cpu;

    let v0 = Vec3::new(2.0, 0.0, 0.0);
    let v1 = Vec3::new(-1.0, 1.0, 0.0);
    let v2 = Vec3::new(0.0, 0.0, 3.0);
    let c = 30.0_f32;

    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity { thermostat: None },
        sim_box: SimBoxInit::Fixed((Vec3::new(0., 0., 0.), Vec3::new(60., 60., 60.))),
        overrides: MdOverrides {
            skip_solvent: true,
            thermo_disabled: true,
            baro_disabled: true,
            bonded_disabled: true,
            coulomb_disabled: true,
            lj_disabled: true,
            long_range_recip_disabled: true,
            ..Default::default()
        },
        max_init_relaxation_iters: None,
        ..Default::default()
    };

    // A bonded pair (1-2 excluded from nb_pairs) plus one isolated atom to satisfy the
    // nb_pairs > 0 guard in step().
    let mol_a = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![
            AtomGeneric {
                serial_number: 1,
                posit: Vec3::new(c - 2.5, c, c).into(),
                force_field_type: Some("ca".to_string()),
                element: Element::Carbon,
                partial_charge: Some(0.0),
                ..Default::default()
            },
            AtomGeneric {
                serial_number: 2,
                posit: Vec3::new(c + 2.5, c, c).into(),
                force_field_type: Some("ca".to_string()),
                element: Element::Carbon,
                partial_charge: Some(0.0),
                ..Default::default()
            },
        ],
        atom_init_velocities: Some(vec![v0, v1]),
        bonds: vec![BondGeneric {
            atom_0_sn: 1,
            atom_1_sn: 2,
            bond_type: BondType::Aromatic,
        }],
        ..Default::default()
    };
    let mol_b = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![AtomGeneric {
            serial_number: 3,
            posit: Vec3::new(c, c - 8.0, c).into(),
            force_field_type: Some("ca".to_string()),
            element: Element::Carbon,
            partial_charge: Some(0.0),
            ..Default::default()
        }],
        atom_init_velocities: Some(vec![v2]),
        ..Default::default()
    };

    let mut md = MdState::new(&dev, &cfg, &[mol_a, mol_b], &param_set).unwrap();
    md.step(&dev, 0.001, None);

    let snap = md.snapshots.last().expect("snapshot must exist");

    // Expected: pure ideal-gas, W = 0. Compute KE from atom masses + initial velocities.
    let ke: f64 = md
        .atoms
        .iter()
        .zip([v0, v1, v2])
        .map(|(a, v)| 0.5 * a.mass as f64 * (v.x * v.x + v.y * v.y + v.z * v.z) as f64)
        .sum::<f64>()
        * NATIVE_TO_KCAL as f64;
    let ext = md.cell.extent;
    let vol = ext.x as f64 * ext.y as f64 * ext.z as f64;
    let expected = 2.0 * ke / (3.0 * vol) * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED;

    assert_close(snap.pressure as f64, expected, 1e-4, "no-force pressure");
}

/// Virial test: two non-bonded atoms with only LJ active, zero velocity.
/// P = W_LJ / (3·V) · BAR_CONST.
///
/// The expected virial is computed from `force_e_lj` directly (already validated
/// in test_lennard_jones), using the actual sigma/eps read from the initialized
/// atoms.  The simulation computes it through a completely different path:
/// f_nonbonded_cpu → barostat.virial → to_kcal_mol() → measure_pressure().
///
/// A unit-conversion bug or accumulation error in the virial path will make
/// these diverge.
#[test]
fn test_pressure_lj_virial() {
    use bio_files::AtomGeneric;
    use na_seq::Element;

    use crate::{
        ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MolDynamics, SimBoxInit,
        forces::force_e_lj, params::FfParamSet,
    };

    let param_set = FfParamSet::new_amber().unwrap();
    let dev = ComputationDevice::Cpu;

    let r = 4.0_f32; // Å — repulsive regime for ca-ca LJ
    let c = 30.0_f32;

    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity { thermostat: None },
        sim_box: SimBoxInit::Fixed((Vec3::new(0., 0., 0.), Vec3::new(60., 60., 60.))),
        overrides: MdOverrides {
            skip_solvent: true,
            thermo_disabled: true,
            baro_disabled: true,
            bonded_disabled: true,
            coulomb_disabled: true,
            lj_disabled: false, // LJ ON — this is what we're testing
            long_range_recip_disabled: true,
            ..Default::default()
        },
        max_init_relaxation_iters: None,
        ..Default::default()
    };

    // Two isolated single-atom molecules — no bond, no 1-2 exclusion, so they
    // interact via LJ.  Zero initial velocities so KE = 0 at pressure measurement.
    let mol_a = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![AtomGeneric {
            serial_number: 1,
            posit: Vec3::new(c - r / 2., c, c).into(),
            force_field_type: Some("ca".to_string()),
            element: Element::Carbon,
            partial_charge: Some(0.0),
            ..Default::default()
        }],
        ..Default::default()
    };
    let mol_b = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![AtomGeneric {
            serial_number: 2,
            posit: Vec3::new(c + r / 2., c, c).into(),
            force_field_type: Some("ca".to_string()),
            element: Element::Carbon,
            partial_charge: Some(0.0),
            ..Default::default()
        }],
        ..Default::default()
    };

    let mut md = MdState::new(&dev, &cfg, &[mol_a, mol_b], &param_set).unwrap();
    md.step(&dev, 0.001, None);

    let snap = md.snapshots.last().expect("snapshot must exist");

    // Read the actual combined LJ params the simulation uses.
    let a0 = &md.atoms[0];
    let a1 = &md.atoms[1];
    let sigma = 0.5 * (a0.lj_sigma + a1.lj_sigma);
    let eps = (a0.lj_eps * a1.lj_eps).sqrt();

    // force_e_lj(dir, inv_dist, sigma, eps): dir is tgt-src unit vector.
    let dir = Vec3::new(1.0, 0.0, 0.0); // atom 1 is in +x from atom 0
    let (f_on_tgt, _energy) = force_e_lj(dir, 1.0 / r, sigma, eps);

    // Virial for this pair: diff · F_tgt, where diff = (r, 0, 0).
    // KE = 0, so P = W / (3·V) · BAR_CONST.
    let virial_expected = (r * f_on_tgt.x) as f64;

    let ext = md.cell.extent;
    let vol = ext.x as f64 * ext.y as f64 * ext.z as f64;
    let p_expected = virial_expected / (3.0 * vol) * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED;

    println!(
        "sigma={sigma:.4} eps={eps:.4} r={r} F_x={:.6} virial={virial_expected:.6} \
         P_expected={p_expected:.4} bar  P_snapshot={:.4} bar",
        f_on_tgt.x, snap.pressure
    );

    assert_close(
        snap.pressure as f64,
        p_expected,
        0.01, // 1% tolerance
        "LJ virial pressure",
    );
}

// ── water at 1 bar ────────────────────────────────────────────────────────────

/// Runs a small pure-water simulation at 310 K and checks that the
/// time-averaged pressure is close to 1 bar. Using default solvent settings, the simulation
/// will initialize water to a standard pressure of 1 bar. It sets the water mol count, and
/// sim box size IOC this.
///
/// The tolerance is wide (±2000 bar) because instantaneous pressure in a small
/// box has large fluctuations — this is normal in MD.  What this test catches
/// is the class of bug where virials are wrong by a large factor (e.g. 418×),
/// which would push the average to tens of thousands of bar.
///
/// The box is sized to give roughly water density (~100 molecules in 30 Å cube
/// ≈ 0.055 mol/Å³ × 18 ≈ 1 g/cm³).  A thermostat keeps the temperature near
/// 300 K.  The barostat is intentionally left off so the measured pressure is
/// independent of any barostat target.
#[test]
fn test_pressure_water_sim_1bar() {
    use crate::{
        ComputationDevice, MdConfig, MdOverrides, MdState, SimBoxInit, integrate::Integrator,
        params::FfParamSet,
    };

    let param_set = FfParamSet::new_amber().unwrap();

    #[cfg(feature = "cuda")]
    let dev = {
        let stream = {
            let ctx = CudaContext::new(0).unwrap();
            ctx.default_stream()
        };

        ComputationDevice::Gpu(stream)
    };

    let dev = ComputationDevice::Cpu;

    // 30 Å cube holds ~100 OPC water molecules at roughly water density.
    // (30³ = 27 000 Å³; water: 1 molecule per ~30 Å³ ≈ 900 molecules — we
    //  deliberately use fewer for speed while still exercising the virial path.)

    // todo: It may be worth trying two setups: One with a fixed number of water molecules and fixed
    // todo

    let cfg_auto_water_count = MdConfig {
        // integrator: Integrator::LangevinMiddle { gamma: 1.0 },
        integrator: Integrator::VerletVelocity { thermostat: None },
        sim_box: SimBoxInit::Fixed((Vec3::new(0., 0., 0.), Vec3::new(35., 35., 35.))),
        temp_target: 310.,
        overrides: MdOverrides {
            baro_disabled: true, // measure pressure; don't control it
            ..Default::default()
        },
        max_init_relaxation_iters: None,
        ..Default::default()
    };

    let cfg_fixed_water_count = MdConfig {
        // todo: QC this is the right number of water mols for a sim box of that size.
        solvent: Solvent::WaterOpcSpecifyMolCount(50),
        ..cfg_auto_water_count.clone()
    };

    // Sim 1: Using an automatically set water count.
    for (i, cfg) in [cfg_auto_water_count, cfg_fixed_water_count]
        .iter()
        .enumerate()
    {
        let mut md = MdState::new(&dev, &cfg, &[], &param_set).unwrap();

        let num_water_mols = md.water.len();

        if i == 0 {
            println!("Simulating with automatic water count. {num_water_mols} mols");
        } else {
            println!("Simulating with fixed water count. {num_water_mols} mols");
        }

        // todo: Steps may not be required; init should be enough to validate the barostat.
        let n_steps = 100;
        for _ in 0..n_steps {
            md.step(&dev, 0.002, None);
        }

        let avg_pressure: f64 =
            md.snapshots.iter().map(|s| s.pressure as f64).sum::<f64>() / md.snapshots.len() as f64;

        println!("avg pressure over {n_steps} steps: {avg_pressure:.1} bar");

        let expected = 1.; // Bar

        assert!(
            (avg_pressure - expected).abs() < 0.2,
            "average pressure {avg_pressure:.1} bar is outside the expected range"
        );
    }

    // Sim 2: Using an automatically set water count.
    {
        let mut md = MdState::new(&dev, &cfg, &[], &param_set).unwrap();

        // todo: Steps may not be required; init should be enough to validate the barostat.
        let n_steps = 100;
        for _ in 0..n_steps {
            md.step(&dev, 0.002, None);
        }

        let avg_pressure: f64 =
            md.snapshots.iter().map(|s| s.pressure as f64).sum::<f64>() / md.snapshots.len() as f64;

        println!("avg pressure over {n_steps} steps: {avg_pressure:.1} bar");

        let expected = 1.; // Bar

        assert!(
            (avg_pressure - expected).abs() < 0.2,
            "average pressure {avg_pressure:.1} bar is outside the expected range"
        );
    }
}

//
// // todo: Replace the sim_1_bar with this A/R.
// #[test]
