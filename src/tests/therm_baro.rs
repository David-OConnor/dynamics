//! Tests for temperature, pressure, and kinetic energy measurement functions.
//!
//! These tests exercise the pure-computation functions directly — no full MD
//! simulation is needed.  All inputs are constructed analytically so the expected
//! output can be computed by hand alongside the code.

use lin_alg::f32::Vec3;

use crate::{
    AtomDynamics, MdState, NATIVE_TO_KCAL,
    barostat::{BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED, SimBox, VirialKcalMol, measure_pressure},
    solvent::{H_MASS, MASS_WATER_MOL, O_MASS, WaterMol},
    thermostat::GAS_CONST_R,
};

// ── helpers ──────────────────────────────────────────────────────────────────

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

// ── measure_kinetic_energy ────────────────────────────────────────────────────

/// KE = 0.5 · Σ(m · v²) · NATIVE_TO_KCAL for all dynamic atoms.
#[test]
fn test_measure_kinetic_energy_atoms() {
    let mut md = MdState::default();
    md.atoms = vec![
        AtomDynamics {
            mass: 12.0,
            vel: Vec3::new(3.0, 0.0, 0.0), // v² = 9
            ..Default::default()
        },
        AtomDynamics {
            mass: 1.0,
            vel: Vec3::new(0.0, 2.0, 0.0), // v² = 4
            ..Default::default()
        },
    ];
    // KE = 0.5 * (12*9 + 1*4) * NATIVE_TO_KCAL
    let expected = 0.5 * (12.0 * 9.0 + 1.0 * 4.0) * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy();
    assert_close(got, expected, 1e-6, "KE from two atoms");
}

/// Static atoms must NOT contribute to kinetic energy.
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
            vel: Vec3::new(10.0, 10.0, 10.0), // large but static
            static_: true,
            ..Default::default()
        },
    ];
    // Only the dynamic atom contributes: 0.5 * 12 * 1 * NATIVE_TO_KCAL
    let expected = 0.5 * 12.0 * 1.0 * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy();
    assert_close(got, expected, 1e-6, "KE excludes static atom");
}

/// Water molecules (O, H0, H1) all contribute to measure_kinetic_energy.
#[test]
fn test_measure_kinetic_energy_includes_water() {
    let mut md = MdState::default();
    let v_o = Vec3::new(2.0, 0.0, 0.0); // v² = 4
    let v_h = Vec3::new(0.0, 1.0, 0.0); // v² = 1

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

    // KE = 0.5 * (O_MASS*4 + H_MASS*1 + H_MASS*1) * NATIVE_TO_KCAL
    let expected = 0.5
        * (O_MASS as f64 * 4.0 + H_MASS as f64 * 1.0 + H_MASS as f64 * 1.0)
        * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy();
    assert_close(got, expected, 1e-5, "KE from water molecule");
}

// ── measure_kinetic_energy_translational ─────────────────────────────────────

/// When all water sites move at the same velocity, the COM velocity equals
/// that velocity; translational KE = 0.5 · M_water · v².
#[test]
fn test_measure_kinetic_energy_translational_rigid_translation() {
    let mut md = MdState::default();
    let v_com = Vec3::new(3.0, 0.0, 0.0); // v² = 9

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

    // KE_trans = 0.5 * MASS_WATER_MOL * 9 * NATIVE_TO_KCAL
    let expected = 0.5 * MASS_WATER_MOL as f64 * 9.0 * NATIVE_TO_KCAL as f64;
    let got = md.measure_kinetic_energy_translational();
    assert_close(got, expected, 1e-5, "translational KE — rigid translation");
}

/// Rotational motion around the COM contributes to the *full* KE but must be
/// absent from the *translational* KE (molecular virial theorem).
///
/// Velocity field: v_H0 = v_H1 = (0, 1, 0),
///                 v_O  = -(2·H_MASS / O_MASS) · (0, 1, 0)   → COM vel ≈ 0.
#[test]
fn test_measure_kinetic_energy_translational_excludes_rotation() {
    let mut md = MdState::default();

    // Choose velocities so M_O·v_O + M_H·v_H0 + M_H·v_H1 = 0 (COM stationary).
    let v_h = Vec3::new(0.0, 1.0, 0.0);
    let v_o = Vec3::new(0.0, -2.0 * H_MASS / O_MASS, 0.0);

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

    assert!(
        ke_full > 1e-10,
        "full KE must be nonzero (atoms are moving)"
    );
    // Translational KE ≈ 0 because the COM is stationary.
    assert!(
        ke_trans < 1e-5 * ke_full,
        "translational KE should be ~0 for pure internal motion; \
         got ke_trans={ke_trans:.4e}, ke_full={ke_full:.4e}"
    );
}

// ── measure_temperature ───────────────────────────────────────────────────────

/// T = 2·KE / (DOF · R).
#[test]
fn test_measure_temperature() {
    let mut md = MdState::default();
    let dof = 6_usize;
    let ke = 3.0_f64; // kcal/mol

    md.kinetic_energy = ke;
    md.thermo_dof = dof;

    let expected = 2.0 * ke / (dof as f64 * GAS_CONST_R);
    let got = md.measure_temperature();
    assert_close(got, expected, 1e-9, "temperature from KE and DOF");
}

/// Setting KE = (DOF/2)·R·T should round-trip back to T.
#[test]
fn test_measure_temperature_roundtrip_300k() {
    let mut md = MdState::default();
    let dof = 9_usize;
    let t_target = 300.0_f64;

    // Equipartition: KE = (DOF / 2) * R * T
    md.kinetic_energy = 0.5 * dof as f64 * GAS_CONST_R * t_target;
    md.thermo_dof = dof;

    let got = md.measure_temperature();
    assert_close(got, t_target, 1e-9, "round-trip temperature at 300 K");
}

// ── measure_pressure ─────────────────────────────────────────────────────────

/// P = (2·KE + W_total) / (3·V) · BAR_CONVERSION  [bar]
#[test]
fn test_measure_pressure_formula() {
    let ke = 5.0_f64; // kcal/mol
    let virial = VirialKcalMol {
        bonded: 1.0,
        nonbonded_short_range: 2.0,
        nonbonded_long_range: 0.5,
        constraints: 0.0,
    };
    let side = 30.0_f32; // Å
    let simbox = SimBox {
        bounds_low: Vec3::new(0.0, 0.0, 0.0),
        bounds_high: Vec3::new(side, side, side),
        extent: Vec3::new(side, side, side),
    };

    let vol = (side as f64).powi(3);
    let w_total = virial.total();
    let expected = (2.0 * ke + w_total) / (3.0 * vol) * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED;

    let got = measure_pressure(ke, &simbox, &virial);
    assert_close(got, expected, 1e-9, "pressure formula");
}

/// Zero KE and zero virial must yield zero pressure.
#[test]
fn test_measure_pressure_zero() {
    let simbox = SimBox {
        bounds_low: Vec3::new(0.0, 0.0, 0.0),
        bounds_high: Vec3::new(30.0, 30.0, 30.0),
        extent: Vec3::new(30.0, 30.0, 30.0),
    };
    let got = measure_pressure(0.0, &simbox, &VirialKcalMol::default());
    assert_eq!(got, 0.0, "pressure is zero when KE=0 and W=0");
}

/// Pressure is linear in kinetic energy (with zero virial).
#[test]
fn test_measure_pressure_ke_scaling() {
    let simbox = SimBox {
        bounds_low: Vec3::new(0.0, 0.0, 0.0),
        bounds_high: Vec3::new(30.0, 30.0, 30.0),
        extent: Vec3::new(30.0, 30.0, 30.0),
    };
    let virial = VirialKcalMol::default();

    let p1 = measure_pressure(1.0, &simbox, &virial);
    let p2 = measure_pressure(2.0, &simbox, &virial);
    assert_close(p2, 2.0 * p1, 1e-9, "pressure is linear in KE");
}

/// Larger volume → lower pressure for fixed KE and virial.
#[test]
fn test_measure_pressure_volume_scaling() {
    let ke = 10.0_f64;
    let virial = VirialKcalMol::default();

    let make_box = |side: f32| SimBox {
        bounds_low: Vec3::new(0.0, 0.0, 0.0),
        bounds_high: Vec3::new(side, side, side),
        extent: Vec3::new(side, side, side),
    };

    let p_small = measure_pressure(ke, &make_box(20.0), &virial);
    let p_large = measure_pressure(ke, &make_box(40.0), &virial);

    // Volume ratio = (40/20)^3 = 8  →  pressure ratio = 1/8
    assert_close(p_large, p_small / 8.0, 1e-6, "pressure ∝ 1/V");
}
