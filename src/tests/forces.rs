use bio_files::{AtomGeneric, BondGeneric, BondType};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use ewald::force_coulomb_short_range;
use lin_alg::f32::Vec3;
use na_seq::Element;

use crate::{
    ComputationDevice, FfMolType, MdConfig, MdOverrides, MdState, MolDynamics, SimBoxInit,
    forces::force_e_lj,
    non_bonded::{CHARGE_UNIT_SCALER, LONG_RANGE_CUTOFF},
    params::FfParamSet,
};

const EWALD_ALPHA: f32 = 0.35;

/// Build two test "ca" carbons positioned symmetrically around (30, 30, 30) in a 60 Å box.
/// The atoms are given distinct serial numbers so that bonds can refer to them.
fn setup_test_pair(dist: f32) -> [AtomGeneric; 2] {
    let c = 30.0; // centre of the 60 Å box

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
fn test_forces_general_on_pair() {
    let dists = [2., 3., 5., 8., 10., 12.];

    // Load once outside the loop — it reads AMBER parameter files.
    let param_set = FfParamSet::new_amber().unwrap();

    #[cfg(feature = "cuda")]
    let dev = {
        let stream = {
            let ctx = CudaContext::new(0).unwrap();
            ctx.default_stream()
        };

        ComputationDevice::Gpu(stream)
    };

    #[cfg(not(feature = "cuda"))]
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
        let c = 30.;
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

// ── helpers ──────────────────────────────────────────────────────────────────

fn assert_close_f32(got: f32, expected: f32, rel_tol: f32, label: &str) {
    if expected.abs() < 1e-6 {
        assert!(got.abs() < 1e-4, "{label}: expected ≈0, got {got:.6e}");
        return;
    }
    let rel = ((got - expected) / expected).abs();
    assert!(
        rel < rel_tol,
        "{label}: got {got:.6}, expected {expected:.6}, rel_err={rel:.4} (tol={rel_tol})"
    );
}

// ── Lennard-Jones ─────────────────────────────────────────────────────────────

/// E = 4ε[(σ/r)¹² − (σ/r)⁶]
fn lj_energy_analytic(sigma: f32, eps: f32, r: f32) -> f32 {
    let sr6 = (sigma / r).powi(6);
    4.0 * eps * (sr6 * sr6 - sr6)
}

/// |F| = 24ε(2(σ/r)¹² − (σ/r)⁶) / r
/// Positive → repulsive (tgt pushed away from src along dir = tgt − src).
fn lj_force_mag_analytic(sigma: f32, eps: f32, r: f32) -> f32 {
    let sr6 = (sigma / r).powi(6);
    24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r
}

/// Verify `force_e_lj` against the analytic 12-6 Lennard-Jones formulas.
///
/// Checks:
/// 1. Force x-component matches the analytic magnitude (sign included).
/// 2. Returned energy matches the 12-6 formula.
/// 3. Force is repulsive at r < r_min and attractive at r > r_min.
/// 4. Energy is minimised at −ε when r = 2^(1/6)·σ.
#[test]
fn test_lennard_jones() {
    let sigma = 3.4_f32; // Å — representative for sp2 carbon
    let eps = 0.086_f32; // kcal/mol
    let r_min = 2.0_f32.powf(1.0 / 6.0) * sigma; // ≈ 3.817 Å

    // dir = tgt − src; tgt is to the right of src along x.
    let dir = Vec3::new(1.0, 0.0, 0.0);

    for r in [2.0_f32, 2.5, 3.0, 4.5, 5.0, 7.0] {
        let (force_vec, energy) = force_e_lj(dir, 1.0 / r, sigma, eps);

        let f_analytic = lj_force_mag_analytic(sigma, eps, r);
        let e_analytic = lj_energy_analytic(sigma, eps, r);

        println!(
            "LJ r={r:.2}Å  Fx={:.5}  F_analytic={f_analytic:.5}  \
             E={energy:.5}  E_analytic={e_analytic:.5} kcal/mol",
            force_vec.x
        );

        // Force x-component must equal the analytic magnitude (dir is along +x).
        assert_close_f32(
            force_vec.x,
            f_analytic,
            1e-4,
            &format!("LJ force at r={r}Å"),
        );

        // y and z are zero because dir is along x.
        assert_eq!(force_vec.y, 0.0, "LJ: Fy should be 0 at r={r}Å");
        assert_eq!(force_vec.z, 0.0, "LJ: Fz should be 0 at r={r}Å");

        // Energy must match the analytic formula.
        assert_close_f32(energy, e_analytic, 1e-4, &format!("LJ energy at r={r}Å"));

        // Direction check.
        if r < r_min {
            assert!(
                force_vec.x > 0.0,
                "LJ: repulsive expected at r={r}Å < r_min"
            );
        } else {
            assert!(
                force_vec.x < 0.0,
                "LJ: attractive expected at r={r}Å > r_min"
            );
        }
    }

    // Energy minimum: E(r_min) = −ε  (within rounding).
    let (_, e_at_min) = force_e_lj(dir, 1.0 / r_min, sigma, eps);
    assert_close_f32(e_at_min, -eps, 1e-4, "LJ energy minimum = −ε");
}

/// Newton's third law: force on tgt from src = −(force on src from tgt).
#[test]
fn test_lennard_jones_newton3() {
    let sigma = 3.4_f32;
    let eps = 0.086_f32;
    let dir = Vec3::new(1.0, 0.0, 0.0);

    for r in [2.0_f32, 3.817, 5.5] {
        let (f_on_tgt, _) = force_e_lj(dir, 1.0 / r, sigma, eps);
        let (f_on_src, _) = force_e_lj(-dir, 1.0 / r, sigma, eps);

        let sum = f_on_tgt + f_on_src;
        assert!(
            sum.magnitude_squared() < 1e-10,
            "LJ Newton3 violated at r={r}Å: |f_tgt + f_src| = {:.4e}",
            sum.magnitude()
        );
    }
}

// ── Coulomb short-range ───────────────────────────────────────────────────────

/// Force–energy consistency for the erfc-screened short-range Coulomb potential.
///
/// For each (charge pair, distance) we compute the x-force from the function
/// and compare it to the numerical gradient −ΔE/Δx obtained by shifting the
/// separation by ±δ.  This verifies both the energy and force implementations
/// simultaneously without needing to evaluate erfc in the test.
#[test]
fn test_coulomb_short_range_force_energy_consistency() {
    let alpha = EWALD_ALPHA;
    let cutoff = LONG_RANGE_CUTOFF;
    let delta = 0.001_f32; // Å

    let charge_pairs: [(f32, f32); 3] = [
        (1.0, -1.0), // opposite, unit
        (1.0, 1.0),  // like, unit
        (0.5, -0.3), // fractional opposite
    ];

    // Stay well inside the cutoff so the force is well above numerical noise.
    let dists = [3.0_f32, 5.0, 8.0];

    for (q1e, q2e) in charge_pairs {
        let q1 = q1e * CHARGE_UNIT_SCALER;
        let q2 = q2e * CHARGE_UNIT_SCALER;

        for r in dists {
            let dir = Vec3::new(1.0, 0.0, 0.0);

            // Analytic force at r.
            let (f, _e) = force_coulomb_short_range(dir, r, 1.0 / r, q1, q2, cutoff, alpha);

            // Numerical gradient of the energy.
            let (_, e_plus) =
                force_coulomb_short_range(dir, r + delta, 1.0 / (r + delta), q1, q2, cutoff, alpha);
            let (_, e_minus) =
                force_coulomb_short_range(dir, r - delta, 1.0 / (r - delta), q1, q2, cutoff, alpha);
            let f_numerical = -(e_plus - e_minus) / (2.0 * delta);

            println!(
                "Coulomb SR q=({q1e:+.2},{q2e:+.2}) r={r:.1}Å  \
                 Fx={:.5}  F_numerical={f_numerical:.5} kcal/(mol·Å)",
                f.x
            );

            assert_close_f32(
                f.x,
                f_numerical,
                0.02, // 2 % — same as SPME gradient test in spme.rs
                &format!("SR Coulomb F=−dE/dx: q=({q1e},{q2e}) r={r}Å"),
            );
        }
    }
}

/// Like charges produce a repulsive force and positive energy;
/// opposite charges produce an attractive force and negative energy.
#[test]
fn test_coulomb_short_range_direction() {
    let alpha = EWALD_ALPHA;
    let cutoff = LONG_RANGE_CUTOFF;
    // dir = (r_tgt − r_src) / |...|; tgt is to the right of src.
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let q = CHARGE_UNIT_SCALER; // +1 e in Amber units

    for r in [3.0_f32, 5.0, 8.0] {
        // Like charges: force on tgt pushed further right (+x), energy > 0.
        let (f_like, e_like) = force_coulomb_short_range(dir, r, 1.0 / r, q, q, cutoff, alpha);
        assert!(
            f_like.x > 0.0,
            "like charges: SR Coulomb should be repulsive at r={r}Å, got Fx={}",
            f_like.x
        );
        assert!(
            e_like > 0.0,
            "like charges: SR energy should be positive at r={r}Å, got E={}",
            e_like
        );

        // Opposite charges: force on tgt pulled toward src (−x), energy < 0.
        let (f_opp, e_opp) = force_coulomb_short_range(dir, r, 1.0 / r, q, -q, cutoff, alpha);
        assert!(
            f_opp.x < 0.0,
            "opposite charges: SR Coulomb should be attractive at r={r}Å, got Fx={}",
            f_opp.x
        );
        assert!(
            e_opp < 0.0,
            "opposite charges: SR energy should be negative at r={r}Å, got E={}",
            e_opp
        );
    }
}

/// Newton's third law: F on particle 1 from particle 2 = −(F on particle 2 from particle 1).
#[test]
fn test_coulomb_short_range_newton3() {
    let alpha = EWALD_ALPHA;
    let cutoff = LONG_RANGE_CUTOFF;
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let q = CHARGE_UNIT_SCALER;

    for r in [3.0_f32, 5.0, 8.0] {
        // Force on particle 1 (at +x relative to particle 2): q1=+q, q2=−q, dir=(r1−r2)
        let (f1, _) = force_coulomb_short_range(dir, r, 1.0 / r, q, -q, cutoff, alpha);
        // Force on particle 2 (at −x relative to particle 1): charges swapped, dir negated
        let (f2, _) = force_coulomb_short_range(-dir, r, 1.0 / r, -q, q, cutoff, alpha);

        let sum = f1 + f2;
        assert!(
            sum.magnitude() < 1e-5 * f1.magnitude(),
            "SR Coulomb Newton3 violated at r={r}Å: |f1+f2|={:.4e}, |f1|={:.4e}",
            sum.magnitude(),
            f1.magnitude()
        );
    }
}

/// Force and energy are exactly zero at and beyond the cutoff radius.
#[test]
fn test_coulomb_short_range_cutoff() {
    let alpha = EWALD_ALPHA;
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let q = CHARGE_UNIT_SCALER;

    for r in [
        LONG_RANGE_CUTOFF,
        LONG_RANGE_CUTOFF + 0.5,
        LONG_RANGE_CUTOFF + 2.0,
    ] {
        let (f, e) = force_coulomb_short_range(dir, r, 1.0 / r, q, -q, LONG_RANGE_CUTOFF, alpha);
        assert_eq!(
            f.magnitude_squared(),
            0.0,
            "SR Coulomb force should be 0 at r={r:.2}Å (≥ cutoff), got {f:?}"
        );
        assert_eq!(
            e, 0.0,
            "SR Coulomb energy should be 0 at r={r:.2}Å (≥ cutoff), got E={e}"
        );
    }
}

// todo: This would be a good place to run a sample of the geostd set to validate
// todo: FF types, partial charges, and FRCMOD overrides.
