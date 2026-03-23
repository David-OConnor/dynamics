//! Validation helpers and tests for the SPME/Ewald electrostatics implementation.
//!
//! Tests the combined short-range + long-range Coulomb energy and forces against
//! analytic values for simple configurations (isolated charge pairs in a large box).
//!
//! # Reference values
//! For a pair of elementary charges q₁, q₂ separated by distance r in a very large
//! box, the total Coulomb energy approaches the vacuum value:
//!   E_vac = K · q₁ · q₂ / r    (K = 332.0522 kcal·Å / (mol·e²))
//!
//! SPME splits this into:
//!   E_short = erfc(α·r)/r · q₁·q₂ · K
//!   E_long  ≈ erf(α·r)/r  · q₁·q₂ · K   (plus small image contributions)
//!   E_total = E_short + E_long  →  K·q₁·q₂/r   as L → ∞
//!
//! Run without features to test on CPU. Run `cargo test --features " cufft"` (or vkfft) to test
//! on GPU. Test both.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use ewald::{PmeRecip, force_coulomb_short_range, get_grid_n};
use lin_alg::f32::Vec3;

use crate::non_bonded::{CHARGE_UNIT_SCALER, EWALD_ALPHA, LONG_RANGE_CUTOFF};

/// Electrostatic constant in Amber kcal/mol units: K = CHARGE_UNIT_SCALER².
/// `E = K · q₁_e · q₂_e / r`  (charges in elementary units, r in Å → kcal/mol)
pub const K_ELEC: f32 = CHARGE_UNIT_SCALER * CHARGE_UNIT_SCALER;

/// Wrap a single coordinate component into [0, L).
fn wrap1(x: f32, l: f32) -> f32 {
    x.rem_euclid(l)
}

/// Wrap a Vec3 position into the primary box [0, L) per axis.
fn wrap_pos(p: Vec3, l: (f32, f32, f32)) -> Vec3 {
    Vec3::new(wrap1(p.x, l.0), wrap1(p.y, l.1), wrap1(p.z, l.2))
}

#[cfg(feature = "cuda")]
/// Build a PmeRecip with the given box and mesh spacing.
/// Feature-gates match dynamics/src/non_bonded.rs.
fn make_pme(l: (f32, f32, f32), alpha: f32, mesh_spacing: f32) -> PmeRecip {
    let stream = {
        let ctx = CudaContext::new(0).unwrap();
        ctx.default_stream()
    };

    let dims = get_grid_n(l, mesh_spacing);
    PmeRecip::new(Some(&stream), dims, l, alpha)
}

#[cfg(not(feature = "cuda"))]
fn make_pme(l: (f32, f32, f32), alpha: f32, mesh_spacing: f32) -> PmeRecip {
    let dims = get_grid_n(l, mesh_spacing);
    PmeRecip::new(dims, l, alpha)
}

/// Compute SPME forces and energy on a pair of charges in a cubic box.
///
/// Charges `q1_e` and `q2_e` are in **elementary units** (e); internally they are
/// scaled to Amber units before passing into the SPME and short-range routines.
///
/// Returns `((f_short_on_q1, f_long_on_q1, f_long_on_q2), (e_short, e_long, e_total))`
/// with forces in kcal/(mol·Å) and energies in kcal/mol.
pub fn spme_pair_forces_energy(
    r1: Vec3,
    r2: Vec3,
    q1_e: f32,
    q2_e: f32,
    box_len: f32,
    alpha: f32,
) -> ((Vec3, Vec3, Vec3), (f32, f32, f32)) {
    #[cfg(feature = "cuda")]
    let stream = {
        let ctx = CudaContext::new(0).unwrap();
        ctx.default_stream()
    };

    let q1 = q1_e * CHARGE_UNIT_SCALER;
    let q2 = q2_e * CHARGE_UNIT_SCALER;

    // Short-range force on q1 (from q2)
    let diff = {
        let mut v = r1 - r2;
        v.x -= box_len * (v.x / box_len).round();
        v.y -= box_len * (v.y / box_len).round();
        v.z -= box_len * (v.z / box_len).round();

        v
    };
    let dist = diff.magnitude();

    let inv_dist = 1.0 / dist;
    let dir = diff * inv_dist;

    let (f_sr_1, e_sr) =
        force_coulomb_short_range(dir, dist, inv_dist, q1, q2, LONG_RANGE_CUTOFF, alpha);

    // Long-range forces
    let l = (box_len, box_len, box_len);
    let mut pme = make_pme(l, alpha, 1.0);
    let pos = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
    let q_arr = vec![q1, q2];

    #[cfg(feature = "cuda")]
    let (f_recip, e_lr) = pme.forces_gpu(&stream, &pos, &q_arr);

    #[cfg(not(feature = "cuda"))]
    let (f_recip, e_lr) = pme.forces(&pos, &q_arr);

    let e_total = e_sr + e_lr;

    ((f_sr_1, f_recip[0], f_recip[1]), (e_sr, e_lr, e_total))
}

/// Expected vacuum Coulomb energy in kcal/mol.
/// `E = K · q₁_e · q₂_e / r`
pub fn vacuum_coulomb_energy(q1_e: f32, q2_e: f32, dist: f32) -> f32 {
    K_ELEC * q1_e * q2_e / dist
}

/// Fraction of the expected value used as the pass/fail threshold.
const REL_TOL: f32 = 0.01; // 1 %

fn assert_rel_close(got: f32, expected: f32, tol: f32, label: &str) {
    if expected.abs() < 1e-6 {
        assert!(got.abs() < 1e-4, "{label}: expected ≈ 0, got {got:.6e}");
        return;
    }
    let rel = ((got - expected) / expected).abs();
    assert!(
        rel < tol,
        "{label}: got {got:.6}, expected {expected:.6}, rel_err = {rel:.4} (tol = {tol})"
    );
}

// ------------------------------------------------------------------
// Energy tests
// ------------------------------------------------------------------

/// For a neutral pair (+q, −q) in a large box the total SPME energy must
/// converge to the vacuum Coulomb energy  E = −K/r.
#[test]
fn test_spme_energy_opposite_charges() {
    let box_len = 50.0;
    let alpha = EWALD_ALPHA;

    for (dist, tag) in [(3.0, "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
        let center = box_len / 2.0;
        let r1 = Vec3::new(center - dist / 2.0, center, center);
        let r2 = Vec3::new(center + dist / 2.0, center, center);

        let (_, (e_sr, e_lr, e_total)) = spme_pair_forces_energy(r1, r2, 1.0, -1.0, box_len, alpha);

        let e_vac = vacuum_coulomb_energy(1.0, -1.0, dist);

        println!(
            "+1/−1 at {tag}: e_sr={e_sr:.4}  e_lr={e_lr:.4}  \
                 e_total={e_total:.4}  e_vac={e_vac:.4} kcal/mol"
        );

        assert_rel_close(e_total, e_vac, REL_TOL, &format!("energy +1/-1 at {tag}"));
    }
}

/// Same as above but with fractional charges to ensure charge-scaling is linear.
#[test]
fn test_spme_energy_fractional_charges() {
    let box_len = 50.0;
    let alpha = EWALD_ALPHA;
    let dist = 5.0;
    let center = box_len / 2.0;
    let r1 = Vec3::new(center - dist / 2.0, center, center);
    let r2 = Vec3::new(center + dist / 2.0, center, center);

    for (q1e, q2e, tag) in [(0.5f32, -0.5f32, "q=±0.5"), (0.25, -0.25, "q=±0.25")] {
        let (_, (_, _, e_total)) = spme_pair_forces_energy(r1, r2, q1e, q2e, box_len, alpha);

        let e_vac = vacuum_coulomb_energy(q1e, q2e, dist);
        println!("{tag}: e_total={e_total:.4}  e_vac={e_vac:.4} kcal/mol");
        assert_rel_close(e_total, e_vac, REL_TOL, tag);
    }
}

/// Energy should converge toward vacuum Coulomb as the box grows.
/// Relative error must be < 1 % at L = 50 Å for r = 5 Å.
#[test]
fn test_spme_energy_box_convergence() {
    let dist = 5.0;
    let alpha = EWALD_ALPHA;
    let e_vac = vacuum_coulomb_energy(1.0, -1.0, dist);

    println!("Box-size convergence test (e_vac = {e_vac:.4} kcal/mol):");
    for box_len in [20.0, 30.0, 50.0] {
        let c = box_len / 2.0;
        let r1 = Vec3::new(c - dist / 2.0, c, c);
        let r2 = Vec3::new(c + dist / 2.0, c, c);

        let (_, (_, _, e_total)) = spme_pair_forces_energy(r1, r2, 1.0, -1.0, box_len, alpha);

        let rel = ((e_total - e_vac) / e_vac).abs();
        println!("  L={box_len:.0} Å:  e_total={e_total:.4}  rel_err={rel:.4}");
    }

    // Assert only for the largest box where images are negligible.
    let c = 50.0 / 2.0;
    let r1 = Vec3::new(c - dist / 2.0, c, c);
    let r2 = Vec3::new(c + dist / 2.0, c, c);

    let (_, (_, _, e_total)) = spme_pair_forces_energy(r1, r2, 1.0, -1.0, 50.0, alpha);

    assert_rel_close(e_total, e_vac, REL_TOL, "energy at L=50 Å");
}

// ------------------------------------------------------------------
// Force tests
// ------------------------------------------------------------------

/// For a +1/−1 pair along x, the total x-force on charge 1 must match
/// vacuum Coulomb: F_x = +K/r² (attractive, toward the −1 charge at +x).
#[test]
fn test_spme_force_magnitude() {
    let box_len = 50.;

    for (dist, tag) in [(3., "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
        let center = box_len / 2.0;
        // q1 (+1) to the left, q2 (−1) to the right
        let r1 = Vec3::new(center - dist / 2.0, center, center);
        let r2 = Vec3::new(center + dist / 2.0, center, center);

        let ((f_sr_1, f_lr_1, _), _) =
            spme_pair_forces_energy(r1, r2, 1.0, -1.0, box_len, EWALD_ALPHA);

        let fx_total = f_sr_1.x + f_lr_1.x;

        // Vacuum force on q1: attractive, in +x direction.
        // dir = (r1 − r2)/|r1 − r2| = −x̂
        // F = dir · (q1_sc · q2_sc · (erfc/r² + ...))
        //   = (−x̂) · (negative magnitude)  =  +x̂  component is positive.
        // |F_total_x| ≈ K · |q1_e| · |q2_e| / r² = K / r²
        let f_vac_x = K_ELEC / (dist * dist); // positive (attractive toward +x)

        println!(
            "+1/−1 at {tag}: f_sr_x={:.4}  f_lr_x={:.4}  \
                 f_total_x={fx_total:.4}  f_vac_x={f_vac_x:.4} kcal/(mol·Å)",
            f_sr_1.x, f_lr_1.x
        );

        // At 8 Å in a 50 Å box, periodic images contribute ~2 % to the force,
        // so use a slightly looser tolerance there.
        let ftol = if dist >= 7.0 { 2.0 * REL_TOL } else { REL_TOL };
        assert_rel_close(fx_total, f_vac_x, ftol, &format!("Fx on q1 at {tag}"));

        // y and z components should be near zero (symmetry).
        let fy = f_sr_1.y + f_lr_1.y;
        let fz = f_sr_1.z + f_lr_1.z;
        assert!(
            fy.abs() < 0.01 * f_vac_x,
            "{tag}: Fy should be ~0, got {fy:.4e}"
        );
        assert!(
            fz.abs() < 0.01 * f_vac_x,
            "{tag}: Fz should be ~0, got {fz:.4e}"
        );
    }
}

/// Newton's third law: for a neutral pair the forces on the two charges
/// must sum to (near) zero.  Exact only in the limit L → ∞.
#[test]
fn test_spme_force_newton3() {
    let box_len = 50.;
    let alpha = EWALD_ALPHA;

    for (dist, tag) in [(3., "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
        let center = box_len / 2.0;
        let r1 = Vec3::new(center - dist / 2.0, center, center);
        let r2 = Vec3::new(center + dist / 2.0, center, center);

        let q1 = 1. * CHARGE_UNIT_SCALER;
        let q2 = -1.0 * CHARGE_UNIT_SCALER;

        // Short-range forces
        let diff = r1 - r2;
        let inv_d = 1.0 / dist;
        let dir = diff * inv_d;
        let (f_sr_1, _) =
            force_coulomb_short_range(dir, dist, inv_d, q1, q2, LONG_RANGE_CUTOFF, alpha);
        let (f_sr_2, _) =
            force_coulomb_short_range(-dir, dist, inv_d, q2, q1, LONG_RANGE_CUTOFF, alpha);

        // Long-range forces
        let l = (box_len, box_len, box_len);
        let mut pme = make_pme(l, alpha, 1.0);
        let pos = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
        let q_arr = vec![q1, q2];
        let (f_recip, _) = pme.forces(&pos, &q_arr);

        let f1 = f_sr_1 + f_recip[0];
        let f2 = f_sr_2 + f_recip[1];

        let sum_x = f1.x + f2.x;
        let sum_y = f1.y + f2.y;
        let sum_z = f1.z + f2.z;
        let sum_mag = (sum_x * sum_x + sum_y * sum_y + sum_z * sum_z).sqrt();
        let f1_mag = f1.magnitude();

        println!("Newton3 at {tag}: |f1|={f1_mag:.4}  |f1+f2|={sum_mag:.4e}");

        assert!(
            sum_mag < 0.02 * f1_mag,
            "{tag}: Newton 3rd law violated: |f1+f2| = {sum_mag:.4e}, |f1| = {f1_mag:.4}"
        );
    }
}

// ------------------------------------------------------------------
// Short-range cutoff
// ------------------------------------------------------------------

/// Forces and energies must be exactly zero at and beyond the cutoff.
#[test]
fn test_short_range_cutoff() {
    let q = CHARGE_UNIT_SCALER;
    let dir = Vec3::new(1.0, 0.0, 0.0);

    for dist in [
        LONG_RANGE_CUTOFF,
        LONG_RANGE_CUTOFF + 0.1,
        LONG_RANGE_CUTOFF + 1.0,
    ] {
        let (f, e) =
            force_coulomb_short_range(dir, dist, 1.0 / dist, q, q, LONG_RANGE_CUTOFF, EWALD_ALPHA);
        assert_eq!(
            f.magnitude_squared(),
            0.0,
            "force should be 0 at dist={dist:.2}: {f:?}"
        );
        assert_eq!(e, 0.0, "energy should be 0 at dist={dist:.2}: {e}");
    }
}

// ------------------------------------------------------------------
// Different box shapes
// ------------------------------------------------------------------

/// Test a non-cubic (elongated) box to catch any axis-mixing bugs.
#[test]
fn test_spme_energy_non_cubic_box() {
    let alpha = EWALD_ALPHA;
    let dist = 5.0;

    // Elongated box: charges along the long axis
    let lx = 60.0;
    let ly = 30.0;
    let lz = 30.0;

    let r1 = Vec3::new(lx / 2.0 - dist / 2.0, ly / 2.0, lz / 2.0);
    let r2 = Vec3::new(lx / 2.0 + dist / 2.0, ly / 2.0, lz / 2.0);

    let q1 = 1.0 * CHARGE_UNIT_SCALER;
    let q2 = -1.0 * CHARGE_UNIT_SCALER;

    let l = (lx, ly, lz);
    let mut pme = make_pme(l, alpha, 1.0);

    let pos = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
    let q_arr = vec![q1, q2];
    let (_, e_lr) = pme.forces(&pos, &q_arr);

    let diff = r1 - r2;
    let inv_d = 1.0 / dist;
    let dir = diff * inv_d;
    let (_, e_sr) = force_coulomb_short_range(dir, dist, inv_d, q1, q2, LONG_RANGE_CUTOFF, alpha);

    let e_total = e_sr + e_lr;
    let e_vac = vacuum_coulomb_energy(1.0, -1.0, dist);

    println!("Non-cubic box (60×30×30): e_total={e_total:.4}  e_vac={e_vac:.4} kcal/mol");
    assert_rel_close(e_total, e_vac, REL_TOL, "non-cubic box energy");
}

// ------------------------------------------------------------------
// Like-charges (repulsive) force test
// ------------------------------------------------------------------

/// For a +1/+1 pair along x the total x-force on charge 1 must match vacuum
/// Coulomb with the correct repulsive sign: F_x = −K/r² (pointing away from q2).
///
/// dir = (r1 − r2)/|r1 − r2| = −x̂; q1·q2 > 0 → force_mag > 0
/// → force = (−x̂) · (positive) → F_x is negative (repulsive).
///
/// Note: only forces (not energies) are tested here because SPME energies for a
/// non-neutral pair include an implicit neutralizing-background correction that
/// shifts the energy but contributes zero gradient.
#[test]
fn test_spme_force_like_charges() {
    let box_len = 50.0;
    let alpha = EWALD_ALPHA;

    for (dist, tag) in [(3.0, "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
        let center = box_len / 2.0;
        let r1 = Vec3::new(center - dist / 2.0, center, center);
        let r2 = Vec3::new(center + dist / 2.0, center, center);

        let ((f_sr_1, f_lr_1, _), _) = spme_pair_forces_energy(r1, r2, 1.0, 1.0, box_len, alpha);

        let fx_total = f_sr_1.x + f_lr_1.x;

        // Vacuum Coulomb: F_x = K·q1·q2/r² projected on (r1−r2), which is −x̂.
        // Result is negative (repulsive).
        let f_vac_x = -K_ELEC / (dist * dist);

        println!(
            "+1/+1 at {tag}: f_sr_x={:.4}  f_lr_x={:.4}  \
                 f_total_x={fx_total:.4}  f_vac_x={f_vac_x:.4} kcal/(mol·Å)",
            f_sr_1.x, f_lr_1.x
        );

        let ftol = if dist >= 7.0 { 2.0 * REL_TOL } else { REL_TOL };
        assert_rel_close(
            fx_total,
            f_vac_x,
            ftol,
            &format!("Fx on q1 (like) at {tag}"),
        );

        // y/z components should be near zero (symmetry).
        let fy = f_sr_1.y + f_lr_1.y;
        let fz = f_sr_1.z + f_lr_1.z;
        assert!(
            fy.abs() < 0.01 * f_vac_x.abs(),
            "{tag}: Fy should be ~0, got {fy:.4e}"
        );
        assert!(
            fz.abs() < 0.01 * f_vac_x.abs(),
            "{tag}: Fz should be ~0, got {fz:.4e}"
        );
    }
}

// ------------------------------------------------------------------
// Force–energy consistency (numerical gradient)
// ------------------------------------------------------------------

/// Gold-standard consistency check: the x-force on charge 1 must equal
/// −∂E/∂x₁ computed via a central-difference numerical gradient of the total
/// SPME energy.
///
/// This catches sign errors, missing prefactors, or broken B-spline derivative
/// weights in `gather_forces_from_potential`.  Uses 2 % relative tolerance to
/// account for mesh-discretisation differences between the energy and gradient
/// paths.
#[test]
fn test_spme_force_matches_energy_gradient() {
    let box_len = 50.0;
    let alpha = EWALD_ALPHA;
    let delta = 0.01; // Å

    for (dist, tag) in [(3.0, "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
        let center = box_len / 2.0;
        let r1 = Vec3::new(center - dist / 2.0, center, center);
        let r2 = Vec3::new(center + dist / 2.0, center, center);

        // Central-difference numerical gradient of total energy w.r.t. r1.x.
        let r1_plus = Vec3::new(r1.x + delta, r1.y, r1.z);
        let r1_minus = Vec3::new(r1.x - delta, r1.y, r1.z);

        let (_, (_, _, e_plus)) = spme_pair_forces_energy(r1_plus, r2, 1.0, -1.0, box_len, alpha);
        let (_, (_, _, e_minus)) = spme_pair_forces_energy(r1_minus, r2, 1.0, -1.0, box_len, alpha);

        let fx_numerical = -(e_plus - e_minus) / (2.0 * delta);

        // Analytic SPME force on charge 1.
        let ((f_sr, f_lr, _), _) = spme_pair_forces_energy(r1, r2, 1.0, -1.0, box_len, alpha);

        let fx_computed = f_sr.x + f_lr.x;

        println!(
            "Force–gradient check at {tag}: fx_computed={fx_computed:.4}  \
                 fx_numerical={fx_numerical:.4} kcal/(mol·Å)"
        );

        assert_rel_close(
            fx_computed,
            fx_numerical,
            0.02,
            &format!("force = -dE/dx at {tag}"),
        );
    }
}
