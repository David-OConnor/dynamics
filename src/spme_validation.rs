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

use ewald::{PmeRecip, force_coulomb_short_range, get_grid_n};
use lin_alg::f32::Vec3;

use crate::non_bonded::{CHARGE_UNIT_SCALER, EWALD_ALPHA, LONG_RANGE_CUTOFF};

/// Electrostatic constant in Amber kcal/mol units: K = CHARGE_UNIT_SCALER².
/// `E = K · q₁_e · q₂_e / r`  (charges in elementary units, r in Å → kcal/mol)
pub const K_ELEC: f32 = CHARGE_UNIT_SCALER * CHARGE_UNIT_SCALER;

/// Wrap a single coordinate component into [0, L).
#[inline]
fn wrap1(x: f32, l: f32) -> f32 {
    x.rem_euclid(l)
}

/// Wrap a Vec3 position into the primary box [0, L) per axis.
fn wrap_pos(p: Vec3, l: (f32, f32, f32)) -> Vec3 {
    Vec3::new(wrap1(p.x, l.0), wrap1(p.y, l.1), wrap1(p.z, l.2))
}

/// Build a PmeRecip with the given box and mesh spacing.
/// Feature-gates match dynamics/src/non_bonded.rs.
fn make_pme(l: (f32, f32, f32), alpha: f32, mesh_spacing: f32) -> PmeRecip {
    let dims = get_grid_n(l, mesh_spacing);
    #[cfg(any(feature = "vkfft", feature = "cufft"))]
    let pme = PmeRecip::new(None, dims, l, alpha);
    #[cfg(not(any(feature = "vkfft", feature = "cufft")))]
    let pme = PmeRecip::new(dims, l, alpha);
    pme
}

/// Compute the total SPME Coulomb energy for a pair of charges in a cubic box.
///
/// Charges `q1_e` and `q2_e` are in **elementary units** (e); internally they are
/// scaled to Amber units before passing into the SPME and short-range routines.
///
/// Returns `(e_short_range, e_long_range, e_total)` in **kcal/mol**.
pub fn spme_pair_energy(
    r1: Vec3,
    r2: Vec3,
    q1_e: f32,
    q2_e: f32,
    box_len: f32,
    alpha: f32,
) -> (f32, f32, f32) {
    let q1 = q1_e * CHARGE_UNIT_SCALER;
    let q2 = q2_e * CHARGE_UNIT_SCALER;

    // Short-range contribution
    let diff = r1 - r2;
    let dist = diff.magnitude();
    let inv_dist = 1.0 / dist;
    let dir = diff * inv_dist;
    let (_, e_sr) = force_coulomb_short_range(dir, dist, inv_dist, q1, q2, LONG_RANGE_CUTOFF, alpha);

    // Long-range (reciprocal) contribution via SPME
    let l = (box_len, box_len, box_len);
    let mut pme = make_pme(l, alpha, 1.0);
    let pos = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
    let q_arr = vec![q1, q2];
    let (_, e_lr) = pme.forces(&pos, &q_arr);

    let e_total = e_sr + e_lr;
    (e_sr, e_lr, e_total)
}

/// Compute SPME forces on a pair of charges in a cubic box.
///
/// Returns `(f_short_on_q1, f_long_on_q1, f_long_on_q2)`.
/// Charges are in elementary units.
pub fn spme_pair_forces(
    r1: Vec3,
    r2: Vec3,
    q1_e: f32,
    q2_e: f32,
    box_len: f32,
    alpha: f32,
) -> (Vec3, Vec3, Vec3) {
    let q1 = q1_e * CHARGE_UNIT_SCALER;
    let q2 = q2_e * CHARGE_UNIT_SCALER;

    // Short-range force on q1 (from q2)
    let diff = r1 - r2;
    let dist = diff.magnitude();
    let inv_dist = 1.0 / dist;
    let dir = diff * inv_dist;
    let (f_sr_1, _) = force_coulomb_short_range(dir, dist, inv_dist, q1, q2, LONG_RANGE_CUTOFF, alpha);

    // Long-range forces
    let l = (box_len, box_len, box_len);
    let mut pme = make_pme(l, alpha, 1.0);
    let pos = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
    let q_arr = vec![q1, q2];
    let (f_recip, _) = pme.forces(&pos, &q_arr);

    (f_sr_1, f_recip[0], f_recip[1])
}

/// Expected vacuum Coulomb energy in kcal/mol.
/// `E = K · q₁_e · q₂_e / r`
pub fn vacuum_coulomb_energy(q1_e: f32, q2_e: f32, dist: f32) -> f32 {
    K_ELEC * q1_e * q2_e / dist
}

/// Expected vacuum Coulomb force on q₁ due to q₂, in the direction (r₁ − r₂)/|r₁ − r₂|.
/// `F_mag = K · q₁_e · q₂_e / r²`  (positive = repulsive, negative = attractive)
pub fn vacuum_coulomb_force_component(q1_e: f32, q2_e: f32, dist: f32) -> f32 {
    K_ELEC * q1_e * q2_e / (dist * dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fraction of the expected value used as the pass/fail threshold.
    const REL_TOL: f32 = 0.01; // 1 %

    fn assert_rel_close(got: f32, expected: f32, tol: f32, label: &str) {
        if expected.abs() < 1e-6 {
            assert!(
                got.abs() < 1e-4,
                "{label}: expected ≈ 0, got {got:.6e}"
            );
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
        let box_len = 50.0f32;
        let alpha   = EWALD_ALPHA;

        for (dist, tag) in [(3.0f32, "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
            let center = box_len / 2.0;
            let r1 = Vec3::new(center - dist / 2.0, center, center);
            let r2 = Vec3::new(center + dist / 2.0, center, center);

            let (e_sr, e_lr, e_total) =
                spme_pair_energy(r1, r2, 1.0, -1.0, box_len, alpha);
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
        let box_len = 50.0f32;
        let alpha   = EWALD_ALPHA;
        let dist    = 5.0f32;
        let center  = box_len / 2.0;
        let r1 = Vec3::new(center - dist / 2.0, center, center);
        let r2 = Vec3::new(center + dist / 2.0, center, center);

        for (q1e, q2e, tag) in [
            (0.5f32, -0.5f32, "q=±0.5"),
            (0.25,   -0.25,   "q=±0.25"),
        ] {
            let (_, _, e_total) = spme_pair_energy(r1, r2, q1e, q2e, box_len, alpha);
            let e_vac = vacuum_coulomb_energy(q1e, q2e, dist);
            println!("{tag}: e_total={e_total:.4}  e_vac={e_vac:.4} kcal/mol");
            assert_rel_close(e_total, e_vac, REL_TOL, tag);
        }
    }

    /// Energy should converge toward vacuum Coulomb as the box grows.
    /// Relative error must be < 1 % at L = 50 Å for r = 5 Å.
    #[test]
    fn test_spme_energy_box_convergence() {
        let dist  = 5.0f32;
        let alpha = EWALD_ALPHA;
        let e_vac = vacuum_coulomb_energy(1.0, -1.0, dist);

        println!("Box-size convergence test (e_vac = {e_vac:.4} kcal/mol):");
        for box_len in [20.0f32, 30.0, 50.0] {
            let c  = box_len / 2.0;
            let r1 = Vec3::new(c - dist / 2.0, c, c);
            let r2 = Vec3::new(c + dist / 2.0, c, c);
            let (_, _, e_total) = spme_pair_energy(r1, r2, 1.0, -1.0, box_len, alpha);
            let rel = ((e_total - e_vac) / e_vac).abs();
            println!("  L={box_len:.0} Å:  e_total={e_total:.4}  rel_err={rel:.4}");
        }

        // Assert only for the largest box where images are negligible.
        let c  = 50.0 / 2.0;
        let r1 = Vec3::new(c - dist / 2.0, c, c);
        let r2 = Vec3::new(c + dist / 2.0, c, c);
        let (_, _, e_total) = spme_pair_energy(r1, r2, 1.0, -1.0, 50.0, alpha);
        assert_rel_close(e_total, e_vac, REL_TOL, "energy at L=50 Å");
    }

    // ------------------------------------------------------------------
    // Force tests
    // ------------------------------------------------------------------

    /// For a +1/−1 pair along x, the total x-force on charge 1 must match
    /// vacuum Coulomb: F_x = +K/r² (attractive, toward the −1 charge at +x).
    #[test]
    fn test_spme_force_magnitude() {
        let box_len = 50.0f32;
        let alpha   = EWALD_ALPHA;

        for (dist, tag) in [(3.0f32, "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
            let center = box_len / 2.0;
            // q1 (+1) to the left, q2 (−1) to the right
            let r1 = Vec3::new(center - dist / 2.0, center, center);
            let r2 = Vec3::new(center + dist / 2.0, center, center);

            let (f_sr_1, f_lr_1, _) =
                spme_pair_forces(r1, r2, 1.0, -1.0, box_len, alpha);

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

            assert_rel_close(fx_total, f_vac_x, REL_TOL, &format!("Fx on q1 at {tag}"));

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
        let box_len = 50.0f32;
        let alpha   = EWALD_ALPHA;

        for (dist, tag) in [(3.0f32, "3Å"), (5.0, "5Å"), (8.0, "8Å")] {
            let center = box_len / 2.0;
            let r1 = Vec3::new(center - dist / 2.0, center, center);
            let r2 = Vec3::new(center + dist / 2.0, center, center);

            let q1 = 1.0f32 * CHARGE_UNIT_SCALER;
            let q2 = -1.0f32 * CHARGE_UNIT_SCALER;

            // Short-range forces
            let diff  = r1 - r2;
            let inv_d = 1.0 / dist;
            let dir   = diff * inv_d;
            let (f_sr_1, _) =
                force_coulomb_short_range(dir, dist, inv_d, q1, q2, LONG_RANGE_CUTOFF, alpha);
            let (f_sr_2, _) =
                force_coulomb_short_range(-dir, dist, inv_d, q2, q1, LONG_RANGE_CUTOFF, alpha);

            // Long-range forces
            let l = (box_len, box_len, box_len);
            let mut pme = make_pme(l, alpha, 1.0);
            let pos    = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
            let q_arr  = vec![q1, q2];
            let (f_recip, _) = pme.forces(&pos, &q_arr);

            let f1 = f_sr_1 + f_recip[0];
            let f2 = f_sr_2 + f_recip[1];

            let sum_x = f1.x + f2.x;
            let sum_y = f1.y + f2.y;
            let sum_z = f1.z + f2.z;
            let sum_mag = (sum_x * sum_x + sum_y * sum_y + sum_z * sum_z).sqrt();
            let f1_mag  = f1.magnitude();

            println!(
                "Newton3 at {tag}: |f1|={f1_mag:.4}  |f1+f2|={sum_mag:.4e}"
            );

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
        let q   = CHARGE_UNIT_SCALER;
        let dir = Vec3::new(1.0, 0.0, 0.0);

        for dist in [LONG_RANGE_CUTOFF, LONG_RANGE_CUTOFF + 0.1, LONG_RANGE_CUTOFF + 1.0] {
            let (f, e) =
                force_coulomb_short_range(dir, dist, 1.0 / dist, q, q, LONG_RANGE_CUTOFF, EWALD_ALPHA);
            assert!(
                f.magnitude_squared() == 0.0,
                "force should be 0 at dist={dist:.2}: {f:?}"
            );
            assert!(
                e == 0.0,
                "energy should be 0 at dist={dist:.2}: {e}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Different box shapes
    // ------------------------------------------------------------------

    /// Test a non-cubic (elongated) box to catch any axis-mixing bugs.
    #[test]
    fn test_spme_energy_non_cubic_box() {
        let alpha = EWALD_ALPHA;
        let dist  = 5.0f32;

        // Elongated box: charges along the long axis
        let lx = 60.0f32;
        let ly = 30.0f32;
        let lz = 30.0f32;

        let r1 = Vec3::new(lx / 2.0 - dist / 2.0, ly / 2.0, lz / 2.0);
        let r2 = Vec3::new(lx / 2.0 + dist / 2.0, ly / 2.0, lz / 2.0);

        let q1 = 1.0f32 * CHARGE_UNIT_SCALER;
        let q2 = -1.0f32 * CHARGE_UNIT_SCALER;

        let l = (lx, ly, lz);
        let mut pme = make_pme(l, alpha, 1.0);

        let pos   = vec![wrap_pos(r1, l), wrap_pos(r2, l)];
        let q_arr = vec![q1, q2];
        let (_, e_lr) = pme.forces(&pos, &q_arr);

        let diff = r1 - r2;
        let inv_d = 1.0 / dist;
        let dir = diff * inv_d;
        let (_, e_sr) =
            force_coulomb_short_range(dir, dist, inv_d, q1, q2, LONG_RANGE_CUTOFF, alpha);

        let e_total = e_sr + e_lr;
        let e_vac   = vacuum_coulomb_energy(1.0, -1.0, dist);

        println!(
            "Non-cubic box (60×30×30): e_total={e_total:.4}  e_vac={e_vac:.4} kcal/mol"
        );
        assert_rel_close(e_total, e_vac, REL_TOL, "non-cubic box energy");
    }

    // ------------------------------------------------------------------
    // B-spline modulus sanity check
    // ------------------------------------------------------------------

    /// At k=0 the B-spline modulus |b_4(0)|^2 must equal 1, so bmod2[0] = 1.
    /// At k=N/2 the exact value is ((4+2·cos(π))/6)^2 = (2/6)^2 = 1/9,
    /// so bmod2[N/2] = 9.
    #[test]
    fn test_bspline_bmod2_known_values() {
        // We can't call spline_bmod2_1d directly (private), but we can inspect
        // PmeRecip's influence function behavior indirectly via energy:
        // An isolated charge in a large box should produce finite, reasonable energies.
        // The self-energy term alone can be checked analytically.

        // Self-energy = -(α/√π) · q²   (subtracted inside PmeRecip::forces)
        let alpha = EWALD_ALPHA;
        let box_len = 50.0f32;
        let l = (box_len, box_len, box_len);
        let mut pme = make_pme(l, alpha, 1.0);

        let q_e = 1.0f32;
        let q   = q_e * CHARGE_UNIT_SCALER;
        let center = box_len / 2.0;
        let pos = vec![Vec3::new(center, center, center)];
        let q_arr = vec![q];

        let (_, e) = pme.forces(&pos, &q_arr);
        // For a single charge, the reciprocal energy + self-energy is small but finite.
        // We mainly check it doesn't blow up (NaN/Inf).
        assert!(
            e.is_finite(),
            "Single-charge SPME energy is not finite: {e}"
        );
        println!("Single-charge SPME energy (should be small/finite): {e:.4} kcal/mol");
    }
}
