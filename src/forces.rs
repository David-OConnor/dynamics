use lin_alg::f32::Vec3;
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};

use crate::non_bonded::LjModifier;

/// Coefficients that apply an `LjModifier`, precomputed for a given cutoff. Following the GROMACS
/// reference manual ("Modified non-bonded interactions"), each of the r⁻⁶ and r⁻¹² terms becomes
/// Φ(r) = r⁻ᵅ − A/3·d³ − B/4·d⁴ − C, with force α·r⁻⁽ᵅ⁺¹⁾ + A·d² + B·d³, where
/// d = max(r − r_switch, 0). A potential shift is the case A = B = 0, C = r_c⁻ᵅ.
///
/// We pass these to the CUDA kernels as-is, so the CPU and GPU share one set of coefficients.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LjModCoeffs {
    /// False for `LjModifier::None`, where we skip the correction.
    pub enabled: bool,
    /// Å
    pub cutoff: f32,
    /// Å
    pub r_switch: f32,
    pub a6: f32,
    pub b6: f32,
    pub c6: f32,
    pub a12: f32,
    pub b12: f32,
    pub c12: f32,
}

impl LjModCoeffs {
    pub fn new(modifier: LjModifier, cutoff: f32) -> Self {
        let mut result = Self {
            cutoff,
            ..Default::default()
        };

        // Compute in f64; these involve high powers of the cutoff.
        let rc = cutoff as f64;

        match modifier {
            LjModifier::None => (),
            LjModifier::PotentialShift => {
                result.enabled = true;
                result.c6 = rc.powi(-6) as f32;
                result.c12 = rc.powi(-12) as f32;
            }
            LjModifier::ForceSwitch { r_switch } => {
                let r1 = r_switch as f64;
                let w = rc - r1;

                let coeffs = |α: f64| {
                    let rc_pow = rc.powf(α + 2.);
                    let a = -α * ((α + 4.) * rc - (α + 1.) * r1) / (rc_pow * w.powi(2));
                    let b = α * ((α + 3.) * rc - (α + 1.) * r1) / (rc_pow * w.powi(3));
                    let c = rc.powf(-α) - a / 3. * w.powi(3) - b / 4. * w.powi(4);
                    (a as f32, b as f32, c as f32)
                };

                (result.a6, result.b6, result.c6) = coeffs(6.);
                (result.a12, result.b12, result.c12) = coeffs(12.);

                result.enabled = true;
                result.r_switch = r_switch;
            }
        }

        result
    }

    /// Returns (force correction, energy correction), at distance r. Add the force correction to
    /// the unmodified force magnitude (along tgt - src), and subtract the energy correction from
    /// the unmodified energy. Only valid for r ≤ the cutoff.
    pub fn correction(&self, r: f32, sigma: f32, eps: f32) -> (f32, f32) {
        let s2 = sigma * sigma;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;

        let d = (r - self.r_switch).max(0.);
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d2 * d2;

        let f = s12 * (self.a12 * d2 + self.b12 * d3) - s6 * (self.a6 * d2 + self.b6 * d3);
        let e = s12 * (self.a12 / 3. * d3 + self.b12 / 4. * d4 + self.c12)
            - s6 * (self.a6 / 3. * d3 + self.b6 / 4. * d4 + self.c6);

        (4. * eps * f, 4. * eps * e)
    }
}

/// CPU LJ force. See notes on `V_lj()`. We set up the inv_dist param to share computation
/// with short-range Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
/// This variant also computes energy.
pub fn force_e_lj(dir: Vec3, inv_dist: f32, sigma: f32, eps: f32) -> (Vec3, f32) {
    let sr = sigma * inv_dist;
    let sr2 = sr * sr;
    let sr4 = sr2 * sr2;
    let sr6 = sr4 * sr2;
    let sr12 = sr6 * sr6;

    let mag = 24. * eps * 2.0f32.mul_add(sr12, -sr6) * inv_dist;

    let energy = 4. * eps * (sr12 - sr6);
    (dir * mag, energy)
}

/// LJ force and energy, with a cutoff modifier applied. Callers apply the cutoff itself.
pub(crate) fn force_e_lj_mod(
    dir: Vec3,
    dist: f32,
    inv_dist: f32,
    sigma: f32,
    eps: f32,
    lj_mod: &LjModCoeffs,
) -> (Vec3, f32) {
    let (f, e) = force_e_lj(dir, inv_dist, sigma, eps);
    if !lj_mod.enabled {
        return (f, e);
    }

    let (f_corr, e_corr) = lj_mod.correction(dist, sigma, eps);
    (f + dir * f_corr, e - e_corr)
}

/// SIMD variant
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub fn force_e_lj_x8(dir: Vec3x8, inv_dist: f32x8, sigma: f32x8, eps: f32x8) -> (Vec3x8, f32x8) {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = f32x8::splat(24.) * eps * (f32x8::splat(2.) * sr12 - sr6) * inv_dist;

    let energy = f32x8::splat(4.) * eps * (sr12 - sr6);
    (dir * mag, energy)
}

/// SIMD variant of `force_e_lj_mod`. The modifier is branch-free across lanes; callers apply the
/// cutoff itself.
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub(crate) fn force_e_lj_mod_x8(
    dir: Vec3x8,
    dist: f32x8,
    inv_dist: f32x8,
    sigma: f32x8,
    eps: f32x8,
    lj_mod: &LjModCoeffs,
) -> (Vec3x8, f32x8) {
    let (f, e) = force_e_lj_x8(dir, inv_dist, sigma, eps);
    if !lj_mod.enabled {
        return (f, e);
    }

    let s6 = sigma.powi(6);
    let s12 = s6 * s6;

    let d = (dist - f32x8::splat(lj_mod.r_switch)).max(f32x8::splat(0.));
    let d2 = d * d;
    let d3 = d2 * d;
    let d4 = d2 * d2;

    let s = f32x8::splat;
    let f_corr = s12 * (s(lj_mod.a12) * d2 + s(lj_mod.b12) * d3)
        - s6 * (s(lj_mod.a6) * d2 + s(lj_mod.b6) * d3);
    let e_corr = s12 * (s(lj_mod.a12 / 3.) * d3 + s(lj_mod.b12 / 4.) * d4 + s(lj_mod.c12))
        - s6 * (s(lj_mod.a6 / 3.) * d3 + s(lj_mod.b6 / 4.) * d4 + s(lj_mod.c6));

    let four_eps = s(4.) * eps;
    (f + dir * (four_eps * f_corr), e - four_eps * e_corr)
}

/// SIMD variant. Note: Having this code compiled, then run on an AVX-512 system is fine;
/// just don't run it.
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub fn force_e_lj_x16(
    dir: Vec3x16,
    inv_dist: f32x16,
    sigma: f32x16,
    eps: f32x16,
) -> (Vec3x16, f32x16) {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = f32x16::splat(24.) * eps * (f32x16::splat(2.) * sr12 - sr6) * inv_dist;

    let energy = f32x16::splat(4.) * eps * (sr12 - sr6);
    (dir * mag, energy)
}

/// SIMD variant of `force_e_lj_mod`. The modifier is branch-free across lanes; callers apply the
/// cutoff itself.
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub(crate) fn force_e_lj_mod_x16(
    dir: Vec3x16,
    dist: f32x16,
    inv_dist: f32x16,
    sigma: f32x16,
    eps: f32x16,
    lj_mod: &LjModCoeffs,
) -> (Vec3x16, f32x16) {
    let (f, e) = force_e_lj_x16(dir, inv_dist, sigma, eps);
    if !lj_mod.enabled {
        return (f, e);
    }

    let s6 = sigma.powi(6);
    let s12 = s6 * s6;

    let d = (dist - f32x16::splat(lj_mod.r_switch)).max(f32x16::splat(0.));
    let d2 = d * d;
    let d3 = d2 * d;
    let d4 = d2 * d2;

    let s = f32x16::splat;
    let f_corr = s12 * (s(lj_mod.a12) * d2 + s(lj_mod.b12) * d3)
        - s6 * (s(lj_mod.a6) * d2 + s(lj_mod.b6) * d3);
    let e_corr = s12 * (s(lj_mod.a12 / 3.) * d3 + s(lj_mod.b12 / 4.) * d4 + s(lj_mod.c12))
        - s6 * (s(lj_mod.a6 / 3.) * d3 + s(lj_mod.b6 / 4.) * d4 + s(lj_mod.c6));

    let four_eps = s(4.) * eps;
    (f + dir * (four_eps * f_corr), e - four_eps * e_corr)
}
