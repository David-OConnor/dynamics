use lin_alg::f32::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};

#[cfg(target_arch = "x86_64")]

/// CPU LJ force. See notes on `V_lj()`. We set up the inv_dist param to share computation
/// with short-range Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
/// This variant also computes energy.
pub fn force_e_lj(dir: Vec3, inv_dist: f32, sigma: f32, eps: f32) -> (Vec3, f32) {
    // return (Vec3::new_zero(), 0.);
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = 24. * eps * 2.0f32.mul_add(sr12,  -sr6) * inv_dist;

    let energy = 4. * eps * (sr12 - sr6);
    (dir * mag, energy)
}

/// SIMD variant
#[cfg(target_arch = "x86_64")]
pub fn force_e_lj_x8(dir: Vec3x8, inv_dist: f32x8, sigma: f32x8, eps: f32x8) -> (Vec3x8, f32x8) {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = f32x8::splat(24.) * eps * (f32x8::splat(2.) * sr12 - sr6) * inv_dist;

    let energy = f32x8::splat(4.) * eps * (sr12 - sr6);
    (dir * mag, energy)
}

/// SIMD variant. Note: Having this code compiled, then run on an AVX-512 system is fine;
/// just don't run it.
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
