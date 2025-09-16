use lin_alg::f32::Vec3;

/// See notes on `V_lj()`. We set up the dist params we do to share computation
/// with Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
/// This variant also computes energy.
pub fn force_e_lj(dir: Vec3, inv_dist: f32, sigma: f32, eps: f32) -> (Vec3, f32) {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    // todo: mul_add is unstable. add later
    let mag = 24. * eps * (2. * sr12 - sr6) * inv_dist;
    // let mag = 24. * eps * mul_add(2., sr12,  -sr6) * inv_dist;

    let energy = 4. * eps * (sr12 - sr6);
    (dir * mag, energy)
}
