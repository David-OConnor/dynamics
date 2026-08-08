use bio_files::md_params::{AngleBendingParams, BondStretchingParams, DihedralParams};
use lin_alg::f32::{Vec3, calc_dihedral_angle_v2};

use crate::barostat::SimBox;

const EPS: f32 = 1e-8;

/// Returns the force on the atom at position 0. Negate this for the force on posit 1.
/// Also returns potential energy.
/// For a reference, see the `Bond stretching` section of [this guide](https://manual.gromacs.org/2024.4/reference-manual/functions/bonded-interactions.html)
pub fn f_bond_stretching(
    posit_0: Vec3,
    posit_1: Vec3,
    params: &BondStretchingParams,
    cell: &SimBox,
) -> (Vec3, f32) {
    let diff = cell.min_image(posit_1 - posit_0);
    let r_meas = diff.magnitude();

    let r_delta = r_meas - params.r_0;

    // Derivative of the Amber harmonic potential k_b * (r - r_0)^2.
    let term_1 = 2. * params.k_b * r_delta;

    // Unit check: kcal/mol/Å² * Å² = kcal/mol. (Energy).
    let f_mag = term_1 / r_meas.max(EPS);

    // Amber harmonic bond potential: U = k_b (r - r_0)^2.
    // Its derivative is the 2*k_b*r_delta term used for the force above.
    let energy = params.k_b * r_delta * r_delta;

    (diff * f_mag, energy)
}

/// Valence angle; angle between 3 atoms.
/// Also returns potential energy.
pub fn f_angle_bending(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    params: &AngleBendingParams,
    cell: &SimBox,
) -> ((Vec3, Vec3, Vec3), f32) {
    // Bond vectors with atom 1 at the vertex.
    let bond_vec_01 = cell.min_image(posit_0 - posit_1);
    let bond_vec_21 = cell.min_image(posit_2 - posit_1);

    let b_vec_01_sq = bond_vec_01.magnitude_squared();
    let b_vec_21_sq = bond_vec_21.magnitude_squared();

    // Quit early if atoms are on top of each other
    if b_vec_01_sq < EPS || b_vec_21_sq < EPS {
        return ((Vec3::new_zero(), Vec3::new_zero(), Vec3::new_zero()), 0.);
    }

    let b_vec_01_len = b_vec_01_sq.sqrt();
    let b_vec_21_len = b_vec_21_sq.sqrt();

    let unit_01 = bond_vec_01 / b_vec_01_len;
    let unit_21 = bond_vec_21 / b_vec_21_len;
    let cos_theta = unit_01.dot(unit_21).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(EPS);

    // Amber harmonic angle potential: U = k (theta - theta_0)^2.
    let delta_theta = theta - params.theta_0;
    let d_energy_d_theta = 2.0 * params.k * delta_theta;

    // Cartesian gradients of theta for the two outer atoms.
    let grad_0 = (unit_01 * cos_theta - unit_21) / (b_vec_01_len * sin_theta);
    let grad_2 = (unit_21 * cos_theta - unit_01) / (b_vec_21_len * sin_theta);

    let f_0 = -grad_0 * d_energy_d_theta;
    let f_2 = -grad_2 * d_energy_d_theta;
    let f_1 = -(f_0 + f_2);

    let f = (f_0, f_1, f_2);
    let energy = params.k * delta_theta * delta_theta;

    (f, energy)
}

/// See Amber reference manual 2025, section 15.1: Torsion Terms and Out-of-Plane Terms.
pub fn f_dihedral(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    posit_3: Vec3,
    // There can be multiple terms.
    params: &[DihedralParams],
    cell: &SimBox,
) -> ((Vec3, Vec3, Vec3, Vec3), f32) {
    let b1 = cell.min_image(posit_1 - posit_0); // r_ij
    let b2 = cell.min_image(posit_2 - posit_1); // r_kj
    let b3 = cell.min_image(posit_3 - posit_2); // r_lk

    // Normal vectors to the two planes
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);

    let n1_sq = n1.magnitude_squared();
    let n2_sq = n2.magnitude_squared();
    let b2_sq = b2.magnitude_squared();
    let b2_len = b2_sq.sqrt();

    // Bail out if dihedral is ill-defined (prevents singular impulses)
    const DIH_TOL: f32 = 1.0e-6;
    if n1_sq < DIH_TOL || n2_sq < DIH_TOL || b2_len < DIH_TOL {
        return (
            (
                Vec3::new_zero(),
                Vec3::new_zero(),
                Vec3::new_zero(),
                Vec3::new_zero(),
            ),
            0.0,
        );
    }

    // Reconstruct PBC-consistent positions from the already-wrapped bond vectors,
    // so the measured angle is correct even if atoms straddle a periodic boundary.
    let p1 = posit_0 + b1;
    let p2 = p1 + b2;
    let p3 = p2 + b3;
    // `lin_alg` measures this signed rotation in the opposite direction from
    // the Amber/GROMACS proper-dihedral convention. Keep the raw angle for
    // the Cartesian gradient below, but reverse it when evaluating the
    // potential.
    let dihe_raw = calc_dihedral_angle_v2(&(posit_0, p1, p2, p3));
    let dihe_measured = std::f32::consts::TAU - dihe_raw;

    let mut energy = 0.;
    // Derivative with respect to `dihe_raw` (not `dihe_measured`).
    let mut dV_dφ = 0.; // Scalar torque magnitude

    for param in params {
        // Note: We have already divided barrier height by the integer divisor when setting up
        // the Indexed params.
        let k = param.barrier_height;
        let per = param.periodicity as f32;

        let dφ = per * dihe_measured - param.phase;
        dV_dφ += k * per * dφ.sin();
        energy += k * (1.0 + dφ.cos());
    }

    // Force Vector Calculation (Blondel-Karplus Method)
    // This formulation automatically handles the "lever arm" effect on atoms 1 and 2.

    // Gradient terms for the outer atoms
    // F0 acts along n1 (perpendicular to bond b1 and b2)
    // F3 acts along n2 (perpendicular to bond b2 and b3)
    let f0_factor = -dV_dφ * b2_len / n1_sq;
    let f3_factor = dV_dφ * b2_len / n2_sq; // Note the sign flip relative to f0

    let f_0 = n1 * f0_factor;
    let f_3 = n2 * f3_factor;

    // Gradient terms for the inner atoms (1 and 2)
    // We project the dot products of bonds to distribute torque
    let dot_b1_b2 = b1.dot(b2);
    let dot_b3_b2 = b3.dot(b2);

    let term_a = dot_b1_b2 / b2_sq;
    let term_b = dot_b3_b2 / b2_sq;

    // F_1 = -F_0  -  term_a * F_0  +  term_b * F_3
    // Explanation:
    // -F_0: Newton's 3rd law reaction to Atom 0
    // -term_a * F_0: Additional torque because Atom 1 is the hinge for Plane 1
    // +term_b * F_3: Reaction to the torque from Plane 2
    let f_1 = -f_0 - (f_0 * term_a) + (f_3 * term_b);

    // F_2 = -F_3  -  term_b * F_3  +  term_a * F_0
    let f_2 = -f_3 - (f_3 * term_b) + (f_0 * term_a);

    ((f_0, f_1, f_2, f_3), energy)
}
