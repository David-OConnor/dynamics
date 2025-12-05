use bio_files::md_params::{AngleBendingParams, BondStretchingParams, DihedralParams};
use lin_alg::f32::{Vec3, calc_dihedral_angle_v2};

const EPS: f32 = 1e-8;

/// Returns the force on the atom at position 0. Negate this for the force on posit 1.
/// Also returns potential energy.
/// For a reference, see the `Bond stretching` section of [this guide](https://manual.gromacs.org/2024.4/reference-manual/functions/bonded-interactions.html)
pub fn f_bond_stretching(
    posit_0: Vec3,
    posit_1: Vec3,
    params: &BondStretchingParams,
) -> (Vec3, f32) {
    let diff = posit_1 - posit_0;
    let r_meas = diff.magnitude();

    let r_delta = r_meas - params.r_0;

    // Pre-scaled by 2 when building the Indexed params.
    let term_1 = 2. * params.k_b * r_delta; // Shared bewteen force and energy.

    // Unit check: kcal/mol/Å² * Å² = kcal/mol. (Energy).
    let f_mag = term_1 / r_meas.max(EPS);

    // U = 2k_b x rΔ^2
    let energy = term_1 * r_delta;

    (diff * f_mag, energy)
}

/// Valence angle; angle between 3 atoms.
/// Also returns potential energy.
pub fn f_angle_bending(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    params: &AngleBendingParams,
) -> ((Vec3, Vec3, Vec3), f32) {
    // Bond vectors with atom 1 at the vertex.
    let bond_vec_01 = posit_0 - posit_1;
    let bond_vec_21 = posit_2 - posit_1;

    let b_vec_01_sq = bond_vec_01.magnitude_squared();
    let b_vec_21_sq = bond_vec_21.magnitude_squared();

    // Quit early if atoms are on top of each other
    if b_vec_01_sq < EPS || b_vec_21_sq < EPS {
        return ((Vec3::new_zero(), Vec3::new_zero(), Vec3::new_zero()), 0.);
    }

    let b_vec_01_len = b_vec_01_sq.sqrt();
    let b_vec_21_len = b_vec_21_sq.sqrt();

    let inv_ab = 1.0 / (b_vec_01_len * b_vec_21_len);

    let cos_θ = (bond_vec_01.dot(bond_vec_21) * inv_ab).clamp(-1.0, 1.0);
    let θ = cos_θ.acos();

    let Δθ = params.theta_0 - θ;
    let dV_dθ = 2. * params.k * Δθ;

    // c is the vector normal to the plane defined by the 3 atoms
    let c = bond_vec_01.cross(bond_vec_21);
    let c_len_sq = c.magnitude_squared().max(EPS);

    let grad_atom0 = (c.cross(bond_vec_01) * b_vec_21_len) / (c_len_sq * b_vec_01_len);
    let grad_atom1 = (bond_vec_21.cross(c) * b_vec_01_len) / (c_len_sq * b_vec_21_len);

    let f_0 = -grad_atom0 * dV_dθ;
    let f_2 = -grad_atom1 * dV_dθ;
    let f_1 = -(f_0 + f_2); // Newton's 3rd law

    let f = (f_0, f_1, f_2);
    let energy = dV_dθ * Δθ;

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
) -> ((Vec3, Vec3, Vec3, Vec3), f32) {
    let b1 = posit_1 - posit_0; // r_ij
    let b2 = posit_2 - posit_1; // r_kj
    let b3 = posit_3 - posit_2; // r_lk

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

    let dihe_measured = calc_dihedral_angle_v2(&(posit_0, posit_1, posit_2, posit_3));

    let mut energy = 0.;
    let mut dV_dφ = 0.; // Scalar torque magnitude

    for param in params {
        // Note: We have already divided barrier height by the integer divisor when setting up
        // the Indexed params.
        let k = param.barrier_height;
        let per = param.periodicity as f32;

        let dφ = per * dihe_measured - param.phase;
        dV_dφ += -k * per * dφ.sin();
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
