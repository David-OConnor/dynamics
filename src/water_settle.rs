//! This module implements the SETTLE algorithm for rigid water molecules.

use lin_alg::f32::Vec3;

use crate::{
    ACCEL_CONVERSION_INV,
    ambient::SimBox,
    water_opc::{H_MASS, H_O_H_θ, MASS_WATER_MOL, O_EP_R, O_H_R, O_MASS, WaterMol},
};

// Reset the water angle to the defined parameter every this many steps,
// to counter numerical drift
pub(crate) const RESET_ANGLE_RATIO: usize = 1_000;

// ---------- SETTLE constants

// Pre-calculate these constants for OPC geometry!
// RA: Distance from O to the midpoint of the H-H line.
// RB: Distance from the H-H midpoint to an H atom (half the H-H distance).
// RC: Distance from O to the Center of Mass.
// geometry:
//       O
//       | (ra)
//   H --+-- H
//     (rb)

// Example for standard OPC (verify your specific constants):
// theta_rad = 103.6 * pi / 180
// bond_len = 0.872433
// ra = bond_len * cos(theta_rad / 2.0)
// rb = bond_len * sin(theta_rad / 2.0)
// rc = ra * (2.0 * H_MASS) / (O_MASS + 2.0 * H_MASS)

// todo: Replace these with calculated values.

// Pre-calcualted, as consts don't support COS, SIN.
pub(crate) const RA: f32 = 0.5395199719801114; // O_H_R * (H_O_H_θ / 2.).cos()
const RB: f32 = 0.6856075890450577; // O_H_R * (H_O_H_θ / 2.).sin()
const RC: f32 = RA * (2.0 * H_MASS) / (O_MASS + 2.0 * H_MASS);

// ---------- end SETTLE constants

/// Analytic SETTLE for OPC water.
/// 1. Drifts atoms forward by dt.
/// 2. Applies constraints analytically.
/// 3. Updates velocities based on the constraint correction.
/// 4. Updates the M-site (Virtual Site).
pub(crate) fn settle_drift_analytic(
    mol: &mut WaterMol,
    dt: f32,
    cell: &SimBox,
    virial_constr: &mut f64,
) {
    let dt_inv = 1.0 / dt;

    // 1. Unconstrained Drift (Predictor)
    // ---------------------------------------------------------
    let mut o_pos = mol.o.posit + mol.o.vel * dt;
    // We work in the O-centered frame for H's to handle PBC safely immediately
    let mut h0_pos = o_pos + cell.min_image((mol.h0.posit + mol.h0.vel * dt) - o_pos);
    let mut h1_pos = o_pos + cell.min_image((mol.h1.posit + mol.h1.vel * dt) - o_pos);

    // 2. Center of Mass & Coordinate System Construction
    // ---------------------------------------------------------
    // Current (unconstrained) COM
    let com = (o_pos * O_MASS + h0_pos * H_MASS + h1_pos * H_MASS) / MASS_WATER_MOL;

    // Shift positions relative to COM
    let mut r_o = o_pos - com;
    let mut r_h0 = h0_pos - com;
    let mut r_h1 = h1_pos - com;

    // Construct local orthonormal coordinate system (a, b, c)
    // 'a' points along the bisector (roughly O -> H-H line)
    // 'b' points along the H-H vector
    // 'c' is perpendicular to the plane

    // a_vec = r_o (vector from COM to O).
    // Note: In unconstrained step, O might drift slightly off bisector,
    // but usually we define the axis based on the triangle orientation.
    // Canonical SETTLE defines axes based on the positions *relative to COM*.

    // Vector from O to midpoint of H0-H1
    // let midpoint = (r_h0 + r_h1) * 0.5;
    let mut a_vec = r_o; // Initial guess for bisector axis (pointing to O)

    // Gram-Schmidt / Cross-product method to ensure orthogonality
    let d_hh = r_h1 - r_h0; // Vector between H's
    let c_vec = d_hh.cross(a_vec); // Normal to plane

    // Recalculate 'b' and 'a' to be perfectly orthogonal
    let b_vec = d_hh;
    // a_vec = b x c (points from H-H line towards O)
    a_vec = b_vec.cross(c_vec);

    // Normalize
    let ax = a_vec.to_normalized();
    let bx = b_vec.to_normalized();
    // let cx = c_vec.to_normalized();

    // 3. Analytic Constraint Solution
    // ---------------------------------------------------------
    // We project the unconstrained positions onto these axes.
    // Due to symmetry, we only need the Z-coordinate (along 'a') and Y-coordinate (along 'b').
    // (Using standard SETTLE notation where Z is bisector, Y is HH vector)

    // Distances of atoms from COM along axis 'a' (bisector)
    let o_z = r_o.dot(ax);
    // let h0_z = r_h0.dot(ax);
    // let h1_z = r_h1.dot(ax);

    // Distances along axis 'b' (H-H vector)
    // O is at 0 in 'b' by definition of the bisector construction
    let h0_y = r_h0.dot(bx);
    let h1_y = r_h1.dot(bx);

    // -- Solve for H-H separation (linear) --
    // We need to move H0 and H1 along 'b' so their distance becomes 2*RB.
    // The separation in unconstrained is (h1_y - h0_y).
    // Weighting factor: mH / (mH + mH) = 0.5 for relative motion, but we use reduced mass concept.
    // Actually, SETTLE simplifies this:
    // We simply reset y-coords to perfect symmetry +/- RB.
    // The shift required:
    let d_hh_unconstr = h1_y - h0_y;
    let target_d_hh = 2.0 * RB;
    // let gamma = (target_d_hh - d_hh_unconstr) * 0.5;

    // Only H's move along 'b', equal and opposite.

    // -- Solve for O-(HH midpoint) separation (quadratic) --
    // We need to move O and H's along 'a' (bisector) to restore geometry.
    // Target: O is at +RC (relative to COM), H's are at -(RA-RC).
    // Let phi be the shift factor.
    // This is the classic SETTLE quadratic: A*phi^2 + B*phi + C = 0
    // But since we built the frame *on* the atoms, we can often just snap them
    // to the known geometry relative to the COM if we trust the COM velocity is constant.

    // Simplified SETTLE reconstruction (Standard "SHAKE-like" SETTLE):
    // 1. Calculate 'sin phi' needed to restore OH bond length given the fixed HH width.
    // This is actually much simpler:
    // The COM is invariant. The axes (ax, bx, cx) represent the rigid body orientation.
    // We just place the atoms at their defined coordinates in this frame!

    // Reconstruct positions in global frame:
    // O is at distance RC from COM along -ax (or +ax depending on direction)
    // H's are at distance (RA-RC) from COM along +ax, and +/- RB along bx.

    // Verify direction of ax: defined as b x c.
    // b = H0->H1. c = plane normal. a points roughly O -> Midpoint?
    // Let's check: r_o dot ax. If positive, O is in +ax direction.

    // Re-assign rigid body positions relative to COM
    let sign = if o_z > 0.0 { 1.0 } else { -1.0 };

    // New local positions
    let new_ro = ax * (sign * RC);
    let center_h = ax * (sign * (RC - RA)); // Midpoint of H-H
    let new_rh0 = center_h - bx * RB;
    let new_rh1 = center_h + bx * RB;

    // 4. Update Global State
    // ---------------------------------------------------------
    let final_o = com + new_ro;
    let final_h0 = com + new_rh0;
    let final_h1 = com + new_rh1;

    // Wrap and commit positions
    mol.o.posit = cell.wrap(final_o);
    // Use min image to keep molecule together visually/topologically
    mol.h0.posit = mol.o.posit + cell.min_image(final_h0 - final_o);
    mol.h1.posit = mol.o.posit + cell.min_image(final_h1 - final_o);

    // 5. Update Velocities (Constraint Force application)
    // ---------------------------------------------------------
    // v_new = (r_new - r_old_at_start_of_step) / dt
    // BUT we drifted r_old at step 1.
    // We need the position *before* the drift.
    // r_old = (r_drifted - v_old * dt).
    // Actually, simplest is: v_new = (final_pos - initial_pos) / dt.

    // We need the original positions passed in? Or we can back-calculate:
    // r_initial = o_pos_drifted - mol.o.vel * dt;
    // v_new = (final_o - r_initial) / dt;
    // Simplifies to: v_new = mol.o.vel + (final_o - o_pos_drifted) / dt;

    let correction_o = (final_o - o_pos) * dt_inv;
    let correction_h0 = (final_h0 - h0_pos) * dt_inv;
    let correction_h1 = (final_h1 - h1_pos) * dt_inv;

    mol.o.vel += correction_o;
    mol.h0.vel += correction_h0;
    mol.h1.vel += correction_h1;

    // 6. Calculate Constraint Virial
    // ---------------------------------------------------------
    // W = Sum( r_ref . F_constraint )
    // F_constraint = m * correction / dt  (units: Mass * Length / Time^2)
    // We assume Virial is in Energy units.
    // Note: Use the midpoint or the new position for r_ref?
    // Standard is usually r_new.

    let fc_o = correction_o * (O_MASS * dt_inv);
    let fc_h0 = correction_h0 * (H_MASS * dt_inv);
    let fc_h1 = correction_h1 * (H_MASS * dt_inv);

    // accumulate virial (dot product)
    *virial_constr += (final_o.dot(fc_o) + final_h0.dot(fc_h0) + final_h1.dot(fc_h1)) as f64;

    // 7. Update Virtual Site (M-Site)
    // ---------------------------------------------------------
    // ALWAYS update this after moving O and H.
    let bisector = (mol.h0.posit - mol.o.posit) + (mol.h1.posit - mol.o.posit);
    mol.m.posit = mol.o.posit + bisector.to_normalized() * O_EP_R;
    mol.m.vel = (mol.h0.vel + mol.h1.vel) * 0.5;
}

/// Analytic SETTLE implementation for 3‑site rigid water (Miyamoto & Kollman, JCC 1992).
/// Works for any bond length / HOH angle. This handles the drift (position updates) for water
/// molecules. It also places M on the bisector, and performs a rigid wrap.
///
/// All distances & masses are in MD internal units (Å, ps, amu, kcal/mol).
///
/// This is handles the Verlet "drift" for a rigid molecule. It is the equivalent
/// of updating position by adding velocity x dt, but also maintains the rigid
/// geometry of 3-atom molecules.
pub(crate) fn integrate_rigid_water(
    mol: &mut WaterMol,
    dt: f32,
    cell: &SimBox,
    virial_constr_kcal: &mut f64,
) {
    let o_pos = mol.o.posit;
    let h0_pos_local = o_pos + cell.min_image(mol.h0.posit - o_pos);
    let h1_pos_local = o_pos + cell.min_image(mol.h1.posit - o_pos);

    // COM position & velocity at start of the drift/rotation substep
    let r_com =
        (mol.o.posit * O_MASS + h0_pos_local * H_MASS + h1_pos_local * H_MASS) / MASS_WATER_MOL;
    let v_com = (mol.o.vel * O_MASS + mol.h0.vel * H_MASS + mol.h1.vel * H_MASS) / MASS_WATER_MOL;

    // Shift to COM frame
    let (rO, rH0, rH1) = (o_pos - r_com, h0_pos_local - r_com, h1_pos_local - r_com);
    let (vO, vH0, vH1) = (mol.o.vel - v_com, mol.h0.vel - v_com, mol.h1.vel - v_com);

    // Angular momentum about COM
    let L = rO.cross(vO) * O_MASS + rH0.cross(vH0) * H_MASS + rH1.cross(vH1) * H_MASS;

    // inertia tensor about COM (symmetric 3×3)
    let accI = |r: Vec3, m: f32| {
        let x = r.x;
        let y = r.y;
        let z = r.z;
        let r2 = r.dot(r);
        (
            m * (r2 - x * x),
            m * (r2 - y * y),
            m * (r2 - z * z),
            -m * x * y,
            -m * x * z,
            -m * y * z,
        )
    };
    let (iOxx, iOyy, iOzz, iOxy, iOxz, iOyz) = accI(rO, O_MASS);
    let (iH0x, iH0y, iH0z, iH0xy, iH0xz, iH0yz) = accI(rH0, H_MASS);
    let (iH1x, iH1y, iH1z, iH1xy, iH1xz, iH1yz) = accI(rH1, H_MASS);

    let (ixx, iyy, izz, ixy, ixz, iyz) = (
        iOxx + iH0x + iH1x,
        iOyy + iH0y + iH1y,
        iOzz + iH0z + iH1z,
        iOxy + iH0xy + iH1xy,
        iOxz + iH0xz + iH1xz,
        iOyz + iH0yz + iH1yz,
    );

    // ω from I·ω = L
    let ω = solve_symmetric3(ixx, iyy, izz, ixy, ixz, iyz, L);

    // pure translation of COM + rigid rotation about COM
    let Δ = v_com * dt;
    let rO2 = rodrigues_rotate(rO, ω, dt);
    let rH02 = rodrigues_rotate(rH0, ω, dt);
    let rH12 = rodrigues_rotate(rH1, ω, dt);

    let new_o = r_com + Δ + rO2;
    let new_h0 = r_com + Δ + rH02;
    let new_h1 = r_com + Δ + rH12;

    // Wrap O, then pull each H next to O using min_image
    // wrap_water(mol, cell);
    mol.o.posit = cell.wrap(new_o);
    mol.h0.posit = mol.o.posit + cell.min_image(new_h0 - mol.o.posit);
    mol.h1.posit = mol.o.posit + cell.min_image(new_h1 - mol.o.posit);

    // COM-frame velocities after rotation
    let vO2 = ω.cross(rO2);
    let vH02 = ω.cross(rH02);
    let vH12 = ω.cross(rH12);

    let dvO = vO2 - vO;
    let dvH0 = vH02 - vH0;
    let dvH1 = vH12 - vH1;

    // Average constraint force over the drift interval (amu·Å/ps²)
    let fO_amu = dvO * O_MASS / dt;
    let fH0_amu = dvH0 * H_MASS / dt;
    let fH1_amu = dvH1 * H_MASS / dt;

    // Convert to kcal·mol⁻¹·Å⁻¹ to match your pair-virial units
    let fO_kcal = fO_amu * ACCEL_CONVERSION_INV;
    let fH0_kcal = fH0_amu * ACCEL_CONVERSION_INV;
    let fH1_kcal = fH1_amu * ACCEL_CONVERSION_INV;

    // Midpoint COM-frame positions
    let rO_mid = (rO + rO2) * 0.5;
    let rH0_mid = (rH0 + rH02) * 0.5;
    let rH1_mid = (rH1 + rH12) * 0.5;

    *virial_constr_kcal +=
        (rO_mid.dot(fO_kcal) + rH0_mid.dot(fH0_kcal) + rH1_mid.dot(fH1_kcal)) as f64;
    // ---------------------------------------------------------

    // Final absolute velocities
    mol.o.vel = v_com + vO2;
    mol.h0.vel = v_com + vH02;
    mol.h1.vel = v_com + vH12;

    // Place EP on the HOH bisector
    {
        let bisector = (mol.h0.posit - mol.o.posit) + (mol.h1.posit - mol.o.posit);
        mol.m.posit = mol.o.posit + bisector.to_normalized() * O_EP_R;
        mol.m.vel = (mol.h0.vel + mol.h1.vel) * 0.5;
    }
}

/// Periodically run this to re-establish the initial water geometry; this should be maintained
/// rigid normally, but numerical errors will accumulate. RUn this periodically to reset it.
pub(crate) fn reset_angle(mol: &mut WaterMol, cell: &SimBox) {
    // Rebuild u (bisector) and v (in-plane) from the updated positions
    let o_pos = mol.o.posit;
    let h0_local = o_pos + cell.min_image(mol.h0.posit - o_pos);
    let h1_local = o_pos + cell.min_image(mol.h1.posit - o_pos);

    let u = (h0_local + h1_local - o_pos * 2.0).to_normalized();
    let mut v = (h0_local - h1_local).to_normalized();
    v = (v - u * u.dot(v)).to_normalized();

    let c = H_O_H_θ * 0.5;
    let new_h0 = o_pos + (u * c.cos() + v * c.sin()) * O_H_R;
    let new_h1 = o_pos + (u * c.cos() - v * c.sin()) * O_H_R;

    // Commit with min-image consistency
    mol.h0.posit = mol.o.posit + cell.min_image(new_h0 - mol.o.posit);
    mol.h1.posit = mol.o.posit + cell.min_image(new_h1 - mol.o.posit);
}

/// Used in relaxation/energy minimization
pub(crate) fn settle_no_dt(mol: &mut WaterMol, cell: &SimBox) {
    // --- Position-only rigid projection for minimization ---
    // 1) Work in O-centered, minimum-image frame
    let o_pos = mol.o.posit;
    let h0_pos_local = o_pos + cell.min_image(mol.h0.posit - o_pos);
    let h1_pos_local = o_pos + cell.min_image(mol.h1.posit - o_pos);

    // 2) Build an orthonormal basis using current water orientation
    let v0 = h0_pos_local - o_pos;
    let v1 = h1_pos_local - o_pos;

    // Bisector (direction of HOH bisector)
    let u = (v0 + v1).to_normalized(); // bisector
    // In-plane perpendicular (points toward H0 minus H1 direction)
    let mut v = (v0 - v1).to_normalized(); // difference axis
    // Ensure v is orthogonalized and normalized against u (robustness)
    v = (v - u * u.dot(v)).to_normalized();

    // 3) Place hydrogens at target geometry (|OH| and angle fixed)
    //    Use your model constants—example names shown:
    let r_oh = O_H_R; // Å
    let theta = H_O_H_θ; // radians

    let c = theta * 0.5;
    let cos_c = c.cos();
    let sin_c = c.sin();

    let new_h0 = o_pos + (u * cos_c + v * sin_c) * r_oh;
    let new_h1 = o_pos + (u * cos_c - v * sin_c) * r_oh;

    // 4) Commit positions and rigid-wrap
    mol.o.posit = cell.wrap(o_pos);
    mol.h0.posit = mol.o.posit + cell.min_image(new_h0 - mol.o.posit);
    mol.h1.posit = mol.o.posit + cell.min_image(new_h1 - mol.o.posit);

    // 5) Update M on the bisector; zero or keep velocities as you prefer
    let bis = (mol.h0.posit - mol.o.posit) + (mol.h1.posit - mol.o.posit);
    mol.m.posit = mol.o.posit + bis.to_normalized() * crate::water_opc::O_EP_R;
}

/// Solve I · x = b for a 3×3 *symmetric* matrix I.
/// The six unique elements are
///     [ ixx  ixy  ixz ]
/// I = [ ixy  iyy  iyz ]
///     [ ixz  iyz  izz ]
///
/// Returns x as a Vec3.  Panics if det(I) ≃ 0.
fn solve_symmetric3(ixx: f32, iyy: f32, izz: f32, ixy: f32, ixz: f32, iyz: f32, b: Vec3) -> Vec3 {
    let det = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz);

    const TOL: f32 = 1.0e-12;
    if det.abs() < TOL {
        // Practically no rotation this step; keep ω = 0
        return Vec3::new_zero();
    }

    let inv_det = 1.0 / det;

    // Adjugate / inverse elements
    let inv00 = (iyy * izz - iyz * iyz) * inv_det;
    let inv01 = (ixz * iyz - ixy * izz) * inv_det;
    let inv02 = (ixy * iyz - ixz * iyy) * inv_det;
    let inv11 = (ixx * izz - ixz * ixz) * inv_det;
    let inv12 = (ixz * ixy - ixx * iyz) * inv_det;
    let inv22 = (ixx * iyy - ixy * ixy) * inv_det;

    // x = I⁻¹ · b
    Vec3::new(
        inv00 * b.x + inv01 * b.y + inv02 * b.z,
        inv01 * b.x + inv11 * b.y + inv12 * b.z,
        inv02 * b.x + inv12 * b.y + inv22 * b.z,
    )
}

fn rodrigues_rotate(r: Vec3, omega: Vec3, dt: f32) -> Vec3 {
    // Rotate vector r by angle θ = |ω| dt about axis n = ω/|ω|
    // Use series for tiny θ to avoid loss of precision.
    let omega_dt = omega * dt;
    let theta = omega_dt.magnitude();

    if theta < 1e-12 {
        let wxr = omega_dt.cross(r);
        return r + wxr + omega_dt.cross(wxr) * 0.5;
    }

    let n = omega_dt / theta; // unit axis
    let c = theta.cos();
    let s = theta.sin();

    r * c + n.cross(r) * s + n * (n.dot(r)) * (1.0 - c)
}
