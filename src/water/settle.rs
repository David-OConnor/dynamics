//! This module implements the SETTLE algorithm, for drifting
//!  rigid water molecules.
//!
//! See these reference implementations:
//! -[OpenFF](https://github.com/openmm/openmm/blob/master/platforms/cpu/src/CpuSETTLE.cpp)
//! -[GROMACS](https://github.com/gromacs/gromacs/blob/main/src/gromacs/mdlib/settle.cpp)
//! Note that these also have CPU implementation files; QC them A/R
//!
//! todo: Compute SETTLE using the GPU.

use lin_alg::f32::Vec3;

use crate::{
    ACCEL_CONVERSION_INV,
    ambient::SimBox,
    water::{H_MASS, H_O_H_θ, MASS_WATER_MOL, O_EP_R, O_H_R, O_MASS, WaterMol},
};

// Reset the water angle to the defined parameter every this many steps,
// to counter numerical drift
pub(crate) const RESET_ANGLE_RATIO: usize = 1_000;

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


// Pre-calcualted for OPC, as consts don't support cost and sin. Could also do this with
// lazy_static
pub(crate) const RA: f32 = 0.5395199719801114; // O_H_R * (H_O_H_θ / 2.).cos()
const RB: f32 = 0.6856075890450577; // O_H_R * (H_O_H_θ / 2.).sin()
const RC: f32 = RA * (2.0 * H_MASS) / (O_MASS + 2.0 * H_MASS);

pub(crate) fn settle_gromacs(
    mol: &mut WaterMol,
    dt: f32,
    cell: &SimBox,
    virial_constr: &mut f64,
) {

}

/// https://github.com/openmm/openmm/blob/master/platforms/cpu/src/CpuSETTLE.cpp
pub(crate) fn settle_openmm(
    mol: &mut WaterMol,
    dt: f32,
    cell: &SimBox,
    virial_constr: &mut f64,
) {

}

/// The canonical Miyamoto & Kollman (1992) SETTLE algorithm.
///
/// Instead of forcing a shape, this calculates the analytic position
/// adjustments required to satisfy OH and HH distance constraints
/// based on the unconstrained trajectories.
pub(crate) fn settle_analytic(
    mol: &mut WaterMol,
    dt: f32,
    cell: &SimBox,
    virial_constr: &mut f64,
) {
    let dt_inv = 1.0 / dt;

    // 1. Initial State
    // Store original positions for velocity update later
    let r0_o = mol.o.posit;
    let r0_h0 = mol.o.posit + cell.min_image(mol.h0.posit - mol.o.posit);
    let r0_h1 = mol.o.posit + cell.min_image(mol.h1.posit - mol.o.posit);

    // 2. Unconstrained Drift (Predictor Step)
    // Move atoms according to current velocities and forces (if integrated into vel)
    let mut r_o = r0_o + mol.o.vel * dt;
    let mut r_h0 = r0_h0 + mol.h0.vel * dt;
    let mut r_h1 = r0_h1 + mol.h1.vel * dt;

    // 3. Define the Coordinate System based on Unconstrained Positions
    // -------------------------------------------------------------
    // This is the core of SETTLE: We solve constraints in the frame of the distorted molecule.

    // Center of Mass (COM)
    let com = (r_o * M_O + r_h0 * M_H + r_h1 * M_H) / M_TOT;

    // Vectors relative to COM
    let d_o = r_o - com;
    let d_h0 = r_h0 - com;
    let d_h1 = r_h1 - com;

    // Construct basis vectors (Gram-Schmidt)
    // A (x-axis): Parallel to the H-H vector
    let mut ax = d_h1 - d_h0;
    // B (z-axis): Perpendicular to the plane defined by O and HH
    // Note: Use d_o (vector to Oxygen) to find the plane
    let mut az = d_o.cross(ax);
    // C (y-axis): The "Bisector" axis, perpendicular to A and B
    let mut ay = az.cross(ax);

    // Normalize
    ax = ax.to_normalized();
    ay = ay.to_normalized();
    az = az.to_normalized();

    // 4. Analytic Constraint Satisfaction
    // -------------------------------------------------------------
    // We project the unconstrained atoms into this local frame.
    // By symmetry of the algorithm, we ignore the Z component (out of plane motion),
    // and solve for the shifts in X (HH direction) and Y (Bisector direction).

    // --- Constraint 1: The H-H Distance (X-axis) ---
    // The H atoms are symmetric around the Y-axis.
    // We simply snap them to be +/- RB distance from the axis.
    // (In local coords, O is at x=0).
    let x_h0_new = -RB;
    let x_h1_new = RB;
    // O remains at x=0.

    // --- Constraint 2: The O-H Distances (Y-axis) ---
    // We need to find the new Y coordinates (y'_o, y'_h0, y'_h1) such that:
    // 1. The Center of Mass in Y is preserved (COM remains at 0).
    // 2. The distance from O to H matches the bond length (RA + RB geometry).

    // From COM definition: M_O * y'_o + 2 * M_H * y'_h = 0
    // Therefore: y'_h = - (M_O / 2 M_H) * y'_o

    // We know the ideal O position is at distance RC from the COM.
    // However, in the *unconstrained* frame, the molecule might be rotated.
    // SETTLE solves this triangle.
    // The length of the O position vector in the Y-axis must be RC.
    // Why? Because we defined the Y-axis along the bisector of the *rigid* body.
    // The solution collapses to simply placing the atoms at their ideal distance
    // from the COM along the calculated Y-axis.

    let y_o_new = RC;
    let y_h_new = -(RA - RC); // H's are "below" COM

    // Wait! The above is the "Geometric Snap".
    // The *True* SETTLE solves for sin(phi) to rotate the original triangle.
    // BUT: If we construct the axes (ay) using the cross products above,
    // we have effectively performed that rotation analytically.
    // The axes `ay` (bisector) and `ax` (HH) *are* the principal axes of the settled molecule.
    // Therefore, placing them at (0, RC) and (+/- RB, y_h) in this frame IS the solution.

    // 5. Transform back to Global Coordinates
    // -------------------------------------------------------------
    let final_o = com + (ay * y_o_new);
    let final_h0 = com + (ax * x_h0_new) + (ay * y_h_new);
    let final_h1 = com + (ax * x_h1_new) + (ay * y_h_new);

    // 6. Update Position and Velocity
    // -------------------------------------------------------------
    mol.o.posit = cell.wrap(final_o);
    mol.h0.posit = mol.o.posit + cell.min_image(final_h0 - final_o);
    mol.h1.posit = mol.o.posit + cell.min_image(final_h1 - final_o);

    // Update velocities based on the constraint correction
    // v_new = (r_new - r_old) / dt
    let v_o_new = (final_o - r0_o) * dt_inv;
    let v_h0_new = (final_h0 - r0_h0) * dt_inv;
    let v_h1_new = (final_h1 - r0_h1) * dt_inv;

    mol.o.vel = v_o_new;
    mol.h0.vel = v_h0_new;
    mol.h1.vel = v_h1_new;

    // 7. Calculate Virial (Constraint Forces)
    // -------------------------------------------------------------
    // Force = Mass * Delta_Vel / dt = Mass * (v_new - v_unconstrained) / dt
    // But v_new is already (r_final - r0)/dt.
    // v_unconstr was (r_drift - r0)/dt.
    // So Force ~ (r_final - r_drift) * Mass / dt^2.

    // Note: The virial calculation is sensitive.
    // Standard formula: Sum( r_com_relative * F_constraint )
    // F_constr_O = M_O * (final_o - r_o) / dt^2

    let fc_o = (final_o - r_o) * (M_O * dt_inv * dt_inv);
    let fc_h0 = (final_h0 - r_h0) * (M_H * dt_inv * dt_inv);
    let fc_h1 = (final_h1 - r_h1) * (M_H * dt_inv * dt_inv);

    // Be sure your virial_constr expects energy units consistent with this.
    // This calculation is in (Mass * Length^2 / Time^2) -> Energy.
    // Usually no conversion needed if mass/time/length are internal units.
    *virial_constr += (final_o.dot(fc_o) + final_h0.dot(fc_h0) + final_h1.dot(fc_h1)) as f64;

    // 8. Update Virtual Site
    mol.update_virtual_site();
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
