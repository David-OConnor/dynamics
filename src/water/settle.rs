//! This module implements the SETTLE algorithm, for drifting
//!  rigid water molecules.
//!
//! See these reference implementations:
//! -[OpenFF](https://github.com/openmm/openmm/blob/master/platforms/cpu/src/CpuSETTLE.cpp)
//! -[GROMACS](https://github.com/openmm/openmm/blob/b0c2c4d84ef1ac82984a577c7506912f5f91bafa/platforms/reference/include/ReferenceSETTLEAlgorithm.h#L4)
//! Note that these also have CPU implementation files; QC them A/R
//!
//! todo: Compute SETTLE using the GPU.

use lin_alg::f32::Vec3;

use crate::{
    // ACCEL_CONVERSION_INV,
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
// const RB: f32 = 0.6856075890450577; // O_H_R * (H_O_H_θ / 2.).sin()
// const RC: f32 = RA * (2.0 * H_MASS) / (O_MASS + 2.0 * H_MASS);

// /// https://github.com/gromacs/gromacs/blob/main/src/gromacs/mdlib/settle.cpp
// pub(crate) fn settle_gromacs(mol: &mut WaterMol, dt: f32, cell: &SimBox, virial_constr: &mut f64) {}

// /// https://github.com/openmm/openmm/blob/b0c2c4d84ef1ac82984a577c7506912f5f91bafa/platforms/reference/include/ReferenceSETTLEAlgorithm.h#L4
// pub(crate) fn settle_openmm(mol: &mut WaterMol, dt: f32, cell: &SimBox, virial_constr: &mut f64) {}

// /// The canonical Miyamoto & Kollman (1992) SETTLE algorithm.
// ///
// /// Instead of forcing a shape, this calculates the analytic position
// /// adjustments required to satisfy OH and HH distance constraints
// /// based on the unconstrained trajectories.
// pub(crate) fn _settle_analytic(mol: &mut WaterMol, dt: f32, cell: &SimBox, virial_constr: &mut f64) {
//     let dt_inv = 1.0 / dt;

//     // 1. Initial State
//     // Store original positions for velocity update later
//     let r0_o = mol.o.posit;
//     let r0_h0 = mol.o.posit + cell.min_image(mol.h0.posit - mol.o.posit);
//     let r0_h1 = mol.o.posit + cell.min_image(mol.h1.posit - mol.o.posit);

//     // 2. Unconstrained Drift (Predictor Step)
//     // Move atoms according to current velocities and forces (if integrated into vel)
//     let mut r_o = r0_o + mol.o.vel * dt;
//     let mut r_h0 = r0_h0 + mol.h0.vel * dt;
//     let mut r_h1 = r0_h1 + mol.h1.vel * dt;

//     // 3. Define the Coordinate System based on Unconstrained Positions
//     // -------------------------------------------------------------
//     // This is the core of SETTLE: We solve constraints in the frame of the distorted molecule.

//     // Center of Mass (COM)
//     let com = (r_o * O_MASS + r_h0 * H_MASS + r_h1 * H_MASS) / MASS_WATER_MOL;

//     // Vectors relative to COM
//     let d_o = r_o - com;
//     let d_h0 = r_h0 - com;
//     let d_h1 = r_h1 - com;

//     // Construct basis vectors (Gram-Schmidt)
//     // A (x-axis): Parallel to the H-H vector
//     let mut ax = d_h1 - d_h0;
//     // B (z-axis): Perpendicular to the plane defined by O and HH
//     // Note: Use d_o (vector to Oxygen) to find the plane
//     let mut az = d_o.cross(ax);
//     // C (y-axis): The "Bisector" axis, perpendicular to A and B
//     let mut ay = az.cross(ax);

//     // Normalize
//     ax = ax.to_normalized();
//     ay = ay.to_normalized();
//     az = az.to_normalized();

//     // 4. Analytic Constraint Satisfaction
//     // -------------------------------------------------------------
//     // We project the unconstrained atoms into this local frame.
//     // By symmetry of the algorithm, we ignore the Z component (out of plane motion),
//     // and solve for the shifts in X (HH direction) and Y (Bisector direction).

//     // --- Constraint 1: The H-H Distance (X-axis) ---
//     // The H atoms are symmetric around the Y-axis.
//     // We simply snap them to be +/- RB distance from the axis.
//     // (In local coords, O is at x=0).
//     let x_h0_new = -RB;
//     let x_h1_new = RB;
//     // O remains at x=0.

//     // --- Constraint 2: The O-H Distances (Y-axis) ---
//     // We need to find the new Y coordinates (y'_o, y'_h0, y'_h1) such that:
//     // 1. The Center of Mass in Y is preserved (COM remains at 0).
//     // 2. The distance from O to H matches the bond length (RA + RB geometry).

//     // From COM definition: M_O * y'_o + 2 * M_H * y'_h = 0
//     // Therefore: y'_h = - (M_O / 2 M_H) * y'_o

//     // We know the ideal O position is at distance RC from the COM.
//     // However, in the *unconstrained* frame, the molecule might be rotated.
//     // SETTLE solves this triangle.
//     // The length of the O position vector in the Y-axis must be RC.
//     // Why? Because we defined the Y-axis along the bisector of the *rigid* body.
//     // The solution collapses to simply placing the atoms at their ideal distance
//     // from the COM along the calculated Y-axis.

//     let y_o_new = RC;
//     let y_h_new = -(RA - RC); // H's are "below" COM

//     // Wait! The above is the "Geometric Snap".
//     // The *True* SETTLE solves for sin(phi) to rotate the original triangle.
//     // BUT: If we construct the axes (ay) using the cross products above,
//     // we have effectively performed that rotation analytically.
//     // The axes `ay` (bisector) and `ax` (HH) *are* the principal axes of the settled molecule.
//     // Therefore, placing them at (0, RC) and (+/- RB, y_h) in this frame IS the solution.

//     // 5. Transform back to Global Coordinates
//     // -------------------------------------------------------------
//     let final_o = com + (ay * y_o_new);
//     let final_h0 = com + (ax * x_h0_new) + (ay * y_h_new);
//     let final_h1 = com + (ax * x_h1_new) + (ay * y_h_new);

//     // 6. Update Position and Velocity
//     // -------------------------------------------------------------
//     mol.o.posit = cell.wrap(final_o);
//     mol.h0.posit = mol.o.posit + cell.min_image(final_h0 - final_o);
//     mol.h1.posit = mol.o.posit + cell.min_image(final_h1 - final_o);

//     // Update velocities based on the constraint correction
//     // v_new = (r_new - r_old) / dt
//     let v_o_new = (final_o - r0_o) * dt_inv;
//     let v_h0_new = (final_h0 - r0_h0) * dt_inv;
//     let v_h1_new = (final_h1 - r0_h1) * dt_inv;

//     mol.o.vel = v_o_new;
//     mol.h0.vel = v_h0_new;
//     mol.h1.vel = v_h1_new;

//     // 7. Calculate Virial (Constraint Forces)
//     // -------------------------------------------------------------
//     // Force = Mass * Delta_Vel / dt = Mass * (v_new - v_unconstrained) / dt
//     // But v_new is already (r_final - r0)/dt.
//     // v_unconstr was (r_drift - r0)/dt.
//     // So Force ~ (r_final - r_drift) * Mass / dt^2.

//     // Note: The virial calculation is sensitive.
//     // Standard formula: Sum( r_com_relative * F_constraint )
//     // F_constr_O = M_O * (final_o - r_o) / dt^2

//     let fc_o = (final_o - r_o) * (O_MASS * dt_inv * dt_inv);
//     let fc_h0 = (final_h0 - r_h0) * (H_MASS * dt_inv * dt_inv);
//     let fc_h1 = (final_h1 - r_h1) * (H_MASS * dt_inv * dt_inv);

//     // Be sure your virial_constr expects energy units consistent with this.
//     // This calculation is in (Mass * Length^2 / Time^2) -> Energy.
//     // Usually no conversion needed if mass/time/length are internal units.
//     *virial_constr += (final_o.dot(fc_o) + final_h0.dot(fc_h0) + final_h1.dot(fc_h1)) as f64;

//     // 8. Update Virtual Site
//     mol.update_virtual_site();
// }

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

// Below: Experimenting ------------
/// Pre-calculated constants struct (equivalent to GROMACS SettleParameters)
pub struct SettleParams {
    pub wh: f32,   // Weighted mass factor
    pub ra: f32,   // Geometry constant A
    pub rb: f32,   // Geometry constant B
    pub rc: f32,   // Geometry constant C
    pub irc2: f32, // Inverse H-H distance (reused)
    pub d_oh: f32, // Target O-H bond length
    pub d_hh: f32, // Target H-H bond length
}

impl SettleParams {
    pub fn new() -> Self {
        let m_o = O_MASS;
        let m_h = H_MASS;

        // Target geometry
        let d_oh = O_H_R;
        // Law of cosines for d_hh
        let d_hh = (2.0 * d_oh * d_oh * (1.0 - H_O_H_θ.cos())).sqrt();

        // GROMACS derivation for constants
        let wohh = m_o + 2.0 * m_h;
        let wh = m_h / wohh;
        let rc = d_hh / 2.0;

        // ra is the distance from O to the H-H axis, weighted by mass ratios
        let ra = 2.0 * m_h * (d_oh * d_oh - rc * rc).sqrt() / wohh;
        let rb = (d_oh * d_oh - rc * rc).sqrt() - ra;
        let irc2 = 1.0 / d_hh;

        Self {
            wh,
            ra,
            rb,
            rc,
            irc2,
            d_oh,
            d_hh,
        }
    }
}

// todo: ALternatively, place this in MdState.
lazy_static::lazy_static! {
    pub static ref SETTLE_PARAMS: SettleParams = SettleParams::new();
}

/// The GROMACS "COM-free" SETTLE implementation.
///
/// Inputs:
/// - mol: Contains `xprime` (positions after drift) and `v` (current velocities).
/// - dt: Time step.
pub(crate) fn settle_drift(mol: &mut WaterMol, dt: f32, cell: &SimBox, virial_constr: &mut f64) {
    let params = &SETTLE_PARAMS;
    let inv_dt = 1.0 / dt;

    // 1. Recover "Old" Positions (x)
    // GROMACS uses the *old* geometry to define the stable coordinate frame axis.
    // x_old = x_new - v * dt
    let x_o = mol.o.posit - mol.o.vel * dt;
    let x_h1 = mol.h0.posit - mol.h0.vel * dt;
    let x_h2 = mol.h1.posit - mol.h1.vel * dt;

    // 2. Load "New" Positions (xprime)
    // We work relative to Oxygen to avoid COM calculation.
    let xp_o = mol.o.posit;

    // Calculate unconstrained bond vectors in NEW frame (applying PBC)
    // doh2 = vector from O to H1 (new)
    let doh2 = cell.min_image(mol.h0.posit - xp_o);
    let doh3 = cell.min_image(mol.h1.posit - xp_o);

    // Calculate bond vectors in OLD frame (defining the axes)
    // dist21 = vector from O to H1 (old)
    let dist21 = cell.min_image(x_h1 - x_o);
    let dist31 = cell.min_image(x_h2 - x_o);

    // 3. Coordinate System Construction (Implicit COM)
    // "a1" is effectively the weighted bisector offset in the new frame
    let a1 = (doh2 + doh3) * -params.wh;
    let b1 = doh2 + a1;
    let c1 = doh3 + a1;

    // Build the rotation matrix (trns) from the OLD vectors (dist21, dist31).
    // This ensures the constraint frame rotates with the rigid body.

    // Cross products to find axes (unnormalized)
    // x axis (H1 -> H2 directionish), y axis (bisectorish), z axis (plane normal)
    // GROMACS calculates these components explicitly to avoid overhead.
    let xakszd = dist21.y * dist31.z - dist21.z * dist31.y;
    let yakszd = dist21.z * dist31.x - dist21.x * dist31.z;
    let zakszd = dist21.x * dist31.y - dist21.y * dist31.x;

    let xaksxd = a1.y * zakszd - a1.z * yakszd;
    let yaksxd = a1.z * xakszd - a1.x * zakszd;
    let zaksxd = a1.x * yakszd - a1.y * xakszd;

    let xaksyd = yakszd * zaksxd - zakszd * yaksxd;
    let yaksyd = zakszd * xaksxd - xakszd * zaksxd;
    let zaksyd = xakszd * yaksxd - yakszd * xaksxd;

    // Inverse lengths (normalizers)
    let axlng = 1.0 / (xaksxd * xaksxd + yaksxd * yaksxd + zaksxd * zaksxd).sqrt();
    let aylng = 1.0 / (xaksyd * xaksyd + yaksyd * yaksyd + zaksyd * zaksyd).sqrt();
    let azlng = 1.0 / (xakszd * xakszd + yakszd * yakszd + zakszd * zakszd).sqrt();

    // The Rotation Matrix (rows are trns1, trns2, trns3)
    let trns1 = Vec3::new(xaksxd * axlng, yaksxd * axlng, zaksxd * axlng);
    let trns2 = Vec3::new(xaksyd * aylng, yaksyd * aylng, zaksyd * aylng);
    let trns3 = Vec3::new(xakszd * azlng, yakszd * azlng, zakszd * azlng);

    // 4. Project unconstrained vectors onto local axes
    // b0d = H1 projection, c0d = H2 projection (on OLD frame)
    let b0d_x = trns1.dot(dist21);
    let b0d_y = trns2.dot(dist21); // z is 0 by definition of plane
    let c0d_x = trns1.dot(dist31);
    let c0d_y = trns2.dot(dist31);

    // Project NEW vectors (a1, b1, c1) onto local axes
    let a1d_z = trns3.dot(a1);
    let b1d_z = trns3.dot(b1);
    let c1d_z = trns3.dot(c1);

    // 5. Solve Analytic Geometry (Theta, Phi, Psi)
    // "sinphi" corrects the bond lengths based on the mass-weighted shift
    let sinphi = a1d_z / params.ra;
    let tmp2 = 1.0 - sinphi * sinphi;

    // Safety check for degenerate triangles (numerical instability)
    if tmp2 <= 1e-12 {
        // In GROMACS this sets an error flag. Here we just clamp or return.
        eprintln!("SETTLE Error: Water molecule distorted.");
        return;
    }

    let cosphi = tmp2.sqrt();
    let sinpsi = (b1d_z - c1d_z) * params.irc2 / cosphi;
    let cospsi = (1.0 - sinpsi * sinpsi).sqrt();

    // Intermediate geometric terms
    let a2d_y = params.ra * cosphi;
    let b2d_x = -params.rc * cospsi;
    let t1 = -params.rb * cosphi;
    let t2 = params.rc * sinpsi * sinphi;
    let b2d_y = t1 - t2;
    let c2d_y = t1 + t2;

    // Calculate rotation angles alpha, beta, gamma
    let alpha = b2d_x * (b0d_x - c0d_x) + b0d_y * b2d_y + c0d_y * c2d_y;
    let beta = b2d_x * (c0d_y - b0d_y) + b0d_x * b2d_y + c0d_x * c2d_y;
    let gamma = b0d_x * trns2.dot(b1) - trns1.dot(b1) * b0d_y + c0d_x * trns2.dot(c1)
        - trns1.dot(c1) * c0d_y;

    let al2be2 = alpha * alpha + beta * beta;
    let tmp_sq = al2be2 - gamma * gamma;
    let sinthe = (alpha * gamma - beta * tmp_sq.sqrt()) / al2be2;
    let costhe = (1.0 - sinthe * sinthe).sqrt();

    // 6. Reconstruct Corrected Local Coordinates (Step 4 in GROMACS)
    let a3d_x = -a2d_y * sinthe;
    let a3d_y = a2d_y * costhe;
    let a3d_z = a1d_z;

    let b3d_x = b2d_x * costhe - b2d_y * sinthe;
    let b3d_y = b2d_x * sinthe + b2d_y * costhe;
    let b3d_z = b1d_z;

    let c3d_x = -b2d_x * costhe - c2d_y * sinthe;
    let c3d_y = -b2d_x * sinthe + c2d_y * costhe;
    let c3d_z = c1d_z;

    // 7. Rotate back to Global Coordinates
    // These are the correction vectors (dx) + the original projection?
    // GROMACS computes a3, b3, c3 as the final vector positions
    let a3 = trns1 * a3d_x + trns2 * a3d_y + trns3 * a3d_z;
    let b3 = trns1 * b3d_x + trns2 * b3d_y + trns3 * b3d_z;
    let c3 = trns1 * c3d_x + trns2 * c3d_y + trns3 * c3d_z;

    // 8. Apply Corrections
    // GROMACS: dx = a3 - a1.
    // xprime = xprime + dx.
    let dx_o = a3 - a1;
    let dx_h1 = b3 - b1;
    let dx_h2 = c3 - c1;

    mol.o.posit += dx_o;
    mol.h0.posit += dx_h1;
    mol.h1.posit += dx_h2;

    // 9. Update Velocities
    // v = v + dx / dt
    mol.o.vel += dx_o * inv_dt;
    mol.h0.vel += dx_h1 * inv_dt;
    mol.h1.vel += dx_h2 * inv_dt;

    // 10. Virial Calculation
    // GROMACS "sum_r_m_dr" logic
    // sum_r_m_dr = sum_r_m_dr - (x_ow1 * mdo + dist21 * mdb + dist31 * mdc)
    // where mdo = m_o * dx_o + m_h * dx_h1 + m_h * dx_h2 ...
    // This is calculating the constraint force * distance.
    // Let's use the simplified form: Force * r
    // Force_O = m_O * dx_o / dt^2
    // Virial = r_O . Force_O + ...

    let f_o = dx_o * (O_MASS * inv_dt * inv_dt);
    let f_h1 = dx_h1 * (H_MASS * inv_dt * inv_dt);
    let f_h2 = dx_h2 * (H_MASS * inv_dt * inv_dt);

    *virial_constr +=
        (mol.o.posit.dot(f_o) + mol.h0.posit.dot(f_h1) + mol.h1.posit.dot(f_h2)) as f64;

    // 11. CRITICAL: Update Virtual Site
    mol.update_virtual_site();
}
