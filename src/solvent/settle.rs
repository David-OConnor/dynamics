//! This module implements the SETTLE algorithm, for drifting
//!  rigid solvent molecules.
//!
//! See these reference implementations:
//! -[OpenFF](https://github.com/openmm/openmm/blob/master/platforms/cpu/src/CpuSETTLE.cpp)
//! -[GROMACS](https://github.com/openmm/openmm/blob/b0c2c4d84ef1ac82984a577c7506912f5f91bafa/platforms/reference/include/ReferenceSETTLEAlgorithm.h#L4)
//! Note that these also have CPU implementation files; QC them A/R
//!
//! todo: Compute SETTLE using the GPU.

use lin_alg::f32::Vec3;

use crate::{
    barostat::{SimBox, Virial},
    solvent::{H_MASS, H_O_H_θ, MASS_WATER_MOL, O_EP_R, O_H_R, O_MASS, WaterMol},
};

// Reset the solvent angle to the defined parameter every this many steps,
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

// /// https://github.com/gromacs/gromacs/blob/main/src/gromacs/mdlib/settle.cpp
// pub(crate) fn settle_gromacs(mol: &mut WaterMol, dt: f32, cell: &SimBox, virial_constr: &mut f64) {}

// /// https://github.com/openmm/openmm/blob/b0c2c4d84ef1ac82984a577c7506912f5f91bafa/platforms/reference/include/ReferenceSETTLEAlgorithm.h#L4
// pub(crate) fn settle_openmm(mol: &mut WaterMol, dt: f32, cell: &SimBox, virial_constr: &mut f64) {}

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

/// Analytic SETTLE implementation for 3‑site rigid solvent (Miyamoto & Kollman, JCC 1992).
/// Works for any bond length / HOH angle. This handles the drift (position updates) for solvent
/// molecules. It also places M on the bisector, and performs a rigid wrap.
///
/// All distances & masses are in MD internal units (Å, ps, amu, kcal/mol).
///
/// This is handles the Verlet "drift" for a rigid molecule. It is the equivalent
/// of updating position by adding velocity x dt, but also maintains the rigid
/// geometry of 3-atom molecules.
/// Returns the constraint virial contribution for this molecule, in native units (amu·Å²/ps²).
pub(crate) fn integrate_rigid_water(mol: &mut WaterMol, dt: f32, cell: &SimBox) -> f64 {
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

    // Constraint virial — GROMACS settle.cpp style:
    // Use O-relative bond vectors at final positions (dist21/dist31 in GROMACS).
    // The O term drops out because SETTLE conserves COM (m_O·Δv_O + m_H·Δv_H0 + m_H·Δv_H1 = 0).
    // Δv_Hi = v_constrained − v_unconstrained (COM-frame, = absolute Δv).
    // Units: Å · (amu · Å/ps) / ps = amu·Å²/ps² (native).
    let dv_h0 = vH02 - vH0;
    let dv_h1 = vH12 - vH1;
    let r_oh0 = rH02 - rO2; // O→H0 bond vector (final, COM frame = absolute bond)
    let r_oh1 = rH12 - rO2; // O→H1 bond vector (final)
    (r_oh0.dot(dv_h0 * (H_MASS / dt)) + r_oh1.dot(dv_h1 * (H_MASS / dt))) as f64
}

/// Periodically run this to re-establish the initial solvent geometry; this should be maintained
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
