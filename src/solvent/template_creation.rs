#![allow(unused)]

//! This module is used to generate templates of water and other solvents. We don't
//! use it at run time. Example workflow: Populate a sim box of fixed size with the correct
//! number of solvent molecules for that solvent at a given temperature and pressure. Run a simulation
//! to equilibriate. Save to a template file. (e.g. .gro, or a binary format). Use that template
//! during MD runs in this library, or pass it to another MD engine.

use std::f32::consts::TAU;

use lin_alg::f32::{Mat3 as Mat3F32, Quaternion, Vec3};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
    prelude::ThreadRng,
};
use rand_distr::Normal;

use crate::{
    AtomDynamics, ComputationDevice, MdConfig, MdOverrides, MdState, MolDynamics, NATIVE_TO_KCAL,
    ParamError,
    barostat::SimBox,
    params::FfParamSet,
    solvent::{
        WaterMolOpc, init,
        init::{MIN_WATER_O_O_DIST_SQ, n_water_mols},
    },
    thermostat::{GAS_CONST_R, KB_A2_PS2_PER_K_PER_AMU},
};
// todo: Add a function which automatically creates a template. Sets up the grid, equilibriates,
// todo and saves to `.gro` and/or `.water_init_template`

/// Creates a regular lattice of water molecules. We use this as the first part of creating
/// a solvent template. Use this,  run a sim with thermostat and barostat, then store the result
/// in a `WaterInitTemplate`. We can save and load this to disk as binary, or in `.gro` format.
///
/// Generate solvent molecules to meet a temperature target, using standard density assumptions.
/// We deconflict with (solute) atoms in the simulation, and base the number of molecules to add
/// on the free space, not the total cell volume.
///
/// Process:
/// - Compute the number of molecules to add
/// - Add them on a regular grid with random orientations, and velocities in a random distribution
///   that matches the target temperature. Move molecules to the edge that are too close to
///   solute atoms.
///
/// Note: If we're able to place most, but not all waters, the barostat should adjust the sim box size
/// to account for the lower-than-specific pressure.
pub fn make_water_mols_grid(
    cell: &SimBox,
    temperature_tgt: f32,
    zero_com_drift: bool,
) -> Vec<WaterMolOpc> {
    println!("Initializing a solvent grid, as part of template preparation...");
    // Initialize an RNG for orientations.
    let mut rng = rand::rng();
    let distro = Uniform::<f32>::new(0.0, 1.0).unwrap();

    let n_mols = n_water_mols(cell, &[]);

    let mut result: Vec<WaterMolOpc> = Vec::with_capacity(n_mols);

    // Initialize the correct number of solvent molecules on a uniform grid. We ignore the solute for
    let lx = cell.bounds_high.x - cell.bounds_low.x;
    let ly = cell.bounds_high.y - cell.bounds_low.y;
    let lz = cell.bounds_high.z - cell.bounds_low.z;

    let base = (n_mols as f32).cbrt().round().max(1.0) as usize;
    let n_x = base;
    let n_y = base;
    let n_z = n_mols.div_ceil(n_x * n_y);

    let spacing_x = lx / n_x as f32;
    let spacing_y = ly / n_y as f32;
    let spacing_z = lz / n_z as f32;

    // Prevents unbounded looping. A higher value means we're more likely to succed,
    // but the run time could be higher.
    let fault_ratio = 3;

    let mut num_added = 0;
    let mut loops_used = 0;

    'outer: for i in 0..n_mols * fault_ratio {
        let a = i % n_x;
        let b = (i / n_x) % n_y;
        let c = (i / (n_x * n_y)) % n_z;

        let posit = Vec3::new(
            cell.bounds_low.x + (a as f32 + 0.5) * spacing_x,
            cell.bounds_low.y + (b as f32 + 0.5) * spacing_y,
            cell.bounds_low.z + (c as f32 + 0.5) * spacing_z,
        );

        // Check for an overlap with existing solvent molecules.
        for w in &result {
            let dist_sq = (w.o.posit - posit).magnitude_squared();
            if dist_sq < MIN_WATER_O_O_DIST_SQ {
                loops_used += 1;
                continue 'outer;
            }
        }

        result.push(WaterMolOpc::new(
            posit,
            Vec3::new_zero(),
            random_quaternion(&mut rng, distro),
        ));
        num_added += 1;

        if num_added == n_mols {
            break;
        }
        loops_used += 1;
    }

    // Set velocities consistent with the temperature target.
    init_velocities(&mut result, temperature_tgt, zero_com_drift, &mut rng);

    println!(
        "Added {} / {n_mols} solvent mols. Used {loops_used} loops",
        result.len()
    );
    result
}

/// We use this as part of our water template generation.
///
/// Note: This sets a reasonable default, but our thermostat, applied notably during
/// our initial solvent simulation, determines the actual temperature set at proper sim init.
/// Note: We've deprecated this in favor of velocities pre-initialized in the template.
fn init_velocities(
    mols: &mut [WaterMolOpc],
    t_target: f32,
    zero_com_drift: bool,
    rng: &mut ThreadRng,
) {
    let kT = KB_A2_PS2_PER_K_PER_AMU * t_target;

    for m in mols.iter_mut() {
        // COM and relative positions
        let (r_com, m_tot) = {
            let mut r = Vec3::new_zero();
            let mut m_tot = 0.0;
            for a in [&m.o, &m.h0, &m.h1] {
                r += a.posit * a.mass;
                m_tot += a.mass;
            }
            (r / m_tot, m_tot)
        };

        let r_0 = m.o.posit - r_com;
        let r_h0 = m.h0.posit - r_com;
        let r_h1 = m.h1.posit - r_com;

        // Sample COM velocity
        let sigma_v = (kT / m_tot).sqrt();
        let n = Normal::new(0.0, sigma_v).unwrap();
        let v_com = Vec3::new(n.sample(rng), n.sample(rng), n.sample(rng));

        // Inertia tensor about COM (world frame)
        // Build as arrays (your code)
        let inertia = |r: Vec3, mass: f32| {
            let r2 = r.dot(r);
            [
                [
                    mass * (r2 - r.x * r.x),
                    -mass * r.x * r.y,
                    -mass * r.x * r.z,
                ],
                [
                    -mass * r.y * r.x,
                    mass * (r2 - r.y * r.y),
                    -mass * r.y * r.z,
                ],
                [
                    -mass * r.z * r.x,
                    -mass * r.z * r.y,
                    mass * (r2 - r.z * r.z),
                ],
            ]
        };
        let mut I_arr = inertia(r_0, m.o.mass);
        let add_I = |I: &mut [[f32; 3]; 3], J: [[f32; 3]; 3]| {
            for i in 0..3 {
                for j in 0..3 {
                    I[i][j] += J[i][j];
                }
            }
        };
        add_I(&mut I_arr, inertia(r_h0, m.h0.mass));
        add_I(&mut I_arr, inertia(r_h1, m.h1.mass));

        let I = Mat3F32::from_arr(I_arr);

        // Diagonalize and solve with the Mat3 methods
        let (eigvecs, eigvals) = I.eigen_vecs_vals();
        let L_principal = Vec3::new(
            Normal::new(0.0, (kT * eigvals.x.max(0.0)).sqrt())
                .unwrap()
                .sample(rng),
            Normal::new(0.0, (kT * eigvals.y.max(0.0)).sqrt())
                .unwrap()
                .sample(rng),
            Normal::new(0.0, (kT * eigvals.z.max(0.0)).sqrt())
                .unwrap()
                .sample(rng),
        );
        let L_world = eigvecs * L_principal; // assumes Mat3 * Vec3 is implemented
        let omega = I.solve_system(L_world); // ω = I^{-1} L

        // Set atomic velocities
        m.o.vel = v_com + omega.cross(r_0);
        m.h0.vel = v_com + omega.cross(r_h0);
        m.h1.vel = v_com + omega.cross(r_h1);
    }

    if zero_com_drift {
        // Remove global COM drift
        remove_com_velocity(mols);
    }

    let (ke_raw, dof) = kinetic_energy_and_dof(mols, zero_com_drift);

    // current T = 2 KE / (dof * R)
    let temperature_meas = (2.0 * ke_raw) / (dof as f32 * GAS_CONST_R as f32);
    let lambda = (t_target / temperature_meas).sqrt();

    for a in atoms_mut(mols) {
        if a.mass > 0.0 {
            a.vel *= lambda;
        }
    }
}

/// Calculate kinetic energy in kcal/mol, and DOF for solvent only.
/// Water is rigid, so 3 DOF per molecule.
fn kinetic_energy_and_dof(mols: &[WaterMolOpc], zero_com_drift: bool) -> (f32, usize) {
    let mut ke = 0.;
    for w in mols {
        ke += (w.o.mass * w.o.vel.magnitude_squared()) as f64;
        ke += (w.h0.mass * w.h0.vel.magnitude_squared()) as f64;
        ke += (w.h1.mass * w.h1.vel.magnitude_squared()) as f64;
    }

    let mut dof = mols.len() * 3;

    if zero_com_drift {
        dof = dof.saturating_sub(3);
    }

    // Add in the 0.5 factor, and convert from amu • (Å/ps)² to kcal/mol.
    (ke as f32 * 0.5 * NATIVE_TO_KCAL, dof)
}

pub fn atoms_mut(mols: &mut [WaterMolOpc]) -> impl Iterator<Item = &mut AtomDynamics> {
    mols.iter_mut()
        .flat_map(|m| [&mut m.o, &mut m.h0, &mut m.h1].into_iter())
}

/// Removes center-of-mass drift. Use in template generation
fn remove_com_velocity(mols: &mut [WaterMolOpc]) {
    let mut p = Vec3::new_zero();
    let mut m_tot = 0.0;
    for a in atoms_mut(mols) {
        p += a.vel * a.mass;
        m_tot += a.mass;
    }

    let v_com = p / m_tot;
    for a in atoms_mut(mols) {
        a.vel -= v_com;
    }
}

/// Used in template generation
// todo: It might be nice to have this in lin_alg, although I don't want to add the rand
// todo dependency to it.
fn random_quaternion(rng: &mut ThreadRng, distro: Uniform<f32>) -> Quaternion {
    let (u1, u2, u3) = (rng.sample(distro), rng.sample(distro), rng.sample(distro));
    let sqrt1_minus_u1 = (1.0 - u1).sqrt();
    let sqrt_u1 = u1.sqrt();
    let (theta1, theta2) = (TAU * u2, TAU * u3);

    Quaternion::new(
        sqrt1_minus_u1 * theta1.sin(),
        sqrt1_minus_u1 * theta1.cos(),
        sqrt_u1 * theta2.sin(),
        sqrt_u1 * theta2.cos(),
    )
    .to_normalized()
}

/// Create a solvent template by packing with a box which starts out too large, and gradually shrinks
/// to the proper size. This may work in cases where a grid-based or other packing doens't work in the cases
/// or long molecules like octanol. We initialize in a grid, then run a sim box. We shrink it graually enough
/// so that the molecules bend and move into deconflicted positions naturally. The final result has the correct
/// density (i.e. pressure) characteristics, and is equilibriated.
pub(crate) fn pack_solvent_with_shrinking_box(
    dev: &ComputationDevice,
    mol_solvent: &MolDynamics,
    solvent_count: usize,
    water_count: usize, // E.g. OPC water with the custom solvent.
    mut cell: SimBox,
    param_set: &FfParamSet,
) -> Result<Vec<AtomDynamics>, ParamError> {
    // todo: A/R. Ideally this scale is dynamic; it should be large enough so the
    // todo initial packign goes smoothly.
    // this scale is of side len; not volume.
    let initial_box_scale = 2.;

    let dt = 0.001; // todo: A/R

    // Å edge len per step. This must be low enough
    let box_scale_rate = 0.01;
    let box_scale_rate_div2 = box_scale_rate / 2.;

    // todo: Instead of a fixed step count, perhaps do it dynamically based on the situation.
    // todo: Or based on the number of steps to get between initial and final box sizes plus a pad.
    let steps = 4_000;

    // Uncomment as required for validating individual processes.
    let cfg = MdConfig {
        overrides: MdOverrides {
            skip_water_relaxation: true,
            snapshots_during_equilibration: true,
            // Merge with caller-supplied overrides so flags like `skip_water` are preserved.
            ..Default::default()
        },
        barostat_cfg: None,
        ..Default::default()
    };

    let mut mols = Vec::with_capacity(solvent_count);
    // todo: Position molecules evenly, e.g. in a grid for simplicity. Perhaps
    // todo with fixed orientation.
    for _ in 0..solvent_count {}

    let (mut md_state, _) = MdState::new(dev, &cfg, &mols, param_set)?;

    for step in 0..steps {
        md_state.step(dev, dt, None);

        cell.bounds_low += Vec3::splat(box_scale_rate_div2);
        cell.bounds_high -= Vec3::splat(box_scale_rate_div2);
    }

    // Recenter so the sim box is around the origin.
    let diff = cell.center();
    for a in &mut md_state.atoms {
        a.posit -= diff;
    }

    Ok(md_state.atoms)
}
