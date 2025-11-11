#![allow(clippy::excessive_precision)]

//! Code for initializing water molecules, including assigning quantity, initial positions, and
//! velocities. Set up to meet density, pressure, and or temperature targets. Not specific to the
//! water model used.

use std::{f32::consts::TAU, time::Instant};

use lin_alg::f32::{Mat3 as Mat3F32, Quaternion as QuaternionF32, Vec3};
use rand::{Rng, distr::Uniform, rngs::ThreadRng};
use rand_distr::{Distribution, Normal};

use crate::{
    ACCEL_CONVERSION_INV, AtomDynamics, ComputationDevice, MdState,
    ambient::{GAS_CONST_R, KB_A2_PS2_PER_K_PER_AMU, SimBox},
    sa_surface,
    water_opc::WaterMol,
};

// 0.997 g cm⁻³ is a good default density for biological pressures. We use this for initializing
// and maintaining the water density and molecule count.
const WATER_DENSITY: f32 = 0.997;

// g / mol (or AMU per molecule)
// This is similar to the Amber H and O masses we used summed, and could be explained
// by precision limits. We use it for generating atoms based on mass density.
const MASS_WATER: f32 = 18.015_28;

// Avogadro's constant. mol^-1.
const N_A: f32 = 6.022_140_76e23;

// This is ~0.0333 mol/Å³
// Multiplying this by volume in Angstrom^3 gives us AMU g cm^-3 Å^3 mol^-1
const WATER_MOLS_PER_VOL: f32 = WATER_DENSITY * N_A / (MASS_WATER * 1.0e24);

// Don't generate water molecules that are too close to other atoms.
// Vdw contact distance between water molecules and organic molecules is roughly 3.5 Å.
const MIN_NONWATER_DIST: f32 = 3.75; // todo: Lower this?
const MIN_NONWATER_DIST_SQ: f32 = MIN_NONWATER_DIST * MIN_NONWATER_DIST;

// Higher is better, but slower. After hydrogen bond networks are settled, higher doens't
// improve things.
const NUM_SIM_STEPS: usize = 600;
const SIM_DT: f32 = 0.002;

/// Generate water molecules to meet a temperature target, using standard density assumptions.
/// We deconflict with (solute) atoms in the simulation, and base the number of molecules to add
/// on the free space, not the total cell volume.
///
/// Process:
/// - Compute the solvent-free volume using an isosurface
/// - Compute the number of molecules to add
/// - Add them on a regular grid with random orientations, and velocities in a random distribution
/// that matches the target temperature. Move molecules to the edge that are too close to
/// solute atoms.
/// - Run a brief simulation with the solute
/// atoms as static, to intially position water molecules realistically. This
/// takes advantage of our simulations' acceleration limits to set up realistic geometry using
/// hydrogen bond networks, and breaks the crystal lattice.

pub fn make_water_mols(
    cell: &SimBox,
    temperature_tgt: f32,
    atoms: &[AtomDynamics],
    zero_com_drift: bool,
) -> Vec<WaterMol> {
    println!("Initializing water molecules...");
    let start = Instant::now();

    // Initialize an RNG for orientations.
    let mut rng = rand::rng();
    let distro = Uniform::<f32>::new(0.0, 1.0).unwrap();

    let cell_volume = cell.volume();
    let mol_volume = sa_surface::vol_take_up_by_atoms(atoms);
    let free_vol = cell_volume - mol_volume;

    // Estimate free volume & n_mols from it
    let n_mols = (WATER_MOLS_PER_VOL * free_vol).round() as usize;

    println!(
        "Solvent-free vol: {:.2} Cell vol: {:.2} (Å³ / 1,000)",
        free_vol / 1_000.,
        cell_volume / 1_000.
    );

    let mut result = Vec::with_capacity(n_mols);

    if n_mols == 0 {
        println!("Complete in {} ms.", start.elapsed().as_millis());
        return result;
    }

    // Initialize the correct number of water molecules on a uniform grid. We ignore the solute for
    let lx = cell.bounds_high.x - cell.bounds_low.x;
    let ly = cell.bounds_high.y - cell.bounds_low.y;
    let lz = cell.bounds_high.z - cell.bounds_low.z;

    let base = (n_mols as f32).cbrt().round().max(1.0) as usize;
    let n_x = base;
    let n_y = base;
    let n_z = ((n_mols + (n_x * n_y) - 1) / (n_x * n_y)).max(1);

    let spacing_x = lx / n_x as f32;
    let spacing_y = ly / n_y as f32;
    let spacing_z = lz / n_z as f32;

    // Solute
    let atom_posits: Vec<_> = atoms.iter().map(|a| a.posit).collect();

    let fault_ratio = 2; // Prevents unbounded looping.
    let mut num_added = 0;
    let mut loops_used = 0;

    for i in 0..n_mols * fault_ratio {
        let a = i % n_x;
        let b = (i / n_x) % n_y;
        let c = i / (n_x * n_y);

        let posit = Vec3::new(
            cell.bounds_low.x + (a as f32 + 0.5) * spacing_x,
            cell.bounds_low.y + (b as f32 + 0.5) * spacing_y,
            cell.bounds_low.z + (c as f32 + 0.5) * spacing_z,
        );

        // If overlapping with a solute atom, don't place. We'll catch it at the end.
        // todo: I'm unclear on how this works. Adds to the outside? Potentially wraps?
        // todo: Just fills up part of the last grid "rect" in the cube?
        let mut skip_this = false;
        for atom_p in &atom_posits {
            let dist_sq = (*atom_p - posit).magnitude_squared();
            if dist_sq < MIN_NONWATER_DIST_SQ {
                skip_this = true;
                break;
            }
        }
        if skip_this {
            loops_used += 1;
            continue;
        }

        result.push(WaterMol::new(
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
    init_velocities_rigid(&mut result, temperature_tgt, zero_com_drift, &mut rng);

    let elapsed = start.elapsed().as_millis();
    println!(
        "Added {} / {n_mols} water mols in {elapsed} ms. Used {loops_used} loops",
        result.len()
    );
    result
}

fn init_velocities_rigid(
    mols: &mut [WaterMol],
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

        // Convert to Mat3 once, then use
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

    // Remove global COM drift
    remove_com_velocity(mols);

    // Optional: compute KE (translation+rotation == sum ½ m v^2 now) and rescale to T_target
    let (ke_raw, dof) = kinetic_energy_and_dof(mols, zero_com_drift);

    let lambda = (t_target / (2.0 * ke_raw) / (dof as f32 * GAS_CONST_R as f32)).sqrt();

    for a in atoms_mut(mols) {
        if a.mass > 0.0 {
            a.vel *= lambda;
        }
    }
}

/// Calculate kinetic energy in kcal/mol, and DOF for water only.
/// Water is rigid, so 3 DOF per molecule.
fn kinetic_energy_and_dof(mols: &[WaterMol], zero_com_drift: bool) -> (f32, usize) {
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
    (ke as f32 * 0.5 * ACCEL_CONVERSION_INV, dof)
}

fn atoms_mut(mols: &mut [WaterMol]) -> impl Iterator<Item = &mut AtomDynamics> {
    mols.iter_mut()
        .flat_map(|m| [&mut m.o, &mut m.h0, &mut m.h1].into_iter())
}

/// Removes center-of-mass drift.
fn remove_com_velocity(mols: &mut [WaterMol]) {
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

// todo: It might be nice to have this in lin_alg, although I don't want to add the rand
// todo dependency to it.
fn random_quaternion(rng: &mut ThreadRng, distro: Uniform<f32>) -> QuaternionF32 {
    let (u1, u2, u3) = (rng.sample(distro), rng.sample(distro), rng.sample(distro));
    let sqrt1_minus_u1 = (1.0 - u1).sqrt();
    let sqrt_u1 = u1.sqrt();
    let (theta1, theta2) = (TAU * u2, TAU * u3);

    QuaternionF32::new(
        sqrt1_minus_u1 * theta1.sin(),
        sqrt1_minus_u1 * theta1.cos(),
        sqrt_u1 * theta2.sin(),
        sqrt_u1 * theta2.cos(),
    )
    .to_normalized()
}

impl MdState {
    /// Use this to help initialize water molecules to realistic geometry of hydrogen bond networks,
    /// prior to the first proper simulation step. Runs MD on water only.
    /// Make sure to only run this after state is properly initialized, e.g. towards the end
    /// of init; not immediately after populating waters.
    pub fn md_on_water_only(&mut self, dev: &ComputationDevice) {
        println!("Initializing water H bond networks...");
        let start = Instant::now();

        // This disables things like snapshot saving, and certain prints.
        self.water_only_sim_at_init = true;

        // Mark all non-water atoms as static; keep track of their original state here.
        let mut static_state = Vec::with_capacity(self.atoms.len());
        for a in &mut self.atoms {
            static_state.push(a.static_);
            a.static_ = true;
        }

        for _ in 0..NUM_SIM_STEPS {
            self.step(dev, SIM_DT);
        }

        // Restore the original static state.
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.static_ = static_state[i];
        }

        self.water_only_sim_at_init = false;
        self.step_count = 0; // Reset.

        let elapsed = start.elapsed().as_millis();
        println!("Water H bond networks complete in {elapsed} ms");
    }
}
