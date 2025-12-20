#![allow(clippy::excessive_precision)]

//! Code for initializing water molecules, including assigning quantity, initial positions, and
//! velocities. Set up to meet density, pressure, and or temperature targets. Not specific to the
//! water model used.

use crate::partial_charge_inference::files::{load, load_from_bytes};
use crate::partial_charge_inference::save;
use crate::{
    ACCEL_CONVERSION_INV, AtomDynamics, ComputationDevice, MdState,
    ambient::{GAS_CONST_R, KB_A2_PS2_PER_K_PER_AMU, SimBox},
    sa_surface,
    water::WaterMol,
};
use bincode::{Decode, Encode};
use lin_alg::f32::{Mat3 as Mat3F32, Quaternion, Vec3};
use rand::{Rng, distr::Uniform, rngs::ThreadRng};
use rand_distr::{Distribution, Normal};
use std::path::Path;
use std::{f32::consts::TAU, io, time::Instant};

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
// todo: Hmm. We could get lower, but there's some risk of an H atom being too close,
// todo and we're currently only measuring water o dist to atoms.
const MIN_NONWATER_DIST: f32 = 1.7;
const MIN_NONWATER_DIST_SQ: f32 = MIN_NONWATER_DIST * MIN_NONWATER_DIST;

// This seems low, but allows more flexibility in placement, and isn't so low
// as to cause the system to blow up. Distances will be set up accurately
// in the initial water-only sim.
// in the initial water-only sim.
const MIN_WATER_O_O_DIST: f32 = 1.7;
const MIN_WATER_O_O_DIST_SQ: f32 = MIN_WATER_O_O_DIST * MIN_WATER_O_O_DIST;

// Higher is better, but slower. After hydrogen bond networks are settled, higher doensn't
// improve things. Note that we initialize from a pre-equilibrated template, so we shouldn't
// need many effects. This mainly deals with template tiling effects, and water-solute conflicts.

// todo: Make this dynamic based on thngs like solute size relative to sim box?
// const NUM_SIM_STEPS: usize = 1_000;
const NUM_SIM_STEPS: usize = 300;
// Like in our normal setup with constraint H, 0.002ps may be the safe upper bound.
// We seem to get better settling results with a low dt.
// const SIM_DT: f32 = 0.001;
const SIM_DT: f32 = 0.0005;

const INIT_TEMPLATE: &[u8] = include_bytes!("../../param_data/water_60A.water_init_template");

/// We store pre-equilibrated water molecules in a template, and use it to initialize water for a simulation.
/// This keeps the equilibration steps relatively low. Note that edge effects from tiling will require
/// equilibration, as well as adjusting a template for the runtime temperature target.
///
/// Struct-of-array layout. (Corresponding indices)
/// Public so it can be created by the application after a run.
///
/// 108 bytes/mol. Size on disk/mem: for a 60Å side len: ~780kb. (Hmm: We're getting a bit less)
/// 80Å/side: 1.20Mb.
#[derive(Encode, Decode)]
pub struct WaterInitTemplate {
    // todo: This could be made more compact by storing positions and orientations, but this would
    // todo require some care. (7 numerical vals vs 9 per mol). Similar concept for storing angular
    // velocity and o velocity, instead of 3 separate velocities
    // posits: Vec<Vec3>,
    o_posits: Vec<Vec3>,
    h0_posits: Vec<Vec3>,
    h1_posits: Vec<Vec3>,
    // todo: Is this OK,
    // orientations: Vec<Quaternion>,
    o_velocities: Vec<Vec3>,
    h0_velocities: Vec<Vec3>,
    h1_velocities: Vec<Vec3>,
    // todo: Cache these, or ifer?
    /// This must correspond to the positions. Cached.
    bounds: (Vec3, Vec3),
}

impl WaterInitTemplate {
    /// Construct from the current state, and save to file.
    /// Call this explicitly. (todo: Determine a formal or informal approach)
    pub fn save(water: &[WaterMol], bounds: (Vec3, Vec3), path: &Path) -> io::Result<()> {
        let n = water.len();

        let mut o_posits = Vec::with_capacity(n);
        let mut h0_posits = Vec::with_capacity(n);
        let mut h1_posits = Vec::with_capacity(n);

        let mut o_velocities = Vec::with_capacity(n);
        let mut h0_velocities = Vec::with_capacity(n);
        let mut h1_velocities = Vec::with_capacity(n);

        // let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));

        // Sort water by position so that it iterates out from the center. This makes initialization
        // easier for cases where this template is larger than the target sim box.
        let water = {
            let ctr = (bounds.1 + bounds.0) / 2.;

            let mut w = water.to_vec();
            w.sort_by(|a, b| {
                let da = (a.o.posit - ctr).magnitude_squared();
                let db = (b.o.posit - ctr).magnitude_squared();
                da.total_cmp(&db)
            });

            w
        };

        for mol in water {
            o_posits.push(mol.o.posit);
            h0_posits.push(mol.h0.posit);
            h1_posits.push(mol.h1.posit);

            o_velocities.push(mol.o.vel);
            h0_velocities.push(mol.h0.vel);
            h1_velocities.push(mol.h1.vel);

            // min = min.min(mol.o.posit);
            // max = max.max(mol.o.posit);
        }

        let result = Self {
            o_posits,
            h0_posits,
            h1_posits,
            o_velocities,
            h0_velocities,
            h1_velocities,
            bounds,
        };

        save(path, &result)
    }
}

fn calc_n_mols(cell: &SimBox, atoms: &[AtomDynamics]) -> usize {
    let cell_volume = cell.volume();
    let mol_volume = sa_surface::vol_take_up_by_atoms(atoms);
    let free_vol = cell_volume - mol_volume;

    println!(
        "Solvent-free vol: {:.2} Cell vol: {:.2} (Å³ / 1,000)",
        free_vol / 1_000.,
        cell_volume / 1_000.
    );

    // Estimate free volume & n_mols from it
    (WATER_MOLS_PER_VOL * free_vol).round() as usize
}

/// Create water molecules from a template. If the target sim box is smaller than the template, we
/// iterate from the center out. Otherwise, we tile. Tiling requires more intitialization steps
/// due to conflicts at the touching faces.
///
/// Contains repetition with the deprecated latice-based approach.
pub fn make_water_mols(
    cell: &SimBox,
    _temperature_tgt: f32,
    atoms: &[AtomDynamics],
    _zero_com_drift: bool,
) -> Vec<WaterMol> {
    println!("Initializing water molecules...");
    let start = Instant::now();

    let template: WaterInitTemplate = load_from_bytes(INIT_TEMPLATE).unwrap();

    let n_mols = calc_n_mols(cell, atoms);

    let mut result = Vec::with_capacity(n_mols);

    if n_mols == 0 {
        println!("Complete in {} ms.", start.elapsed().as_millis());
        return result;
    }

    // Solute
    let atom_posits: Vec<_> = atoms.iter().map(|a| a.posit).collect();

    // Prevents unbounded looping. A higher value means we're more likely to succeed,
    // but the run time could be higher.
    let fault_ratio = 4;

    let mut num_added = 0;
    let mut loops_used = 0;

    let template_size = template.bounds.1 - template.bounds.0;
    let cell_size = cell.bounds_high - cell.bounds_low;

    let cell_fits_in_template = cell_size.x <= template_size.x
        && cell_size.y <= template_size.y
        && cell_size.z <= template_size.z;

    // todo: QC this offset
    let mut offset = Vec3::new_zero();

    // Iterate inside out using the template
    if cell_fits_in_template {
        // Align the template center to the cell center.
        let template_ctr = (template.bounds.0 + template.bounds.1) / 2.;
        let cell_ctr = (cell.bounds_low + cell.bounds_high) / 2.;
        offset = cell_ctr - template_ctr;

        // This loop requires the template to iterate inside-out.
        'outer: for i in 0..n_mols * fault_ratio {
            let o_posit = template.o_posits[i] + offset;

            // If overlapping with a solute atom, don't place. We'll catch populate it towards the end.
            for atom_p in &atom_posits {
                let dist_sq = (*atom_p - o_posit).magnitude_squared();
                if dist_sq < MIN_NONWATER_DIST_SQ {
                    loops_used += 1;
                    continue 'outer;
                }
            }

            // Check for an overlap with existing water molecules.
            for w in &result {
                // todo: QC , and if you even need this
                // todo: If the problem you have lies elsewhere, you can remove this check.
                let dist_sq =
                    (cell.min_image(w.o.posit) - cell.min_image(o_posit)).magnitude_squared();

                if dist_sq < MIN_WATER_O_O_DIST_SQ {
                    loops_used += 1;
                    continue 'outer;
                }
            }

            // todo: You must check sim box edge effects of water-water interactions.

            // Check if the molecule is inside the cell. Our templates iterate in a spherical pattern
            // outwards, so this will happen for molecules outside the cell faces towards their center,
            // until the corners are filled.
            if !cell.contains(o_posit) {
                loops_used += 1;
                continue 'outer;
            }

            // This template sets up the charge, mass, element, etc. We override posits and vels.
            let mut mol = WaterMol::new(
                Vec3::new_zero(),
                Vec3::new_zero(),
                Quaternion::new_identity(),
            );
            mol.o.posit = o_posit;
            mol.h0.posit = template.h0_posits[i] + offset;
            mol.h1.posit = template.h1_posits[i] + offset;

            // todo: Temp rm
            // mol.o.vel = template.o_velocities[i];
            // mol.h0.vel = template.h0_velocities[i];
            // mol.h1.vel = template.h1_velocities[i];

            println!("MOL: {mol:?}");

            result.push(mol);
            num_added += 1;

            if num_added == n_mols {
                break;
            }
            loops_used += 1;
        }
    } else {
        // Tile the template.
        let min0 = template.bounds.0 + offset;
        let max1 = template.bounds.1 + offset;

        let ix_min = ((cell.bounds_low.x - max1.x) / template_size.x).floor() as i32;
        let iy_min = ((cell.bounds_low.y - max1.y) / template_size.y).floor() as i32;
        let iz_min = ((cell.bounds_low.z - max1.z) / template_size.z).floor() as i32;

        let ix_max = ((cell.bounds_high.x - min0.x) / template_size.x).ceil() as i32;
        let iy_max = ((cell.bounds_high.y - min0.y) / template_size.y).ceil() as i32;
        let iz_max = ((cell.bounds_high.z - min0.z) / template_size.z).ceil() as i32;

        'tiles: for ix in ix_min..=ix_max {
            for iy in iy_min..=iy_max {
                for iz in iz_min..=iz_max {
                    let tile_offset = offset
                        + Vec3::new(
                            ix as f32 * template_size.x,
                            iy as f32 * template_size.y,
                            iz as f32 * template_size.z,
                        );

                    for i in 0..template.o_posits.len() {
                        let o_posit = template.o_posits[i] + tile_offset;
                        let h0_posit = template.h0_posits[i] + tile_offset;
                        let h1_posit = template.h1_posits[i] + tile_offset;

                        loops_used += 1;

                        if !cell.contains(o_posit) {
                            continue;
                        }

                        // todo: Put in
                        // if overlaps_solute_sites(o_posit, h0_posit, h1_posit, &atom_posits) {
                        //     continue;
                        // }
                        //
                        // if overlaps_water_sites(o_posit, h0_posit, h1_posit, &result) {
                        //     continue;
                        // }

                        let mut mol = WaterMol::new(
                            Vec3::new_zero(),
                            Vec3::new_zero(),
                            Quaternion::new_identity(),
                        );
                        mol.o.posit = o_posit;
                        mol.h0.posit = h0_posit;
                        mol.h1.posit = h1_posit;

                        mol.o.vel = template.o_velocities[i];
                        mol.h0.vel = template.h0_velocities[i];
                        mol.h1.vel = template.h1_velocities[i];

                        result.push(mol);
                        num_added += 1;

                        if num_added == n_mols {
                            break 'tiles;
                        }
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!(
        "Added {} / {n_mols} water mols in {elapsed} ms. Used {loops_used} loops",
        result.len()
    );

    result
}

/// Generate water molecules to meet a temperature target, using standard density assumptions.
/// We deconflict with (solute) atoms in the simulation, and base the number of molecules to add
/// on the free space, not the total cell volume.
///
/// Process:
/// - Compute the solvent-free volume using an isosurface
/// - Compute the number of molecules to add
/// - Add them on a regular grid with random orientations, and velocities in a random distribution
///   that matches the target temperature. Move molecules to the edge that are too close to
///   solute atoms.
/// - Run a brief simulation with the solute
///   atoms as static, to intially position water molecules realistically. This
///   takes advantage of our simulations' acceleration limits to set up realistic geometry using
///   hydrogen bond networks, and breaks the crystal lattice.
///
/// Note: If we're able to place most, but not all waters, the barostat should adjust the sim box size
/// to account for the lower-than-specific pressure.
///
/// todo: Update this so it creates realistic orientations and molecules intead of a lattice.
/// todo: This will require (much?) less equilibration.
pub fn _make_water_mols_grid(
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

    let n_mols = calc_n_mols(cell, atoms);

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
    let n_z = n_mols.div_ceil(n_x * n_y);

    let spacing_x = lx / n_x as f32;
    let spacing_y = ly / n_y as f32;
    let spacing_z = lz / n_z as f32;

    // Solute
    let atom_posits: Vec<_> = atoms.iter().map(|a| a.posit).collect();

    // Prevents unbounded looping. A higher value means we're more likely to succed,
    // but the run time could be higher.
    let fault_ratio = 3;

    let mut num_added = 0;
    let mut loops_used = 0;

    'outer: for i in 0..n_mols * fault_ratio {
        let a = i % n_x;
        let b = (i / n_x) % n_y;
        let c = (i / (n_x * n_y)) % n_z;
        // let c = i / (n_x * n_y);

        let posit = Vec3::new(
            cell.bounds_low.x + (a as f32 + 0.5) * spacing_x,
            cell.bounds_low.y + (b as f32 + 0.5) * spacing_y,
            cell.bounds_low.z + (c as f32 + 0.5) * spacing_z,
        );

        // If overlapping with a solute atom, don't place. We'll catch it at the end.
        for atom_p in &atom_posits {
            let dist_sq = (*atom_p - posit).magnitude_squared();
            if dist_sq < MIN_NONWATER_DIST_SQ {
                loops_used += 1;
                continue 'outer;
            }
        }

        // Check for an overlap with existing water molecules.
        for w in &result {
            let dist_sq = (w.o.posit - posit).magnitude_squared();
            if dist_sq < MIN_WATER_O_O_DIST_SQ {
                loops_used += 1;
                continue 'outer;
            }
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
    _init_velocities(&mut result, temperature_tgt, zero_com_drift, &mut rng);

    let elapsed = start.elapsed().as_millis();
    println!(
        "Added {} / {n_mols} water mols in {elapsed} ms. Used {loops_used} loops",
        result.len()
    );
    result
}

/// Note: This sets a reasonable default, but our thermostat, applied notably during
/// our initial water simulation, determines the actual temperature set at proper sim init.
/// Note: We've deprecated this in favor of velocities pre-initialized in the template.
fn _init_velocities(
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

impl MdState {
    /// Use this to help initialize water molecules to realistic geometry of hydrogen bond networks,
    /// prior to the first proper simulation step. Runs MD on water only.
    /// Make sure to only run this after state is properly initialized, e.g. towards the end
    /// of init; not immediately after populating waters.
    ///
    /// This will result in an immediate energy bump as water positions settle from their grid
    /// into position. As they settle, the thermostat will bring the velocities down to set
    /// the target temp. This sim should run long enough to the water is stable by the time
    /// the main sim starts.
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
