#![allow(unused)]

//! This module is used to generate templates of water and other solvents. We don't
//! use it at run time. Example workflow: Populate a sim box of fixed size with the correct
//! number of solvent molecules for that solvent at a given temperature and pressure. Run a simulation
//! to equilibriate. Save to a template file. (e.g. .gro, or a binary format). Use that template
//! during MD runs in this library, or pass it to another MD engine.

use std::{f32::consts::TAU, path::Path};

use bio_files::gromacs::{
    OutputControl,
    gro::{AtomGro, Gro},
};
use lin_alg::{
    f32::{Mat3 as Mat3F32, Quaternion, Vec3},
    f64::{Quaternion as QuaternionF64, Vec3 as Vec3F64},
};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
    prelude::ThreadRng,
};
use rand_distr::Normal;

use crate::{
    AtomDynamics, ComputationDevice, MdConfig, MdOverrides, MdState, MolDynamics, NATIVE_TO_KCAL,
    ParamError, SimBoxInit,
    barostat::SimBox,
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers},
    solvent::{
        Solvent, WaterMolOpc, init,
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

/// Perhaps formally called *gradual isotropic compression*.
///
/// Create a solvent template by packing with a box which starts out too large, and gradually shrinks
/// to the proper size. This may work in cases where a grid-based or other packing doens't work in the cases
/// or long molecules like octanol. We initialize in a grid, then run a sim box. We shrink it graually enough
/// so that the molecules bend and move into deconflicted positions naturally. The final result has the correct
/// density (i.e. pressure) characteristics, and is equilibriated.
///
/// Note: This generates its own `MdState`, and exists outside our normal pipeline. IJt can, therefor,
/// be called externally.
///
/// We save a trajectory to disk, and a .gro file of the molecule set used, for playikng it back.
pub fn pack_solvent_with_shrinking_box(
    dev: &ComputationDevice,
    mol_solvent: &MolDynamics,
    mol_name: &str, // Residue name used in the saved .gro (e.g. "OCT", "MOL").
    solvent_count: usize,
    water_count: usize, // E.g. OPC water with the custom solvent.
    cell: SimBox,       // Final, after shrinking
    param_set: &FfParamSet,
    save_dir: &Path, // Path to write the equilibrated template as a GROMACS .gro file.
) -> Result<(Vec<MolDynamics>, Vec<Snapshot>), ParamError> {
    println!("Packing a custom solvent using a shrinking box...");

    // Side-length scale factor for the initial (large) box. Volume scales by the cube of this.
    // 2.0 seems insufficient for a naive packing of octanol.
    let initial_box_scale = 3.0;
    let dt = 0.001;
    // Å shrunk per axis per step (total, both sides combined).
    let box_shrink_per_step = 0.02;

    // Steps of pure MD to run after the box has reached its target size.
    // As this will be used for a template, err on the side of too many steps.
    let equilibration_steps = 10_000;

    // ── 1. Build the large initial cell, centered on the same point as the target ────────────
    let target_center = cell.center();
    let large_half = cell.extent * (initial_box_scale / 2.);
    let large_cell = SimBox::new(target_center - large_half, target_center + large_half);

    // ── 2. Grid-place `solvent_count` copies of `mol_solvent` with random orientations ───────
    // Template: prefer atom_posits override, fall back to atom positions.
    let template_world: Vec<Vec3F64> = mol_solvent
        .atom_posits
        .as_ref()
        .cloned()
        .unwrap_or_else(|| mol_solvent.atoms.iter().map(|a| a.posit).collect());

    let n_atoms = template_world.len();
    // Centroid-relative local positions (for rotation).
    let centroid = template_world
        .iter()
        .fold(Vec3F64::new(0., 0., 0.), |s, &p| s + p)
        * (1.0 / n_atoms as f64);
    let local: Vec<Vec3F64> = template_world.iter().map(|&p| p - centroid).collect();

    // Regular 3-D grid inside the large cell.
    let lx = large_cell.extent.x as f64;
    let ly = large_cell.extent.y as f64;
    let lz = large_cell.extent.z as f64;
    let base = (solvent_count as f64).cbrt().round().max(1.0) as usize;
    let n_x = base;
    let n_y = base;
    let n_z = solvent_count.div_ceil(n_x * n_y);
    let (sx, sy, sz) = (lx / n_x as f64, ly / n_y as f64, lz / n_z as f64);

    let mut rng = rand::rng();
    let mut mols: Vec<MolDynamics> = Vec::with_capacity(solvent_count);

    let mut rng = rand::rng();
    let distro = Uniform::<f32>::new(0.0, 1.0).unwrap();

    for idx in 0..solvent_count {
        let a = idx % n_x;
        let b = (idx / n_x) % n_y;
        let c = idx / (n_x * n_y);

        let world_ctr = Vec3F64::new(
            large_cell.bounds_low.x as f64 + (a as f64 + 0.5) * sx,
            large_cell.bounds_low.y as f64 + (b as f64 + 0.5) * sy,
            large_cell.bounds_low.z as f64 + (c as f64 + 0.5) * sz,
        );

        // Random uniform quaternion.
        let rot: QuaternionF64 = random_quaternion(&mut rng, distro).into();
        let posits: Vec<Vec3F64> = local
            .iter()
            .map(|&l| rot.rotate_vec(l) + world_ctr)
            .collect();

        let mut mol_copy = mol_solvent.clone();
        mol_copy.atom_posits = Some(posits);
        mols.push(mol_copy);
    }

    // ── 3. Create MdState with the large cell ────────────────────────────────────────────────
    let cfg = MdConfig {
        sim_box: SimBoxInit::Fixed((large_cell.bounds_low, large_cell.bounds_high)),
        // We place the solvent ourselves above; don't let MdState::new add any.
        solvent: Solvent::WaterOpcSpecifyMolCount(water_count),
        // No barostat: we drive the pressure manually by shrinking the box.
        barostat_cfg: None,
        snapshot_handlers: SnapshotHandlers {
            memory: Some(1),
            dcd: None,
            gromacs: OutputControl {
                // We only need posits and velocity at the end, but this may help us
                // visualize and validate this process.
                nstxout: Some(1),
                nstvout: Some(1),
                ..Default::default()
            },
        },
        overrides: MdOverrides {
            skip_water_relaxation: true,
            snapshots_during_equilibration: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let (mut md_state, _) = MdState::new(dev, &cfg, &mols, param_set)?;

    // ── 4. Shrink loop ───────────────────────────────────────────────────────────────────────
    // Compute the number of steps required to reach the target cell size on the slowest axis.
    let shrink_needed_x = large_cell.extent.x - cell.extent.x;
    let shrink_needed_y = large_cell.extent.y - cell.extent.y;
    let shrink_needed_z = large_cell.extent.z - cell.extent.z;
    let max_shrink = shrink_needed_x.max(shrink_needed_y).max(shrink_needed_z);
    let n_shrink_steps = (max_shrink / box_shrink_per_step).ceil() as usize;
    let total_steps = n_shrink_steps + equilibration_steps;

    println!(
        "pack_solvent_with_shrinking_box: shrinking from {:.1}×{:.1}×{:.1} Å to \
         {:.1}×{:.1}×{:.1} Å over {n_shrink_steps} steps, then {equilibration_steps} \
         equilibration steps.",
        large_cell.extent.x,
        large_cell.extent.y,
        large_cell.extent.z,
        cell.extent.x,
        cell.extent.y,
        cell.extent.z,
    );

    for step in 0..total_steps {
        md_state.step(dev, dt, None);

        if step < n_shrink_steps {
            let cur = &md_state.cell;
            let half = box_shrink_per_step / 2.;

            let new_low = Vec3::new(
                (cur.bounds_low.x + half).min(cell.bounds_low.x),
                (cur.bounds_low.y + half).min(cell.bounds_low.y),
                (cur.bounds_low.z + half).min(cell.bounds_low.z),
            );
            let new_high = Vec3::new(
                (cur.bounds_high.x - half).max(cell.bounds_high.x),
                (cur.bounds_high.y - half).max(cell.bounds_high.y),
                (cur.bounds_high.z - half).max(cell.bounds_high.z),
            );
            let new_cell = SimBox::new(new_low, new_high);

            // Calculate scale factors relative to the current cell
            let scale_x = (new_cell.extent.x / cur.extent.x) as f64;
            let scale_y = (new_cell.extent.y / cur.extent.y) as f64;
            let scale_z = (new_cell.extent.z / cur.extent.z) as f64;

            // --- Replace the atom wrapping loop with molecule scaling ---
            for mol_atoms in md_state.atoms.chunks_mut(n_atoms) {
                // 1. Calculate the current centroid of the molecule
                let mut centroid: Vec3F64 = Vec3F64::new(0., 0., 0.);
                for a in mol_atoms.iter() {
                    let p: Vec3F64 = a.posit.into();
                    centroid = centroid + p;
                }
                centroid *= 1.0 / n_atoms as f64;

                // 2. Calculate vector from target_center to centroid, and scale it
                let p: Vec3F64 = target_center.into();
                let relative_pos: Vec3F64 = centroid - p;
                let scaled_relative_pos = Vec3F64::new(
                    relative_pos.x * scale_x,
                    relative_pos.y * scale_y,
                    relative_pos.z * scale_z,
                );

                // 3. Determine the new centroid and calculate the displacement vector
                let p: Vec3 = scaled_relative_pos.into();
                let new_centroid: Vec3F64 = (target_center + p).into();
                let displacement: Vec3 = (new_centroid - centroid).into();

                // 4. Shift all atoms in the molecule by the displacement vector
                for a in mol_atoms.iter_mut() {
                    a.posit += displacement;
                }
            }

            md_state.cell = new_cell;
        }
    }

    // ── 5. Recenter the sim box on the origin ────────────────────────────────────────────────
    let final_center = md_state.cell.center();
    for a in &mut md_state.atoms {
        a.posit -= final_center;
    }
    for w in &mut md_state.water {
        w.o.posit -= final_center;
        w.h0.posit -= final_center;
        w.h1.posit -= final_center;
        w.m.posit -= final_center;
    }

    // ── 6. Save .gro file ────────────────────────────────────────────────────────────────────
    {
        let a_to_nm = |v: Vec3| -> Vec3F64 {
            Vec3F64::new(v.x as f64 / 10.0, v.y as f64 / 10.0, v.z as f64 / 10.0)
        };

        let mut gro_atoms: Vec<AtomGro> = Vec::new();
        let mut atom_serial = 1u32;

        // Custom solvent molecules
        for (mol_idx, chunk) in md_state.atoms.chunks(n_atoms).enumerate() {
            let mol_id = (mol_idx + 1) as u32;
            for a in chunk {
                gro_atoms.push(AtomGro {
                    mol_id,
                    mol_name: mol_name.to_string(),
                    element: a.element.clone(),
                    atom_type: a.force_field_type.clone(),
                    serial_number: atom_serial,
                    posit: a_to_nm(a.posit),
                    velocity: Some(a_to_nm(a.vel)),
                });
                atom_serial += 1;
            }
        }

        // OPC water molecules (O, H1, H2, M virtual site)
        for (w_idx, w) in md_state.water.iter().enumerate() {
            let mol_id = (solvent_count + w_idx + 1) as u32;
            // Note: We don't save MW; we can compute that.
            for (atom, name) in [(&w.o, "OW"), (&w.h0, "HW1"), (&w.h1, "HW2")] {
                gro_atoms.push(AtomGro {
                    mol_id,
                    mol_name: "SOL".to_string(),
                    element: atom.element.clone(),
                    atom_type: name.to_string(),
                    serial_number: atom_serial,
                    posit: a_to_nm(atom.posit),
                    velocity: Some(a_to_nm(atom.vel)),
                });
                atom_serial += 1;
            }
        }

        let ext = md_state.cell.extent;
        let gro = Gro {
            atoms: gro_atoms,
            head_text: format!("Solvent template: {mol_name}"),
            box_vec: Vec3F64::new(
                ext.x as f64 / 10.0,
                ext.y as f64 / 10.0,
                ext.z as f64 / 10.0,
            ),
        };

        let path = save_dir.join(format!("solvent_template_{mol_name}.gro"));

        match gro.save(&path) {
            Ok(()) => println!("Saved solvent template to {}", path.display()),
            Err(e) => {
                eprintln!(
                    "pack_solvent_with_shrinking_box: failed to save .gro file to path {path:?}: {e}"
                )
            }
        }
    }

    // ── 7. Reconstruct Vec<MolDynamics> from the flat atom list ─────────────────────────────
    let result_mols: Vec<MolDynamics> = md_state
        .atoms
        .chunks(n_atoms)
        .map(|chunk| {
            let mut mol = mol_solvent.clone();
            mol.atom_posits = Some(chunk.iter().map(|a| a.posit.into()).collect());
            mol.atom_init_velocities = Some(chunk.iter().map(|a| a.vel).collect());
            mol
        })
        .collect();

    println!("Shrinking box packing complete.");

    // Ok((result_mols, md_state.snapshots))
    Ok((result_mols, md_state.snapshots))
}
