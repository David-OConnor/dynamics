//! Packs solvent, with or without a solute, by initially creating a large box with low density,
//! and gradually shrinking it. We use this in several variants. It applies both to template creation,
//! And certain other workflows, e.g. for assessing molecular properties as part of the "main" (non-template-gen) sim.

use std::path::Path;

use bio_files::gromacs::{
    OutputControl,
    gro::{AtomGro, Gro},
};
use lin_alg::{
    f32::Vec3,
    f64::{Quaternion as QuaternionF64, Vec3 as Vec3F64},
};
use rand::distr::Uniform;

use crate::{
    ComputationDevice, CustomSolventCount, MdConfig, MdOverrides, MdState, MolDynamics, ParamError,
    SimBox, SimBoxInit, Solvent,
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers},
};

// The volume taken up by a single 4-point water molecule, in Å³. 0.997 g/cm³ water → ~0.03337 molecules/Å³ → ~29.97 Å³
// per OPC water molecule. Used only to
// reserve volume for co-packed water when auto-selecting the solvent count (mixed solvents).
const OPC_WATER_VOLUME: f64 = 29.97;

/// Apparent packing fraction φ = (Σ atomic van der Waals sphere volumes) / (bulk-liquid molar
/// volume), computed with *naive* (overlapping) spheres. Bond overlap shrinks the true molecular
/// volume by roughly as much as the liquid's void space inflates the molar volume, so across common
/// solvents (water, alcohols, alkanes, aromatics) φ clusters near ~0.95. This lets us estimate a
/// realistic molecule count for an arbitrary solvent without knowing its experimental density.
const DEFAULT_LIQUID_PACKING_FRACTION: f64 = 0.95;

/// Settings shared by template preparation and driven shrinking-box simulations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShrinkingBoxCfg {
    /// Side-length scale factor applied to the target box at the beginning of the run.
    pub initial_box_scale: f32,
    /// Angstroms removed from each box axis per simulation step.
    pub box_shrink_per_step: f32,
}

impl ShrinkingBoxCfg {
    pub fn initial_cell(self, target_cell: SimBox) -> SimBox {
        let scale = self.initial_box_scale.max(1.0);
        let center = target_cell.center();
        let half = target_cell.extent * (scale / 2.0);
        SimBox::new(center - half, center + half)
    }

    pub fn shrink_step_count(self, target_cell: SimBox) -> usize {
        let initial_cell = self.initial_cell(target_cell);
        let shrink_needed = initial_cell.extent - target_cell.extent;
        let max_shrink = shrink_needed.x.max(shrink_needed.y).max(shrink_needed.z);
        let shrink_per_step = self.box_shrink_per_step.max(f32::EPSILON);

        (max_shrink / shrink_per_step).ceil() as usize
    }

    /// Constant GROMACS `deform` rates in nm/ps for a run that reaches the target box
    /// at its final step. The remaining three triclinic shear rates are zero.
    pub fn gromacs_deform_nm_ps(self, target_cell: SimBox, dt: f32) -> [f32; 6] {
        let initial_cell = self.initial_cell(target_cell);
        let duration_ps = self.shrink_step_count(target_cell).max(1) as f32 * dt.max(f32::EPSILON);
        let to_nm_ps = |initial: f32, target: f32| (target - initial) * 0.1 / duration_ps;

        [
            to_nm_ps(initial_cell.extent.x, target_cell.extent.x),
            to_nm_ps(initial_cell.extent.y, target_cell.extent.y),
            to_nm_ps(initial_cell.extent.z, target_cell.extent.z),
            0.0,
            0.0,
            0.0,
        ]
    }

    pub(in crate::solvent) fn next_cell(
        self,
        current_cell: SimBox,
        target_cell: SimBox,
    ) -> Option<SimBox> {
        if current_cell.extent.x <= target_cell.extent.x
            && current_cell.extent.y <= target_cell.extent.y
            && current_cell.extent.z <= target_cell.extent.z
        {
            return None;
        }

        let half = self.box_shrink_per_step.max(0.0) / 2.0;
        if half <= f32::EPSILON {
            return None;
        }

        Some(SimBox::new(
            Vec3::new(
                (current_cell.bounds_low.x + half).min(target_cell.bounds_low.x),
                (current_cell.bounds_low.y + half).min(target_cell.bounds_low.y),
                (current_cell.bounds_low.z + half).min(target_cell.bounds_low.z),
            ),
            Vec3::new(
                (current_cell.bounds_high.x - half).max(target_cell.bounds_high.x),
                (current_cell.bounds_high.y - half).max(target_cell.bounds_high.y),
                (current_cell.bounds_high.z - half).max(target_cell.bounds_high.z),
            ),
        ))
    }
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
/// We save a trajectory to disk, and a .gro file of the molecule set used, for playing it back.
/// The `.gro` file can be used as a template; it has the final atom positions.
#[derive(Clone, Copy, Debug)]
pub struct ShrinkingBoxPackingCfg {
    pub initial_box_scale: f32,
    pub dt: f32,
    pub box_shrink_per_step: f32,
    pub equilibration_steps: usize,
    pub snapshot_interval: Option<usize>,
    pub gromacs_output_interval: Option<u32>,
    pub save_gro: bool,
    pub count: CustomSolventCount,
}

impl Default for ShrinkingBoxPackingCfg {
    fn default() -> Self {
        Self {
            // Side-length scale factor for the initial (large) box. Volume scales by the cube
            // of this. 2.0 seems insufficient for a naive packing of octanol.
            initial_box_scale: 3.0,
            dt: 0.001,
            // Å shrunk per axis per step (total, both sides combined).
            box_shrink_per_step: 0.02,
            // Steps of pure MD to run after the box has reached its target size. As this will
            // be used for a template, err on the side of too many steps.
            equilibration_steps: 20_000,
            snapshot_interval: Some(1),
            gromacs_output_interval: Some(1),
            save_gro: true,
            count: CustomSolventCount::Auto(DEFAULT_LIQUID_PACKING_FRACTION),
        }
    }
}

/// A think wrapper over `pack_solvent_with_shrinking_box_cfg`, but with the default config.
pub fn pack_solvent_with_shrinking_box(
    dev: &ComputationDevice,
    mol_solvent: &MolDynamics,
    mol_name: &str,     // Residue name used in the saved .gro (e.g. "OCT", "MOL").
    water_count: usize, // E.g. OPC water with the custom solvent; specifies the OPC (etc) count.
    cell: SimBox,       // Final, after shrinking
    param_set: &FfParamSet,
    save_dir: &Path, // Path to write the equilibrated template as a GROMACS .gro file.
) -> Result<(Vec<MolDynamics>, Vec<Snapshot>), ParamError> {
    pack_solvent_with_shrinking_box_cfg(
        dev,
        mol_solvent,
        mol_name,
        water_count,
        cell,
        param_set,
        save_dir,
        ShrinkingBoxPackingCfg::default(),
    )
}

pub fn pack_solvent_with_shrinking_box_cfg(
    dev: &ComputationDevice,
    mol_solvent: &MolDynamics,
    mol_name: &str,
    water_count: usize, // E.g. OPC water with the custom solvent.
    cell: SimBox,       // Final, after shrinking
    param_set: &FfParamSet,
    save_dir: &Path, // Path to write the equilibrated template as a GROMACS .gro file.
    packing_cfg: ShrinkingBoxPackingCfg,
) -> Result<(Vec<MolDynamics>, Vec<Snapshot>), ParamError> {
    println!("Packing a custom solvent using a shrinking box...");

    let solvent_count = match packing_cfg.count {
        CustomSolventCount::Specified(count) => count,
        CustomSolventCount::Auto(packing_fraction) => {
            estimate_solvent_count(mol_solvent, &cell, water_count, packing_fraction)
        }
    };

    let initial_box_scale = packing_cfg.initial_box_scale;
    let dt = packing_cfg.dt;
    let box_shrink_per_step = packing_cfg.box_shrink_per_step;
    let equilibration_steps = packing_cfg.equilibration_steps;

    let shrink_cfg = ShrinkingBoxCfg {
        initial_box_scale,
        box_shrink_per_step,
    };

    // Build the large initial cell, centered on the same point as the target.
    let large_cell = shrink_cfg.initial_cell(cell);

    // Grid-place `solvent_count` copies of `mol_solvent` with random orientations
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

    let mut mols: Vec<MolDynamics> = Vec::with_capacity(solvent_count);
    let mut solvent_centers: Vec<Vec3F64> = Vec::with_capacity(solvent_count);
    let solvent_spacing = Vec3F64::new(sx, sy, sz);

    let mut rng = rand::rng();
    let distro = Uniform::<f64>::new(0.0, 1.0).unwrap();

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
        let rot = QuaternionF64::random(&mut rng, Some(distro));
        let posits: Vec<Vec3F64> = local
            .iter()
            .map(|&l| rot.rotate_vec(l) + world_ctr)
            .collect();

        let mut mol_copy = mol_solvent.clone();
        mol_copy.atom_posits = Some(posits);
        mols.push(mol_copy);
        solvent_centers.push(world_ctr);
    }

    // Create MdState with the large cell
    let cfg = MdConfig {
        sim_box: SimBoxInit::Fixed((large_cell.bounds_low, large_cell.bounds_high)),
        // We place the solvent ourselves above; don't let MdState::new add any.
        solvent: if water_count == 0 {
            Solvent::None
        } else {
            Solvent::WaterOpcSpecifyMolCount(water_count)
        },
        // No barostat: we drive the pressure manually by shrinking the box.
        barostat_cfg: None,
        snapshot_handlers: SnapshotHandlers {
            memory: packing_cfg.snapshot_interval,
            dcd: None,
            gromacs: if let Some(interval) = packing_cfg.gromacs_output_interval {
                OutputControl {
                    // We only need posits and velocity at the end, but this may help us
                    // visualize and validate this process.
                    nstxout: Some(interval),
                    nstvout: Some(interval),
                    ..Default::default()
                }
            } else {
                OutputControl {
                    nstxout: None,
                    nstvout: None,
                    nstfout: None,
                    nstlog: None,
                    nstcalcenergy: None,
                    nstenergy: None,
                    nstxout_compressed: None,
                    ..Default::default()
                }
            },
        },
        overrides: MdOverrides {
            skip_water_relaxation: true,
            snapshots_during_equilibration: packing_cfg.snapshot_interval.is_some(),
            ..Default::default()
        },
        ..Default::default()
    };

    let (mut md_state, _) = MdState::new(dev, &cfg, &mols, param_set)?;

    if water_count > 0 {
        md_state.redistribute_interleaved_opc_waters(dev, &solvent_centers, solvent_spacing);
    }

    // Shrink loop
    // Compute the number of steps required to reach the target cell size on the slowest axis.
    let n_shrink_steps = shrink_cfg.shrink_step_count(cell);
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
            md_state.shrink_cell_towards(dev, cell, shrink_cfg);
        }
    }

    // Recenter the sim box on the origin
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

    if packing_cfg.save_gro {
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

    // Reconstruct Vec<MolDynamics> from the flat atom list
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

/// Estimate how many copies of `mol_solvent` reproduce a realistic bulk-liquid density inside
/// `cell` — i.e. the count we pack when the caller uses `CustomSolventCount::Auto`. We approximate
/// each molecule's occupied volume as the sum of its atoms' van der Waals sphere volumes, divide
/// the available box volume (minus volume reserved for `water_count` co-packed waters) by the
/// per-molecule molar volume implied by `packing_fraction`, and round.
///
/// This is a heuristic: it targets densities within ~10-15% for typical organic solvents, which is
/// a good starting configuration for the shrink + equilibration run that follows. (Once pressure
/// measurement / the barostat are re-enabled, this could be refined by instead shrinking the box
/// until the target pressure is reached, then reading off the resulting density.)
fn estimate_solvent_count(
    mol_solvent: &MolDynamics,
    cell: &SimBox,
    water_count: usize,
    packing_fraction: f64,
) -> usize {
    // Sum of atomic van der Waals sphere volumes for one solvent molecule (Å³).
    let mut vdw_volume = 0.0_f64;
    for atom in &mol_solvent.atoms {
        // `vdw_radius` returns 0 for elements without a tabulated radius; fall back to a
        // carbon-like radius so exotic atoms still contribute a sane volume.
        let r = match atom.element.vdw_radius() as f64 {
            r if r > 0.0 => r,
            _ => 1.7,
        };
        vdw_volume += (4.0 / 3.0) * std::f64::consts::PI * r * r * r;
    }

    if vdw_volume <= 0.0 || packing_fraction <= 0.0 {
        eprintln!(
            "estimate_solvent_count: cannot size solvent (vdW vol {vdw_volume:.2}, φ \
             {packing_fraction:.2}); defaulting to 1 molecule."
        );
        return 1;
    }

    // Molar volume of the bulk liquid implied by the apparent packing fraction.
    let molar_volume = vdw_volume / packing_fraction;

    let available_vol = (cell.volume() as f64 - water_count as f64 * OPC_WATER_VOLUME).max(0.0);

    let n = ((available_vol / molar_volume).round() as usize).max(1);

    println!(
        "Auto solvent count: {n} mols of {} atoms (vdW vol {vdw_volume:.1} Å³, molar vol \
         {molar_volume:.1} Å³) to fill {:.0} Å³ (less {water_count} waters) at φ={packing_fraction:.2}.",
        mol_solvent.atoms.len(),
        cell.volume(),
    );

    n
}
