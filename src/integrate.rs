//! Contains integration code, including the primary time step.

use std::{
    fmt,
    fmt::{Display, Formatter},
    time::Instant,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use ewald::{PmeRecip, ewald_comp_force};
use lin_alg::f32::Vec3;

use crate::{
    ACCEL_CONVERSION, CENTER_SIMBOX_RATIO, ComputationDevice, HydrogenConstraint, MdState,
    PMEIndex,
    non_bonded::{EWALD_ALPHA, SCALE_COUL_14, SPME_N},
    snapshot::{FILE_SAVE_INTERVAL, SaveType, append_dcd},
};

// todo: Make this Thermostat instead of Integrator? And have a WIP Integrator with just VV.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub enum Integrator {
    VerletVelocity,
    /// Velocity-verlet with a Langevin thermometer. Good temperature control
    /// and ergodicity, but the friction parameter damps real dynamics as it grows.
    /// γ is friction in 1/ps. Typical values are 1–5. for proteins in implicit/weak solvent.
    /// With explicit solvents, we can often go lower to 0.1 – 1.
    /// A higher value has strong damping and is rougher. A lower value is gentler.
    Langevin {
        gamma: f32,
    },
    /// See notes on Langevin. This version splits drift into two halves, and is likely more accurate.
    LangevinMiddle {
        gamma: f32,
    },
}

impl Default for Integrator {
    fn default() -> Self {
        Self::LangevinMiddle { gamma: 1. }
    }
}

impl Display for Integrator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Integrator::VerletVelocity => write!(f, "Verlet Vel"),
            Integrator::Langevin { gamma } => write!(f, "Langevin. γ: {gamma}"),
            Integrator::LangevinMiddle { gamma } => write!(f, "Langevin Mid. γ: {gamma}"),
        }
    }
}

impl MdState {
    /// For Langevin middle.
    fn drift_atoms(&mut self, dt: f32) {
        for a in &mut self.atoms {
            a.posit += a.vel * dt;
            a.posit = self.cell.wrap(a.posit);
            self.neighbors_nb.max_displacement_sq = self
                .neighbors_nb
                .max_displacement_sq
                .max((a.vel * dt).magnitude_squared());
        }
    }

    /// For Langevin middle.
    fn water_drift(&mut self, dt: f32) {
        for w in &mut self.water {
            w.o.posit += w.o.vel * dt;
            w.h0.posit += w.h0.vel * dt;
            w.h1.posit += w.h1.vel * dt;
            w.m.posit += w.m.vel * dt;

            w.o.posit = self.cell.wrap(w.o.posit);
            w.h0.posit = self.cell.wrap(w.h0.posit);
            w.h1.posit = self.cell.wrap(w.h1.posit);
            w.m.posit = self.cell.wrap(w.m.posit);
        }
    }

    /// One **Velocity-Verlet** step (leap-frog style) of length `dt` is in picoseconds (10^-12),
    /// with typical values of 0.001, or 0.002ps (1 or 2fs).
    /// This method orchestrates the dynamics at each time step.
    pub fn step(&mut self, dev: &ComputationDevice, dt: f32) {
        let dt_half = 0.5 * dt;

        match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                // See notes in the below branch.

                // Half-kick
                for a in &mut self.atoms {
                    a.vel += a.accel * dt_half;
                }

                self.water_vv_first_half_and_drift(dt, dt_half);

                self.drift_atoms(dt_half);

                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.shake_hydrogens();
                }

                self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);

                self.drift_atoms(dt_half);
                self.water_drift(dt_half);

                self.reset_accels();
                self.apply_all_forces(dev);

                let start = Instant::now();
                self.handle_spme_recip(dev);
                if self.step_count == 0 {
                    let elapsed = start.elapsed();
                    println!("SPME recip time: {:?} μs", elapsed.as_micros());
                }

                for a in &mut self.atoms {
                    a.accel *= ACCEL_CONVERSION as f32 / a.mass;
                    a.vel += a.accel * dt_half;
                }

                // self.water_vv_second_half(&mut self.forces_on_water, dt_half);
                self.water_vv_second_half(dt_half);

                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    // todo: Sort this index out!
                    self.rattle_hydrogens(0);
                }
            }
            _ => {
                // O(dt/2)
                if let Integrator::Langevin { gamma } = self.cfg.integrator {
                    self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);
                }

                // First half-kick (v += a dt/2) and drift (x += v dt)
                // todo: Do we want traditional verlet instead of velocity verlet (VV)?
                // Note: We do not apply the accel unit conversion, nor mass division here; they're already
                // included in this values from the previous step.
                for a in &mut self.atoms {
                    a.vel += a.accel * dt_half; // Half-kick

                    a.posit += a.vel * dt; // Drift
                    a.posit = self.cell.wrap(a.posit);

                    // todo: What is this? Implement it, or remove it?
                    // todo: Should this take water displacements into account?
                    // track the largest squared displacement to know when to rebuild the list
                    self.neighbors_nb.max_displacement_sq = self
                        .neighbors_nb
                        .max_displacement_sq
                        .max((a.vel * dt).magnitude_squared());
                }

                // todo: Consider applying the thermostat between the first half-kick and drift.
                // todo: e.g. half-kick, then shake H and settle velocity water (?), then thermostat, then drift. (?) ,
                // todo then settle positions?

                self.water_vv_first_half_and_drift(dt, dt_half);

                // The order we perform these steps is important.
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.shake_hydrogens();
                }

                self.reset_accels();
                self.apply_all_forces(dev);

                let start = Instant::now();
                // todo: YOu need to update potential energy from LR PME as well.
                self.handle_spme_recip(dev);
                if self.step_count == 0 {
                    let elapsed = start.elapsed();
                    println!("SPME recip time: {:?} μs", elapsed.as_micros());
                }

                // Forces (bonded and nonbonded, to dynamic and water atoms) have been applied; perform other
                // steps required for integration; second half-kick, RATTLE for hydrogens; SETTLE for water. -----

                // Second half-kick using the forces calculated this step, and update accelerations using the atom's mass;
                // Between the accel reset and this step, the accelerations have been missing those factors; this is an optimization to
                // do it once at the end.
                for a in &mut self.atoms {
                    // We divide by mass here, once accelerations have been computed in parts above; this
                    // is an optimization to prevent dividing each accel component by it.
                    // This is the step where we A: convert force to accel, and B: Convert units from the param
                    // units to the ones we use in dynamics.
                    a.accel *= ACCEL_CONVERSION as f32 / a.mass;
                    a.vel += a.accel * dt_half;
                }

                // self.water_vv_second_half(&mut self.forces_on_water, dt_half);
                self.water_vv_second_half(dt_half);

                // O(dt/2)
                if let Integrator::Langevin { gamma } = self.cfg.integrator {
                    self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);
                }

                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    // todo: Sort this index out!
                    self.rattle_hydrogens(0);
                }
            }
        }

        let dt_f64 = dt as f64;

        // I believe we must run barostat prior to thermostat, in our current configuration.
        self.apply_barostat_berendsen(dt_f64);

        if let Integrator::VerletVelocity = self.cfg.integrator {
            self.apply_thermostat_csvr(dt_f64, self.cfg.temp_target as f64);
        }

        self.time += dt as f64;
        self.step_count += 1;

        // todo: Ratio for this too?
        self.build_neighbors_if_needed();

        // We keeping the cell centered on the dynamics atoms. Note that we don't change the dimensions,
        // as these are under management by the barostat.
        if self.step_count % CENTER_SIMBOX_RATIO == 0 {
            self.cell.recenter(&self.atoms);

            // todo: Will this interfere with carrying over state from the previous step?
            self.regen_pme();
        }

        let mut updated_ke = false;
        let mut take_ss = false;
        let mut take_ss_file = false;

        for handler in &self.cfg.snapshot_handlers {
            if self.step_count % handler.ratio != 0 {
                continue;
            }

            // We currently only use kinetic energy in snapshots, so update it only when
            // calling a handler.
            if !updated_ke {
                updated_ke = true;
                self.kinetic_energy = self.current_kinetic_energy();
            }

            match &handler.save_type {
                // No action if multiple Memory savetypes are specified.
                SaveType::Memory => {
                    take_ss = true;
                }
                SaveType::Dcd(path) => {
                    take_ss_file = true;

                    // todo: Handle the case of the final step!
                    if self.step_count % FILE_SAVE_INTERVAL == 0 {
                        if let Err(e) = append_dcd(&self.snapshot_queue_for_file, &path) {
                            eprintln!("Error saving snapshot as DCD: {e:?}");
                        }
                        self.snapshot_queue_for_file = Vec::new();
                    }
                }
            }
        }

        if take_ss || take_ss_file {
            let snapshot = self.take_snapshot();

            if take_ss {
                // todo: DOn't clone.
                self.snapshots.push(snapshot.clone());
            }
            if take_ss_file {
                self.snapshot_queue_for_file.push(snapshot);
            }
        }
    }

    fn handle_spme_recip(&mut self, dev: &ComputationDevice) {
        const K_COUL: f32 = 1.; // todo: ChatGPT really wants this, but I don't think I need it.

        let (pos_all, q_all, map) = self.gather_pme_particles_wrapped();

        let (mut f_recip, e_recip) = match &mut self.pme_recip {
            Some(pme_recip) => {
                match dev {
                    ComputationDevice::Cpu => pme_recip.forces(&pos_all, &q_all),
                    #[cfg(feature = "cuda")]
                    ComputationDevice::Gpu((stream, module)) => {
                        // self.pme_recip.forces_gpu(stream, module, &pos_all, &q_all)
                        // todo: GPU isn't improving this, but it should be
                        pme_recip.forces(&pos_all, &q_all)
                    }
                }
            }
            None => {
                panic!("No PME recip available; not computing SPME recip.");
            }
        };

        self.potential_energy += e_recip as f64;

        // println!("F RECIP: {:?}", &f_recip[0..20]);

        // todo: QC this.
        // Scale to Amber force units
        for f in f_recip.iter_mut() {
            *f *= K_COUL;
        }

        let mut w_recip = 0.0;
        for (k, tag) in map.iter().enumerate() {
            match *tag {
                PMEIndex::Dyn(i) => {
                    self.atoms[i].accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]); // tin-foil virial
                }
                PMEIndex::WatO(i) => {
                    self.water[i].o.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::WatM(i) => {
                    self.water[i].m.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::WatH0(i) => {
                    self.water[i].h0.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::WatH1(i) => {
                    self.water[i].h1.accel += f_recip[k];
                    w_recip += 0.5 * pos_all[k].dot(f_recip[k]);
                }
                PMEIndex::Static(_) => { /* contributes to field, no accel update */ }
            }
        }
        self.barostat.virial_pair_kcal += w_recip as f64;

        // 1–4 Coulomb scaling correction
        for &(i, j) in &self.pairs_14_scaled {
            let diff = self
                .cell
                .min_image(self.atoms[i].posit - self.atoms[j].posit);
            let r = diff.magnitude();
            let dir = diff / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            let Some(pme_recip) = &mut self.pme_recip else {
                panic!("Missing PME recip; code error");
            };
            let df = ewald_comp_force(dir, r, qi, qj, pme_recip.alpha)
                * (SCALE_COUL_14 - 1.0) // todo: Cache this.
                * K_COUL;

            self.atoms[i].accel += df;
            self.atoms[j].accel -= df;
            self.barostat.virial_pair_kcal += (dir * r).dot(df) as f64; // r·F
        }
    }

    // todo: QC, and simplify as required.
    /// Gather all particles that contribute to PME (dyn, water sites, statics).
    /// Returns positions wrapped to the primary box, their charges, and a map telling
    /// us which original DOF each entry corresponds to.
    fn gather_pme_particles_wrapped(&self) -> (Vec<Vec3>, Vec<f32>, Vec<PMEIndex>) {
        let n_dyn = self.atoms.len();
        let n_wat = self.water.len();
        // let n_st = self.atoms_static.len();

        // Capacity hint: dyn + 4*water + statics
        // let mut pos = Vec::with_capacity(n_dyn + 4 * n_wat + n_st);
        let mut pos = Vec::with_capacity(n_dyn + 4 * n_wat);
        let mut q = Vec::with_capacity(pos.capacity());
        let mut map = Vec::with_capacity(pos.capacity());

        // Dynamic atoms
        for (i, a) in self.atoms.iter().enumerate() {
            pos.push(self.cell.wrap(a.posit)); // [0,L) per axis
            q.push(a.partial_charge); // already scaled to Amber units
            map.push(PMEIndex::Dyn(i));
        }

        // Water sites (OPC: O usually has 0 charge; include anyway—cost is negligible)
        for (i, w) in self.water.iter().enumerate() {
            pos.push(self.cell.wrap(w.o.posit));
            q.push(w.o.partial_charge);
            map.push(PMEIndex::WatO(i));
            pos.push(self.cell.wrap(w.m.posit));
            q.push(w.m.partial_charge);
            map.push(PMEIndex::WatM(i));
            pos.push(self.cell.wrap(w.h0.posit));
            q.push(w.h0.partial_charge);
            map.push(PMEIndex::WatH0(i));
            pos.push(self.cell.wrap(w.h1.posit));
            q.push(w.h1.partial_charge);
            map.push(PMEIndex::WatH1(i));
        }

        // // Static atoms (contribute to field but you won't update accel)
        // for (i, a) in self.atoms_static.iter().enumerate() {
        //     pos.push(self.cell.wrap(a.posit));
        //     q.push(a.partial_charge);
        //     map.push(PMEIndex::Static(i));
        // }

        // Optional sanity check (debug only): near-neutral total charge
        #[cfg(debug_assertions)]
        {
            let qsum: f64 = q.iter().map(|q_| (*q_) as f64).sum();
            if qsum.abs() > 1e-6 {
                eprintln!(
                    "[PME] Warning: net charge = {qsum:.6e} (PME assumes neutral or a uniform background)"
                );
            }
        }

        (pos, q, map)
    }

    /// Run this at init, and whenever you update the sim box.
    pub(crate) fn regen_pme(&mut self) {
        let [lx, ly, lz] = self.cell.extent.to_arr();
        self.pme_recip = Some(PmeRecip::new(
            (SPME_N, SPME_N, SPME_N),
            (lx, ly, lz),
            EWALD_ALPHA,
        ));
    }
}
