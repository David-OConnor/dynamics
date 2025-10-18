//! Contains integration code, including the primary time step.

use std::{
    fmt,
    fmt::{Display, Formatter},
    time::Instant,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use ewald::PmeRecip;
use lin_alg::f32::Vec3;

const COM_REMOVAL_RATIO_LINEAR: usize = 10;
const COM_REMOVAL_RATIO_ANGULAR: usize = 20;

use crate::{
    CENTER_SIMBOX_RATIO, COMPUTATION_TIME_RATIO, ComputationDevice, HydrogenConstraint, MdState,
    PMEIndex, SPME_RATIO,
    ambient::{BAR_PER_KCAL_MOL_PER_A3, GAS_CONST_R, measure_instantaneous_pressure},
    non_bonded::{EWALD_ALPHA, SCALE_COUL_14, SPME_MESH_SPACING},
    snapshot::{FILE_SAVE_INTERVAL, SaveType, append_dcd},
    water_opc::{ACCEL_CONV_WATER_H, ACCEL_CONV_WATER_O},
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
    // fn default() -> Self {
    //     Self::LangevinMiddle { gamma: 1. }
    // }
    // todo: Until we fix both Langevin integrators/thermostats.
    fn default() -> Self {
        Self::VerletVelocity
    }
}

impl Display for Integrator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Integrator::VerletVelocity => write!(f, "Verlet Vel"),
            Integrator::Langevin { gamma: _ } => write!(f, "Langevin"),
            Integrator::LangevinMiddle { gamma: _ } => write!(f, "Langevin Mid"),
        }
    }
}

impl MdState {
    /// For Langevin middle.
    fn drift_atoms(&mut self, dt: f32) {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.posit += a.vel * dt;
        }
    }

    /// For Langevin middle. This is a rigid shift, with one shift for all sites on a given
    /// molecule.
    fn water_drift(&mut self, dt: f32) {
        for w in &mut self.water {
            // Drift
            w.o.posit += w.o.vel * dt;
            w.h0.posit += w.h0.vel * dt;
            w.h1.posit += w.h1.vel * dt;
            w.m.posit += w.m.vel * dt;

            // Compute a single wrap shift using O (or COM) and apply to all sites
            let o_wrapped = self.cell.wrap(w.o.posit);
            let shift = o_wrapped - w.o.posit; // translation to bring O into [0,L)
            w.o.posit += shift;
            w.h0.posit += shift;
            w.h1.posit += shift;
            w.m.posit += shift;
        }
    }

    /// One step of length `dt` is in picoseconds (10^-12),
    /// with typical values of 0.001, or 0.002ps (1 or 2fs).
    /// This method orchestrates the dynamics at each time step. Uses a Verlet Velocity base,
    /// with different thermostat approaches depending on configuration.
    pub fn step(&mut self, dev: &ComputationDevice, dt: f32) {
        let mut start_entire_step = Instant::now();
        let mut start = Instant::now(); // Per-item

        let log_time = self.step_count.is_multiple_of(COMPUTATION_TIME_RATIO);

        let dt_half = 0.5 * dt;

        // todo: YOu can remove this once we crush the root cause.
        if self.nb_pairs.len() == 0 {
            eprintln!("UHoh. Pairs count is 0. THis likely means the system blew up. :(");
            return;
        }

        match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                if log_time {
                    start = Instant::now();
                }

                // B: half-kick (atoms)
                for a in &mut self.atoms {
                    if !a.static_ {
                        a.vel += a.accel * dt_half;
                    }
                }
                // B: half-kick (water) — inline version; do NOT touch M here
                for w in &mut self.water {
                    w.o.vel += w.o.accel * dt_half;
                    w.h0.vel += w.h0.accel * dt_half;
                    w.h1.vel += w.h1.accel * dt_half;
                }

                // A: half-drift
                self.drift_atoms(dt_half);
                self.water_drift(dt_half);

                // Position constraints (only hydrogens available here)
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.shake_hydrogens();
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                // O: thermostat
                if log_time {
                    start = Instant::now();
                }
                self.apply_langevin_thermostat(dt, gamma, self.cfg.temp_target);
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.rattle_hydrogens(dt);
                }
                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // A: second half-drift
                if log_time {
                    start = Instant::now();
                }
                self.drift_atoms(dt_half);
                self.water_drift(dt_half);
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.shake_hydrogens();
                }
                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                // Forces for final B
                self.reset_accel_e();
                if log_time {
                    start = Instant::now();
                }
                let _p_inst_bar = measure_instantaneous_pressure(
                    &self.atoms,
                    &self.water,
                    &self.cell,
                    self.barostat.virial_bonded
                        + self.barostat.virial_coulomb
                        + self.barostat.virial_lj
                        + self.barostat.virial_constraints
                        + self.barostat.virial_nonbonded_long_range,
                );
                self.apply_all_forces(dev);
                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // Optional: only if your box changed elsewhere; otherwise skip.
                self.regen_pme();

                // B: final half-kick (atoms with mass/units conversion)
                if log_time {
                    start = Instant::now();
                }
                for (i, a) in self.atoms.iter_mut().enumerate() {
                    if !a.static_ {
                        a.accel *= self.mass_accel_factor[i];
                        a.vel += a.accel * dt_half;
                    }
                }
                // B: final half-kick (water, with your existing helper that also handles any M back-prop)
                self.water_vv_second_half(dt_half);

                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.rattle_hydrogens(dt);
                }

                if self.step_count % COM_REMOVAL_RATIO_LINEAR == 0 {
                    self.zero_linear_momentum_atoms();
                }
                if self.step_count % COM_REMOVAL_RATIO_ANGULAR == 0 {
                    self.zero_angular_momentum_atoms();
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }
            }
            _ => {
                // O(dt/2)
                if let Integrator::Langevin { gamma } = self.cfg.integrator {
                    if log_time {
                        start = Instant::now();
                    }
                    self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);

                    if log_time {
                        let elapsed = start.elapsed().as_micros() as u64;
                        self.computation_time.ambient_sum += elapsed;
                    }

                    if log_time {
                        start = Instant::now();
                    }

                    if self.cfg.hydrogen_constraint == HydrogenConstraint::Constrained {
                        self.rattle_hydrogens(dt);
                    }

                    if log_time {
                        let elapsed = start.elapsed().as_micros() as u64;
                        self.computation_time.integration_sum += elapsed;
                    }
                }

                if log_time {
                    start = Instant::now();
                }

                // First half-kick (v += a dt/2) and drift (x += v dt)
                // Note: We do not apply the accel unit conversion, nor mass division here; they're already
                // included in this values from the previous step.
                // for (i, a) in self.atoms.iter_mut().enumerate() {
                for a in &mut self.atoms {
                    if a.static_ {
                        continue;
                    }

                    a.vel += a.accel * dt_half; // Half-kick
                    a.posit += a.vel * dt; // Drift
                }

                // todo: Consider applying the thermostat between the first half-kick and drift.
                // todo: e.g. half-kick, then shake H and settle velocity water (?), then thermostat, then drift. (?) ,
                // todo then settle positions?

                self.water_vv_first_half_and_drift(dt, dt_half);

                // The order we perform these steps is important.
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.shake_hydrogens();
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                self.reset_accel_e();
                self.apply_all_forces(dev);

                if log_time {
                    start = Instant::now();
                }

                let p_inst_bar = measure_instantaneous_pressure(
                    &self.atoms,
                    &self.water,
                    &self.cell,
                    self.barostat.virial_coulomb
                        + self.barostat.virial_lj
                        + self.barostat.virial_bonded
                        + self.barostat.virial_constraints
                        + self.barostat.virial_nonbonded_long_range,
                );

                if self.step_count.is_multiple_of(100) {
                    self.print_ambient_data(p_inst_bar);
                }

                // todo: Troubleshooting. causes systme to blow up. Note that the pressure reading
                // todo is showing *much* to high.
                // self.barostat.apply_isotropic(
                //     dt as f64,
                //     p_inst_bar,
                //     &mut self.cell,
                //     &mut self.atoms,
                //     &mut self.water,
                // );

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                self.regen_pme();

                // Forces (bonded and nonbonded, to non-water and water atoms) have been applied; perform other
                // steps required for integration; second half-kick, RATTLE for hydrogens; SETTLE for water. -----

                // Second half-kick using the forces calculated this step, and update accelerations using the atom's mass;
                // Between the accel reset and this step, the accelerations have been missing those factors; this is an optimization to
                // do it once at the end.
                if log_time {
                    start = Instant::now();
                }
                for (i, a) in self.atoms.iter_mut().enumerate() {
                    if a.static_ {
                        continue;
                    }
                    // We divide by mass here, once accelerations have been computed in parts above; this
                    // is an optimization to prevent dividing each accel component by it.
                    // This is the step where we A: convert force to accel, and B: Convert units from the param
                    // units to the ones we use in dynamics.

                    a.accel *= self.mass_accel_factor[i];
                    a.vel += a.accel * dt_half;
                }

                // self.water_vv_second_half(&mut self.forces_on_water, dt_half);
                self.water_vv_second_half(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                // O(dt/2)
                if let Integrator::Langevin { gamma } = self.cfg.integrator {
                    self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.rattle_hydrogens(dt);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if self.step_count % COM_REMOVAL_RATIO_LINEAR == 0 {
                    self.zero_linear_momentum_atoms();
                }
                if self.step_count % COM_REMOVAL_RATIO_ANGULAR == 0 {
                    self.zero_angular_momentum_atoms();
                }
            }
        }

        if let Integrator::VerletVelocity = self.cfg.integrator {
            if log_time {
                start = Instant::now();
            }

            self.apply_thermostat_csvr(dt as f64, self.cfg.temp_target as f64);

            if log_time {
                let elapsed = start.elapsed().as_micros() as u64;
                self.computation_time.ambient_sum += elapsed;
            }
        }

        self.time += dt as f64;
        self.step_count += 1;

        start = Instant::now(); // No ratio for neighbor times.

        self.update_max_displacement_since_rebuild();
        self.build_neighbors_if_needed(dev);

        let elapsed = start.elapsed().as_micros() as u64;
        self.computation_time.neighbor_all_sum += elapsed;

        // We keeping the cell centered on the dynamics atoms. Note that we don't change the dimensions,
        // as these are under management by the barostat.
        if self.step_count.is_multiple_of(CENTER_SIMBOX_RATIO) {
            self.cell.recenter(&self.atoms);
            // todo: Will this interfere with carrying over state from the previous step?
            self.regen_pme();
        }

        let start = Instant::now(); // Not sure how else to handle. (Option would work)
        self.handle_snapshots();

        if log_time {
            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.snapshot_sum += elapsed;
        }

        if log_time {
            let elapsed = start_entire_step.elapsed().as_micros() as u64;
            self.computation_time.total += elapsed;
        }
    }

    fn handle_snapshots(&mut self) {
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

    pub(crate) fn handle_spme_recip(&mut self, dev: &ComputationDevice) {
        const K_COUL: f32 = 1.; // todo: ChatGPT really wants this, but I don't think I need it.

        let (pos_all, q_all, map) = self.gather_pme_particles_wrapped();

        let (mut f_recip, e_recip) = match &mut self.pme_recip {
            Some(pme_recip) => match dev {
                ComputationDevice::Cpu => pme_recip.forces(&pos_all, &q_all),
                #[cfg(feature = "cuda")]
                ComputationDevice::Gpu(modules) => {
                    // todo for now
                    // pme_recip.forces(&pos_all, &q_all)
                    pme_recip.forces_gpu(&modules.stream, &modules.ewald, &pos_all, &q_all)
                    // pme_recip.forces_gpu(&modules.stream, &modules.ewald, stream, &pos_all, &q_all)
                }
            },
            None => {
                panic!("No PME recip available; not computing SPME recip.");
            }
        };

        self.potential_energy += e_recip as f64;

        // todo: QC this.
        // Scale to Amber force units
        for f in f_recip.iter_mut() {
            *f *= K_COUL;
        }

        // println!("F prior accel a0: {}",  self.atoms[1].accel);

        let mut virial_lr_recip = 0.0;
        for (k, tag) in map.iter().enumerate() {
            match *tag {
                PMEIndex::NonWat(i) => {
                    self.atoms[i].accel += f_recip[k];
                    if i == 0 {
                        // f_0 = f_recip[k];
                    }
                }
                PMEIndex::WatO(i) => {
                    self.water[i].o.accel += f_recip[k];
                }
                PMEIndex::WatM(i) => {
                    let fM = f_recip[k];
                    self.water[i].m.accel += fM; // stash PME M force; back-prop happens later
                }
                PMEIndex::WatH0(i) => {
                    self.water[i].h0.accel += f_recip[k];
                }
                PMEIndex::WatH1(i) => {
                    self.water[i].h1.accel += f_recip[k];
                }
                PMEIndex::Static(_) => { /* contributes to field, no accel update */ }
            }

            // todo: Debug
            // if self.step_count.is_multiple_of(20) {
            //     println!("F RECIP a0: {}", f_0);
            //     println!("F RECIP a1: {}", f_1);
            // }

            // todo: QC that you don't want the 1/2 factor.
            // virial_lr_recip += 0.5 * pos_all[k].dot(f_recip[k]); // tin-foil virial
            virial_lr_recip += pos_all[k].dot(f_recip[k]); // tin-foil virial

            // println!("LR. i: {k}, f: {:?}", f_recip[k]);
        }

        self.barostat.virial_nonbonded_long_range += virial_lr_recip as f64;

        // 1–4 Coulomb scaling correction (vacuum correction)
        for &(i, j) in &self.pairs_14_scaled {
            let diff = self
                .cell
                .min_image(self.atoms[i].posit - self.atoms[j].posit);
            let r = diff.magnitude();
            if r == 0.0 {
                continue;
            } // guard
            let dir = diff / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            // Vacuum Coulomb force (K=1 if charges are Amber-scaled)
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let f_vac = dir * (qi * qj * inv_r2);

            // We run this once every `SPME_RATIO` steps, so multiply the force by it.
            let df = f_vac * (SCALE_COUL_14 - 1.0) * SPME_RATIO as f32;

            self.atoms[i].accel += df;
            self.atoms[j].accel -= df;

            self.barostat.virial_nonbonded_long_range += (dir * r).dot(df) as f64; // r·F
        }
    }

    // todo: QC, and simplify as required.
    /// Gather all particles that contribute to PME (non-water atoms, water sites).
    /// Returns positions wrapped to the primary box, their charges, and a map telling
    /// us which original DOF each entry corresponds to.
    fn gather_pme_particles_wrapped(&self) -> (Vec<Vec3>, Vec<f32>, Vec<PMEIndex>) {
        let n_std = self.atoms.len();
        let n_wat = self.water.len();

        // Capacity hint: std + 4*water + statics
        let mut pos = Vec::with_capacity(n_std + 4 * n_wat);
        let mut q = Vec::with_capacity(pos.capacity());
        let mut map = Vec::with_capacity(pos.capacity());

        // Non-water atoms.
        for (i, a) in self.atoms.iter().enumerate() {
            pos.push(self.cell.wrap(a.posit)); // [0,L) per axis
            q.push(a.partial_charge); // already scaled to Amber units
            map.push(PMEIndex::NonWat(i));
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

        (pos, q, map)
    }

    /// Re-initializes the SPME based on sim box dimensions. Run this at init, and whenever you
    /// update the sim box. Sets FFT planner dimensions.
    pub(crate) fn regen_pme(&mut self) {
        let [lx, ly, lz] = self.cell.extent.to_arr();

        // todo: This is awkward.
        fn next_planner_n(mut n: usize) -> usize {
            fn good(mut x: usize) -> bool {
                for p in [2, 3, 5, 7] {
                    while x % p == 0 {
                        x /= p;
                    }
                }
                x == 1
            }
            if n < 2 {
                n = 2;
            }
            while !good(n) {
                n += 1;
            }
            n
        }

        let nx0 = (lx / SPME_MESH_SPACING).round().max(2.0) as usize;
        let ny0 = (ly / SPME_MESH_SPACING).round().max(2.0) as usize;
        let nz0 = (lz / SPME_MESH_SPACING).round().max(2.0) as usize;

        let nx = next_planner_n(nx0);
        let ny = next_planner_n(ny0);
        let mut nz = next_planner_n(nz0);
        if nz % 2 != 0 {
            nz = next_planner_n(nz + 1);
        }

        self.pme_recip = Some(PmeRecip::new((nx, ny, nz), (lx, ly, lz), EWALD_ALPHA));
    }

    /// Print ambient parameters, as a sanity check.
    fn print_ambient_data(&self, pressure: f64) {
        println!("\nPressure: {pressure} bar");
        println!("------------------------");
        let temp = self.temperature_kelvin();
        println!("Temp: {temp} K");

        let mut water_v = 0.;
        for mol in &self.water {
            water_v += mol.o.vel.magnitude();
        }
        println!("Water O vel: {:?}", water_v / self.water.len() as f32);

        let K_kcal = self.kinetic_energy_kcal();
        let V_a3 = self.cell.volume() as f64;
        eprintln!(
            "K[kcal/mol]={:.3}  V[Å^3]={:.3} W_bonded: {:.3}  W_Coul: [kcal/mol]={:.3} W_LJ: [kcal/mol]={:.3} W_long range: {:.3} W_constraint: {:.5}",
            K_kcal,
            V_a3,
            self.barostat.virial_bonded,
            self.barostat.virial_coulomb,
            self.barostat.virial_lj,
            self.barostat.virial_nonbonded_long_range,
            self.barostat.virial_constraints
        );

        // reconstruct pressure path
        let p_kcal_per_a3 = (2.0 * K_kcal
            + self.barostat.virial_coulomb
            + self.barostat.virial_lj
            + self.barostat.virial_bonded
            + self.barostat.virial_constraints
            + self.barostat.virial_nonbonded_long_range)
            / (3.0 * V_a3);
        eprintln!(
            "P_from_terms[bar]={:.3}",
            p_kcal_per_a3 * BAR_PER_KCAL_MOL_PER_A3
        );

        // temperature path
        let ndof = 3 * self.atoms.iter().filter(|a| !a.static_).count() + 6 * self.water.len() - 3;
        let T_k = (2.0 * K_kcal) / (ndof as f64 * GAS_CONST_R as f64);
        eprintln!("T[K]={:.1}  ndof={}", T_k, ndof);
    }
}
