//! Contains integration code, including the primary time step.

use std::{
    fmt,
    fmt::{Display, Formatter},
    time::Instant,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use lin_alg::f32::Vec3;

use crate::{
    CENTER_SIMBOX_RATIO, COMPUTATION_TIME_RATIO, ComputationDevice, HydrogenConstraint, MdState,
    barostat::{
        LANGEVIN_GAMMA_DEFAULT, LANGEVIN_GAMMA_WATER_INIT, TAU_TEMP_WATER_INIT, measure_pressure,
    },
    solvent::{
        ACCEL_CONV_WATER_H, ACCEL_CONV_WATER_O, MASS_WATER_MOL,
        settle::{RESET_ANGLE_RATIO, integrate_rigid_water, reset_angle},
    },
};

const COM_REMOVAL_RATIO_LINEAR: usize = 10;
const COM_REMOVAL_RATIO_ANGULAR: usize = 20;

// The maximum allowed acceleration, in Å/ps^2.
// For example, pathological starting conditions including hydrogen placement.
const MAX_ACCEL: f32 = 1e5;
const MAX_ACCEL_SQ: f32 = MAX_ACCEL * MAX_ACCEL;

// todo: Make this Thermostat instead of Integrator? And have a WIP Integrator with just VV.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub enum Integrator {
    /// The inner value is the temperature-coupling time constant if the thermostat is enabled.
    /// This value is in ps.
    /// Lower means more sensitive. 1ps is a good default.
    VerletVelocity { thermostat: Option<f64> },
    /// Velocity-verlet with a Langevin thermometer. Good temperature control
    /// and ergodicity, but the friction parameter damps real dynamics as it grows.
    /// γ is friction in 1/ps. Typical values are 1–5. for proteins in implicit/weak solvent.
    /// With explicit solvents, we can often go lower to 0.1 – 1.
    /// A higher value has strong damping and is rougher. A lower value is gentler.
    LangevinMiddle { gamma: f32 },
}

impl Default for Integrator {
    fn default() -> Self {
        Self::LangevinMiddle {
            gamma: LANGEVIN_GAMMA_DEFAULT,
        }
    }
}

impl Display for Integrator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Integrator::VerletVelocity { thermostat: _ } => write!(f, "Verlet Vel"),
            Integrator::LangevinMiddle { gamma: _ } => write!(f, "Langevin Mid"),
        }
    }
}

impl MdState {
    /// Perform one integration step. This is the entry point for running the simulation.
    /// One step of length `dt` is in picoseconds (10^-12),
    /// with typical values of 0.001, or 0.002ps (1 or 2fs).
    /// This method orchestrates the dynamics at each time step. Uses a Verlet Velocity base,
    /// with different thermostat approaches depending on configuration.
    ///
    /// `External force` is indexed by atom.
    pub fn step(&mut self, dev: &ComputationDevice, dt: f32, external_force: Option<Vec<Vec3>>) {
        if let Some(f_ext) = &external_force
            && f_ext.len() != self.atoms.len()
        {
            eprintln!("Error: External force vector length does not match number of atoms.");
            return;
        }

        if self.atoms.is_empty() && self.water.is_empty() {
            return;
        }

        let start_entire_step = Instant::now();
        let mut start = Instant::now(); // Re-used for different items

        let log_time = self.step_count.is_multiple_of(COMPUTATION_TIME_RATIO);

        let dt_half = 0.5 * dt;

        // todo: YOu can remove this once we crush the root cause.
        if self.nb_pairs.len() == 0 {
            eprintln!("UHoh. Pairs count is 0. THis likely means the system blew up. :(");
            return;
        }

        let pressure = match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                if log_time {
                    start = Instant::now();
                }

                self.barostat.virial.constraints = 0.; // todo: Re-evaluate how you handle this.

                self.kick_and_drift(dt_half, dt_half);

                // We carry this over the reset.
                let virial_constr = self.barostat.virial.constraints;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                if !self.cfg.overrides.thermo_disabled && !self.solvent_only_sim_at_init {
                    self.apply_langevin_thermostat(dt, gamma, self.cfg.temp_target);
                    // Update KE after vel updates from the thermostat, prior to barostat.
                    self.kinetic_energy = self.kinetic_energy();
                } else if self.solvent_only_sim_at_init {
                    self.apply_langevin_thermostat(
                        dt,
                        LANGEVIN_GAMMA_WATER_INIT,
                        self.cfg.temp_target,
                    );
                    self.kinetic_energy = self.kinetic_energy();
                }

                // Rattle after the thermostat run, as it updates velocities in a non-uniform manner.
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.rattle_hydrogens(dt);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                self.drift(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                // ------- Below: Compute new forces and accelerations.
                if log_time {
                    start = Instant::now();
                }

                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, &external_force);

                // Applying from our pre-reset calcs.
                self.barostat.virial.constraints = virial_constr;

                // Molecular virial theorem: use COM-only KE for solvent (rigid molecules
                // contribute no rotational KE to pressure; no SETTLE constraint virial needed).
                let pressure = measure_pressure(
                    self.kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );

                if !self.cfg.overrides.baro_disabled {
                    self.barostat.apply_isotropic(
                        dt as f64,
                        pressure,
                        self.cfg.temp_target as f64,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    );
                    // The box dimensions changed; update PME so the next force computation
                    // uses the correct reciprocal lattice vectors.
                    self.regen_pme(dev);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // Final half-kick (atoms with mass/units conversion)
                if log_time {
                    start = Instant::now();
                }

                self.kick_and_calc_accel(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                pressure
            }
            Integrator::VerletVelocity { thermostat } => {
                if log_time {
                    start = Instant::now();
                }

                self.barostat.virial.constraints = 0.;

                self.kick_and_drift(dt_half, dt);

                // We carry this over the reset.
                let virial_constr = self.barostat.virial.constraints;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, &external_force);

                if log_time {
                    start = Instant::now();
                }

                // Applying from our pre-reset calcs.
                self.barostat.virial.constraints = virial_constr;

                // Molecular virial theorem: COM-only KE for solvent (no SETTLE constraint virial).
                let pressure = measure_pressure(
                    self.kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // Forces (bonded and nonbonded, to non-solvent and solvent atoms) have been applied; perform other
                // steps required for integration; second half-kick, RATTLE for hydrogens; SETTLE for solvent. -----

                // Second half-kick using the forces calculated this step, and update accelerations using the atom's mass;
                // Between the accel reset and this step, the accelerations have been missing those factors; this is an optimization to
                // do it once at the end.
                if log_time {
                    start = Instant::now();
                }

                self.kick_and_calc_accel(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                // Note: We don't need to RATTLE hydrogens after applying the CSVR thermostat, because
                // it updates all velocites uniformly.
                if let Some(tau_temp) = thermostat
                    && !self.cfg.overrides.thermo_disabled
                    && !self.solvent_only_sim_at_init
                {
                    self.apply_thermostat_csvr(dt as f64, tau_temp, self.cfg.temp_target as f64);
                    self.kinetic_energy = self.kinetic_energy();
                } else if self.solvent_only_sim_at_init {
                    self.apply_thermostat_csvr(
                        dt as f64,
                        TAU_TEMP_WATER_INIT,
                        self.cfg.temp_target as f64,
                    );
                    self.kinetic_energy = self.kinetic_energy();
                }

                // Barostat runs last in VV: velocities are fully updated and the thermostat has
                // already set the correct KE, so the box/coordinate scaling happens cleanly.
                // Scaled positions feed into the next step's force computation.
                if !self.cfg.overrides.baro_disabled {
                    self.barostat.apply_isotropic(
                        dt as f64,
                        pressure,
                        self.cfg.temp_target as f64,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    );
                    // The box dimensions changed; update PME so the next force computation
                    // uses the correct reciprocal lattice vectors.
                    self.regen_pme(dev);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }
                pressure
            }
        };

        if self.cfg.zero_com_drift {
            // Linear calls angular, which is why we don't run both at the same time.
            if self.step_count.is_multiple_of(COM_REMOVAL_RATIO_ANGULAR) {
                self.zero_angular_momentum();
            } else if self.step_count.is_multiple_of(COM_REMOVAL_RATIO_LINEAR) {
                self.zero_linear_momentum();
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
            self.regen_pme(dev);
        }

        if self.step_count.is_multiple_of(RESET_ANGLE_RATIO) && self.step_count != 0 {
            for mol in &mut self.water {
                reset_angle(mol, &self.cell);
            }
        }

        if !self.solvent_only_sim_at_init {
            // if self.step_count.is_multiple_of(200) {
            //     self.print_ambient_data(pressure);
            // }

            let start = Instant::now(); // Not sure how else to handle. (Option would work)
            self.take_snapshot_if_required(pressure);

            if log_time {
                let elapsed = start.elapsed().as_micros() as u64;
                self.computation_time.snapshot_sum += elapsed;
            }

            if log_time {
                let elapsed = start_entire_step.elapsed().as_micros() as u64;
                self.computation_time.total += elapsed;
            }
        }

        if self.cfg.overrides.snapshots_during_equilibration && self.solvent_only_sim_at_init {
            self.take_snapshot_if_required(pressure);
        }
    }

    /// Half kick and drift for non-solvent and solvent. We call this one or more time
    /// in the various integration approaches. Includes the SETTLE application for solvent,
    /// and SHAKE + RATTLE for hydrogens, if applicable. Updates kinetic energy.
    fn kick_and_drift(&mut self, dt_kick: f32, dt_drift: f32) {
        // Half-kick
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.vel += a.accel * dt_kick; // kick
            a.posit += a.vel * dt_drift; // drift
        }

        for w in &mut self.water {
            // Kick
            w.o.vel += w.o.accel * dt_kick;
            w.h0.vel += w.h0.accel * dt_kick;
            w.h1.vel += w.h1.accel * dt_kick;

            let _ = integrate_rigid_water(w, dt_drift, &self.cell);
        }

        if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
            self.shake_hydrogens(dt_kick);
            self.rattle_hydrogens(dt_kick);
        }

        self.kinetic_energy = self.kinetic_energy();
    }

    /// Half kick for non-solvent and solvent. We call this one or more time
    /// in the various integration approaches. Updates kinetic energy.
    fn kick_and_calc_accel(&mut self, dt: f32) {
        for (i, a) in self.atoms.iter_mut().enumerate() {
            if a.static_ {
                continue;
            }

            a.accel = a.force * self.mass_accel_factor[i];
            if a.accel.magnitude_squared() > MAX_ACCEL_SQ {
                println!(
                    "Error: Acceleration out of bounds for atom {} on step {}. Clamping {:.3} to {:.3}",
                    i,
                    self.step_count,
                    a.accel.magnitude(),
                    MAX_ACCEL
                );
                a.accel = a.accel.to_normalized() * MAX_ACCEL;
            }

            a.vel += a.accel * dt;
        }

        for w in &mut self.water {
            // Take the force on M/EP, and instead apply it to the other atoms. This leaves it at 0.
            // w.project_ep_force_to_real_sites(&self.cell);
            w.project_ep_force_optimized();

            w.o.accel = w.o.force * ACCEL_CONV_WATER_O;
            w.h0.accel = w.h0.force * ACCEL_CONV_WATER_H;
            w.h1.accel = w.h1.force * ACCEL_CONV_WATER_H;

            w.o.vel += w.o.accel * dt;
            w.h0.vel += w.h0.accel * dt;
            w.h1.vel += w.h1.accel * dt;
        }

        if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
            self.rattle_hydrogens(dt);
        }

        self.kinetic_energy = self.kinetic_energy();
    }

    /// Drifts all non-static atoms in the system.  Includes the SETTLE application for solvent,
    /// and SHAKE + RATTLE for hydrogens, if applicable.
    fn drift(&mut self, dt: f32) {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }
            a.posit += a.vel * dt;
        }

        for w in &mut self.water {
            let _ = integrate_rigid_water(w, dt, &self.cell);
        }

        if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
            self.shake_hydrogens(dt);
        }
    }
}
