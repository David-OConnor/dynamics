//! Contains integration code, including the primary time step.

use std::{
    fmt,
    fmt::{Display, Formatter},
    time::Instant,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};

const COM_REMOVAL_RATIO_LINEAR: usize = 10;
const COM_REMOVAL_RATIO_ANGULAR: usize = 20;

use crate::{
    CENTER_SIMBOX_RATIO, COMPUTATION_TIME_RATIO, ComputationDevice, HydrogenConstraint, MdState,
    ambient::{BAR_PER_KCAL_MOL_PER_A3, GAS_CONST_R, measure_instantaneous_pressure},
    water_opc::{ACCEL_CONV_WATER_H, ACCEL_CONV_WATER_O},
    water_settle,
    water_settle::{RESET_ANGLE_RATIO, settle_drift},
};

// todo: Make this Thermostat instead of Integrator? And have a WIP Integrator with just VV.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub enum Integrator {
    VerletVelocity,
    /// Deprecated
    Langevin {
        gamma: f32,
    },
    /// Velocity-verlet with a Langevin thermometer. Good temperature control
    /// and ergodicity, but the friction parameter damps real dynamics as it grows.
    /// γ is friction in 1/ps. Typical values are 1–5. for proteins in implicit/weak solvent.
    /// With explicit solvents, we can often go lower to 0.1 – 1.
    /// A higher value has strong damping and is rougher. A lower value is gentler.
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
            Integrator::Langevin { gamma: _ } => write!(f, "Langevin"),
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
    pub fn step(&mut self, dev: &ComputationDevice, dt: f32) {
        let mut start_entire_step = Instant::now();
        let mut start = Instant::now(); // Re-used for different items

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

                self.kick_and_drift(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                if !self.cfg.overrides.thermo_disabled {
                    self.apply_langevin_thermostat(dt, gamma, self.cfg.temp_target);
                }
                // Rattle after the thermostat run.
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

                // Final half-kick (atoms with mass/units conversion)
                if log_time {
                    start = Instant::now();
                }

                self.kick_and_calc_accel(dt_half);

                if self.step_count.is_multiple_of(COM_REMOVAL_RATIO_LINEAR) {
                    self.zero_linear_momentum_atoms();
                }
                if self.step_count.is_multiple_of(COM_REMOVAL_RATIO_ANGULAR) {
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
                    if !self.cfg.overrides.thermo_disabled {
                        self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);
                    }

                    if log_time {
                        let elapsed = start.elapsed().as_micros() as u64;
                        self.computation_time.ambient_sum += elapsed;
                    }

                    if log_time {
                        start = Instant::now();
                    }

                    // Rattle after application of thermostat.
                    if self.cfg.hydrogen_constraint == HydrogenConstraint::Constrained {
                        self.rattle_hydrogens(dt_half);
                    }

                    if log_time {
                        let elapsed = start.elapsed().as_micros() as u64;
                        self.computation_time.integration_sum += elapsed;
                    }
                }

                if log_time {
                    start = Instant::now();
                }

                self.kick_and_drift(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                self.reset_accel_e();
                self.apply_all_forces(dev);

                if log_time {
                    start = Instant::now();
                }

                if !self.cfg.overrides.baro_disabled {
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

                    if self.step_count.is_multiple_of(1_000) {
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
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // Forces (bonded and nonbonded, to non-water and water atoms) have been applied; perform other
                // steps required for integration; second half-kick, RATTLE for hydrogens; SETTLE for water. -----

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

                // O(dt/2)
                if let Integrator::Langevin { gamma } = self.cfg.integrator
                    && !self.cfg.overrides.thermo_disabled
                {
                    self.apply_langevin_thermostat(dt_half, gamma, self.cfg.temp_target);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                // Rattle after applying the thermostat.
                if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                    self.rattle_hydrogens(dt_half);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if self.step_count.is_multiple_of(COM_REMOVAL_RATIO_LINEAR) {
                    self.zero_linear_momentum_atoms();
                }
                if self.step_count.is_multiple_of(COM_REMOVAL_RATIO_ANGULAR) {
                    self.zero_angular_momentum_atoms();
                }
            }
        }

        if let Integrator::VerletVelocity = self.cfg.integrator {
            if log_time {
                start = Instant::now();
            }

            if !self.cfg.overrides.thermo_disabled {
                self.apply_thermostat_csvr(dt as f64, self.cfg.temp_target as f64);
            }
            if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                self.rattle_hydrogens(dt);
            }

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
            self.regen_pme(dev);
        }

        if self.step_count.is_multiple_of(RESET_ANGLE_RATIO) && self.step_count != 0 {
            for mol in &mut self.water {
                water_settle::reset_angle(mol, &self.cell);
            }
        }

        let start = Instant::now(); // Not sure how else to handle. (Option would work)
        self.take_snapshot_if_required();

        if log_time {
            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.snapshot_sum += elapsed;
        }

        if log_time {
            let elapsed = start_entire_step.elapsed().as_micros() as u64;
            self.computation_time.total += elapsed;
        }
    }

    /// Half kick and drift for non-water and water. We call this one or more time
    /// in the various integration approaches. Includes the SETTLE application for water,
    /// and SHAKE + RATTLE for hydrogens, if applicable.
    fn kick_and_drift(&mut self, dt: f32) {
        // Half-kick
        // for (i, a) in self.atoms.iter_mut().enumerate() {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.vel += a.accel * dt; // kick
            a.posit += a.vel * dt; // drift
        }

        for w in &mut self.water {
            // Kick
            w.o.vel += w.o.accel * dt;
            w.h0.vel += w.h0.accel * dt;
            w.h1.vel += w.h1.accel * dt;

            settle_drift(w, dt, &self.cell, &mut self.barostat.virial_coulomb);
        }

        if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
            self.shake_hydrogens();
            self.rattle_hydrogens(dt);
        }
    }

    /// Half kick for non-water and water. We call this one or more time
    /// in the various integration approaches.
    fn kick_and_calc_accel(&mut self, dt: f32) {
        for (i, a) in self.atoms.iter_mut().enumerate() {
            if a.static_ {
                continue;
            }

            a.accel = a.force * self.mass_accel_factor[i];
            a.vel += a.accel * dt;
        }

        for w in &mut self.water {
            // Take the force on M/EP, and instead apply it to the other atoms. This leaves it at 0.
            w.project_ep_force_to_real_sites(&self.cell);

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
    }

    /// Drifts all non-static atoms in the system.  Includes the SETTLE application for water,
    /// and SHAKE + RATTLE for hydrogens, if applicable.
    fn drift(&mut self, dt: f32) {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }
            a.posit += a.vel * dt;
        }

        for w in &mut self.water {
            settle_drift(w, dt, &self.cell, &mut self.barostat.virial_coulomb);
        }

        if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
            self.shake_hydrogens();
        }
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
