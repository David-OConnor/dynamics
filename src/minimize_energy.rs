use std::time::Instant;

use lin_alg::f32::Vec3;

use crate::{ComputationDevice, MdState, water_settle::settle_no_dt};

/// Force/E at current geometry
fn compute_forces_and_energy(state: &mut MdState, dev: &ComputationDevice) {
    state.reset_accel_e();
    state.potential_energy = 0.0;

    state.apply_all_forces(dev);
}

fn force_stats(state: &MdState) -> (f32, f32) {
    let mut max_f_loc = 0.0f32;
    let mut sum = 0.0f32;

    let mut n = 0;
    for a in &state.atoms {
        if a.static_ {
            continue;
        }

        let m = a.force.magnitude();
        max_f_loc = max_f_loc.max(m);

        sum += m * m;
        n += 1;
    }

    // Sum water too, as long as we are including it in the simulation; can stall out otherwise.
    for w in &state.water {
        for p in [&w.o, &w.h0, &w.h1, &w.m] {
            let f_mag = p.force.magnitude();

            max_f_loc = max_f_loc.max(f_mag);
            sum += f_mag * f_mag;
            n += 1;
        }
    }

    let rms = if n > 0 { (sum / n as f32).sqrt() } else { 0.0 };

    (max_f_loc, rms)
}

impl MdState {
    /// Relaxes the molecules. Use this at the start of the simulation to control kinetic energy that
    /// arrises from differences between atom positions, and bonded parameters. It can also be called
    /// externally. It also stabilizes the water molecules, so that their hydrogen bond
    /// structure is correct at initialization.
    ///
    /// Uses flexible bonds to hydrogen. (Not Shake/Rattle constraints)
    ///
    /// We don't apply this to water molecules, as we have a pre-sim set up for them that runs
    /// prior to this.
    pub fn minimize_energy(&mut self, dev: &ComputationDevice, max_iters: usize) {
        println!("Minimizing energy...");
        let start = Instant::now();

        // We disable long range forces here, as they're slow and not required.
        let prev_long_range = self.cfg.overrides.long_range_recip_disabled;
        self.cfg.overrides.long_range_recip_disabled = true;

        const F_TOL: f32 = 1.0e-3; // stop when max |F| is below this (force units used in your accel pre-division)
        const STEP_INIT: f32 = 1.0e-4; // initial step along +F (Å per force-unit)
        const STEP_MAX: f32 = 0.2; // cap per-atom displacement per iteration (Å)
        const GROW: f32 = 1.2; // expand step if energy decreased
        const SHRINK: f32 = 0.5; // backtrack factor if energy increased
        const ALPHA_MIN: f32 = 1.0e-8;
        const ALPHA_MAX: f32 = 1.0e-2;

        // Zero velocities; we’re minimizing, not integrating. Note that accel and force are
        // zeroed downstream.
        // Store initial velocities, and re-apply at the end.
        let mut initial_velocities = Vec::with_capacity(self.atoms.len());
        for a in &mut self.atoms {
            initial_velocities.push(a.vel);
            a.vel = Vec3::new_zero();
        }

        compute_forces_and_energy(self, dev);

        // Helper to measure convergence
        let (max_f, _rms_f) = force_stats(self);
        if max_f <= F_TOL {
            // Undo our config change.
            self.cfg.overrides.long_range_recip_disabled = prev_long_range;

            let elapsed = start.elapsed().as_millis();
            println!("Complete in {elapsed} ms. (early return)");

            return;
        }

        let mut alpha = STEP_INIT;
        let mut e_prev = self.potential_energy;

        // Per-atom last step for backtracking
        let n_atoms = self.atoms.len();
        let mut last_step: Vec<_> = vec![Vec3::new_zero(); n_atoms];

        let mut iters = 0;
        'outer: for _iter in 0..max_iters {
            iters += 1;

            for (i, a) in self.atoms.iter_mut().enumerate() {
                last_step[i] = Vec3::new_zero();
                if a.static_ {
                    continue;
                }

                let f_mag = a.force.magnitude();
                if !f_mag.is_finite() || f_mag == 0.0 {
                    continue;
                }
                let step_mag = (alpha * f_mag).min(STEP_MAX);
                let s = a.force * (step_mag / f_mag);
                a.posit += s;

                last_step[i] = s;
                self.neighbors_nb.max_displacement_sq = self
                    .neighbors_nb
                    .max_displacement_sq
                    .max(s.magnitude_squared());
            }

            self.build_neighbors_if_needed(dev);

            compute_forces_and_energy(self, dev);
            let e_new = self.potential_energy;

            if e_new <= e_prev {
                e_prev = e_new;
                alpha = (alpha * GROW).min(ALPHA_MAX);

                let (max_f, _rms_f) = force_stats(self);
                if max_f <= F_TOL {
                    break 'outer;
                }
            } else {
                // Revert positions of atoms and water.
                for (i, a) in self.atoms.iter_mut().enumerate() {
                    let s = last_step[i];
                    if s.magnitude_squared() > 0.0 {
                        a.posit -= s;
                    }
                }

                compute_forces_and_energy(self, dev);

                alpha *= SHRINK;
                if alpha < ALPHA_MIN {
                    break 'outer;
                }
                continue;
            }
        }

        // Cleanup: zero velocities, recenter, and refresh PME grid if you do this routinely elsewhere.
        for a in &mut self.atoms {
            a.vel = Vec3::new_zero();
        }

        self.reset_accel_e();

        // Keep consistent with the normal cadence.
        self.cell.recenter(&self.atoms);
        self.regen_pme(dev);

        // Undo our config change.
        self.cfg.overrides.long_range_recip_disabled = prev_long_range;

        // Re-apply our initial velocities.
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.vel = initial_velocities[i];
        }

        let elapsed = start.elapsed().as_millis();
        println!("Complete in {elapsed} ms. Used {iters} of {max_iters} iters");
    }
}
