use std::time::Instant;

use lin_alg::f32::Vec3;

use crate::{ComputationDevice, HydrogenConstraint, MdState};

// Force/E at current geometry
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
        let f = a.accel; // pre-mass, pre-unit-conversion accumulator is your net force vector
        let m = f.magnitude();
        max_f_loc = max_f_loc.max(m);
        sum += m * m;
        n += 1;
    }

    // Sum water too, as long as we are including it in the simulation; can stall out otherwise.
    for w in &state.water {
        for p in [&w.o, &w.h0, &w.h1, &w.m] {
            let m = p.accel.magnitude();
            max_f_loc = max_f_loc.max(m);
            sum += m * m;
            n += 1;
        }
    }

    let rms = if n > 0 { (sum / n as f32).sqrt() } else { 0.0 };

    (max_f_loc, rms)
}

impl MdState {
    /// Relaxes the molecules. Use this at the start of the simulation to control kinetic energy that
    /// arrises from differences between atom positions, and bonded parameters. It can also be called
    /// externally.
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

        // Zero velocities; we’re minimizing, not integrating.
        for a in &mut self.atoms {
            a.vel = Vec3::new_zero();
            a.accel = Vec3::new_zero();
        }
        for w in &mut self.water {
            w.o.vel = Vec3::new_zero();
            w.h0.vel = Vec3::new_zero();
            w.h1.vel = Vec3::new_zero();
            w.m.vel = Vec3::new_zero();
            w.o.accel = Vec3::new_zero();
            w.h0.accel = Vec3::new_zero();
            w.h1.accel = Vec3::new_zero();
            w.m.accel = Vec3::new_zero();
        }

        compute_forces_and_energy(self, dev);

        // Helper to measure convergence
        let mut max_f = 0.0f32;
        let mut _rms_f = 0.0f32;

        (max_f, _rms_f) = force_stats(self);
        if max_f <= F_TOL {
            return;
        }

        let mut alpha = STEP_INIT;
        let mut e_prev = self.potential_energy;

        // Per-atom last step for backtracking
        let n_atoms = self.atoms.len();
        let mut last_step: Vec<Vec3> = vec![Vec3::new_zero(); n_atoms];
        let mut water_last_steps: Vec<[Vec3; 4]> = vec![[Vec3::new_zero(); 4]; self.water.len()];

        let mut iters = 0_u32;
        'outer: for iter in 0..max_iters {
            iters += 1;

            for (i, a) in self.atoms.iter_mut().enumerate() {
                last_step[i] = Vec3::new_zero();
                if a.static_ {
                    continue;
                }
                let f = a.accel;
                let fm = f.magnitude();
                if !fm.is_finite() || fm == 0.0 {
                    continue;
                }
                let step_mag = (alpha * fm).min(STEP_MAX);
                let s = f * (step_mag / fm);
                a.posit += s;
                last_step[i] = s;
                self.neighbors_nb.max_displacement_sq = self
                    .neighbors_nb
                    .max_displacement_sq
                    .max(s.magnitude_squared());
            }

            // for (wi, w) in self.water.iter_mut().enumerate() {
            //     let mut s_o = Vec3::new_zero();
            //     let mut s_h0 = Vec3::new_zero();
            //     let mut s_h1 = Vec3::new_zero();
            //     let mut s_m = Vec3::new_zero();
            //
            //     for (site, s_ref) in [
            //         (&mut w.o, &mut s_o),
            //         (&mut w.h0, &mut s_h0),
            //         (&mut w.h1, &mut s_h1),
            //         (&mut w.m, &mut s_m),
            //     ] {
            //         if site.static_ { continue; }
            //         let f = site.accel;
            //         let fm = f.magnitude();
            //         if !fm.is_finite() || fm == 0.0 { continue; }
            //         let step_mag = (alpha * fm).min(STEP_MAX);
            //         let s = f * (step_mag / fm);
            //         site.posit += s;
            //         *s_ref = s;
            //         self.neighbors_nb.max_displacement_sq =
            //             self.neighbors_nb.max_displacement_sq.max(s.magnitude_squared());
            //     }
            //     water_last_steps[wi] = [s_o, s_h0, s_h1, s_m];
            // }

            if let HydrogenConstraint::Constrained = self.cfg.hydrogen_constraint {
                self.shake_hydrogens();
            }

            // for w in &mut self.water {
            //     w.o.posit = self.cell.wrap(w.o.posit);
            //     w.h0.posit = self.cell.wrap(w.h0.posit);
            //     w.h1.posit = self.cell.wrap(w.h1.posit);
            //     w.m.posit = self.cell.wrap(w.m.posit);
            // }

            self.build_neighbors_if_needed(dev);

            compute_forces_and_energy(self, dev);
            let e_new = self.potential_energy;

            if e_new <= e_prev {
                e_prev = e_new;
                alpha = (alpha * GROW).min(ALPHA_MAX);

                (max_f, _rms_f) = force_stats(self);
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

                // for (wi, w) in self.water.iter_mut().enumerate() {
                //     let [s_o, s_h0, s_h1, s_m] = water_last_steps[wi];
                //     if s_o.magnitude_squared() > 0.0 { w.o.posit -= s_o; }
                //     if s_h0.magnitude_squared() > 0.0 { w.h0.posit -= s_h0; }
                //     if s_h1.magnitude_squared() > 0.0 { w.h1.posit -= s_h1; }
                //     if s_m.magnitude_squared() > 0.0 { w.m.posit -= s_m; }
                // }

                compute_forces_and_energy(self, dev);

                alpha *= SHRINK;
                if alpha < ALPHA_MIN {
                    break 'outer;
                }
                continue;
            }
        }

        // Final cleanups: zero velocities, recenter, and refresh PME grid if you do this routinely elsewhere.
        for a in &mut self.atoms {
            a.vel = Vec3::new_zero();
        }
        // for w in &mut self.water {
        //     w.o.vel = Vec3::new_zero();
        //     w.h0.vel = Vec3::new_zero();
        //     w.h1.vel = Vec3::new_zero();
        //     w.m.vel = Vec3::new_zero();
        // }

        // Keep consistent with the normal cadence.
        self.cell.recenter(&self.atoms);
        self.regen_pme();

        // Undo our config change.
        self.cfg.overrides.long_range_recip_disabled = prev_long_range;

        let elapsed = start.elapsed().as_millis();
        println!("Complete in {elapsed} ms. Used {iters} of {max_iters} iters");
    }
}
