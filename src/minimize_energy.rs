use std::time::Instant;

use lin_alg::f32::Vec3;

use crate::{ComputationDevice, MdState};

const F_TOL: f32 = 1.0e-3; // stop when max |F| is below this (force units used in your accel pre-division)
const STEP_INIT: f32 = 1.0e-4; // initial step along +F (Å per force-unit)
const STEP_MAX: f32 = 0.2; // cap per-atom displacement per iteration (Å)
const GROW: f32 = 1.2; // expand step if energy decreased
const SHRINK: f32 = 0.5; // backtrack factor if energy increased
const ALPHA_MIN: f32 = 1.0e-8;
const ALPHA_MAX: f32 = 1.0e-2;

/// Force/E at current geometry
fn compute_forces_and_energy(
    state: &mut MdState,
    dev: &ComputationDevice,
    external_force: &Option<Vec<Vec3>>,
) {
    state.reset_f_acc_pe_virial();
    state.potential_energy = 0.0;

    state.apply_all_forces(dev, external_force);
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

    let rms = if n > 0 { (sum / n as f32).sqrt() } else { 0.0 };

    (max_f_loc, rms)
}

impl MdState {
    /// Relaxes the molecules using a steepest-descent energy minimizer. Use this at the start of the simulation
    /// to control kinetic energy that
    /// arrises from differences between atom positions, and bonded parameters. It can also be called
    /// externally. It also stabilizes the solvent molecules, so that their hydrogen bond
    /// structure is correct at initialization.
    ///
    /// Uses flexible bonds to hydrogen. (Not Shake/Rattle constraints)
    ///
    /// We don't apply this to solvent molecules, as we have a pre-sim set up for them that runs
    /// prior to this.
    pub fn minimize_energy(
        &mut self,
        dev: &ComputationDevice,
        max_iters: usize,
        external_force: Option<Vec<Vec3>>,
    ) {
        println!("Minimizing energy...");
        let start = Instant::now();

        let (mut last_step, mut alpha, mut e_prev, initial_velocities, prev_recip) =
            self.minimize_energy_setup(dev, &external_force);

        let mut iters = 0;
        for _ in 0..max_iters {
            iters += 1;
            if self.step_energy_min(
                dev,
                &mut last_step,
                &mut alpha,
                &mut e_prev,
                &external_force,
            ) {
                break; // Converged.
            }
        }

        self.minimize_energy_cleanup(dev, prev_recip, &initial_velocities);

        let elapsed = start.elapsed().as_millis();
        println!("Complete in {elapsed} ms. Used {iters} of {max_iters} iters");
    }

    /// Separate, so can be called `separately by an application, e.g. if it needs to
    /// apply a new external force each step.
    pub fn minimize_energy_setup(
        &mut self,
        dev: &ComputationDevice,
        external_force: &Option<Vec<Vec3>>,
    ) -> (Vec<Vec3>, f32, f64, Vec<Vec3>, bool) {
        // We disable long range forces here, as they're slow and not required.
        let prev_recip = self.cfg.overrides.long_range_recip_disabled;
        self.cfg.overrides.long_range_recip_disabled = true;

        // Zero velocities; we’re minimizing, not integrating. Note that accel and force are
        // zeroed downstream.
        // Store initial velocities, and re-apply at the end.
        let mut initial_velocities = Vec::with_capacity(self.atoms.len());
        for a in &mut self.atoms {
            initial_velocities.push(a.vel);
            a.vel = Vec3::new_zero();
        }

        compute_forces_and_energy(self, dev, external_force);

        let alpha = STEP_INIT;
        let e_prev = self.potential_energy;

        // Per-atom last step for backtracking
        let n_atoms = self.atoms.len();
        let last_step: Vec<_> = vec![Vec3::new_zero(); n_atoms];

        // // Helper to measure convergence
        // let (max_f, _rms_f) = force_stats(self);
        // if max_f <= F_TOL {
        //     // Undo our config change.
        //     self.cfg.overrides.long_range_recip_disabled = prev_long_range;
        //     return;
        // }

        (last_step, alpha, e_prev, initial_velocities, prev_recip)
    }

    /// See the note on `minimize_energy_setup`; this is broken out so it can be called separately
    /// by an application.
    pub fn minimize_energy_cleanup(
        &mut self,
        dev: &ComputationDevice,
        prev_long_range: bool,
        initial_velocities: &[Vec3],
    ) {
        // Cleanup: zero velocities, recenter, and refresh PME grid if you do this routinely elsewhere.
        for a in &mut self.atoms {
            a.vel = Vec3::new_zero();
        }

        self.reset_f_acc_pe_virial();

        // Keep consistent with the normal cadence.
        self.cell.recenter(&self.atoms);

        // Undo our config change.
        self.cfg.overrides.long_range_recip_disabled = prev_long_range;
        self.regen_pme(dev);

        // Re-apply our initial velocities.
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.vel = initial_velocities[i];
        }
    }

    /// One iteration of energy minimization. Returns `true` if the energy is converged, indicating
    /// to abort further steps.
    pub fn step_energy_min(
        &mut self,
        dev: &ComputationDevice,
        last_step: &mut [Vec3],
        alpha: &mut f32,
        e_prev: &mut f64,
        external_force: &Option<Vec<Vec3>>,
    ) -> bool {
        let mut alpha_try = *alpha;

        loop {
            self.neighbors_nb.max_displacement_sq = 0.0;

            for (i, a) in self.atoms.iter_mut().enumerate() {
                last_step[i] = Vec3::new_zero();
                if a.static_ {
                    continue;
                }

                let f_mag = a.force.magnitude();
                if !f_mag.is_finite() || f_mag == 0.0 {
                    continue;
                }

                let step_mag = (alpha_try * f_mag).min(STEP_MAX);
                let s = a.force * (step_mag / f_mag);
                a.posit += s;
                last_step[i] = s;

                self.neighbors_nb.max_displacement_sq = self
                    .neighbors_nb
                    .max_displacement_sq
                    .max(s.magnitude_squared());
            }

            self.build_neighbors_if_needed(dev);

            compute_forces_and_energy(self, dev, external_force);
            let e_new = self.potential_energy;

            if self.cfg.overrides.snapshots_during_energy_min {
                self.take_snapshot_if_required(self.cfg.pressure_target as f64); // Pressure: todo?
            }

            if e_new <= *e_prev {
                *e_prev = e_new;

                *alpha = (alpha_try * GROW).min(ALPHA_MAX);

                let (max_f, _rms_f) = force_stats(self);
                return max_f <= F_TOL;
            }

            // Reject: revert positions
            let mut max_revert_sq = 0.0f32;
            for (i, a) in self.atoms.iter_mut().enumerate() {
                let s = last_step[i];
                let s2 = s.magnitude_squared();
                if s2 > 0.0 {
                    a.posit -= s;
                    max_revert_sq = max_revert_sq.max(s2);
                }
            }

            alpha_try *= SHRINK;

            if alpha_try < ALPHA_MIN {
                *alpha = alpha_try;
                self.neighbors_nb.max_displacement_sq = max_revert_sq;
                self.build_neighbors_if_needed(dev);
                compute_forces_and_energy(self, dev, external_force);
                return true;
            }

            // Ensure neighbors/forces are valid at the reverted geometry, then retry with smaller alpha_try
            self.neighbors_nb.max_displacement_sq = max_revert_sq;
            self.build_neighbors_if_needed(dev);
            compute_forces_and_energy(self, dev, external_force);
        }
    }
}
