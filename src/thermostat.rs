//! Note: We keep most thermostat and barostat code as f64, although we use f32 in most sections.

use na_seq::Element;
use rand::{Rng, distr::Distribution};
use rand_distr::{ChiSquared, StandardNormal};

use crate::{
    HydrogenConstraint, MdState, NATIVE_TO_KCAL,
    solvent::{H_MASS, MASS_WATER_MOL, O_MASS},
};

// Per-molecule Boltzmann, in kcal/mol/K.
// For assigning velocities from temperature, and other thermostat/barostat use.
pub(crate) const GAS_CONST_R: f64 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹ (Amber-style units)

// Boltzmann constant in (amu · Å²/ps²) K⁻¹
// We use this for the Langevin and Anderson thermostat, where we need per-particle Gaussian noise or variance.
pub(crate) const KB_A2_PS2_PER_K_PER_AMU: f32 = 0.831_446_26;

// TAU is for the CSVR thermostat. In ps. Lower means more sensitive.
// We set an aggressive thermostat during solvent initialization, then a more relaxed one at runtime.
// This is for the VV/CVSR themostat only.
// Note: These are publically exposed, for use in applications.
pub const TAU_TEMP_DEFAULT: f64 = 0.1; // GROMACS default.
pub const TAU_TEMP_WATER_INIT: f64 = 0.01; // for CSVR

// These are in 1/ps. 1 ps^-1 is a good default for explicit solvent and constrained H bonds.
// Lower is closer to Newtonian dynamics. 2ps (0.5ps^-1) is Gromac's default.
pub const LANGEVIN_GAMMA_DEFAULT: f32 = 0.5;
pub const LANGEVIN_GAMMA_WATER_INIT: f32 = 15.;

impl MdState {
    /// Computes total kinetic energy, in native units.
    /// Includes all non-static atoms, including solvent.
    pub(crate) fn measure_kinetic_energy(&self) -> f64 {
        let mut result = 0.0;

        for a in &self.atoms {
            if !a.static_ {
                result += (a.mass * a.vel.magnitude_squared()) as f64;
            }
        }

        // Do not include the M/EP site.
        for w in &self.water {
            result += (w.o.mass * w.o.vel.magnitude_squared()) as f64;
            result += (w.h0.mass * w.h0.vel.magnitude_squared()) as f64;
            result += (w.h1.mass * w.h1.vel.magnitude_squared()) as f64;
        }

        // Add in the 0.5 factor, and convert from amu • (Å/ps)² to kcal/mol.
        result * 0.5 * NATIVE_TO_KCAL as f64
    }

    /// COM-only kinetic energy for solvent + full atomic KE for non-solvent atoms, in kcal/mol.
    /// Used for pressure via the molecular virial theorem: rigid solvent molecules contribute
    /// only their translational (COM) KE, so no SETTLE constraint virial is needed.
    pub(crate) fn measure_kinetic_energy_translational(&self) -> f64 {
        let mut result = 0.0;

        for a in &self.atoms {
            if !a.static_ {
                result += (a.mass * a.vel.magnitude_squared()) as f64;
            }
        }

        for w in &self.water {
            let v_com = (w.o.vel * O_MASS + w.h0.vel * H_MASS + w.h1.vel * H_MASS) / MASS_WATER_MOL;
            result += (MASS_WATER_MOL * v_com.magnitude_squared()) as f64;
        }

        result * 0.5 * NATIVE_TO_KCAL as f64
    }

    /// Instantaneous temperature [K]
    pub(crate) fn measure_temperature(&self) -> f64 {
        (2.0 * self.kinetic_energy) / (self.thermo_dof as f64 * GAS_CONST_R)
    }

    /// Used in temperature computation. Constraints tracked are Hydrogen if constrained, COM drift removal,
    /// and static atoms.
    /// We cache this at init. Used for kinetic energy and temperature computations.
    pub(crate) fn dof_for_thermo(&self) -> usize {
        // 3 positional + 3 rotational for each solvent mol.
        let mut result = 6 * self.water.len();

        if !self.solvent_only_sim_at_init {
            result += 3 * self.atoms.iter().filter(|a| !a.static_).count();
        }

        let num_constraints = {
            let mut c = 0;

            if !self.solvent_only_sim_at_init {
                for atom in &self.atoms {
                    if matches!(
                        self.cfg.hydrogen_constraint,
                        HydrogenConstraint::Constrained { shake_tolerance: _ }
                    ) && atom.element == Element::Hydrogen
                        && !atom.static_
                    {
                        c += 1;
                    }
                }
            }

            if self.cfg.zero_com_drift {
                c += 6;
            }
            c
        };

        result.saturating_sub(num_constraints)
    }

    /// Canonical Sampling through Velocities Rescaling thermostat. (Also known as Bussi, its
    /// primary author)
    /// [CSVR thermostat](https://arxiv.org/pdf/0803.4060)
    /// A canonical velocity-rescale algorithm.
    /// Cheap with gentle coupling, but doesn't imitate solvent drag.
    pub(crate) fn apply_thermostat_csvr(&mut self, dt: f64, tau: f64, t_target: f64) {
        if tau <= 0.0 {
            return;
        }

        // This value is cached at init.
        let dof = self.thermo_dof.max(2) as f64;

        // Measure current KE from velocities so we get the post-kick value, not a stale cache.
        let ke = self.measure_kinetic_energy(); // In kcal/mol

        if ke < 1e-20 {
            return;
        }

        let c = (-dt / tau).exp();

        // Draw the two random variates used in the exact CSVR update:
        let r: f64 = StandardNormal.sample(&mut self.barostat.rng); // N(0,1)
        let chi = ChiSquared::new(dof - 1.0)
            .unwrap()
            .sample(&mut self.barostat.rng); // χ²_{dof-1}

        let ke_target = 0.5 * dof * GAS_CONST_R * t_target;

        // Discrete-time exact solution for the OU process in K (from Bussi 2007):
        // K' = K*c + ke_bar*(1.0 - c) * [ (chi + r*r)/dof ] + 2.0*r*sqrt(c*(1.0-c)*K*ke_bar/dof)
        let k_prime = ke * c
            + ke_target * (1.0 - c) * ((chi + r * r) / dof)
            + 2.0 * r * ((c * (1.0 - c) * ke * ke_target / dof).sqrt());

        let k_prime = k_prime.max(1e-20);
        let lam = (k_prime / ke).sqrt() as f32;

        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.vel *= lam;
        }
        for w in &mut self.water {
            w.o.vel *= lam;
            w.h0.vel *= lam;
            w.h1.vel *= lam;
        }
    }

    /// A thermostat that integrates the stochastic Langevin equation. Good temperature control
    /// and ergodicity, but the friction parameter damps real dynamics as it grows. This applies an OU update.
    pub(crate) fn apply_langevin_thermostat(&mut self, dt: f32, gamma: f32, temp_tgt_k: f32) {
        let c = (-gamma * dt).exp();
        let s2 = (1.0 - c * c).max(0.0); // numerical guard

        let sigma_num = KB_A2_PS2_PER_K_PER_AMU * temp_tgt_k * s2;
        let sigma_o = (sigma_num / O_MASS).sqrt();
        let sigma_h = (sigma_num / H_MASS).sqrt();

        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            // per-component σ for velocity noise
            let sigma = (sigma_num / a.mass).sqrt();

            let nx: f32 = self.barostat.rng.sample(StandardNormal);
            let ny: f32 = self.barostat.rng.sample(StandardNormal);
            let nz: f32 = self.barostat.rng.sample(StandardNormal);

            a.vel.x = c * a.vel.x + sigma * nx;
            a.vel.y = c * a.vel.y + sigma * ny;
            a.vel.z = c * a.vel.z + sigma * nz;
        }

        for w in &mut self.water {
            let (ox, oy, oz): (f32, f32, f32) = (
                self.barostat.rng.sample(StandardNormal),
                self.barostat.rng.sample(StandardNormal),
                self.barostat.rng.sample(StandardNormal),
            );
            let (h0x, h0y, h0z): (f32, f32, f32) = (
                self.barostat.rng.sample(StandardNormal),
                self.barostat.rng.sample(StandardNormal),
                self.barostat.rng.sample(StandardNormal),
            );
            let (h1x, h1y, h1z): (f32, f32, f32) = (
                self.barostat.rng.sample(StandardNormal),
                self.barostat.rng.sample(StandardNormal),
                self.barostat.rng.sample(StandardNormal),
            );

            w.o.vel.x = c * w.o.vel.x + sigma_o * ox;
            w.o.vel.y = c * w.o.vel.y + sigma_o * oy;
            w.o.vel.z = c * w.o.vel.z + sigma_o * oz;

            w.h0.vel.x = c * w.h0.vel.x + sigma_h * h0x;
            w.h0.vel.y = c * w.h0.vel.y + sigma_h * h0y;
            w.h0.vel.z = c * w.h0.vel.z + sigma_h * h0z;

            w.h1.vel.x = c * w.h1.vel.x + sigma_h * h1x;
            w.h1.vel.y = c * w.h1.vel.y + sigma_h * h1y;
            w.h1.vel.z = c * w.h1.vel.z + sigma_h * h1z;
        }
    }
}
