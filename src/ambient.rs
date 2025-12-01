//! This module deals with the sim box, thermostat, and barostat.
//!
//! We set up Sim box, or cell, which is a rectangular prism (cube currently) which wraps at each face,
//! indefinitely. Its purpose is to simulate an infinity of water molecules. This box covers the atoms of interest,
//! but atoms in the neighboring (tiled) boxes influence the system as well. We use the concept of
//! a "minimum image" to find the closest copy of an item to a given site, among all tiled boxes.
//!
//! Note: We keep most thermostat and barostat code as f64, although we use f32 in most sections.

use lin_alg::f32::Vec3;
use na_seq::Element;
use rand::{Rng, prelude::ThreadRng};
use rand_distr::{ChiSquared, Distribution, StandardNormal};

use crate::{
    ACCEL_CONVERSION, ACCEL_CONVERSION_INV, AtomDynamics, HydrogenConstraint, MdState, SimBoxInit,
    water_opc::{H_MASS, MASS_WATER_MOL, O_MASS, WaterMol},
};

// Per-molecule Boltzmann, in kcal/mol/K.
// For assigning velocities from temperature, and other thermostat/barostat use.
pub(crate) const GAS_CONST_R: f64 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹ (Amber-style units)

// Boltzmann constant in (amu · Å²/ps²) K⁻¹
// We use this for the Langevin and Anderson thermostat, where we need per-particle Gaussian noise or variance.
pub(crate) const KB_A2_PS2_PER_K_PER_AMU: f32 = 0.831_446_26;
pub(crate) const BAR_PER_KCAL_MOL_PER_A3: f64 = 69476.95457055373;

// TAU is for the CSVR thermostat only. Lower means more sensitive.
// We set an aggressive thermostat during water initialization, then a more relaxed one at runtime.
// This is for the VV/CVSR themostat only.
// todo: SPME seems to be injecting energy into the system. Ideally this value should be close to 1.
// todo: Lowered until we sort this out.
pub(crate) const TAU_TEMP_DEFAULT: f64 = 1.0;
pub(crate) const TAU_TEMP_WATER_INIT: f64 = 0.03; // for CSVR

// Gamma is for the Langevin thermostat.
const GAMMA_WATER_INIT: f32 = 1.; // todo: Experiment

/// This bounds the area where atoms are wrapped. For now at least, it is only
/// used for water atoms. Its size and position should be such as to keep the system
/// solvated. We may move it around during the sim.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct SimBox {
    pub bounds_low: Vec3,
    pub bounds_high: Vec3,
    pub extent: Vec3,
}

impl SimBox {
    /// Set up to surround all atoms, with a pad, or with fixed dimensions. `atoms` is whichever we use to center the bix.
    pub fn new(atoms: &[AtomDynamics], box_type: &SimBoxInit) -> Self {
        match box_type {
            SimBoxInit::Pad(pad) => {
                let (mut min, mut max) =
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for a in atoms {
                    min = min.min(a.posit);
                    max = max.max(a.posit);
                }

                let bounds_low = min - Vec3::splat(*pad);
                let bounds_high = max + Vec3::splat(*pad);

                Self {
                    bounds_low,
                    bounds_high,
                    extent: bounds_high - bounds_low,
                }
            }
            SimBoxInit::Fixed((bounds_low, bounds_high)) => {
                let bounds_low: Vec3 = *bounds_low;
                let bounds_high: Vec3 = *bounds_high;
                Self {
                    bounds_low,
                    bounds_high,
                    extent: bounds_high - bounds_low,
                }
            }
        }
    }

    /// We periodically run this to keep the solvent surrounding the dynamic atoms, as they move.
    pub fn recenter(&mut self, atoms: &[AtomDynamics]) {
        let half_ext = self.extent / 2.;

        // todo: DRY with new.
        let mut center = Vec3::new_zero();
        for atom in atoms {
            center += atom.posit;
        }
        center /= atoms.len() as f32;

        self.bounds_low = center - half_ext;
        self.bounds_high = center + half_ext;
    }

    /// Wrap an absolute coordinate back into the unit cell. (orthorhombic). We use it to
    /// keep arbitrary coordinates inside it.
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        let ext = &self.extent;

        assert!(
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0,
            "SimBox edges must be > 0 (lo={:?}, hi={:?})",
            self.bounds_low,
            self.bounds_high
        );

        // rem_euclid keeps the value in [0, ext)
        Vec3::new(
            (p.x - self.bounds_low.x).rem_euclid(ext.x) + self.bounds_low.x,
            (p.y - self.bounds_low.y).rem_euclid(ext.y) + self.bounds_low.y,
            (p.z - self.bounds_low.z).rem_euclid(ext.z) + self.bounds_low.z,
        )
    }

    /// Minimum-image displacement vector. Find the closest copy
    /// of an item to a given site, among all tiled boxes. Maps a displacement vector to the closest
    /// periodic image. Allows distance measurements to use the shortest separation.
    pub fn min_image(&self, dv: Vec3) -> Vec3 {
        let ext = &self.extent;
        debug_assert!(ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0);

        Vec3::new(
            dv.x - (dv.x / ext.x).round() * ext.x,
            dv.y - (dv.y / ext.y).round() * ext.y,
            dv.z - (dv.z / ext.z).round() * ext.z,
        )
    }

    pub fn volume(&self) -> f32 {
        (self.bounds_high.x - self.bounds_low.x).abs()
            * (self.bounds_high.y - self.bounds_low.y).abs()
            * (self.bounds_high.z - self.bounds_low.z).abs()
    }

    pub fn center(&self) -> Vec3 {
        (self.bounds_low + self.bounds_high) * 0.5
    }

    /// For use with the barostat. It will expand or shrink the box if it determines the pressure
    /// is too high or low based on the virial pair sum.
    pub fn scale_isotropic(&mut self, lambda: f32) {
        // todo: QC f32 vs f64 in this fn.

        // Treat non-finite or tiny λ as "no-op"
        let lam = if lambda.is_finite() && lambda.abs() > 1.0e-12 {
            lambda
        } else {
            1.0
        };

        let c = self.center();
        let lo = c + (self.bounds_low - c) * lam;
        let hi = c + (self.bounds_high - c) * lam;

        // Enforce low <= high per component
        self.bounds_low = Vec3::new(lo.x.min(hi.x), lo.y.min(hi.y), lo.z.min(hi.z));
        self.bounds_high = Vec3::new(lo.x.max(hi.x), lo.y.max(hi.y), lo.z.max(hi.z));
        self.extent = self.bounds_high - self.bounds_low;

        debug_assert!({
            let ext = &self.extent;
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0
        });
    }
}

/// Isotropic Berendsen barostat (τ=relaxation time, κT=isothermal compressibility)
pub struct BerendsenBarostat {
    /// bar (kPa / 100)
    pub pressure_target: f64,
    /// picoseconds
    pub tau_pressure: f64,
    /// bar‑1 (≈4.5×10⁻⁵ for water at 300K, 1bar)
    pub kappa_t: f64,
    /// Virials, in kcal. We split these up to make debugging easier.
    pub virial_nonbonded_short_range: f64,
    // pub virial_lj: f64,
    pub virial_bonded: f64,
    pub virial_constraints: f64,
    /// I.e. SPME recip
    pub virial_nonbonded_long_range: f64,
    pub rng: ThreadRng,
}

impl Default for BerendsenBarostat {
    fn default() -> Self {
        Self {
            // Standard atmospheric pressure.
            pressure_target: 1.,
            // Relaxation time: 1 ps ⇒ gentle volume changes every few steps.
            tau_pressure: 1.,
            // Isothermal compressibility of water at 298 K.
            kappa_t: 4.5e-5,
            //These virials init to 0 here, and at the start of each integrator step.
            virial_nonbonded_short_range: 0.0,
            // virial_lj: 0.0,
            virial_bonded: 0.0,
            virial_constraints: 0.0,
            virial_nonbonded_long_range: 0.0,
            rng: rand::rng(),
        }
    }
}

impl BerendsenBarostat {
    pub fn scale_factor(&self, p_inst: f64, dt: f64) -> f64 {
        // Δln V = (κ_T/τ_p) (P - P0) dt
        let mut dlnv = (self.kappa_t / self.tau_pressure) * (p_inst - self.pressure_target) * dt;

        // Cap per-step volume change (e.g., ≤10%)
        const MAX_DLNV: f64 = 0.10;
        dlnv = dlnv.clamp(-MAX_DLNV, MAX_DLNV);

        // λ = exp(ΔlnV/3) — strictly positive and well-behaved
        (dlnv / 3.0).exp()
    }

    pub(crate) fn apply_isotropic(
        &self,
        dt_ps: f64,
        p_inst_bar: f64,
        simbox: &mut SimBox,
        atoms_dyn: &mut [AtomDynamics],
        waters: &mut [WaterMol],
    ) {
        let lam = self.scale_factor(p_inst_bar, dt_ps); // λ for **lengths** (not volume)

        if !(lam.is_finite() && lam > 0.0) || (lam - 1.0).abs() < 1e-12 {
            return; // no-op
        }

        // 1) Scale the box about its center
        simbox.scale_isotropic(lam as f32);

        // 2) Scale all coordinates about the same center (affine dilation)
        let c = simbox.center();
        let lc = lam as f32;

        fn scale_pos(p: &mut Vec3, c: Vec3, s: f32) {
            *p = c + (*p - c) * s;
        }

        for a in atoms_dyn.iter_mut() {
            if !a.static_ {
                scale_pos(&mut a.posit, c, lc);
            }
        }
        for w in waters.iter_mut() {
            scale_pos(&mut w.o.posit, c, lc);
            scale_pos(&mut w.h0.posit, c, lc);
            scale_pos(&mut w.h1.posit, c, lc);
            // If you store relative geometry for rigid bodies, keep it consistent (here we keep absolute).
        }

        // 3) (Optional but recommended) Affine velocity scaling due to box dilation rate.
        // For simple Berendsen, many codes rescale velocities by λ as well; the thermostat will re-set T.
        // If you prefer, omit this and let the thermostat handle KE. Either way is acceptable for Berendsen.
        let lv = lc; // or comment out this block to leave velocities unchanged
        for a in atoms_dyn.iter_mut() {
            if !a.static_ {
                a.vel *= lv;
            }
        }
        for w in waters.iter_mut() {
            w.o.vel *= lv;
            w.h0.vel *= lv;
            w.h1.vel *= lv;
        }
    }

    pub(crate) fn virial_total(&self) -> f64 {
        // + self.virial_lj
        self.virial_bonded
            + self.virial_constraints
            + self.virial_nonbonded_short_range
            + self.virial_nonbonded_long_range
    }
}

impl MdState {
    /// Computes total kinetic energy, in kcal/mol
    /// Includes all non-static atoms, including water.
    pub(crate) fn kinetic_energy(&self) -> f64 {
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
        result * 0.5 * ACCEL_CONVERSION_INV as f64
    }

    /// Instantaneous temperature [K]
    pub(crate) fn temperature(&self) -> f64 {
        (2.0 * self.kinetic_energy) / (self.thermo_dof as f64 * GAS_CONST_R)
    }

    /// Used in temperature computation. Constraints tracked are Hydrogen if configured, and
    /// static atoms.
    /// We cache this at init. Used for kinetic energy and temperature computations.
    pub(crate) fn dof_for_thermo(&self) -> usize {
        // 6 positional + 3 rotational for each water mol.
        // let mut result = 3 * self.water.len();
        let mut result = 6 * self.water.len();

        if !self.water_only_sim_at_init {
            result += 3 * self.atoms.iter().filter(|a| !a.static_).count();
        }

        let num_constraints = {
            let mut c = 0;

            if !self.water_only_sim_at_init {
                for atom in &self.atoms {
                    if self.cfg.hydrogen_constraint == HydrogenConstraint::Constrained
                        && atom.element == Element::Hydrogen
                    {
                        c += 1;
                    }
                }
            }

            if self.cfg.zero_com_drift {
                c += 3;
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
        // This value is cached at init.
        let dof = self.thermo_dof.max(2) as f64;

        // Cached during the kick-and-drift step.
        let ke = self.kinetic_energy; // In kcal/mol

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
    /// todo: Should this be based on f64?
    pub(crate) fn apply_langevin_thermostat(&mut self, dt: f32, gamma_ps: f32, temp_tgt_k: f32) {
        // More aggressive gamma for water init
        let gamma = if self.water_only_sim_at_init {
            GAMMA_WATER_INIT
        } else {
            gamma_ps
        };

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
            // per-component σ for velocity noise
            let nx: f32 = self.barostat.rng.sample(StandardNormal);
            let ny: f32 = self.barostat.rng.sample(StandardNormal);
            let nz: f32 = self.barostat.rng.sample(StandardNormal);

            w.o.vel.x = c * w.o.vel.x + sigma_o * nx;
            w.o.vel.y = c * w.o.vel.y + sigma_o * ny;
            w.o.vel.z = c * w.o.vel.z + sigma_o * nz;

            w.h0.vel.x = c * w.h0.vel.x + sigma_h * nx;
            w.h0.vel.y = c * w.h0.vel.y + sigma_h * ny;
            w.h0.vel.z = c * w.h0.vel.z + sigma_h * nz;

            w.h1.vel.x = c * w.h1.vel.x + sigma_h * nx;
            w.h1.vel.y = c * w.h1.vel.y + sigma_h * ny;
            w.h1.vel.z = c * w.h1.vel.z + sigma_h * nz;
        }
    }
}

/// Instantaneous pressure in bar.
/// P = (2K + W) / (3V), in kcal/mol/Å^3
pub(crate) fn measure_instantaneous_pressure(
    kinetic_energy: f64, // kcal
    simbox: &SimBox,
    virial_total: f64,
) -> f64 {
    // P = (2K + W) / (3V)  in kcal/mol/Å^3
    let v_a3 = simbox.volume() as f64;
    let p_kcal_per_a3 = (2.0 * kinetic_energy + virial_total) / (3.0 * v_a3);

    // Convert to bar
    p_kcal_per_a3 * BAR_PER_KCAL_MOL_PER_A3
}

impl MdState {
    // todo: Consider removing this in favor or exposing these values in snapshots.
    // todo: Then, applications could display in GUI etc.
    /// Print ambient parameters, as a sanity check.
    pub(crate) fn print_ambient_data(&self, pressure: f64) {
        println!("\n------Ambient stats at step {}--------", self.step_count);

        let cell_vol = self.cell.volume() as f64;
        let atom_count = &self.atoms.iter().filter(|a| !a.static_).count();
        println!(
            "Cell vol: {cell_vol:.1} Å^3 num dynamic atoms: {atom_count} num water mols: {}",
            self.water.len()
        );

        let temp = self.temperature();
        println!("\nTemperature: {temp:.2} K");

        let mut water_v = 0.;
        for mol in &self.water {
            water_v += mol.o.vel.magnitude();
        }
        let mut atom_v = 0.;
        for atom in self.atoms.iter().filter(|a| !a.static_) {
            atom_v += atom.vel.magnitude();
        }

        println!(
            "Ke: {:.2} kcal/mol = {:.2} (Å/ps)²  DOF: {}",
            self.kinetic_energy,
            self.kinetic_energy * ACCEL_CONVERSION as f64,
            self.thermo_dof
        );

        println!(
            "Water O avg vel: {:.3} Å/ps | Atom (non-static) avg vel: {:.3} Å/ps",
            water_v / self.water.len() as f32,
            atom_v / *atom_count as f32
        );

        println!("\nPressure: {pressure:.3} bar");

        eprintln!(
            "W_bonded: {:.3} kcal/mol  W Short range={:.3}  W long range: {:.3}  W_constraint: {:.5}",
            self.barostat.virial_bonded,
            self.barostat.virial_nonbonded_short_range,
            self.barostat.virial_nonbonded_long_range,
            self.barostat.virial_constraints,
        );

        println!("------------------------");
    }
}
