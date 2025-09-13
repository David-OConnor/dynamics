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
use rand_distr::StandardNormal;

use crate::{
    ACCEL_CONVERSION_INV, AtomDynamics, HydrogenConstraint, KB, KB_A2_PS2_PER_K_PER_AMU, MdState,
    SimBoxInit,
    water_opc::{H_MASS, O_MASS},
};

const BAR_PER_KCAL_MOL_PER_A3: f64 = 69476.95457055373;

/// This bounds the area where atoms are wrapped. For now at least, it is only
/// used for water atoms. Its size and position should be such as to keep the system
/// solvated. We may move it around during the sim.
#[derive(Clone, Copy, Default, Debug)]
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
                let bounds_low: Vec3 = (*bounds_low).into();
                let bounds_high: Vec3 = (*bounds_high).into();
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
    pub tau_temp: f64,
    /// bar‑1 (≈4.5×10⁻⁵ for water at 300K, 1bar)
    pub kappa_t: f64,
    pub virial_pair_kcal: f64,
    pub rng: ThreadRng,
}

impl Default for BerendsenBarostat {
    fn default() -> Self {
        Self {
            // Standard atmospheric pressure.
            pressure_target: 1.,
            // Relaxation time: 1 ps ⇒ gentle volume changes every few steps.
            tau_pressure: 1.,
            tau_temp: 1.,
            // Isothermal compressibility of water at 298 K.
            kappa_t: 4.5e-5,
            virial_pair_kcal: 0.0, // Inits to 0 here, and at the start of each integrator step.
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
}

impl MdState {
    fn kinetic_energy_kcal(&self) -> f64 {
        // dynamic atoms + waters (skip massless EP)
        let mut ke = 0.0;
        for a in &self.atoms {
            ke += 0.5 * (a.mass * a.vel.magnitude_squared()) as f64;
        }

        for w in &self.water {
            ke += (0.5 * w.o.mass * w.o.vel.magnitude_squared()) as f64;
            ke += (0.5 * w.h0.mass * w.h0.vel.magnitude_squared()) as f64;
            ke += (0.5 * w.h1.mass * w.h1.vel.magnitude_squared()) as f64;
        }
        ke * ACCEL_CONVERSION_INV
    }

    fn num_constraints_estimate(&self) -> usize {
        let mut c = 0;

        // (1) Rigid waters (O,H0,H1 rigid; EP is massless/virtual)
        // 3 constraints per water triad.
        c += 3 * self.water.len();

        // (2) SHAKE/RATTLE on X–H bonds among *dynamic* atoms (not counting waters here)
        // If hydrogens are constrained (your code calls shake_hydrogens() when HydrogenMdType::Fixed),
        // count ≈ number of H atoms among self.atoms (each has one constrained bond).
        if self.cfg.hydrogen_constraint == HydrogenConstraint::Constrained {
            c += self
                .atoms
                .iter()
                .filter(|a| a.element == Element::Hydrogen)
                .count();
        }

        // (3) If you have any extra explicit constraints elsewhere, add them here.
        // e.g., c += self.extra_constraints.len();

        c
    }

    fn dof_for_thermo(&self) -> usize {
        let mut n = 3 * (self.atoms.len() + 3 * self.water.len());
        n -= self.num_constraints_estimate();

        if self.cfg.zero_com_drift {
            n = n.saturating_sub(3);
        }
        n
    }

    /// CSVR/Bussi thermostat: A canonical velocity-rescale algorithm.
    /// Cheap with gentle coupling, but doesn't imitate solvent drag.
    pub(crate) fn apply_thermostat_csvr(&mut self, dt: f64, t_target_k: f64) {
        // todo: QC f32 vs f64 here.
        use rand_distr::{ChiSquared, Distribution, StandardNormal};

        let dof = self.dof_for_thermo().max(2) as f64;
        let ke = self.kinetic_energy_kcal();
        let ke_bar = 0.5 * dof * KB as f64 * t_target_k;

        let c = (-dt / self.barostat.tau_temp).exp();
        // Draw the two random variates used in the exact CSVR update:
        let r: f64 = StandardNormal.sample(&mut self.barostat.rng); // N(0,1)
        let chi = ChiSquared::new(dof - 1.0)
            .unwrap()
            .sample(&mut self.barostat.rng); // χ²_{dof-1}

        // Discrete-time exact solution for the OU process in K (from Bussi 2007):
        // K' = K*c + ke_bar*(1.0 - c) * [ (chi + r*r)/dof ] + 2.0*r*sqrt(c*(1.0-c)*K*ke_bar/dof)
        let kprime = ke * c
            + ke_bar * (1.0 - c) * ((chi + r * r) / dof)
            + 2.0 * r * ((c * (1.0 - c) * ke * ke_bar / dof).sqrt());

        let lam = (kprime / ke).sqrt() as f32;

        for a in &mut self.atoms {
            a.vel *= lam;
        }
        for w in &mut self.water {
            w.o.vel *= lam;
            w.h0.vel *= lam;
            w.h1.vel *= lam;
        }
    }

    /// Instantaneous pressure in **bar** (pair virial only).
    /// P = (2K + W) / (3V)
    pub(crate) fn instantaneous_pressure_bar(&self) -> f64 {
        let vol_a3 = self.cell.volume() as f64; // Å^3
        if !(vol_a3 > 0.0) {
            return f64::NAN;
        }
        let k_kcal = self.kinetic_energy_kcal(); // kcal/mol
        let w_kcal = self.barostat.virial_pair_kcal; // kcal/mol (pairs included this step)
        let p_kcal_per_a3 = (2.0 * k_kcal + w_kcal) / (3.0 * vol_a3);
        p_kcal_per_a3 * BAR_PER_KCAL_MOL_PER_A3
    }

    /// Call each step (or every nstpcouple steps) after thermostat
    pub(crate) fn apply_barostat_berendsen(&mut self, dt: f64) {
        let p_inst_bar = self.instantaneous_pressure_bar();
        if !p_inst_bar.is_finite() {
            return; // don't touch the box if pressure is bad
        }

        let lambda = self.barostat.scale_factor(p_inst_bar, dt) as f32;

        // Scale the cell
        let c = self.cell.center();
        self.cell.scale_isotropic(lambda);

        // Scale flexible atom coordinates about c; scale velocities
        for a in &mut self.atoms {
            a.posit = c + (a.posit - c) * lambda;
            a.vel *= lambda;
        }

        // Translate rigid waters by COM only; scale COM velocity
        for w in &mut self.water {
            let m_tot = w.o.mass + w.h0.mass + w.h1.mass;
            let com =
                (w.o.posit * w.o.mass + w.h0.posit * w.h0.mass + w.h1.posit * w.h1.mass) / m_tot;
            let com_v = (w.o.vel * w.o.mass + w.h0.vel * w.h0.mass + w.h1.vel * w.h1.mass) / m_tot;

            let com_new = c + (com - c) * lambda;
            let d = com_new - com;

            w.o.posit += d;
            w.h0.posit += d;
            w.h1.posit += d;
            w.m.posit += d;

            let dv = com_v * lambda - com_v;
            w.o.vel += dv;
            w.h0.vel += dv;
            w.h1.vel += dv;
        }
    }

    /// A thermostat that integrates the stochastic Langevin equation. Good temperature control
    /// and ergodicity, but the firction parameter damps real dynamics as it grows. This applies an OU update.
    /// todo: Should this be based on f64?
    pub(crate) fn apply_langevin_thermostat(&mut self, dt: f32, gamma_ps: f32, temp_k: f32) {
        let c = (-gamma_ps * dt).exp();
        let s2 = (1.0 - c * c).max(0.0); // numerical guard

        for a in &mut self.atoms {
            // per-component σ for velocity noise
            let sigma = (KB_A2_PS2_PER_K_PER_AMU * temp_k * s2 / a.mass).sqrt();

            // todo: Can we reuse this RNG from barostat here?
            let nx: f32 = self.barostat.rng.sample(StandardNormal);
            let ny: f32 = self.barostat.rng.sample(StandardNormal);
            let nz: f32 = self.barostat.rng.sample(StandardNormal);

            a.vel.x = c * a.vel.x + sigma * nx;
            a.vel.y = c * a.vel.y + sigma * ny;
            a.vel.z = c * a.vel.z + sigma * nz;
        }

        self.apply_langevin_thermostat_water(dt, gamma_ps, temp_k);
    }

    /// Part of the langevin thermostat.
    /// todo: Should this be based on f64?
    fn apply_langevin_thermostat_water(&mut self, dt: f32, gamma_ps: f32, temp_k: f32) {
        let c = (-gamma_ps * dt).exp();
        let s2 = (1.0 - c * c).max(0.0);
        let m_tot = O_MASS + 2.0 * H_MASS;

        let sigma_v = (KB_A2_PS2_PER_K_PER_AMU * temp_k * s2 / m_tot).sqrt();
        let sigma_omega = (KB_A2_PS2_PER_K_PER_AMU * temp_k * s2).sqrt();

        for w in &mut self.water {
            // COM position and velocity
            let rc =
                (w.o.posit * O_MASS + w.h0.posit * H_MASS + w.h1.posit * H_MASS) * (1.0 / m_tot);
            let mut v_com =
                (w.o.vel * O_MASS + w.h0.vel * H_MASS + w.h1.vel * H_MASS) * (1.0 / m_tot);

            // Relative positions
            let r_o = w.o.posit - rc;
            let r_h0 = w.h0.posit - rc;
            let r_h1 = w.h1.posit - rc;
            let r_m = w.m.posit - rc; // massless

            // Relative velocities
            let v_o_rel = w.o.vel - v_com;
            let v_h0_rel = w.h0.vel - v_com;
            let v_h1_rel = w.h1.vel - v_com;

            // Inertia tensor (lab frame), symmetric 3x3
            let mut a = 0.0;
            let mut d = 0.0;
            let mut f = 0.0; // xx, yy, zz
            let mut b = 0.0;
            let mut cxy = 0.0;
            let mut e = 0.0; // xy, xz, yz
            let mut add = |r: Vec3, m: f32| {
                let x = r.x;
                let y = r.y;
                let z = r.z;
                a += m * (y * y + z * z);
                d += m * (x * x + z * z);
                f += m * (x * x + y * y);
                b -= m * x * y;
                cxy -= m * x * z;
                e -= m * y * z;
            };
            add(r_o, O_MASS);
            add(r_h0, H_MASS);
            add(r_h1, H_MASS);

            // Cholesky of SPD matrix [[a,b,cxy],[b,d,e],[cxy,e,f]]
            let l11 = a.max(0.0).sqrt();
            let l21 = if l11 > 0.0 { b / l11 } else { 0.0 };
            let l31 = if l11 > 0.0 { cxy / l11 } else { 0.0 };
            let t22 = (d - l21 * l21).max(0.0);
            let l22 = t22.sqrt();
            let l32 = if l22 > 0.0 {
                (e - l21 * l31) / l22
            } else {
                0.0
            };
            let t33 = (f - l31 * l31 - l32 * l32).max(0.0);
            let l33 = t33.sqrt();

            // Angular momentum L = Σ m r × v_rel
            let l_vec = r_o.cross(v_o_rel) * O_MASS
                + r_h0.cross(v_h0_rel) * H_MASS
                + r_h1.cross(v_h1_rel) * H_MASS;

            // Solve I ω = L via the Cholesky: L y = L, L^T ω = y
            let solve_lower = |lx11: f32,
                               lx21: f32,
                               lx31: f32,
                               lx22: f32,
                               lx32: f32,
                               lx33: f32,
                               rhs: Vec3|
             -> Vec3 {
                let y1 = if lx11 > 0.0 { rhs.x / lx11 } else { 0.0 };
                let y2 = if lx22 > 0.0 {
                    (rhs.y - lx21 * y1) / lx22
                } else {
                    0.0
                };
                let y3 = if lx33 > 0.0 {
                    (rhs.z - lx31 * y1 - lx32 * y2) / lx33
                } else {
                    0.0
                };
                Vec3 {
                    x: y1,
                    y: y2,
                    z: y3,
                }
            };
            let solve_upper = |lx11: f32,
                               lx21: f32,
                               lx31: f32,
                               lx22: f32,
                               lx32: f32,
                               lx33: f32,
                               rhs: Vec3|
             -> Vec3 {
                let z3 = if lx33 > 0.0 { rhs.z / lx33 } else { 0.0 };
                let z2 = if lx22 > 0.0 {
                    (rhs.y - lx32 * z3) / lx22
                } else {
                    0.0
                };
                let z1 = if lx11 > 0.0 {
                    (rhs.x - lx21 * z2 - lx31 * z3) / lx11
                } else {
                    0.0
                };
                Vec3 {
                    x: z1,
                    y: z2,
                    z: z3,
                }
            };
            let y = solve_lower(l11, l21, l31, l22, l32, l33, l_vec);
            let mut omega = solve_upper(l11, l21, l31, l22, l32, l33, y);

            // OU on COM velocity
            let nx: f32 = self.barostat.rng.sample(StandardNormal);
            let ny: f32 = self.barostat.rng.sample(StandardNormal);
            let nz: f32 = self.barostat.rng.sample(StandardNormal);
            v_com.x = c * v_com.x + sigma_v * nx;
            v_com.y = c * v_com.y + sigma_v * ny;
            v_com.z = c * v_com.z + sigma_v * nz;

            // OU on angular velocity: ω ← c ω + σ * I^{-1/2} ξ, with ξ ~ N(0, I)
            let zx: f32 = self.barostat.rng.sample(StandardNormal);
            let zy: f32 = self.barostat.rng.sample(StandardNormal);
            let zz: f32 = self.barostat.rng.sample(StandardNormal);
            // x = I^{-1/2} z by solving L^T x = z
            let x = solve_upper(
                l11,
                l21,
                l31,
                l22,
                l32,
                l33,
                Vec3 {
                    x: zx,
                    y: zy,
                    z: zz,
                },
            );
            omega = omega * c + x * sigma_omega;

            // Reconstruct rigid-body velocities
            w.o.vel = v_com + omega.cross(r_o);
            w.h0.vel = v_com + omega.cross(r_h0);
            w.h1.vel = v_com + omega.cross(r_h1);
            w.m.vel = v_com + omega.cross(r_m);
        }
    }
}
