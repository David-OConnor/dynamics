//! This module deals with the sim box, and barostat.
//!
//! We set up Sim box, or cell, which is a rectangular prism (cube currently) which wraps at each face,
//! indefinitely. Its purpose is to simulate an infinity of solvent molecules. This box covers the atoms of interest,
//! but atoms in the neighboring (tiled) boxes influence the system as well. We use the concept of
//! a "minimum image" to find the closest copy of an item to a given site, among all tiled boxes.
//!
//! Note: We keep most thermostat and barostat code as f64, although we use f32 in most sections.

use std::fmt::Display;

use lin_alg::f32::Vec3;
use rand::prelude::ThreadRng;
use rand_distr::{Distribution, StandardNormal};

use crate::{AtomDynamics, KCAL_TO_NATIVE, MdState, NATIVE_TO_KCAL, SimBoxInit, solvent::WaterMol};

pub(crate) const BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED: f64 = 69476.95457055373;

/// Boltzmann constant in bar·Å³·K⁻¹ (= 1.380649×10⁻²³ J/K × 10⁻⁵ bar/Pa × 10³⁰ Å³/m³).
/// Used for the stochastic term in the C-rescale barostat.
const KB_BAR_A3_PER_K: f64 = 138.064_9;

// TAU is for the CSVR thermostat. In ps. Lower means more sensitive.
// We set an aggressive thermostat during solvent initialization, then a more relaxed one at runtime.
// This is for the VV/CVSR themostat only.
// Note: These are publically exposed, for use in applications.
// pub const TAU_TEMP_DEFAULT: f64 = 1.0;
pub const TAU_TEMP_DEFAULT: f64 = 0.9;
pub const TAU_TEMP_WATER_INIT: f64 = 0.01; // for CSVR

// These are in 1/ps. 1 ps^-1 is a good default for explicit solvent and constrained H bonds.
// Lower is closer to Newtonian dynamics.
pub const LANGEVIN_GAMMA_DEFAULT: f32 = 1.0;
pub const LANGEVIN_GAMMA_WATER_INIT: f32 = 15.;

/// This bounds the area where atoms are wrapped. For now at least, it is only
/// used for solvent atoms. Its size and position should be such as to keep the system
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

                if atoms.is_empty() {
                    min = Vec3::new_zero();
                    max = Vec3::new_zero();
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
        if atoms.is_empty() {
            return;
        }

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

    pub fn contains(&self, posit: Vec3) -> bool {
        !(posit.x < self.bounds_low.x
            || posit.y < self.bounds_low.y
            || posit.z < self.bounds_low.z
            || posit.x > self.bounds_high.x
            || posit.y > self.bounds_high.y
            || posit.z > self.bounds_high.z)
    }
}

/// The virial, in Kcal/Mol. Converted from our native units. We use a
/// separate type to help ensure we are using the correct units.
#[derive(Debug, Default)]
pub struct VirialKcalMol {
    pub bonded: f64,
    pub nonbonded_short_range: f64,
    pub nonbonded_long_range: f64,
    pub constraints: f64,
}

impl VirialKcalMol {
    pub(crate) fn total(&self) -> f64 {
        // todo temp!
        self.bonded + self.nonbonded_short_range + self.nonbonded_long_range + self.constraints
    }
}

impl Display for VirialKcalMol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Virial, kcal/mol. W_bonded: {:.3} kcal/mol  W Short range={:.3}  W long range: {:.3}  W_constraint: {:.3}",
            self.bonded, self.nonbonded_short_range, self.nonbonded_long_range, self.constraints,
        )
    }
}

/// Accumulated during force computations. Used to measure pressure.
/// Units per field:
///   - `bonded`, `nonbonded_short_range`, `nonbonded_long_range`: **kcal/mol**
///     (forces in kcal/(mol·Å) × distances in Å — no conversion needed).
///   - `constraints`: **native units** (amu·Å²/ps²), because SHAKE constraint
///     forces are computed as m·Δr/dt² which is in amu·Å/ps², and the virial
///     r·(m·Δr/dt²) is therefore in amu·Å²/ps².
/// We split this into components to make validating and debugging easier.
#[derive(Debug, Default)]
pub struct Virial {
    pub bonded: f64,
    pub nonbonded_short_range: f64,
    pub nonbonded_long_range: f64,
    pub constraints: f64,
}

impl Virial {
    /// Convert to kcal/mol.
    /// `bonded`/`nonbonded_*` are already in kcal/mol → copied as-is.
    /// `constraints` is in native units (amu·Å²/ps²) → multiplied by NATIVE_TO_KCAL.
    pub(crate) fn to_kcal_mol(&self) -> VirialKcalMol {
        VirialKcalMol {
            bonded: self.bonded,
            nonbonded_short_range: self.nonbonded_short_range,
            nonbonded_long_range: self.nonbonded_long_range,
            constraints: self.constraints * NATIVE_TO_KCAL as f64,
        }
    }
}

impl Display for Virial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Virial, native. W_bonded: {:.3} kcal/mol  W Short range={:.3}  W long range: {:.3}  W_constraint: {:.3}",
            self.bonded, self.nonbonded_short_range, self.nonbonded_long_range, self.constraints,
        )
    }
}

/// Isotropic C-rescale (stochastic cell rescaling) barostat — GROMACS `pcoupl = C-rescale`.
///
/// Reference: Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
///
/// The volume update is:
///   ΔlnV = (κT/τp)(P_inst − P₀)dt  +  √(2κT·kB·T·dt / (τp·V)) · ξ,  ξ ~ N(0,1)
///   μ = exp(ΔlnV/3)   (isotropic length scale factor)
///
/// The deterministic part is identical to Berendsen; the stochastic term restores the
/// correct NpT fluctuations that Berendsen suppresses.
pub struct Barostat {
    /// bar (kPa / 100)
    pub pressure_target: f64,
    /// picoseconds
    pub tau_pressure: f64,
    /// bar‑1 (≈4.5×10⁻⁵ for solvent at 300K, 1bar)
    pub kappa_t: f64,
    pub virial: Virial,
    pub rng: ThreadRng,
}

impl Default for Barostat {
    fn default() -> Self {
        Self {
            // Standard atmospheric pressure.
            pressure_target: 1.,
            // Relaxation time: 1 ps ⇒ gentle volume changes every few steps.
            tau_pressure: 1.,
            // Isothermal compressibility of solvent at 298 K.
            kappa_t: 4.5e-5,
            virial: Default::default(),
            rng: rand::rng(),
        }
    }
}

impl Barostat {
    /// Compute the isotropic length scale factor μ using the C-rescale algorithm.
    ///
    /// `temp_k` should be the reference (target) temperature in K.
    /// `vol_a3` is the current simulation-box volume in Å³.
    pub fn scale_factor(&mut self, p_inst: f64, dt: f64, temp_k: f64, vol_a3: f64) -> f64 {
        // Deterministic term: ΔlnV_det = (κT/τp)(P_inst − P₀)dt
        let dlnv_det = (self.kappa_t / self.tau_pressure) * (p_inst - self.pressure_target) * dt;

        // Stochastic term: σ = √(2κT·kB·T·dt / (τp·V))
        let sigma_lnv = (2.0 * self.kappa_t * KB_BAR_A3_PER_K * temp_k * dt
            / (self.tau_pressure * vol_a3))
            .sqrt();
        let xi: f64 = StandardNormal.sample(&mut self.rng);

        // Cap per-step volume change (≤10%) before computing λ
        const MAX_DLNV: f64 = 0.10;
        let dlnv = (dlnv_det + sigma_lnv * xi).clamp(-MAX_DLNV, MAX_DLNV);

        // λ = exp(ΔlnV/3) — isotropic length scale, strictly positive and well-behaved
        (dlnv / 3.0).exp()
    }

    pub(crate) fn apply_isotropic(
        &mut self,
        dt_ps: f64,
        p_inst_bar: f64,
        temp_k: f64,
        simbox: &mut SimBox,
        atoms_dyn: &mut [AtomDynamics],
        waters: &mut [WaterMol],
    ) {
        // todo: Temporarily disabled  barostat, until pressure measurements are fixed
        return;

        let vol_a3 = simbox.volume() as f64;
        let lam = self.scale_factor(p_inst_bar, dt_ps, temp_k, vol_a3); // λ for lengths (not volume)

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
        }

        // Affine velocity scaling; the thermostat will correct kinetic energy.
        for a in atoms_dyn.iter_mut() {
            if !a.static_ {
                a.vel *= lc;
            }
        }
        for w in waters.iter_mut() {
            w.o.vel *= lc;
            w.h0.vel *= lc;
            w.h1.vel *= lc;

            // We moved O and Hs above; update EP.
            w.update_virtual_site();
        }
    }
}

/// Measure instantaneous pressure, in bar. Inputs have been converted from native units
/// to kcal and kcal/mol.
/// P = (2K + W) / (3V), in kcal/mol/Å³
pub(crate) fn measure_pressure(
    kinetic_energy: f64, // kcal
    simbox: &SimBox,
    virial: &VirialKcalMol,
) -> f64 {
    let vol = simbox.volume() as f64; // Å³

    // This is in kcal/mol/Å³
    let result = (2.0 * kinetic_energy + virial.total()) / (3.0 * vol);

    // Convert from kcal/mol/Å³ to bar
    result * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED
}

impl MdState {
    #[allow(unused)]
    // todo: Consider removing this in favor or exposing these values in snapshots.
    // todo: Then, applications could display in GUI etc.
    /// Print ambient parameters, as a sanity check.
    pub(crate) fn print_ambient_data(&self, pressure: f64) {
        println!(
            "\n\n------Ambient stats at step {}--------",
            self.step_count
        );

        let cell_vol = self.cell.volume() as f64;
        let atom_count = &self.atoms.iter().filter(|a| !a.static_).count();
        println!(
            "Cell vol: {cell_vol:.1} Å^3 num dynamic atoms: {atom_count} num solvent mols: {}",
            self.water.len()
        );

        {
            let p_kin_bar = (2.0 * self.measure_kinetic_energy_translational()) / (3.0 * cell_vol)
                * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED;

            let vir_total_kcal = self.barostat.virial.to_kcal_mol().total();
            let p_vir_bar = vir_total_kcal / (3.0 * cell_vol) * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED;

            println!("P_kin: {p_kin_bar:.3} bar  P_vir: {p_vir_bar:.3} bar");
        }

        let temp = self.measure_temperature();
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
            self.kinetic_energy * KCAL_TO_NATIVE as f64,
            self.thermo_dof
        );

        println!(
            "Water O avg vel: {:.3} Å/ps | Atom (non-static) avg vel: {:.3} Å/ps",
            water_v / self.water.len() as f32,
            atom_v / *atom_count as f32
        );

        println!("\nPressure: {pressure:.3} bar");

        println!("Virial: {}", self.barostat.virial.to_kcal_mol());

        println!("------------------------");
    }
}
