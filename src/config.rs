use crate::{
    MdOverrides, SimBoxInit,
    integrate::Integrator,
    non_bonded::{CUTOFF_VDW, LONG_RANGE_CUTOFF},
    prep::HydrogenConstraint,
    snapshot::{SaveType, SnapshotHandler},
    solvent::Solvent,
    thermostat::TAU_TEMP_DEFAULT,
};
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::gromacs::mdp::PmeConfig;
use bio_files::{
    gromacs,
    gromacs::mdp::{
        Barostat, ConstraintAlgorithm, Constraints, CoulombType, EnergyMinimization,
        Integrator as MdpIntegrator, MdpParams, Pbc, PressureCouplingType, Thermostat, VdwType,
    },
};
use std::io;
use std::path::Path;

/// Abramowitz & Stegun 7.1.26 — max error 1.5×10⁻⁷.
fn erfc_approx(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// Invert erfc via bisection. Only called at setup time so performance is irrelevant.
fn erfc_inv_approx(y: f32) -> f32 {
    let (mut lo, mut hi) = (0.0_f32, 6.0_f32); // erfc(6) ≈ 2e-17, covers all practical rtol
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if erfc_approx(mid) > y {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// This is the primary way of configurating an MD run. It's passed at init, along with the
/// molecule list and FF params.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub struct MdConfig {
    /// Defaults to Velocity Verlet.
    pub integrator: Integrator,
    /// If enabled, zero the drift in center of mass of the system.
    pub zero_com_drift: bool,
    /// Kelvin. Defaults to 310 K.
    pub temp_target: f32,
    /// Bar (Pa/100). Defaults to 1 bar.
    pub pressure_target: f32,
    /// ps. Defaults to 5; this is what GROMACS uses.
    pub tau_pressure: f32,
    /// Allows constraining Hydrogens to be rigid with their bonded atom, using SHAKE and RATTLE
    /// algorithms. This allows for higher time steps.
    pub hydrogen_constraint: HydrogenConstraint,
    pub snapshot_handlers: Vec<SnapshotHandler>,
    pub sim_box: SimBoxInit,
    pub solvent: Solvent,
    /// Prior to the first integrator step, we attempt to relax energy in the system.
    /// Use no more than this many iterations to do so. Higher can produce better results,
    /// but is slower. If None, don't relax.
    pub max_init_relaxation_iters: Option<usize>,
    /// Distance threshold, in Å, used to determine when we rebuild neighbor lists.
    /// 2-4Å are common values. Higher values rebuild less often, and have more computationally-intense
    /// rebuilds. Rebuild the list if an atom moved > skin/2.
    pub neighbor_skin: f32,
    pub overrides: MdOverrides,
    /// Optional path to a pre-equilibrated water template file.
    /// When set, this template is used instead of the built-in 60 Å template.
    /// A box-specific template (e.g. 30 Å) is required for accurate initial pressures,
    /// because the built-in template is cut from a larger equilibrated cell and lacks
    /// the long-range Coulomb correlations appropriate for the target PBC cell size.
    /// Generate one with the `generate_water_init_template` test (marked #[ignore]).
    pub water_template_path: Option<String>,
    /// Skip the PBC-boundary proximity check (2.8 Å cross-boundary O-O filter) when placing
    /// water molecules.  Use this only when generating a pre-equilibrated template at the correct
    /// density: the ~88 boundary molecules that the filter would otherwise reject are needed to
    /// reach 1 g/cm³, and the MD equilibration run will push them to their natural distances.
    pub skip_water_pbc_filter: bool,
    /// We use the GROMACS struct unaltered.
    pub energy_minimization: EnergyMinimization,
    /// SPME mesh spacing in **Å**. Smaller = higher-resolution reciprocal mesh, more accurate
    /// but slower. 1.0 Å is a good default; equivalent to GROMACS `fourierspacing = 0.1 nm`.
    pub spme_mesh_spacing: f32,
    /// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
    /// reciprocal load.
    /// Common rule for α: erfc(α r_c) ≲ 10⁻⁴…10⁻⁵
    /// Å^-1. 0.35 is good for cutoff of 10–12 Å.
    pub spme_alpha: f32,
}

impl Default for MdConfig {
    fn default() -> Self {
        Self {
            integrator: Default::default(),
            zero_com_drift: false, // todo: True?
            temp_target: 300.,     // GROMACS uses this.
            pressure_target: 1.,
            tau_pressure: 5.,
            hydrogen_constraint: Default::default(),
            snapshot_handlers: vec![SnapshotHandler {
                save_type: SaveType::Memory,
                ratio: 1,
            }],
            sim_box: Default::default(),
            solvent: Default::default(),
            max_init_relaxation_iters: Some(1_000), // todo: A/R
            neighbor_skin: 4.0,
            overrides: Default::default(),
            water_template_path: None,
            skip_water_pbc_filter: false,
            energy_minimization: Default::default(),
            spme_mesh_spacing: 1.0,
            spme_alpha: 0.35,
        }
    }
}

impl MdConfig {
    /// Creates a similar config for use with GROMACS. Attempts to replicate this library's
    /// settings where we don't have an applicable MdConfig field.
    ///
    /// Not `From` trait: This lib depends on `bio_files`, but not the other way around.
    pub fn to_gromacs(&self, n_steps: usize, dt: f32) -> MdpParams {
        // Use the first handler's ratio for all output types.
        // todo: We're currently using a single ratio for all output types.
        let ratio = self.snapshot_handlers.first().map(|h| h.ratio).unwrap_or(0) as u32;

        let (integrator, tau_t, thermostat) = match self.integrator {
            Integrator::LangevinMiddle { gamma } => {
                // GROMACS `sd` (stochastic dynamics) is the correct counterpart for Langevin.
                // `tcoupl` must be `no` for `sd`; friction is given via `tau-t` = 1/γ (ps).
                let tau = if gamma > 0.0 { 1.0 / gamma } else { 1.0 };
                (
                    gromacs::mdp::Integrator::Sd,
                    tau,
                    gromacs::mdp::Thermostat::No,
                )
            }
            Integrator::VerletVelocity { thermostat } => {
                let tau = if let Some(t) = thermostat {
                    t as f32
                } else {
                    TAU_TEMP_DEFAULT as f32
                };
                (
                    gromacs::mdp::Integrator::MdVv,
                    tau,
                    gromacs::mdp::Thermostat::VRescale,
                )
            }
        };

        let (constraints, shake_tol) = match self.hydrogen_constraint {
            HydrogenConstraint::Constrained { shake_tolerance } => {
                (gromacs::mdp::Constraints::HBonds, shake_tolerance)
            }
            HydrogenConstraint::Flexible => (gromacs::mdp::Constraints::None, 0.),
        };

        let barostat = if self.overrides.baro_disabled {
            gromacs::mdp::Barostat::No
        } else {
            gromacs::mdp::Barostat::CRescale
        };

        const ANGSTROM_TO_NM: f32 = 0.1;

        // Many of these values are the MdpParams defaults, but we specify here
        // to be explicit, and not miss any.
        MdpParams {
            integrator,
            nsteps: n_steps as u64,
            dt,
            nstxout: ratio,
            nstvout: ratio,
            nstxout_compressed: ratio,
            nstenergy: ratio,
            nstlog: ratio,
            coulombtype: CoulombType::Pme(PmeConfig {
                fourierspacing: self.spme_mesh_spacing * ANGSTROM_TO_NM,
                order: 4, // Hard-coded in `ewald`.
                rtol: erfc_approx(self.spme_alpha * LONG_RANGE_CUTOFF),
                ..Default::default()
            }),
            rcoulomb: LONG_RANGE_CUTOFF * ANGSTROM_TO_NM,
            vdwtype: VdwType::CutOff,
            rvdw: CUTOFF_VDW * ANGSTROM_TO_NM,
            thermostat,
            // We only have one temperature-coupling group in this lib.
            tau_t: vec![tau_t],
            ref_t: vec![self.temp_target],
            pcoupl: barostat,
            pcoupltype: PressureCouplingType::default(),
            tau_p: self.tau_pressure,
            ref_p: self.pressure_target,
            compressibility: 4.5e-5, // Standard water compressibility
            pbc: Pbc::Xyz,
            gen_vel: true,
            gen_temp: self.temp_target,
            gen_seed: -1, // Default
            constraints,
            // todo: Support LINCS in dynamics?
            constraint_algorithm: ConstraintAlgorithm::Shake { tol: shake_tol },
            energy_minimization: Some(self.energy_minimization.clone()),
        }
    }

    /// Convenience function to save in Gromacs' `.mdp` format.
    pub fn save_mdp(&self, path: &Path, n_steps: usize, dt: f32) -> io::Result<()> {
        self.to_gromacs(n_steps, dt).save(path)
    }

    pub fn load_from_mdp(path: &Path) -> io::Result<Self> {
        Ok(MdpParams::load(path)?.into())
    }
}

impl From<MdpParams> for MdConfig {
    fn from(p: MdpParams) -> Self {
        use crate::thermostat::LANGEVIN_GAMMA_DEFAULT;

        // Mirror of to_gromacs(): Sd → LangevinMiddle, everything else → VerletVelocity.
        let integrator = match p.integrator {
            MdpIntegrator::Sd => {
                let tau = p.tau_t.first().copied().unwrap_or(1.0);
                let gamma = if tau > 0.0 {
                    1.0 / tau
                } else {
                    LANGEVIN_GAMMA_DEFAULT
                };
                Integrator::LangevinMiddle { gamma }
            }
            _ => {
                let thermostat = if p.thermostat != Thermostat::No {
                    Some(p.tau_t.first().copied().unwrap_or(TAU_TEMP_DEFAULT as f32) as f64)
                } else {
                    None
                };
                Integrator::VerletVelocity { thermostat }
            }
        };

        // Extract shake tolerance from the constraint algorithm, falling back to the default.
        let shake_tolerance = match p.constraint_algorithm {
            ConstraintAlgorithm::Shake { tol } => tol,
            ConstraintAlgorithm::Lincs { .. } => 0.0001, // SHAKE default tolerance
        };
        let hydrogen_constraint = match p.constraints {
            Constraints::HBonds | Constraints::AllBonds => {
                HydrogenConstraint::Constrained { shake_tolerance }
            }
            Constraints::None => HydrogenConstraint::Flexible,
        };

        let mut overrides = MdOverrides::default();
        overrides.baro_disabled = p.pcoupl == Barostat::No;

        // Use nstlog as the snapshot cadence — it's the natural output ratio.
        let snapshot_handlers = if p.nstlog > 0 {
            vec![SnapshotHandler {
                save_type: SaveType::Memory,
                ratio: p.nstlog as usize,
            }]
        } else {
            MdConfig::default().snapshot_handlers
        };

        const NM_TO_ANGSTROM: f32 = 10.0;
        let (mesh_spacing, spme_alpha) = match &p.coulombtype {
            CoulombType::Pme(pme) => {
                let spacing = pme.fourierspacing * NM_TO_ANGSTROM;
                let rc = p.rcoulomb * NM_TO_ANGSTROM;
                let alpha = erfc_inv_approx(pme.rtol) / rc;
                (spacing, alpha)
            }
            _ => (1.0, 0.35),
        };

        Self {
            integrator,
            zero_com_drift: false,
            temp_target: p.ref_t.first().copied().unwrap_or(300.0),
            pressure_target: p.ref_p,
            tau_pressure: p.tau_p,
            hydrogen_constraint,
            snapshot_handlers,
            sim_box: Default::default(),
            solvent: Default::default(),
            max_init_relaxation_iters: Some(1_000),
            neighbor_skin: 4.0,
            overrides,
            water_template_path: None,
            skip_water_pbc_filter: false,
            energy_minimization: p.energy_minimization.unwrap_or_default(),
            spme_mesh_spacing: mesh_spacing,
            spme_alpha,
        }
    }
}
