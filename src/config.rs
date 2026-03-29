use std::{io, path::Path};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    gromacs,
    gromacs::mdp::{
        Barostat, BarostatCfg, ConstraintAlgorithm, Constraints, CoulombType,
        Integrator as MdpIntegrator, MdpParams, Pbc, PmeConfig, PressureCouplingType, Thermostat,
        VdwType,
    },
};

use bio_files::gromacs::mdp::VdwModifier;

use crate::{
    MdOverrides, SimBoxInit, integrate::Integrator, prep::HydrogenConstraint,
    snapshot::SnapshotHandlers, solvent::Solvent, thermostat::TAU_TEMP_DEFAULT,
};

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
    pub snapshot_handlers: SnapshotHandlers,
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
    // /// We use the GROMACS struct unaltered.
    // pub energy_minimization: EnergyMinimization,
    /// SPME mesh spacing in **Å**. Smaller = higher-resolution reciprocal mesh, more accurate
    /// but slower. 1.0 Å is a good default; equivalent to GROMACS `fourierspacing = 0.1 nm`.
    pub spme_mesh_spacing: f32,
    /// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
    /// reciprocal load.
    /// Common rule for α: erfc(α r_c) ≲ 10⁻⁴…10⁻⁵
    /// Å^-1. 0.35 is good for cutoff of 10–12 Å.
    pub spme_alpha: f32,
    /// The distance at which we cut off short-range (Direct) Coulomb operations, and transtion
    /// to SPME reciprical forces. Å
    pub coulomb_cutoff: f32,
    /// A hard distance cutoff for VDW forces. Å
    pub lj_cutoff: f32,
    /// Simliar to GROMACS' emtol. kcal mol⁻¹ Å⁻¹
    pub energy_minimization_tolerance: f32,
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
            snapshot_handlers: Default::default(),
            sim_box: Default::default(),
            solvent: Default::default(),
            max_init_relaxation_iters: Some(1_000), // todo: A/R
            neighbor_skin: 4.0,
            overrides: Default::default(),
            water_template_path: None,
            skip_water_pbc_filter: false,
            // energy_minimization: Default::default(),
            spme_mesh_spacing: 1.0,
            // Å⁻¹. Chosen so erfc(α × r_c) ≈ 1e-5 at r_c = 12 Å, matching GROMACS' default
            // ewald-rtol. (0.35 Å⁻¹ gives ~2.9e-9 — accurate but pushes k-space costs up.)
            spme_alpha: 0.26,
            coulomb_cutoff: 10.,
            lj_cutoff: 10.,
            // GROMACS nstcgsteep default.
            energy_minimization_tolerance: 0.2390,
        }
    }
}

impl MdConfig {
    /// Creates a similar config for use with GROMACS. Attempts to replicate this library's
    /// settings where we don't have an applicable MdConfig field.
    ///
    /// Not `From` trait: This lib depends on `bio_files`, but not the other way around.
    pub fn to_gromacs(&self, n_steps: usize, dt: f32) -> MdpParams {
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
            Integrator::Leapfrog { thermostat } => {
                let tau = if let Some(t) = thermostat {
                    t as f32
                } else {
                    TAU_TEMP_DEFAULT as f32
                };
                (
                    gromacs::mdp::Integrator::Md,
                    tau,
                    gromacs::mdp::Thermostat::VRescale,
                )
            }
        };

        let constraints = match self.hydrogen_constraint {
            // LINCS maps to the same GROMACS h-bonds constraint type; GROMACS selects LINCS
            // automatically when constraints are enabled with the `md` integrator.
            HydrogenConstraint::Linear { order, iter } => {
                Constraints::HBonds(ConstraintAlgorithm::Lincs { order, iter })
            }
            HydrogenConstraint::Shake { shake_tolerance } => {
                Constraints::HBonds(ConstraintAlgorithm::Shake {
                    tol: shake_tolerance,
                })
            }
            HydrogenConstraint::Flexible => Constraints::None,
        };

        let pcoupl = if self.overrides.bonded_disabled {
            gromacs::mdp::Barostat::No
        } else {
            gromacs::mdp::Barostat::CRescale(BarostatCfg {
                // Standard water compressibility
                pcoupltype: PressureCouplingType::Isotropic {
                    ref_p: self.pressure_target,
                    compressibility: 4.5e-5,
                },
                tau_p: self.tau_pressure,
            })
        };

        const ANGSTROM_TO_NM: f32 = 0.1;

        // Many of these values are the MdpParams defaults, but we specify here
        // to be explicit, and not miss any.
        MdpParams {
            integrator,
            nsteps: n_steps as u64,
            dt,
            output_control: self.snapshot_handlers.gromacs.clone(),
            coulombtype: CoulombType::Pme(PmeConfig {
                fourierspacing: self.spme_mesh_spacing * ANGSTROM_TO_NM,
                order: 4, // Hard-coded in `ewald`.
                // Convert internal Å⁻¹ to GROMACS nm⁻¹ (1 Å⁻¹ = 10 nm⁻¹).
                // PmeConfig::make_inp() derives ewald-rtol = erfc(alpha × rcoulomb).
                alpha: self.spme_alpha / ANGSTROM_TO_NM,
                ..Default::default()
            }),
            rcoulomb: self.coulomb_cutoff * ANGSTROM_TO_NM,
            vdwtype: VdwType::CutOff,
            vdw_modifier: VdwModifier::default(),
            rvdw: self.lj_cutoff * ANGSTROM_TO_NM,
            thermostat,
            // We only have one temperature-coupling group in this lib.
            tau_t: vec![tau_t],
            ref_t: vec![self.temp_target],
            pcoupl,
            pbc: Pbc::Xyz,
            gen_vel: true,
            gen_temp: self.temp_target,
            gen_seed: None,
            constraints: self.hydrogen_constraint.to_gromacs(),
            // energy_minimization: Some(self.energy_minimization.clone()),
            free_energy_calculations: Default::default(),
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

        let hydrogen_constraint = match p.constraints {
            Constraints::HBonds(ca) => match ca {
                ConstraintAlgorithm::Lincs { order, iter } => {
                    HydrogenConstraint::Linear { order, iter }
                }
                ConstraintAlgorithm::Shake { tol } => HydrogenConstraint::Shake {
                    shake_tolerance: tol,
                },
            },
            Constraints::None => HydrogenConstraint::Flexible,
            _ => {
                eprintln!("Can't use this GROMACS' bond constraint; reverting to a safe default");
                Default::default()
            }
        };

        let mut overrides = MdOverrides::default();
        overrides.baro_disabled = p.pcoupl == Barostat::No;

        // Use nstlog as the snapshot cadence — it's the natural output ratio.
        let nstlog = p.output_control.nstlog.unwrap_or(0);

        let snapshot_handlers = SnapshotHandlers {
            memory: None,
            dcd: None,
            gromacs: p.output_control.clone(),
        };

        const NM_TO_ANGSTROM: f32 = 10.0;

        let (mesh_spacing, spme_alpha) = match &p.coulombtype {
            CoulombType::Pme(pme) => {
                let spacing = pme.fourierspacing * NM_TO_ANGSTROM;
                // Convert nm⁻¹ back to Å⁻¹ (1 nm⁻¹ = 0.1 Å⁻¹).
                let alpha = pme.alpha / NM_TO_ANGSTROM;
                (spacing, alpha)
            }
            _ => (1.0, 0.35),
        };

        let (pressure_target, tau_pressure) = match p.pcoupl {
            Barostat::No => (0.0, 0.0),
            Barostat::Berendsen(v)
            | Barostat::CRescale(v)
            | Barostat::ParrinelloRahman(v)
            | Barostat::Mtkk(v) => {
                let ref_p = match v.pcoupltype {
                    PressureCouplingType::Isotropic {
                        ref_p,
                        compressibility: _,
                    } => ref_p,
                    _ => {
                        eprintln!(
                            "Unsupported GROMACS pressure coupling type; reverting to a default"
                        );
                        300.
                    }
                };

                (ref_p, v.tau_p)
            }
        };

        let def = Self::default();

        Self {
            integrator,
            zero_com_drift: false,
            temp_target: p.ref_t.first().copied().unwrap_or(300.0),
            pressure_target,
            tau_pressure,
            hydrogen_constraint,
            snapshot_handlers,
            sim_box: Default::default(),
            solvent: Default::default(),
            max_init_relaxation_iters: def.max_init_relaxation_iters,
            neighbor_skin: def.neighbor_skin,
            overrides,
            water_template_path: None,
            skip_water_pbc_filter: false,
            // energy_minimization: p.energy_minimization.unwrap_or_default(),
            spme_mesh_spacing: mesh_spacing,
            spme_alpha,
            coulomb_cutoff: p.rcoulomb * NM_TO_ANGSTROM,
            lj_cutoff: p.rvdw * NM_TO_ANGSTROM,
            energy_minimization_tolerance: def.energy_minimization_tolerance,
        }
    }
}
