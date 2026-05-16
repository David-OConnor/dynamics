//! Alchemical free-energy calculations for high-level solvation and LogP workflows.
//! This computes a result after varying simulation interactions between solute and solvent,
//! and measuring potential energy at each variation.
//!
//! # Overview
//!
//! Solvation free energies and partition coefficients can be estimated from
//! alchemical free-energy simulations using thermodynamic integration (TI):
//!
//! - Run **N separate MD simulations** at different λ values (e.g., 0.0, 0.05, ... 1.0),
//!     λ = 0 means the solute interacts normally with the solvent; λ = 1 means it is fully decoupled.
//!
//! - At each simulation frame, record **∂H/∂λ** — the derivative of the Hamiltonian
//!    with respect to λ, accumulated during the non-bonded force calculation.
//!    This is stored per frame in [`crate::snapshot::SnapshotEnergyData::dh_dl`].
//!
//! - Average ⟨∂H/∂λ⟩ over each λ window's trajectory into a [`LambdaWindow`].
//!
//! - Compute **ΔG = ∫₀¹ ⟨∂H/∂λ⟩_λ dλ** via [`free_energy_ti`].
//!
//! # Running a λ window
//!
//! Use [`MdState::configure_alchemical_window`] before running each λ window. It
//! validates the molecule index and λ value and rebuilds the cached non-bonded
//! pair list so cross interactions with the alchemical molecule are marked for
//! alchemical handling.
//!
//! # Soft-core potentials
//! [GROMACS docs](https://manual.gromacs.org/nightly/reference-manual/functions/free-energy-interactions.html#soft-core-interactions-beutler-et-al)
//!
//! Near λ = 0 or 1, simple linear LJ coupling gives noisy endpoint derivatives
//! when two atoms overlap. For alchemical cross interactions we replace linear
//! LJ scaling with the Beutler et al. soft-core form used by GROMACS:
//!
//! ```text
//! V_sc(r, λ) = (1 - λ) V_LJ(r_A)
//! r_A = (r^6 + α σ_sc^6 λ^p)^(1/6)
//! ```
//!
//! The electrostatic coupling can remain linear; switch it off before LJ to avoid
//! charge–charge singularities.
//!
//!
use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

use crate::{ComputationDevice, MdState, snapshot::Snapshot};

/// Beutler/GROMACS-style LJ soft-core alpha used for alchemical decoupling.
///
/// GROMACS exposes this as `sc-alpha`; common LJ decoupling setups use `0.5`.
pub const SOFT_CORE_ALPHA: f32 = 0.5;

/// Lambda power `p` in `r_A = (r^6 + alpha * sigma^6 * lambda^p)^(1/6)`.
///
/// GROMACS supports 1 and 2; 1 is the modern smoother default.
pub const SOFT_CORE_POWER: i32 = 1;

/// Minimum soft-core sigma in Angstrom.
///
/// GROMACS' default `sc-sigma` is 0.3 nm, which is 3.0 Angstrom.
pub const SOFT_CORE_SIGMA_MIN: f32 = 3.0;

#[derive(Default)]
pub struct StateAlchemical {
    /// Index into `mol_start_indices` of the molecule being alchemically decoupled.
    ///
    /// When `Some(m)`, `take_snapshot` computes ∂H/∂λ for molecule m and stores it
    /// in each `Snapshot::dh_dl`.  Set `None` (the default) for ordinary MD.
    pub mol_idx: Option<usize>,
    /// Current lambda value for alchemical simulations, in [0, 1].
    ///
    /// λ = 0: solute fully coupled; λ = 1: solute fully decoupled.
    /// For thermodynamic integration, hold this fixed for the duration of one
    /// simulation window and sweep across multiple windows.
    pub lambda: f64,
    /// Instantaneous alchemical non-bonded derivative for the current step, in kcal/mol.
    ///
    /// This includes soft-core short-range LJ contributions and linearly scaled
    /// Coulomb/SPME cross contributions for the configured alchemical molecule.
    pub dh_dl: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AlchemicalError {
    EmptySnapshots,
    MissingDhDl,
    InvalidLambda(f64),
    InvalidTemperature(f64),
    InvalidFreeEnergy(f64),
    NotEnoughWindows(usize),
    NonFiniteMeanDhDl { lambda: f64, mean_dh_dl: f64 },
    UnsortedWindows { previous: f64, next: f64 },
    InvalidMoleculeIndex { mol_idx: usize, mol_count: usize },
    AlchemicalMoleculeNotSet,
    UnsupportedDevice(&'static str),
}

impl Display for AlchemicalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySnapshots => write!(f, "snapshot slice is empty"),
            Self::MissingDhDl => write!(f, "no dh/dlambda values were recorded in the snapshots"),
            Self::InvalidLambda(lambda) => {
                write!(f, "lambda must be finite and in [0, 1], got {lambda}")
            }
            Self::InvalidTemperature(temperature) => {
                write!(
                    f,
                    "temperature must be finite and positive, got {temperature}"
                )
            }
            Self::InvalidFreeEnergy(dg) => write!(f, "free energy must be finite, got {dg}"),
            Self::NotEnoughWindows(n) => {
                write!(f, "at least two lambda windows are required, got {n}")
            }
            Self::NonFiniteMeanDhDl { lambda, mean_dh_dl } => write!(
                f,
                "mean dh/dlambda must be finite for lambda {lambda}, got {mean_dh_dl}"
            ),
            Self::UnsortedWindows { previous, next } => write!(
                f,
                "lambda windows must be strictly increasing, got {previous} followed by {next}"
            ),
            Self::InvalidMoleculeIndex { mol_idx, mol_count } => write!(
                f,
                "alchemical molecule index {mol_idx} is out of range for {mol_count} molecules"
            ),
            Self::AlchemicalMoleculeNotSet => {
                write!(f, "no alchemical molecule has been configured")
            }
            Self::UnsupportedDevice(message) => write!(f, "{message}"),
        }
    }
}

impl Error for AlchemicalError {}

/// Data collected from one MD run at a single fixed λ value.
///
/// Build this from a trajectory slice via [`collect_window`].
#[derive(Clone, Debug)]
pub struct LambdaWindow {
    /// The fixed λ value for this simulation window, in [0, 1].
    pub lambda: f64,
    /// Mean ∂H/∂λ for this window in kcal/mol, averaged over all trajectory frames.
    pub mean_dh_dl: f64,
    /// Standard error of the mean (None if fewer than 2 frames).
    pub sem_dh_dl: Option<f64>,
}

/// Build a [`LambdaWindow`] from a slice of snapshots taken at one fixed λ value.
///
/// All snapshots must come from the same λ window. Uses the `dh_dl` field that
/// is recorded when snapshot energy data is written.
///
/// Snapshots without energy data are ignored, which is useful for trajectories
/// that contain positions more frequently than energies. An error is returned
/// when no usable `dh_dl` samples are present.
pub fn collect_window(
    lambda: f64,
    snapshots: &[Snapshot],
) -> Result<LambdaWindow, AlchemicalError> {
    if snapshots.is_empty() {
        return Err(AlchemicalError::EmptySnapshots);
    }

    let dh_dl: Vec<f64> = snapshots
        .iter()
        .filter_map(|s| s.energy_data.as_ref()?.dh_dl.map(f64::from))
        .collect();

    if dh_dl.is_empty() {
        return Err(AlchemicalError::MissingDhDl);
    }

    for &value in &dh_dl {
        if !value.is_finite() {
            return Err(AlchemicalError::NonFiniteMeanDhDl {
                lambda,
                mean_dh_dl: value,
            });
        }
    }

    let n = dh_dl.len() as f64;
    let mean = dh_dl.iter().sum::<f64>() / n;

    let sem = if dh_dl.len() > 1 {
        let variance = dh_dl.iter().map(|v| (*v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Some((variance / n).sqrt())
    } else {
        None
    };

    Ok(LambdaWindow {
        lambda,
        mean_dh_dl: mean,
        sem_dh_dl: sem,
    })
}

/// Compute ΔG via **Thermodynamic Integration** (TI) using the trapezoidal rule.
///
/// ΔG = ∫₀¹ ⟨∂H/∂λ⟩_λ dλ ≈ Σᵢ ½(⟨∂H/∂λ⟩ᵢ + ⟨∂H/∂λ⟩ᵢ₊₁) · Δλᵢ
///
/// `windows` must be sorted by `lambda` in ascending order and span [0, 1]
/// (or whatever range was simulated — partial ranges give partial ΔG).
///
/// Returns ΔG in **kcal/mol**.  A positive value means decoupling costs energy
/// (solute prefers the solvent); a negative value means it is favourable to remove.
///
/// # Panics
pub fn free_energy_ti(windows: &[LambdaWindow]) -> Result<f64, AlchemicalError> {
    if windows.len() < 2 {
        return Err(AlchemicalError::NotEnoughWindows(windows.len()));
    }

    for window in windows {
        if !window.mean_dh_dl.is_finite() {
            return Err(AlchemicalError::NonFiniteMeanDhDl {
                lambda: window.lambda,
                mean_dh_dl: window.mean_dh_dl,
            });
        }
    }

    for pair in windows.windows(2) {
        if pair[1].lambda <= pair[0].lambda {
            return Err(AlchemicalError::UnsortedWindows {
                previous: pair[0].lambda,
                next: pair[1].lambda,
            });
        }
    }

    Ok(windows
        .windows(2)
        .map(|w| 0.5 * (w[0].mean_dh_dl + w[1].mean_dh_dl) * (w[1].lambda - w[0].lambda))
        .sum())
}

// todo: Other than the build_all_neighbors call, this would make more sense as a
// todo: method on AlchemicalState
impl MdState {
    /// Enable alchemical decoupling for one molecule at a fixed λ value. This is the
    /// entry point to enable an alchemical computation for an MD run. The application
    /// sequences the λ values by setting up MD runs, and calling this with the appropraite
    /// λ.
    ///
    /// It validates the molecule index, stores the λ value, clears cached reciprocal data, and
    /// rebuilds non-bonded pairs so cross interactions with the selected molecule
    /// use alchemical LJ/Coulomb force handling.
    pub fn configure_alchemical_window(
        &mut self,
        dev: &ComputationDevice,
        mol_idx: usize,
        lambda: f64,
    ) -> Result<(), AlchemicalError> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(AlchemicalError::InvalidLambda(lambda));
        }

        let mol_count = self.mol_start_indices.len();
        if mol_idx >= mol_count {
            return Err(AlchemicalError::InvalidMoleculeIndex { mol_idx, mol_count });
        }

        self.alchemical.mol_idx = Some(mol_idx);

        self.alchemical.lambda = lambda;
        self.alchemical.dh_dl = 0.0;
        self.spme_force_prev = None;

        // todo: Why is this here?
        self.build_all_neighbors(dev);

        Ok(())
    }

    /// Disable alchemical scaling and rebuild non-bonded pairs.
    pub fn clear_alchemical_window(&mut self, dev: &ComputationDevice) {
        self.alchemical.mol_idx = None;
        self.alchemical.lambda = 0.0;
        self.alchemical.dh_dl = 0.0;
        self.spme_force_prev = None;

        self.build_all_neighbors(dev);
    }

    pub(crate) fn alchemical_atom_range(&self) -> Option<(usize, usize)> {
        let mol_idx = self.alchemical.mol_idx?;
        let start = *self.mol_start_indices.get(mol_idx)?;

        let end = self
            .mol_start_indices
            .get(mol_idx + 1)
            .copied()
            .unwrap_or(self.atoms.len());

        Some((start, end))
    }
}
