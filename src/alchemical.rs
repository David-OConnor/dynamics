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
//!    with respect to λ. For linear coupling this equals minus the solute–solvent
//!    interaction energy accumulated during the non-bonded force calculation.
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
//! pair list so cross interactions with the alchemical molecule are scaled by
//! `(1 − λ)`.
//!
//! # Soft-core potentials
//! [GROMACS docs](https://manual.gromacs.org/nightly/reference-manual/functions/free-energy-interactions.html#soft-core-interactions-beutler-et-al)
//!
//! Near λ = 0 or 1, the simple linear LJ coupling diverges when two atoms overlap.
//! We Replace linear LJ scaling with a soft-core potential, from Beutler et al. (1994),
//! similar to GROMACS' approach:
//!
//! ```text
//! U_sc(r, λ) = 4·ε·λ · [ 1/(α(1−λ)² + (r/σ)⁶)² − 1/(α(1−λ)² + (r/σ)⁶) ]
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

const GAS_CONST_R_KCAL: f64 = 0.001_987_204_1; // kcal / (mol · K)

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

/// Compute **LogP** from free energies in solvent and octanol.
///
/// Both `dg_water` and `dg_octanol` should be the decoupling free energies
/// (ΔG for turning off solute–solvent interactions), in kcal/mol, obtained from
/// [`free_energy_ti`] run in each solvent.
///
/// ```text
/// LogP = (ΔG_octanol − ΔG_water) / (2.303 · R · T)
/// ```
///
/// `temperature_k` is the simulation temperature in Kelvin (typically 298.15 K).
///
/// A positive LogP means the solute prefers octanol (lipophilic), i.e. the
/// decoupling free energy is larger in octanol than in water.
///
/// # Panics
pub fn log_p(dg_water: f64, dg_octanol: f64, temperature_k: f64) -> Result<f64, AlchemicalError> {
    let rt = GAS_CONST_R_KCAL * temperature_k;
    Ok((dg_octanol - dg_water) / (2.302_585_093 * rt))
}

impl MdState {
    /// Currently configured alchemical molecule index, if any.
    pub fn alchemical_molecule_index(&self) -> Option<usize> {
        self.alch_mol_idx
    }

    /// Current alchemical λ value.
    pub fn alchemical_lambda(&self) -> f64 {
        self.lambda_alch
    }

    /// Enable alchemical decoupling for one molecule at a fixed λ value.
    ///
    /// This is the preferred setup call for each TI window. It validates the
    /// molecule index, stores the λ value, clears cached reciprocal data, and
    /// rebuilds non-bonded pairs so cross interactions with the selected molecule
    /// are marked for `(1 - λ)` scaling.
    pub fn configure_alchemical_window(
        &mut self,
        dev: &ComputationDevice,
        mol_idx: usize,
        lambda: f64,
    ) -> Result<(), AlchemicalError> {
        let mol_count = self.mol_start_indices.len();
        if mol_idx >= mol_count {
            return Err(AlchemicalError::InvalidMoleculeIndex { mol_idx, mol_count });
        }

        self.alch_mol_idx = Some(mol_idx);
        self.lambda_alch = lambda;
        self.alch_interaction_energy = 0.0;
        self.spme_force_prev = None;
        self.build_all_neighbors(dev);

        Ok(())
    }

    /// Update λ for the currently configured alchemical molecule.
    ///
    /// The pair list does not need to be rebuilt when only λ changes.
    pub fn set_alchemical_lambda(&mut self, lambda: f64) -> Result<(), AlchemicalError> {
        if self.alch_mol_idx.is_none() {
            return Err(AlchemicalError::AlchemicalMoleculeNotSet);
        }

        self.lambda_alch = lambda;
        self.alch_interaction_energy = 0.0;
        self.spme_force_prev = None;

        Ok(())
    }

    /// Disable alchemical scaling and rebuild non-bonded pairs.
    pub fn clear_alchemical_window(&mut self, dev: &ComputationDevice) {
        self.alch_mol_idx = None;
        self.lambda_alch = 0.0;
        self.alch_interaction_energy = 0.0;
        self.spme_force_prev = None;
        self.build_all_neighbors(dev);
    }
}
