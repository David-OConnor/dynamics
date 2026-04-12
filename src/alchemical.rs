//! Alchemical free energy calculations for LogP estimation.
//!
//! # Overview
//!
//! LogP (the octanol-solvent partition coefficient) can be estimated from alchemical
//! free energy simulations using thermodynamic integration (TI):
//!
//! 1. Run **N separate MD simulations** at different λ values (e.g., 0.0, 0.05, …, 1.0),
//!    **in both solvent and octanol**. λ = 0 means the solute interacts normally with the
//!    solvent; λ = 1 means it is fully decoupled (non-interacting ghost).
//!
//! 2. At each simulation frame, record **∂H/∂λ** — the derivative of the Hamiltonian
//!    with respect to λ. For linear coupling this equals minus the solute–solvent
//!    interaction energy, which is already tracked in `MdState::potential_energy_between_mols`.
//!    This is stored per frame in [`Snapshot::dh_dl`].
//!
//! 3. Average ⟨∂H/∂λ⟩ over each λ window's trajectory into a [`LambdaWindow`].
//!
//! 4. Compute **ΔG = ∫₀¹ ⟨∂H/∂λ⟩_λ dλ** via [`free_energy_ti`].
//!
//! 5. Repeat for octanol and compute **LogP** via [`log_p`].
//!
//! # Force scaling requirement
//!
//! For physically correct intermediate-λ sampling, the non-bonded forces on the
//! alchemical molecule must be scaled by `(1 − λ)` in `apply_nonbonded_forces()`.
//! Without this, each window samples the fully-coupled (λ = 0) ensemble and TI
//! reduces to a single endpoint calculation.  The infrastructure here is ready;
//! add the force scaling in `non_bonded.rs` using `MdState::alch_mol_idx` and
//! `MdState::lambda` to complete the implementation.
//!
//! # Soft-core potentials
//!
//! Near λ = 0 or 1, the simple linear LJ coupling diverges when two atoms overlap.
//! Production simulations should replace linear LJ scaling with a soft-core potential
//! such as Beutler et al. (1994):
//!
//! ```text
//! U_sc(r, λ) = 4·ε·λ · [ 1/(α(1−λ)² + (r/σ)⁶)² − 1/(α(1−λ)² + (r/σ)⁶) ]
//! ```
//!
//! The electrostatic coupling can remain linear; switch it off before LJ to avoid
//! charge–charge singularities.

use crate::snapshot::Snapshot;

const GAS_CONST_R_KCAL: f64 = 0.001_987_204_1; // kcal / (mol · K)

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
/// All snapshots must come from the same λ window.  Uses the `dh_dl` field that
/// is recorded every step by `MdState::take_snapshot`.
///
/// # Panics
/// Panics if `snapshots` is empty.
pub fn collect_window(lambda: f64, snapshots: &[Snapshot]) -> LambdaWindow {
    assert!(
        !snapshots.is_empty(),
        "collect_window: empty snapshot slice"
    );

    let dh_dl: Vec<f64> = snapshots
        .iter()
        .filter_map(|s| s.energy_data.as_ref()?.dh_dl.map(|v| v as f64))
        .collect();

    assert!(
        !dh_dl.is_empty(),
        "collect_window: no dh/dlambda values were recorded in the snapshots"
    );

    let n = dh_dl.len() as f64;
    let mean = dh_dl.iter().sum::<f64>() / n;

    let sem = if dh_dl.len() > 1 {
        let variance = dh_dl.iter().map(|v| (*v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Some((variance / n).sqrt())
    } else {
        None
    };

    LambdaWindow {
        lambda,
        mean_dh_dl: mean,
        sem_dh_dl: sem,
    }
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
pub fn free_energy_ti(windows: &[LambdaWindow]) -> f64 {
    windows
        .windows(2)
        .map(|w| 0.5 * (w[0].mean_dh_dl + w[1].mean_dh_dl) * (w[1].lambda - w[0].lambda))
        .sum()
}

/// Compute **LogP** from free energies in solvent and octanol.
///
/// Both `dg_water` and `dg_octanol` should be the decoupling free energies
/// (ΔG for turning off solute–solvent interactions), in kcal/mol, obtained from
/// [`free_energy_ti`] run in each solvent.
///
/// ```text
/// LogP = (ΔG_water − ΔG_octanol) / (2.303 · R · T)
/// ```
///
/// `temperature_k` is the simulation temperature in Kelvin (typically 298.15 K).
///
/// A positive LogP means the solute prefers octanol (lipophilic).
pub fn log_p(dg_water: f64, dg_octanol: f64, temperature_k: f64) -> f64 {
    let rt = GAS_CONST_R_KCAL * temperature_k;
    (dg_water - dg_octanol) / (2.302_585_093 * rt)
}

#[cfg(test)]
mod tests {
    use crate::snapshot::{Snapshot, SnapshotEnergyData};

    use super::{collect_window, free_energy_ti, log_p};

    fn snapshot_with_dh_dl(value: f32) -> Snapshot {
        Snapshot {
            energy_data: Some(SnapshotEnergyData {
                energy_kinetic: 0.0,
                energy_potential: 0.0,
                energy_potential_between_mols: Vec::new(),
                energy_potential_nonbonded: 0.0,
                energy_potential_bonded: 0.0,
                hydrogen_bonds: Vec::new(),
                temperature: 0.0,
                pressure: 0.0,
                dh_dl: Some(value),
                volume: 0.0,
                density: 0.0,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn collect_window_reads_dh_dl_from_snapshots() {
        let snaps = vec![
            snapshot_with_dh_dl(1.0),
            snapshot_with_dh_dl(2.0),
            snapshot_with_dh_dl(3.0),
        ];

        let window = collect_window(0.5, &snaps);

        assert!((window.mean_dh_dl - 2.0).abs() < 1.0e-8);
        assert!(window.sem_dh_dl.is_some());
    }

    #[test]
    fn free_energy_ti_uses_trapezoidal_rule() {
        let windows = vec![
            super::LambdaWindow {
                lambda: 0.0,
                mean_dh_dl: 2.0,
                sem_dh_dl: None,
            },
            super::LambdaWindow {
                lambda: 0.5,
                mean_dh_dl: 4.0,
                sem_dh_dl: None,
            },
            super::LambdaWindow {
                lambda: 1.0,
                mean_dh_dl: 6.0,
                sem_dh_dl: None,
            },
        ];

        let dg = free_energy_ti(&windows);
        assert!((dg - 4.0).abs() < 1.0e-8);
    }

    #[test]
    fn log_p_uses_expected_sign() {
        let logp = log_p(5.0, 3.0, 298.15);
        assert!(logp > 0.0);
    }
}
