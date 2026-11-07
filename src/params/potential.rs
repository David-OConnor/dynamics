//! Whole-system potentials that own bonded and long-range evaluation.

use std::fmt::Debug;

use lin_alg::f32::Vec3;

use crate::{AtomDynamics, MdConfig, MolDynamics, ParamError, SimBox};

#[derive(Debug)]
pub struct EnergyComponent {
    pub name: String,
    /// kcal/mol.
    pub energy: f64,
}

#[derive(Debug)]
pub struct PotentialEvaluation {
    /// kcal/mol; forces are -dE/dr in kcal/mol/Å.
    pub energy: f64,
    pub forces: Vec<Vec3>,
    pub charges: Option<Vec<f32>>,
    pub components: Vec<EnergyComponent>,
}

pub trait SystemPotential: Debug + Send {
    fn evaluate(&mut self, positions: &[Vec3]) -> Result<PotentialEvaluation, ParamError>;
}

pub trait SystemPotentialBackend: Debug + Send + Sync {
    /// Reject unsupported boundary, solvent and constraint settings before initialization.
    fn validate_config(&self, config: &MdConfig) -> Result<(), ParamError>;

    fn prepare(&self, molecule: &MolDynamics) -> Result<MolDynamics, ParamError>;

    fn create(
        &self,
        atoms: &[AtomDynamics],
        cell: &SimBox,
        molecules: &[MolDynamics],
    ) -> Result<Box<dyn SystemPotential>, ParamError>;
}
