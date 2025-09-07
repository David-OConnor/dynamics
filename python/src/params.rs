use std::str::FromStr;

use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::from_bio_files::{AtomGeneric, ResidueGeneric};

#[pyclass]
#[derive(Clone)]
struct ParamGeneralPaths {
    pub inner: dynamics_rs::params::ParamGeneralPaths,
}
// todo: Set up this constructor, or this won't work

#[pyclass]
pub struct FfParamSet {
    pub inner: dynamics_rs::params::FfParamSet,
}

#[pymethods]
impl FfParamSet {
    #[new]
    fn new(paths: ParamGeneralPaths) -> PyResult<Self> {
        Ok(Self {
            inner: dynamics_rs::params::FfParamSet::new(&paths.inner)?,
        })
    }
}

#[pyclass]
struct ProtFFTypeChargeMap {
    inner: dynamics_rs::ProtFFTypeChargeMap,
}

// // todo: Impl after converting the atom and residue types.
// #[pyfunction]
// fn populate_peptide_ff_and_q(
//     atoms: &[AtomGeneric],
//     residues: &[ResidueGeneric],
//     ff_type_charge: &ProtFFTypeChargeMap,
// ) -> Result<(), ParamError> {
//     dynamics_rs::populate_peptide_ff_and_q(atoms, residues, &ff_type_charge.inner)
// }
