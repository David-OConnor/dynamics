use std::str::FromStr;

use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::from_bio_files::{AtomGeneric, ChainGeneric, ResidueGeneric};

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

    #[classmethod]
    fn new_amber(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Ok(Self {
            inner: dynamics_rs::params::FfParamSet::new_amber()?,
        })
    }

    #[getter]
    fn peptide_ff_q_map(&self) -> Option<ProtFFTypeChargeMap> {
        match self.inner.peptide_ff_q_map.clone() {
            Some(v) => Some(ProtFFTypeChargeMap { inner: v }),
            None => None,
        }
    }
    #[setter(peptide_ff_q_map)]
    fn peptide_ff_q_map_set(&mut self, v: ProtFFTypeChargeMap) {
        self.inner.peptide_ff_q_map = Some(v.inner);
    }

    // pub peptide: Option<ForceFieldParams>,
    // pub small_mol: Option<ForceFieldParams>,
    // pub dna: Option<ForceFieldParams>,
    // pub rna: Option<ForceFieldParams>,
    // pub lipids: Option<ForceFieldParams>,
    // pub carbohydrates: Option<ForceFieldParams>,
    // /// In addition to charge, this also contains the mapping of res type to FF type; required to map
    // /// other parameters to protein atoms. E.g. from `amino19.lib`, and its N and C-terminus variants.
    // pub peptide_ff_q_map: Option<ProtFFTypeChargeMap>,
}

#[derive(Clone)]
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

#[pyfunction]
// Different from the inner version. Doesn't mutate in-place; returns new values.
// todo: for now, atoms only. And doesn't return dihedrals.
pub fn prepare_peptide(
    py: Python<'_>,
    atoms: Vec<Py<AtomGeneric>>,
    residues: Vec<Py<ResidueGeneric>>,
    chains: Vec<Py<ChainGeneric>>,
    ff_map: ProtFFTypeChargeMap,
    ph: f32,
) -> PyResult<Vec<AtomGeneric>> {
    let mut atoms: Vec<_> = atoms
        .into_iter()
        .map(|a| a.borrow(py).inner.clone())
        .collect();

    let mut residues: Vec<_> = residues
        .into_iter()
        .map(|r| r.borrow(py).inner.clone())
        .collect();

    let mut chains: Vec<_> = chains
        .into_iter()
        .map(|c| c.borrow(py).inner.clone())
        .collect();

    dynamics_rs::params::prepare_peptide(&mut atoms, &mut residues, &mut chains, &ff_map.inner, ph)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("{e:?}")))?;

    let atoms_res = atoms
        .into_iter()
        .map(|a| AtomGeneric { inner: a })
        .collect();
    Ok(atoms_res)
}
