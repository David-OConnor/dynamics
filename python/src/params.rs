use std::str::FromStr;

use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::from_bio_files::{AtomGeneric, ChainGeneric, ResidueGeneric, MmCif};

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
pub fn prepare_peptide(
    py: Python<'_>,
    atoms: Vec<Py<AtomGeneric>>,
    mut bonds: Vec<Py<BondGeneric>>, // we’ll return fresh BondGeneric objects instead of trying to edit this list
    residues: Vec<Py<ResidueGeneric>>,
    chains: Vec<Py<ChainGeneric>>,
    ff_map: ProtFFTypeChargeMap,
    ph: f32,
) -> PyResult<(Vec<BondGeneric>, Vec<Dihedral>)> {
    // Move out inners
    let mut atoms_inner: Vec<_> = atoms
        .iter()
        .map(|a| {
            let mut b = a.borrow_mut(py);
            std::mem::take(&mut b.inner)
        })
        .collect();

    let mut bonds_inner: Vec<_> = bonds
        .iter()
        .map(|bnd| {
            let mut b = bnd.borrow_mut(py);
            std::mem::take(&mut b.inner)
        })
        .collect();

    let mut residues_inner: Vec<_> = residues
        .iter()
        .map(|r| {
            let mut b = r.borrow_mut(py);
            std::mem::take(&mut b.inner)
        })
        .collect();

    let mut chains_inner: Vec<_> = chains
        .iter()
        .map(|c| {
            let mut b = c.borrow_mut(py);
            std::mem::take(&mut b.inner)
        })
        .collect();

    // Run the pure-Rust routine
    let dihedrals = dynamics_rs::params::prepare_peptide(
        &mut atoms_inner,
        &mut bonds_inner,
        &mut residues_inner,
        &mut chains_inner,
        &ff_map.inner,
        ph,
    )
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("{e:?}")))?;

    // Write results back into the *same* Python objects
    for (obj, new_inner) in atoms.iter().zip(atoms_inner.into_iter()) {
        obj.borrow_mut(py).inner = new_inner;
    }
    for (obj, new_inner) in residues.iter().zip(residues_inner.into_iter()) {
        obj.borrow_mut(py).inner = new_inner;
    }
    for (obj, new_inner) in chains.iter().zip(chains_inner.into_iter()) {
        obj.borrow_mut(py).inner = new_inner;
    }

    // Bonds may have been created/removed; return fresh wrappers instead of trying to resize the passed list.
    let bonds_out: Vec<BondGeneric> = bonds_inner.into_iter().map(|inner| BondGeneric { inner }).collect();

    Ok((bonds_out, dihedrals))
}


#[pyfunction]
pub fn prepare_peptide_mmcif(
    py: Python<'_>,
    mol: Bound<'_, PyCell<Mmcif>>,
    ff_map: ProtFFTypeChargeMap,
    ph: f32,
) -> PyResult<(Vec<BondGeneric>, Vec<Dihedral>)> {
    let mut mol_b = mol.borrow_mut(py);

    let (bonds, dihedrals) = dynamics_rs::params::prepare_peptide_mmcif(
        &mut mol_b.inner,
        &ff_map.inner,
        ph,
    )
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("{e:?}")))?;

    let bonds_out = bonds.into_iter().map(|inner| BondGeneric { inner }).collect();
    Ok((bonds_out, dihedrals))
}
