use std::{collections::HashMap, path::PathBuf};

use bio_files_rs;
use pyo3::{prelude::*, types::PyType};

use crate::{AtomGeneric, BondGeneric, ChainGeneric, ResidueGeneric, map_io};

#[pyclass(module = "bio_files")]
pub struct Sdf {
    inner: bio_files_rs::Sdf,
}

#[pymethods]
impl Sdf {
    #[getter]
    fn ident(&self) -> &str {
        &self.inner.ident
    }

    #[getter]
    fn metadata(&self) -> &HashMap<String, String> {
        &self.inner.metadata
    }

    #[getter]
    fn atoms(&self) -> Vec<AtomGeneric> {
        self.inner
            .atoms
            .iter()
            .map(|a| AtomGeneric { inner: a.clone() })
            .collect()
    }

    #[getter]
    fn bonds(&self) -> Vec<BondGeneric> {
        self.inner
            .bonds
            .iter()
            .cloned()
            .map(|b| BondGeneric { inner: b.clone() })
            .collect()
    }

    #[getter]
    fn chains(&self) -> Vec<ChainGeneric> {
        self.inner
            .chains
            .iter()
            .map(|c| ChainGeneric { inner: c.clone() })
            .collect()
    }

    #[getter]
    fn residues(&self) -> Vec<ResidueGeneric> {
        self.inner
            .residues
            .iter()
            .cloned()
            .map(|r| ResidueGeneric { inner: r.clone() })
            .collect()
    }

    #[getter]
    fn pubchem_cid(&self) -> Option<u32> {
        self.inner.pubchem_cid.clone()
    }

    #[getter]
    fn drugbank_id(&self) -> Option<String> {
        self.inner.drugbank_id.clone()
    }

    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::Sdf::new(text))?,
        })
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        map_io(self.inner.save(&path))
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::Sdf::load(&path))?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
