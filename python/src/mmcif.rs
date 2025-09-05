use std::{collections::HashMap, path::PathBuf};

use bio_files_rs;
use pyo3::{prelude::*, types::PyType};

use crate::{
    AtomGeneric, BackboneSS, BondGeneric, ChainGeneric, ExperimentalMethod, ResidueGeneric, map_io,
};

#[pyclass(module = "bio_files")]
pub struct MmCif {
    inner: bio_files_rs::MmCif,
}

//     pub secondary_structure: Vec<BackboneSS>,
//     pub experimental_method: Option<ExperimentalMethod>,
#[pymethods]
impl MmCif {
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

    // todo: If implemented in rust.
    // #[getter]
    // fn bonds(&self) -> Vec<BondGeneric> {
    //     self.inner
    //         .bonds
    //         .iter()
    //         .cloned()
    //         .map(|b| BondGeneric { inner: b.clone() })
    //         .collect()
    // }

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

    // todo: String for now
    #[getter]
    fn secondary_structure(&self) -> Vec<String> {
        self.inner.secondary_structure.iter().map(|s| format!("{s:?}")).collect()
    }

    // todo: String for now
    #[getter]
    fn experimental_method(&self) -> Option<String> {
        match self.inner.experimental_method {
            Some(m) => Some(m.to_string()),
            None => None,
        }
    }

    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::MmCif::new(text))?,
        })
    }

    // todo: When implemented in rust.
    // fn save(&self, path: PathBuf) -> PyResult<()> {
    //     map_io(self.inner.save(&path))
    // }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::MmCif::load(&path))?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
