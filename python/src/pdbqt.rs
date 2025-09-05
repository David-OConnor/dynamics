use std::path::PathBuf;

use bio_files_rs;
use pyo3::{prelude::*, types::PyType};

use crate::map_io;

#[pyclass(module = "bio_files")]
pub struct Pdbqt {
    inner: bio_files_rs::Pdbqt,
}

#[pymethods]
impl Pdbqt {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::Pdbqt::new(text))?,
        })
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        map_io(self.inner.save(&path))
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::Pdbqt::load(&path))?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
