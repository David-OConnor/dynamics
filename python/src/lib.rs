use std::str::FromStr;

use bio_files_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

mod mmcif;
mod mol2;
mod pdbqt;
mod sdf;

fn map_io<T>(r: std::io::Result<T>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass]
struct AtomGeneric {
    inner: bio_files_rs::AtomGeneric,
}

#[pymethods]
impl AtomGeneric {
    #[getter]
    fn serial_number(&self) -> u32 {
        self.inner.serial_number
    }

    #[getter]
    fn posit(&self) -> [f64; 3] {
        self.inner.posit.to_arr()
    }

    #[getter]
    // todo: String for now
    fn element(&self) -> String {
        self.inner.element.to_string()
    }

    #[getter]
    // todo: String for now
    fn type_in_res(&self) -> Option<String> {
        self.inner.type_in_res.as_ref().map(|v| v.to_string())
    }

    #[getter]
    fn force_field_type(&self) -> Option<String> {
        self.inner.force_field_type.clone()
    }

    #[getter]
    fn partial_charge(&self) -> Option<f32> {
        self.inner.partial_charge
    }

    #[getter]
    fn hetero(&self) -> bool {
        self.inner.hetero
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct BondType {
    // todo: Sort out enum variants.
    inner: bio_files_rs::BondType,
}

#[pymethods]
impl BondType {
    fn to_str_sdf(&self) -> String {
        self.inner.to_str_sdf()
    }

    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, str: &str) -> PyResult<Self> {
        Ok(Self {
            inner: map_io(bio_files_rs::BondType::from_str(str))?,
        })
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct BondGeneric {
    inner: bio_files_rs::BondGeneric,
}

#[pymethods]
impl BondGeneric {
    #[getter]
    fn bond_type<'py>(&self, py: Python<'py>) -> PyResult<Py<BondType>> {
        Py::new(
            py,
            BondType {
                inner: self.inner.bond_type,
            },
        )
    }

    #[getter]
    fn atom_0_sn(&self) -> u32 {
        self.inner.atom_0_sn
    }

    #[getter]
    fn atom_1_sn(&self) -> u32 {
        self.inner.atom_1_sn
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
// todo: Enum fields
struct ResidueType {
    inner: bio_files_rs::ResidueType,
}

#[pymethods]
impl ResidueType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, str: &str) -> Self {
        Self {
            inner: bio_files_rs::ResidueType::from_str(str),
        }
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct ResidueGeneric {
    inner: bio_files_rs::ResidueGeneric,
}

#[pymethods]
impl ResidueGeneric {
    #[getter]
    fn serial_number(&self) -> u32 {
        self.inner.serial_number
    }

    #[getter]
    fn res_type<'py>(&self, py: Python<'py>) -> PyResult<Py<ResidueType>> {
        Py::new(
            py,
            ResidueType {
                inner: self.inner.res_type.clone(),
            },
        )
    }

    #[getter]
    fn atom_sns(&self) -> Vec<u32> {
        self.inner.atom_sns.clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct ChainGeneric {
    inner: bio_files_rs::ChainGeneric,
}

#[pymethods]
impl ChainGeneric {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn residue_sns(&self) -> Vec<u32> {
        self.inner.residue_sns.clone()
    }

    #[getter]
    fn atom_sns(&self) -> Vec<u32> {
        self.inner.atom_sns.clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct SecondaryStructure {
    // todo: Enum variants
    inner: bio_files_rs::SecondaryStructure,
}

#[pymethods]
impl SecondaryStructure {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct BackboneSS {
    inner: bio_files_rs::BackboneSS,
}

#[pymethods]
impl BackboneSS {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct ExperimentalMethod {
    // todo: Enum variants
    inner: bio_files_rs::ExperimentalMethod,
}

#[pymethods]
impl ExperimentalMethod {
    fn to_str_short(&self) -> String {
        self.inner.to_str_short()
    }
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, str: &str) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files_rs::ExperimentalMethod::from_str(str)?,
        })
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pymodule]
fn bio_files(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // General
    m.add_class::<AtomGeneric>()?;
    m.add_class::<BondType>()?;
    m.add_class::<BondGeneric>()?;
    m.add_class::<ResidueType>()?;
    m.add_class::<ResidueGeneric>()?;
    m.add_class::<ChainGeneric>()?;
    m.add_class::<SecondaryStructure>()?;
    m.add_class::<BackboneSS>()?;
    m.add_class::<ExperimentalMethod>()?;

    // Small molecules
    m.add_class::<mmcif::MmCif>()?;
    m.add_class::<mol2::Mol2>()?;
    m.add_class::<sdf::Sdf>()?;
    m.add_class::<pdbqt::Pdbqt>()?;

    // m.add_class::<mol2::MolType>()?;

    Ok(())
}
