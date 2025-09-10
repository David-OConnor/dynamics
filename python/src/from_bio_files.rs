//! Todo: Not sure how to handle this. Lots of C+P for now

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[pyclass]
#[derive(Clone)]
pub struct AtomGeneric {
    pub inner: bio_files::AtomGeneric,
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
#[derive(Clone)]
pub struct BondGeneric {
    pub inner: bio_files::BondGeneric,
}

#[pymethods]
impl BondGeneric {
    // #[getter]
    // fn bond_type(&self) -> BondType {
    //     self.bond_type().into()
    // }
    // #[setter(bond_type)]
    // fn bond_type_set(&mut self, val: BondType) {
    //     self.inner.bond_type = val.into();
    // }

    #[getter]
    fn atom_0_sn(&self) -> u32 {
        self.inner.atom_0_sn
    }
    #[setter(atom_0_sn)]
    fn atom_0_sn_set(&mut self, val: u32) {
        self.inner.atom_0_sn = val;
    }

    #[getter]
    fn atom_1_sn(&self) -> u32 {
        self.inner.atom_1_sn
    }
    #[setter(atom_1_sn)]
    fn atom_1_sn_set(&mut self, val: u32) {
        self.inner.atom_1_sn = val;
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
pub struct ResidueGeneric {
    pub inner: bio_files::ResidueGeneric,
}

#[pyclass]
#[derive(Clone)]
pub struct ForceFieldParamsKeyed {
    pub inner: bio_files::amber_params::ForceFieldParamsKeyed,
}
