use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::from_bio_files::ForceFieldParamsKeyed;

#[pyfunction]
fn merge_params(
    generic: &ForceFieldParamsKeyed,
    specific: &ForceFieldParamsKeyed,
) -> ForceFieldParamsKeyed {
    let generic_native = generic.inner.clone();
    let specific_native = specific.inner.clone();
    let result = dynamics_rs::merge_params(&generic_native, Some(&specific_native));

    ForceFieldParamsKeyed { inner: result }
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub enum HydrogenMdType {
    // Skipping the inner part.
    Fixed,
    Flexible,
}

impl HydrogenMdType {
    pub fn to_native(&self) -> dynamics_rs::HydrogenMdType {
        match self {
            Self::Fixed => dynamics_rs::HydrogenMdType::Fixed(Vec::new()),
            Self::Flexible => dynamics_rs::HydrogenMdType::Flexible,
        }
    }

    pub fn from_native(native: dynamics_rs::HydrogenMdType) -> Self {
        match native {
            dynamics_rs::HydrogenMdType::Fixed(_) => Self::Fixed,
            dynamics_rs::HydrogenMdType::Flexible => Self::Flexible,
        }
    }
}
