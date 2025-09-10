use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::{from_bio_files::ForceFieldParamsKeyed, make_enum};

#[pyfunction]
pub fn merge_params(
    generic: &ForceFieldParamsKeyed,
    specific: &ForceFieldParamsKeyed,
) -> ForceFieldParamsKeyed {
    let generic_native = generic.inner.clone();
    let specific_native = specific.inner.clone();
    let result = dynamics_rs::merge_params(&generic_native, Some(&specific_native));

    ForceFieldParamsKeyed { inner: result }
}

make_enum!(HydrogenConstraint, dynamics_rs::HydrogenConstraint, Constrained, Flexible);