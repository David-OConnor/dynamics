use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::{from_bio_files::ForceFieldParams, make_enum};

#[pyfunction]
pub fn merge_params(baseline: &ForceFieldParams, add_this: &ForceFieldParams) -> ForceFieldParams {
    let baseline_native = baseline.inner.clone();
    let add_this_native = add_this.inner.clone();

    let result = dynamics_rs::merge_params(&baseline_native, &add_this_native);
    ForceFieldParams { inner: result }
}

make_enum!(
    HydrogenConstraint,
    dynamics_rs::HydrogenConstraint,
    Constrained,
    Flexible
);
