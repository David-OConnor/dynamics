use std::str::FromStr;

use dynamics_rs;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use lin_alg::f64::Vec3;

/// Candidate for standalone helper lib.
#[macro_export]
macro_rules! make_enum {
    ($Py:ident, $Native:path, $( $Var:ident ),+ $(,)?) => {
        #[pyclass]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $Py { $( $Var ),+ }

        impl ::core::convert::From<$Py> for $Native {
            fn from(v: $Py) -> Self { match v { $( $Py::$Var => <$Native>::$Var ),+ } }
        }

        impl ::core::convert::From<$Native> for $Py {
            fn from(v: $Native) -> Self { match v { $( <$Native>::$Var => $Py::$Var ),+ } }
        }

        impl $Py {
            pub fn to_native(self) -> $Native {
                self.into()
            }

            pub fn from_native(native: $Native) -> Self {
               native.into()
            }
        }
    };
}

// todo: Blocked by PyO3 on macros.
/// Candidate for standalone helper lib.
macro_rules! field {
    ($name:ident, $ty:ty) => {
        #[getter]
        fn $name(&self) -> $ty {
            self.inner.$name.into()
        }

        // #[setter($name)]
        // // todo: Do we need to use paste! here?
        // fn $name##_set(&mut self, val: $ty) -> $ty {
        //     self.inner.$name = val.into();
        //     val
        // }
    };
}


// #[classmethod]
// fn from_str(_cls: &Bound<'_, PyType>, str: &str) -> PyResult<Self> {
//     Ok(bio_files_rs::BondType::from_str(str)?.into())
// }

make_enum!(MdMode, dynamics_rs::MdMode, Docking, Peptide);

#[pyclass]
struct AtomDynamics {
    inner: dynamics_rs::AtomDynamics,
}

#[pymethods]
impl AtomDynamics {
    // field!(serial_number, u32);
    // field!(force_field_type, String);

    // todo: Sort out how you import this.
    // field!(element, Element);

    #[getter]
    fn posit(&self) -> [f32; 3] {
        self.inner.posit.to_array()
    }
    #[setter(posit)]
    fn posit_set(&mut self, posit: [f32; 3])  {
        self.inner.posit = Vec3::from_slice(&posit)
    }
    #[getter]
    fn vel(&self) -> [f32; 3] {
        self.inner.vel.to_array()
    }
    #[setter(vel)]
    fn vel_set(&mut self, vel: [f32; 3])  {
        self.inner.vel = Vec3::from_slice(&vel)
    }
    #[getter]
    fn accel(&self) -> [f32; 3] {
        self.inner.accel.to_array()
    }
    #[setter(accel)]
    fn accel_set(&mut self, accel: [f32; 3])  {
        self.inner.accel = Vec3::from_slice(&accel)
    }

    // field!(mass, f64);
    // field!(partial_charge, f64);
    // field!(lj_sigma, f64);
    // field!(lj_eps, f64);

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pymethods]
impl MdMode {
    fn __repr__(&self) -> String {
        format!("{:?}", self.to_native())
    }
}

#[pyclass]
struct MdState {
    inner: dynamics_rs::MdState,
}

#[pymethods]
impl MdState {
    // #[new]
    // fn new(
    //     mode: MdMode,
    //     atoms_dy: Vec<AtomDynamics>,
    //     atoms_static: Vec<AtomDynamics>,
    //     ff_params_non_static: ForceFieldParamsIndexed,
    //     temp_target: f64,     // K
    //     pressure_target: f64, // k
    //     hydrogen_md_type: HydrogenMdType,
    //     adjacency_list: Vec<Vec<usize>>,
    // ) -> Self {
    //     Self { inner: dynamics_rs::MdState::new(
    //         mode.into(), atoms_dy, atoms_static, ff_params_non_static,
    //         temp_target, pressure_target, hydrogen_md_type.into(), adjacency_list,
    //     )}
    // }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}


#[pymodule]
fn bio_files(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // General
    m.add_class::<AtomDynamics>()?;
    m.add_class::<MdMode>()?;
    m.add_class::<MdState>()?;


    Ok(())
}
