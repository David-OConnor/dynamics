use std::str::FromStr;

use dynamics_rs;
use lin_alg::f64::Vec3;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

mod from_bio_files;
mod params;
mod prep;

use crate::{from_bio_files::*, params::FfParamSet, prep::HydrogenMdType};

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

// #[classmethod]
// fn from_str(_cls: &Bound<'_, PyType>, str: &str) -> PyResult<Self> {
//     Ok(bio_files_rs::BondType::from_str(str)?.into())
// }

#[pyclass]
struct AtomDynamics {
    inner: dynamics_rs::AtomDynamics,
}

// Note: We may not need or want to expose these; used mainly internally.
#[pymethods]
impl AtomDynamics {
    // field!(serial_number, u32);
    // field!(force_field_type, String);

    // todo: Sort out how you import this.
    // field!(element, Element);

    #[getter]
    fn posit(&self) -> [f64; 3] {
        self.inner.posit.to_arr()
    }
    #[setter(posit)]
    fn posit_set(&mut self, posit: [f64; 3]) {
        self.inner.posit = Vec3::from_slice(&posit).unwrap()
    }
    #[getter]
    fn vel(&self) -> [f64; 3] {
        self.inner.vel.to_arr()
    }
    #[setter(vel)]
    fn vel_set(&mut self, vel: [f64; 3]) {
        self.inner.vel = Vec3::from_slice(&vel).unwrap()
    }
    #[getter]
    fn accel(&self) -> [f64; 3] {
        self.inner.accel.to_arr()
    }
    #[setter(accel)]
    fn accel_set(&mut self, accel: [f64; 3]) {
        self.inner.accel = Vec3::from_slice(&accel).unwrap()
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.inner.mass
    }
    #[setter(mass)]
    fn mass_set(&mut self, mass: f64) {
        self.inner.mass = mass
    }
    #[getter]
    fn partial_charge(&self) -> f64 {
        self.inner.partial_charge
    }
    #[setter(partial_charge)]
    fn partial_charge_set(&mut self, partial_charge: f64) {
        self.inner.partial_charge = partial_charge
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

make_enum!(
    FfMolType,
    dynamics_rs::FfMolType,
    Peptide,
    SmallOrganic,
    Dna,
    Rna,
    Lipid,
    Carbohydrate
);

#[pyclass]
struct MolDynamics<'a> {
    inner: dynamics_rs::MolDynamics<'a>,
}

#[pymethods]
impl MolDynamics {
    #[new]
    fn new(
        ff_mol_type: FfMolType,
        atoms: Vec<AtomGeneric>,
        atom_posits: Option<Vec<[f64; 3]>>,
        bonds: Vec<BondGeneric>,
        adjacency_list: Option<Vec<Vec<usize>>>,
        static_: bool,
        mol_specific_params: Option<ForceFieldParamsKeyed>,
    ) -> Self {
        let atoms_native: Vec<_> = atoms.into_iter().map(|v| v.inner).collect();
        let bonds_native: Vec<_> = bonds.into_iter().map(|v| v.inner).collect();

        let atom_posits_native = match atom_posits {
            Some(v) => {
                let v: Vec<_> = v
                    .into_iter()
                    .map(|v| Vec3::from_slice(&v).unwrap())
                    .collect();
                Some(v)
            }
            None => None,
        };

        let mol_specific_params_native = match mol_specific_params {
            Some(v) => Some(v.inner),
            None => None,
        };

        let adjacency_list = match adjacency_list {
            Some(v) => Some(&v),
            None => None,
        };

        Self {
            inner: dynamics_rs::MolDynamics {
                ff_mol_type: ff_mol_type.to_native(),
                atoms: &atoms_native,
                atom_posits: atom_posits_native.as_deref(),
                bonds: &bonds_native,
                adjacency_list: adjacency_list.map(|v| &**v),
                static_,
                mol_specific_params: mol_specific_params_native.as_ref(),
            },
        }
    }
}

#[pyclass]
struct MdState {
    inner: dynamics_rs::MdState,
}

#[pymethods]
impl MdState {
    #[new]
    fn new(
        mols: Vec<MolDynamics>,
        temp_target: f64,
        pressure_target: f64,
        param_set: &FfParamSet,
        hydrogen_md_type: HydrogenMdType,
        snapshot_ratio: usize,
    ) -> PyResult<Self> {
        let mols_native: Vec<_> = mols.into_iter().map(|v| v.inner).collect();

        Ok(Self {
            inner: dynamics_rs::MdState::new(
                mols_native,
                temp_target,
                pressure_target,
                &param_set.inner,
                hydrogen_md_type.to_native(),
                snapshot_ratio,
            )
            .map_err(|e| PyValueError::new_err(e.descrip))?,
        })
    }

    fn step(&mut self, dt: f64) {
        // CPU only is temp.
        self.inner.step(&dynamics_rs::ComputationDevice::Cpu, dt);
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
    // No debug impl
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pymodule]
fn mol_dynamics(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // General
    m.add_class::<AtomDynamics>()?;
    m.add_class::<MolDynamics>()?;

    m.add_class::<MdState>()?;
    m.add_class::<FfMolType>()?;
    m.add_class::<FfParamSet>()?;
    m.add_class::<HydrogenMdType>()?;

    m.add_class::<from_bio_files::AtomGeneric>()?;
    m.add_class::<from_bio_files::BondGeneric>()?;
    m.add_class::<from_bio_files::ResidueGeneric>()?;
    m.add_class::<from_bio_files::ForceFieldParamsKeyed>()?;

    Ok(())
}
