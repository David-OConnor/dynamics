use std::{path::PathBuf, str::FromStr};

use dynamics_rs;
use lin_alg::f64::Vec3;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType, Py};

mod from_bio_files;
mod params;
mod prep;

use crate::{
    from_bio_files::*,
    params::FfParamSet,
    prep::{HydrogenConstraint, merge_params},
};

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

#[pyclass]
struct Snapshot {
    inner: dynamics_rs::Snapshot,
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
struct MolDynamics {
    // We don't use the inner pattern here, as we can't use lifetimes in Pyo3.
    pub ff_mol_type: FfMolType,
    pub atoms: Vec<AtomGeneric>,
    pub atom_posits: Option<Vec<Vec3>>,
    pub bonds: Vec<BondGeneric>,
    pub adjacency_list: Option<Vec<Vec<usize>>>,
    pub static_: bool,
    pub mol_specific_params: Option<ForceFieldParamsKeyed>,
}

#[pymethods]
impl MolDynamics {
    #[new]
    fn new(
        py: Python<'_>,
        ff_mol_type: FfMolType,
        atoms: Vec<Py<from_bio_files::AtomGeneric>>,
        atom_posits: Option<Vec<[f64; 3]>>,
        bonds: Vec<Py<from_bio_files::BondGeneric>>,
        adjacency_list: Option<Vec<Vec<usize>>>,
        static_: bool,
        mol_specific_params: Option<Py<from_bio_files::ForceFieldParamsKeyed>>,
    ) -> Self {
        // NOTE: Py<T>::borrow(py) — no .as_ref(py)
        let atoms: Vec<from_bio_files::AtomGeneric> =
            atoms.into_iter().map(|p| p.borrow(py).clone()).collect();
        let bonds: Vec<from_bio_files::BondGeneric> =
            bonds.into_iter().map(|p| p.borrow(py).clone()).collect();
        let mol_specific_params =
            mol_specific_params.map(|p| p.borrow(py).clone());

        let atom_posits = atom_posits.map(|v| {
            v.into_iter()
                .map(|a| Vec3::from_slice(&a).unwrap())
                .collect()
        });

        Self {
            ff_mol_type,
            atoms,
            atom_posits,
            bonds,
            adjacency_list,
            static_,
            mol_specific_params,
        }
    }
}

make_enum!(Integrator, dynamics_rs::Integrator, VerletVelocity, Langevin, LangevinCenter);

#[pyclass]
struct MdConfig {
    inner: dynamics_rs::MdConfig,
}

#[pymethods]
impl MdConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: dynamics_rs::MdConfig::default(),
        }
    }

    #[getter]
    fn integrator(&self) -> Integrator {
        self.inner.integrator.into()
    }
    #[setter(integrator)]
    fn integrator_set(&mut self, v: Integrator) {
        self.inner.integrator = v.to_native();
    }
    #[getter]
    fn zero_com_drift(&self) -> bool {
        self.inner.zero_com_drift
    }
    #[setter(zero_com_drift)]
    fn zero_com_drift_set(&mut self, v: bool) {
        self.inner.zero_com_drift = v;
    }
    #[getter]
    fn temp_target(&self) -> f32 {
        self.inner.temp_target
    }
    #[setter(temp_target)]
    fn temp_target_set(&mut self, v: f32) {
        self.inner.temp_target = v;
    }
    #[getter]
    fn pressure_target(&self) -> f32 {
        self.inner.pressure_target
    }
    #[setter(pressure_target)]
    fn pressure_target_set(&mut self, v: f32) {
        self.inner.pressure_target = v;
    }
    #[getter]
    fn hydrogen_constraint(&self) -> HydrogenConstraint {
        self.inner.hydrogen_constraint.into()
    }
    #[setter(hydrogen_constraint)]
    fn hydrogen_constraint_set(&mut self, v: HydrogenConstraint) {
        self.inner.hydrogen_constraint = v.to_native();
    }
    #[getter]
    fn snapshot_ratio_memory(&self) -> usize {
        self.inner.snapshot_ratio_memory
    }
    #[setter(snapshot_ratio_memory)]
    fn snapshot_ratio_memory_set(&mut self, v: usize) {
        self.inner.snapshot_ratio_memory = v;
    }
    #[getter]
    fn snapshot_ratio_file(&self) -> usize {
        self.inner.snapshot_ratio_file
    }
    #[setter(snapshot_ratio_file)]
    fn snapshot_ratio_file_set(&mut self, v: usize) {
        self.inner.snapshot_ratio_file = v;
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(unsendable)] // Unsendable due to the RNG in the barostat.
struct MdState {
    inner: dynamics_rs::MdState,
}

#[pymethods]
impl MdState {
    #[new]
    fn new(
        py: Python<'_>,
        cfg: &MdConfig,
        mols: Vec<Py<MolDynamics>>,
        param_set: &FfParamSet,
    ) -> PyResult<Self> {
        let n = mols.len();

        // First pass: copy data out of Python objects while holding the GIL.
        let mut ff_types = Vec::with_capacity(n);
        let mut static_flags = Vec::with_capacity(n);

        let mut atoms_bufs: Vec<bio_files::AtomGeneric> = Vec::with_capacity(n);
        let mut bonds_bufs = Vec::with_capacity(n);

        let mut posits_bufs: Vec<Option<Vec<Vec3>>> = Vec::with_capacity(n);

        let mut adj_bufs: Vec<Option<Vec<Vec<usize>>>> = Vec::with_capacity(n);
        let mut msp_bufs: Vec<Option<from_bio_files::ForceFieldParamsKeyed>> =
            Vec::with_capacity(n);

        for mol_py in &mols {
            let v = mol_py.borrow(py); // PyRef<MolDynamics>

            ff_types.push(v.ff_mol_type.to_native());
            static_flags.push(v.static_);

            atoms_bufs.push(v.atoms.iter().map(|a| a.inner.clone()).collect());
            bonds_bufs.push(v.bonds.iter().map(|b| b.inner.clone()).collect());

            posits_bufs.push(v.atom_posits.clone());
            adj_bufs.push(v.adjacency_list.clone());
            msp_bufs.push(v.mol_specific_params.clone());
        }

        // Second pass: build borrowed views into our owned buffers.
        let mut mols_native = Vec::with_capacity(n);
        for i in 0..n {
            let atoms_slice = atoms_bufs[i].as_slice();
            let bonds_slice = bonds_bufs[i].as_slice();

            let atom_posits_slice = posits_bufs[i].as_ref().map(|v| v.as_slice());
            let adjacency_slice: Option<&[Vec<usize>]> =
                adj_bufs[i].as_ref().map(|v| v.as_slice());
            let mol_specific_params = msp_bufs[i].as_ref().map(|p| &p.inner);

            mols_native.push(dynamics_rs::MolDynamics {
                ff_mol_type: ff_types[i],
                atoms: atoms_slice,
                atom_posits: atom_posits_slice,
                bonds: bonds_slice,
                adjacency_list: adjacency_slice,
                static_: static_flags[i],
                mol_specific_params,
            });
        }

        Ok(Self {
            inner: dynamics_rs::MdState::new(
                &cfg.inner,
                &mols_native,
                &param_set.inner,
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

#[pyfunction]
fn save_snapshots(py: Python<'_>, snapshots: Vec<Py<Snapshot>>, path: PathBuf) -> PyResult<()> {
    let snapshots_native: Vec<_> =
        snapshots.into_iter().map(|p| p.borrow(py).inner.clone()).collect();
    dynamics_rs::save_snapshots(&snapshots_native, &path)?;
    Ok(())
}

#[pyfunction]
fn load_snapshots(path: PathBuf) -> PyResult<Vec<Snapshot>> {
    let snapshots = dynamics_rs::load_snapshots(&path)?;
    Ok(snapshots
        .into_iter()
        .map(|v| Snapshot { inner: v })
        .collect())
}

#[pymodule]
fn mol_dynamics(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // General
    m.add_class::<AtomDynamics>()?;
    m.add_class::<MolDynamics>()?;

    m.add_class::<Integrator>()?;
    m.add_class::<HydrogenConstraint>()?;
    m.add_class::<MdState>()?;

    m.add_class::<FfMolType>()?;
    m.add_class::<FfParamSet>()?;
    m.add_class::<Snapshot>()?;

    m.add_class::<from_bio_files::AtomGeneric>()?;
    m.add_class::<from_bio_files::BondGeneric>()?;
    m.add_class::<from_bio_files::ResidueGeneric>()?;
    m.add_class::<from_bio_files::ForceFieldParamsKeyed>()?;

    m.add_function(wrap_pyfunction!(merge_params, m)?)?;
    m.add_function(wrap_pyfunction!(save_snapshots, m)?)?;
    m.add_function(wrap_pyfunction!(load_snapshots, m)?)?;

    Ok(())
}
