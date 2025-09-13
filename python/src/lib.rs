use std::{path::PathBuf, str::FromStr};

use dynamics_rs;
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use pyo3::{Py, exceptions::PyValueError, prelude::*, types::PyType};

mod from_bio_files;
mod params;
mod prep;

use crate::{
    from_bio_files::*,
    params::{FfParamSet, prepare_peptide},
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
    inner: dynamics_rs::snapshot::Snapshot,
}

#[pymethods]
impl Snapshot {
    //     pub atom_velocities: Vec<Vec3>,
    //     pub water_o_posits: Vec<Vec3>,
    //     pub water_h0_posits: Vec<Vec3>,
    //     pub water_h1_posits: Vec<Vec3>,
    //     /// Single velocity per water molecule, as it's rigid.
    //     pub water_velocities: Vec<Vec3>,
    //     pub energy_kinetic: f32,
    //     pub energy_potential: f32,
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }
    #[setter(time)]
    fn time_set(&mut self, v: f64) {
        self.inner.time = v;
    }
    #[getter]
    fn atom_posits(&self) -> Vec<[f32; 3]> {
        self.inner.atom_posits.iter().map(|v| v.to_arr()).collect()
    }
    #[setter(atom_posits)]
    fn atom_posits_set(&mut self, v: Vec<[f32; 3]>) {
        self.inner.water_o_posits = v.iter().map(|v| Vec3::from_slice(v).unwrap()).collect();
    }
    #[getter]
    fn water_o_posits(&self) -> Vec<[f32; 3]> {
        self.inner.atom_posits.iter().map(|v| v.to_arr()).collect()
    }
    #[setter(water_o_posits)]
    fn water_o_posits_set(&mut self, v: Vec<[f32; 3]>) {
        self.inner.water_o_posits = v.iter().map(|v| Vec3::from_slice(v).unwrap()).collect();
    }
    // todo: Impl teh other fields.

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct SnapshotHandler {
    inner: dynamics_rs::snapshot::SnapshotHandler,
}

#[pymethods]
impl SnapshotHandler {
    // todo: Impl Save type field and its enum
    #[getter]
    fn ratio(&self) -> usize {
        self.inner.ratio
    }
    #[setter(ratio)]
    fn ratio_set(&mut self, v: usize) {
        self.inner.ratio = v;
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
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
    fn posit(&self) -> [f32; 3] {
        self.inner.posit.to_arr()
    }
    #[setter(posit)]
    fn posit_set(&mut self, posit: [f32; 3]) {
        self.inner.posit = Vec3::from_slice(&posit).unwrap()
    }
    #[getter]
    fn vel(&self) -> [f32; 3] {
        self.inner.vel.to_arr()
    }
    #[setter(vel)]
    fn vel_set(&mut self, vel: [f32; 3]) {
        self.inner.vel = Vec3::from_slice(&vel).unwrap()
    }
    #[getter]
    fn accel(&self) -> [f32; 3] {
        self.inner.accel.to_arr()
    }
    #[setter(accel)]
    fn accel_set(&mut self, accel: [f32; 3]) {
        self.inner.accel = Vec3::from_slice(&accel).unwrap()
    }

    #[getter]
    fn mass(&self) -> f32 {
        self.inner.mass
    }
    #[setter(mass)]
    fn mass_set(&mut self, mass: f32) {
        self.inner.mass = mass
    }
    #[getter]
    fn partial_charge(&self) -> f32 {
        self.inner.partial_charge
    }
    #[setter(partial_charge)]
    fn partial_charge_set(&mut self, v: f32) {
        self.inner.partial_charge = v
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
    // This contains owned equivalents.
    pub ff_mol_type: FfMolType,
    pub atoms: Vec<AtomGeneric>,
    pub atom_posits: Option<Vec<Vec3F64>>,
    pub bonds: Vec<BondGeneric>,
    pub adjacency_list: Option<Vec<Vec<usize>>>,
    pub static_: bool,
    pub mol_specific_params: Option<ForceFieldParams>,
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
        mol_specific_params: Option<Py<from_bio_files::ForceFieldParams>>,
    ) -> Self {
        // NOTE: Py<T>::borrow(py) — no .as_ref(py)
        let atoms: Vec<from_bio_files::AtomGeneric> =
            atoms.into_iter().map(|p| p.borrow(py).clone()).collect();
        let bonds: Vec<from_bio_files::BondGeneric> =
            bonds.into_iter().map(|p| p.borrow(py).clone()).collect();
        let mol_specific_params = mol_specific_params.map(|p| p.borrow(py).clone());

        let atom_posits = atom_posits.map(|v| {
            v.into_iter()
                .map(|a| Vec3F64::new(a[0] as f64, a[1] as f64, a[2] as f64))
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

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Integrator {
    VerletVelocity,
    Langevin,
    LangevinMiddle,
}

impl Integrator {
    pub fn to_native(self) -> dynamics_rs::Integrator {
        match self {
            Self::VerletVelocity => dynamics_rs::Integrator::VerletVelocity,
            Self::Langevin => dynamics_rs::Integrator::Langevin { gamma: 1.0 },
            Self::LangevinMiddle => dynamics_rs::Integrator::LangevinMiddle { gamma: 1.0 },
        }
    }
    pub fn from_native(native: dynamics_rs::Integrator) -> Self {
        match native {
            dynamics_rs::Integrator::VerletVelocity => Self::VerletVelocity,
            dynamics_rs::Integrator::Langevin { gamma: _ } => Self::Langevin,
            dynamics_rs::Integrator::LangevinMiddle { gamma: _ } => Self::LangevinMiddle,
        }
    }
}

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
        Integrator::from_native(self.inner.integrator.clone())
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
    fn snapshot_handlers(&self) -> Vec<SnapshotHandler> {
        self.inner
            .snapshot_handlers
            .clone()
            .into_iter()
            .map(|v| SnapshotHandler { inner: v })
            .collect()
    }
    #[setter(snapshot_handlers)]
    fn snapshot_handlers_set(&mut self, v: Vec<PyRef<SnapshotHandler>>) {
        self.inner.snapshot_handlers = v.into_iter().map(|v| v.inner.clone()).collect();
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(unsendable)] // Unsendable due to the RNG in the barostat.
struct MdState {
    inner: dynamics_rs::MdState,
}

// todo: Determine how to implement.
#[cfg(feature = "cuda")]
fn get_dev() -> dynamics_rs::ComputationDevice {
    if cudarc::driver::result::init().is_ok() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // let module = ctx.load_module(Ptx::from_src(PTX));
        let module_dynamics = ctx.load_module(Ptx::from_src(dynamics_rs::PTX));

        match module_dynamics {
            Ok(m) => ComputationDevice::Gpu((stream, m)),
            Err(e) => {
                eprintln!(
                    "Error loading CUDA module: {}; not using CUDA. Error: {e}",
                    dynamics_rs::PTX
                );
                ComputationDevice::Cpu
            }
        }
    } else {
        ComputationDevice::Cpu
    }
}

#[pymethods]
impl MdState {
    #[new]
    fn new(
        py: Python<'_>,
        cfg: &MdConfig,
        mols: Vec<Py<MolDynamics>>, // <-- Py-wrapped so PyO3 can extract
        param_set: &FfParamSet,
    ) -> PyResult<Self> {
        let n_mols = mols.len();

        // Per-molecule ownership
        let mut atoms_bufs: Vec<Vec<bio_files::AtomGeneric>> = Vec::with_capacity(n_mols);
        let mut posit_bufs: Vec<Option<Vec<Vec3F64>>> = Vec::with_capacity(n_mols);
        let mut bonds_bufs: Vec<Vec<bio_files::BondGeneric>> = Vec::with_capacity(n_mols);
        let mut adj_bufs: Vec<Option<Vec<Vec<usize>>>> = Vec::with_capacity(n_mols);
        let mut msp_bufs: Vec<Option<from_bio_files::ForceFieldParams>> =
            Vec::with_capacity(n_mols);

        for mol in &mols {
            let v = mol.borrow(py);

            atoms_bufs.push(v.atoms.iter().map(|a| a.inner.clone()).collect());
            posit_bufs.push(v.atom_posits.clone());
            bonds_bufs.push(v.bonds.iter().map(|b| b.inner.clone()).collect());
            adj_bufs.push(v.adjacency_list.clone());
            msp_bufs.push(v.mol_specific_params.clone());
        }

        let mut mols_native = Vec::with_capacity(n_mols);

        for (i, mol) in mols.iter().enumerate() {
            let v = mol.borrow(py);

            let atoms_slice = atoms_bufs[i].as_slice();
            let bonds_slice = bonds_bufs[i].as_slice();
            let atom_posits_slice: Option<&[Vec3F64]> = posit_bufs[i].as_deref();
            let adjacency_slice: Option<&[Vec<usize>]> = adj_bufs[i].as_deref();
            let mol_specific_params = msp_bufs[i].as_ref().map(|p| &p.inner);

            mols_native.push(dynamics_rs::MolDynamics {
                ff_mol_type: v.ff_mol_type.to_native(),
                atoms: atoms_slice,
                atom_posits: atom_posits_slice,
                bonds: bonds_slice,
                adjacency_list: adjacency_slice,
                static_: v.static_,
                mol_specific_params,
            });
        }

        let inner = dynamics_rs::MdState::new(&cfg.inner, &mols_native, &param_set.inner)
            .map_err(|e| PyValueError::new_err(e.descrip))?;

        Ok(Self { inner })
    }

    fn step(&mut self, dt: f32) {
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
    let snapshots_native: Vec<_> = snapshots
        .into_iter()
        .map(|p| p.borrow(py).inner.clone())
        .collect();
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
    m.add_class::<MdConfig>()?;
    m.add_class::<MdState>()?;

    m.add_class::<FfMolType>()?;
    m.add_class::<FfParamSet>()?;
    m.add_class::<Snapshot>()?;
    m.add_class::<SnapshotHandler>()?;

    m.add_class::<from_bio_files::AtomGeneric>()?;
    m.add_class::<from_bio_files::BondGeneric>()?;
    m.add_class::<from_bio_files::ResidueGeneric>()?;
    m.add_class::<from_bio_files::ChainGeneric>()?;

    m.add_class::<from_bio_files::ForceFieldParams>()?;
    m.add_class::<from_bio_files::Mol2>()?;
    m.add_class::<from_bio_files::MolType>()?;
    m.add_class::<from_bio_files::Sdf>()?;
    m.add_class::<from_bio_files::MmCif>()?;

    m.add_function(wrap_pyfunction!(merge_params, m)?)?;
    m.add_function(wrap_pyfunction!(save_snapshots, m)?)?;
    m.add_function(wrap_pyfunction!(load_snapshots, m)?)?;

    m.add_function(wrap_pyfunction!(prepare_peptide, m)?)?;

    Ok(())
}
