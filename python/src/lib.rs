use pyo3::exceptions::PyIOError;
use std::{path::PathBuf, str::FromStr};

use dynamics_rs;
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use pyo3::{Py, exceptions::PyValueError, prelude::*, types::PyType};

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext},
    nvrtc::Ptx,
};

mod from_bio_files;
mod params;
mod prep;

use crate::{
    from_bio_files::*,
    params::{FfParamSet, prepare_peptide, prepare_peptide_mmcif},
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
pub struct Dihedral {
    pub inner: dynamics_rs::Dihedral,
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
    #[getter]
    fn energy_kinetic(&self) -> f32 {
        self.inner.energy_kinetic
    }
    #[getter]
    fn energy_potential(&self) -> f32 {
        self.inner.energy_potential
    }
    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature
    }
    #[getter]
    fn pressure(&self) -> f32 {
        self.inner.pressure
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
    // // We don't use the inner pattern here, as we can't use lifetimes in Pyo3.
    // // This contains owned equivalents.
    // pub ff_mol_type: FfMolType,
    // pub atoms: Vec<AtomGeneric>,
    // pub atom_posits: Option<Vec<Vec3F64>>,
    // pub bonds: Vec<BondGeneric>,
    // pub adjacency_list: Option<Vec<Vec<usize>>>,
    // pub static_: bool,
    // pub mol_specific_params: Option<ForceFieldParams>,
    inner: dynamics_rs::MolDynamics,
}

#[pymethods]
impl MolDynamics {
    #[new]
    fn new(
        py: Python<'_>,
        ff_mol_type: FfMolType,
        atoms: Vec<Py<from_bio_files::AtomGeneric>>,
        atom_posits: Option<Vec<[f64; 3]>>,
        atom_init_velocities: Option<Vec<[f32; 3]>>,
        bonds: Vec<Py<from_bio_files::BondGeneric>>,
        adjacency_list: Option<Vec<Vec<usize>>>,
        static_: bool,
        mol_specific_params: Option<Py<from_bio_files::ForceFieldParams>>,
    ) -> Self {
        let atoms = atoms
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        let bonds = bonds
            .into_iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();

        let mol_specific_params = mol_specific_params.map(|p| p.borrow(py).inner.clone());

        let atom_posits = atom_posits.map(|v| {
            v.into_iter()
                .map(|a| Vec3F64::new(a[0], a[1], a[2]))
                .collect()
        });
        let atom_init_velocties = match atom_init_velocities {
            Some(vel) => {
                vel.map(|v| {
                    v.into_iter()
                        .map(|a| Vec3::new(a[0], a[1], a[2]))
                        .collect()
                })
            }
            None => None,
        };

        Self {
            inner: dynamics_rs::MolDynamics {
                ff_mol_type: ff_mol_type.into(),
                atoms,
                atom_posits,
                atom_init_velocities,
                bonds,
                adjacency_list,
                static_,
                mol_specific_params,
            },
        }
    }

    #[classmethod]
    fn from_mol2(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        mol: Py<from_bio_files::Mol2>,
        mol_specific_params: Option<Py<from_bio_files::ForceFieldParams>>,
    ) -> PyResult<Self> {
        let mol_specific_params = mol_specific_params.map(|p| p.borrow(py).inner.clone());
        let inner = dynamics_rs::MolDynamics::from_mol2(&mol.borrow(py).inner, mol_specific_params);
        Ok(Self { inner })
    }

    #[classmethod]
    fn from_sdf(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        mol: Py<from_bio_files::Sdf>,
        mol_specific_params: Option<Py<from_bio_files::ForceFieldParams>>,
    ) -> PyResult<Self> {
        let mol_specific_params = mol_specific_params.map(|p| p.borrow(py).inner.clone());
        let inner = dynamics_rs::MolDynamics::from_sdf(&mol.borrow(py).inner, mol_specific_params);
        Ok(Self { inner })
    }

    #[classmethod]
    fn from_amber_geostd(_cls: &Bound<'_, PyType>, ident: &str) -> PyResult<Self> {
        match dynamics_rs::MolDynamics::from_amber_geostd(ident) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(PyIOError::new_err(e.to_string())),
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
    #[getter]
    fn neighbor_skin(&self) -> f32 {
        self.inner.neighbor_skin
    }
    #[setter(neighbor_skin)]
    fn neighbor_skin_set(&mut self, v: f32) {
        self.inner.neighbor_skin = v;
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(unsendable)] // Unsendable due to the RNG in the barostat.
struct MdState {
    inner: dynamics_rs::MdState,
}

fn get_dev() -> dynamics_rs::ComputationDevice {
    #[cfg(not(feature = "cuda"))]
    return dynamics_rs::ComputationDevice::Cpu;

    #[cfg(feature = "cuda")]
    if cudarc::driver::result::init().is_ok() {
        let ctx = CudaContext::new(0).unwrap();

        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(dynamics_rs::PTX));

        match module {
            Ok(m) => dynamics_rs::ComputationDevice::Gpu((stream, m)),
            Err(e) => {
                eprintln!(
                    "Error loading CUDA module: {}; not using CUDA. Error: {e}",
                    dynamics_rs::PTX
                );
                dynamics_rs::ComputationDevice::Cpu
            }
        }
    } else {
        dynamics_rs::ComputationDevice::Cpu
    }
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
        use std::mem::take;

        let n_mols = mols.len();

        // Per-molecule ownership
        let mut atoms_bufs: Vec<Vec<bio_files::AtomGeneric>> = Vec::with_capacity(n_mols);
        let mut posit_bufs: Vec<Option<Vec<Vec3F64>>> = Vec::with_capacity(n_mols);
        let mut bonds_bufs: Vec<Vec<bio_files::BondGeneric>> = Vec::with_capacity(n_mols);
        let mut adj_bufs: Vec<Option<Vec<Vec<usize>>>> = Vec::with_capacity(n_mols);
        let mut msp_bufs: Vec<Option<bio_files::md_params::ForceFieldParams>> =
            Vec::with_capacity(n_mols);

        for mol in &mols {
            let v = mol.borrow(py);

            atoms_bufs.push(v.inner.atoms.iter().map(|a| a.clone()).collect());
            posit_bufs.push(v.inner.atom_posits.clone());
            bonds_bufs.push(v.inner.bonds.iter().map(|b| b.clone()).collect());
            adj_bufs.push(v.inner.adjacency_list.clone());
            msp_bufs.push(v.inner.mol_specific_params.clone());
        }

        let mut mols_native = Vec::with_capacity(n_mols);

        for (i, mol) in mols.iter().enumerate() {
            let v = mol.borrow(py);

            // Move owned buffers out (no extra allocation/copy)
            let atoms = take(&mut atoms_bufs[i]);
            let bonds = take(&mut bonds_bufs[i]);
            let atom_posits = take(&mut posit_bufs[i]); // Option<Vec<Vec3F64>>
            let adjacency_list = take(&mut adj_bufs[i]); // Option<Vec<Vec<usize>>>
            let mol_specific_params = take(&mut msp_bufs[i]).map(|p| p.clone());

            mols_native.push(dynamics_rs::MolDynamics {
                ff_mol_type: v.inner.ff_mol_type.into(),
                atoms,          // Vec<bio_files::AtomGeneric>
                atom_posits,    // Option<Vec<Vec3F64>>
                bonds,          // Vec<bio_files::BondGeneric>
                adjacency_list, // Option<Vec<Vec<usize>>>
                static_: v.inner.static_,
                mol_specific_params, // Option<dynamics_rs::...::ForceFieldParams>
            });
        }

        let dev = get_dev();
        let inner = dynamics_rs::MdState::new(&dev, &cfg.inner, &mols_native, &param_set.inner)
            .map_err(|e| PyValueError::new_err(e.descrip))?;

        Ok(Self { inner })
    }

    fn step(&mut self, dt: f32) {
        let dev = get_dev();
        self.inner.step(&dev, dt);
    }

    /// A string for now to keep this wrapper simple.
    fn computation_time(&self) -> String {
        self.inner.computation_time().to_string()
    }

    fn minimize_energy(&mut self, max_iters: usize) {
        let dev = get_dev();
        self.inner.minimize_energy(&dev, max_iters);
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
    m.add_class::<Dihedral>()?;

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
    m.add_function(wrap_pyfunction!(prepare_peptide_mmcif, m)?)?;

    m.add_function(wrap_pyfunction!(from_bio_files::load_prmtop, m)?)?;
    m.add_function(wrap_pyfunction!(from_bio_files::save_prmtop, m)?)?;

    Ok(())
}
