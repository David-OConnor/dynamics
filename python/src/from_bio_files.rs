//! We must duplicate work from bio_files' python bindings, e.g. copy+paste. We can
//! use the rust bio_files lib directly here, but we must re-export the Python bindings; otherwise
//! the calling code will have compatibility problems. e.g. "AtomGeneric != AtomGeneric".

use std::{collections::HashMap, path::PathBuf};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::make_enum;

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
pub struct ChainGeneric {
    pub inner: bio_files::ChainGeneric,
}

#[pyclass]
#[derive(Clone)]
pub struct ForceFieldParams {
    pub inner: bio_files::md_params::ForceFieldParams,
}

#[pymethods]
impl ForceFieldParams {
    #[classmethod]
    fn from_frcmod(_cls: &Bound<'_, PyType>, text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::md_params::ForceFieldParams::from_frcmod(text)?,
        })
    }

    #[classmethod]
    fn from_dat(_cls: &Bound<'_, PyType>, text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::md_params::ForceFieldParams::from_dat(text)?,
        })
    }

    #[classmethod]
    fn load_frcmod(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::md_params::ForceFieldParams::load_frcmod(&path)?,
        })
    }

    #[classmethod]
    fn load_dat(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::md_params::ForceFieldParams::load_dat(&path)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

make_enum!(
    MolType,
    bio_files::mol2::MolType,
    Small,
    Bipolymer,
    Protein,
    NucleicAcid,
    Saccharide,
);

#[pymethods]
impl MolType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.to_native())
    }
}

// todo: ChargeType as well.

#[pyclass(module = "bio_files")]
pub struct Mol2 {
    pub inner: bio_files::Mol2,
}

#[pymethods]
impl Mol2 {
    // todo: Blocked by Pyo3 on macros here.
    // field!(ident, String);
    // field!(mol_type, MolType);

    #[getter]
    fn ident(&self) -> &str {
        &self.inner.ident
    }
    #[setter(ident)]
    fn ident_set(&mut self, val: String) {
        self.inner.ident = val;
    }

    #[getter]
    fn metadata(&self) -> &HashMap<String, String> {
        &self.inner.metadata
    }
    #[setter(metadata)]
    fn metadata_set(&mut self, val: HashMap<String, String>) {
        self.inner.metadata = val;
    }

    #[getter]
    fn atoms(&self) -> Vec<AtomGeneric> {
        self.inner
            .atoms
            .iter()
            .map(|a| AtomGeneric { inner: a.clone() })
            .collect()
    }
    #[setter(atoms)]
    fn atoms_set(&mut self, val: Vec<PyRef<'_, AtomGeneric>>) {
        let atoms = val.iter().map(|a| a.inner.clone()).collect();

        self.inner.atoms = atoms;
    }

    #[getter]
    fn bonds(&self) -> Vec<BondGeneric> {
        self.inner
            .bonds
            .iter()
            .cloned()
            .map(|b| BondGeneric { inner: b.clone() })
            .collect()
    }
    #[setter(bonds)]
    fn bonds_set(&mut self, val: Vec<PyRef<'_, BondGeneric>>) {
        let bonds = val.iter().map(|a| a.inner.clone()).collect();

        self.inner.bonds = bonds;
    }

    #[getter]
    fn mol_type(&self) -> MolType {
        MolType::from_native(self.inner.mol_type)
    }
    #[setter(mol_type)]
    fn mol_type_set(&mut self, val: MolType) {
        self.inner.mol_type = val.into();
    }

    // todo: str for now
    #[getter]
    // fn charge_type(&self) -> ChargeType {
    fn charge_type(&self) -> String {
        self.inner.charge_type.to_string()
    }

    #[getter]
    fn comment(&self) -> Option<String> {
        self.inner.comment.clone()
    }
    #[setter(comment)]
    fn comment_set(&mut self, val: String) {
        self.inner.comment = val.into();
    }

    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::Mol2::new(text)?,
        })
    }

    fn to_sdf(&self) -> Sdf {
        Sdf {
            inner: self.inner.clone().into(),
        }
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        Ok(self.inner.save(&path)?)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::Mol2::load(&path)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(module = "bio_files")]
pub struct Sdf {
    pub inner: bio_files::Sdf,
}

#[pymethods]
impl Sdf {
    #[getter]
    fn ident(&self) -> &str {
        &self.inner.ident
    }
    #[setter(ident)]
    fn ident_set(&mut self, val: String) {
        self.inner.ident = val;
    }

    #[getter]
    fn metadata(&self) -> &HashMap<String, String> {
        &self.inner.metadata
    }
    #[setter(metadata)]
    fn metadata_set(&mut self, val: HashMap<String, String>) {
        self.inner.metadata = val;
    }

    #[getter]
    fn atoms(&self) -> Vec<AtomGeneric> {
        self.inner
            .atoms
            .iter()
            .map(|a| AtomGeneric { inner: a.clone() })
            .collect()
    }
    #[setter(atoms)]
    fn atoms_set(&mut self, val: Vec<PyRef<'_, AtomGeneric>>) {
        let atoms = val.iter().map(|a| a.inner.clone()).collect();

        self.inner.atoms = atoms;
    }

    #[getter]
    fn bonds(&self) -> Vec<BondGeneric> {
        self.inner
            .bonds
            .iter()
            .cloned()
            .map(|b| BondGeneric { inner: b.clone() })
            .collect()
    }
    #[setter(bonds)]
    fn bonds_set(&mut self, val: Vec<PyRef<'_, BondGeneric>>) {
        let bonds = val.iter().map(|a| a.inner.clone()).collect();

        self.inner.bonds = bonds;
    }

    #[getter]
    fn chains(&self) -> Vec<ChainGeneric> {
        self.inner
            .chains
            .iter()
            .map(|c| ChainGeneric { inner: c.clone() })
            .collect()
    }

    #[getter]
    fn residues(&self) -> Vec<ResidueGeneric> {
        self.inner
            .residues
            .iter()
            .cloned()
            .map(|r| ResidueGeneric { inner: r.clone() })
            .collect()
    }

    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::Sdf::new(text)?,
        })
    }

    fn to_mol2(&self) -> Mol2 {
        Mol2 {
            inner: self.inner.clone().into(),
        }
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        Ok(self.inner.save(&path)?)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::Sdf::load(&path)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(module = "bio_files")]
pub struct MmCif {
    inner: bio_files::MmCif,
}

//     pub secondary_structure: Vec<BackboneSS>,
//     pub experimental_method: Option<ExperimentalMethod>,
#[pymethods]
impl MmCif {
    #[getter]
    fn ident(&self) -> &str {
        &self.inner.ident
    }
    #[setter(ident)]
    fn ident_set(&mut self, val: String) {
        self.inner.ident = val;
    }

    #[getter]
    fn metadata(&self) -> &HashMap<String, String> {
        &self.inner.metadata
    }
    #[setter(metadata)]
    fn metadata_set(&mut self, val: HashMap<String, String>) {
        self.inner.metadata = val;
    }

    #[getter]
    fn atoms(&self) -> Vec<AtomGeneric> {
        self.inner
            .atoms
            .iter()
            .map(|a| AtomGeneric { inner: a.clone() })
            .collect()
    }
    #[setter(atoms)]
    fn atoms_set(&mut self, val: Vec<PyRef<'_, AtomGeneric>>) {
        let atoms = val.iter().map(|a| a.inner.clone()).collect();

        self.inner.atoms = atoms;
    }

    // #[getter]
    // fn bonds(&self) -> Vec<BondGeneric> {
    //     self.inner
    //         .bonds
    //         .iter()
    //         .cloned()
    //         .map(|b| BondGeneric { inner: b.clone() })
    //         .collect()
    // }
    // #[setter(bonds)]
    // fn bonds_set(&mut self, val: Vec<PyRef<'_, BondGeneric>>) {
    //     let bonds = val.iter().map(|a| a.inner.clone()).collect();
    //
    //     self.inner.bonds = bonds;
    // }

    #[getter]
    fn chains(&self) -> Vec<ChainGeneric> {
        self.inner
            .chains
            .iter()
            .map(|c| ChainGeneric { inner: c.clone() })
            .collect()
    }

    #[getter]
    fn residues(&self) -> Vec<ResidueGeneric> {
        self.inner
            .residues
            .iter()
            .cloned()
            .map(|r| ResidueGeneric { inner: r.clone() })
            .collect()
    }

    // todo: String for now
    #[getter]
    fn secondary_structure(&self) -> Vec<String> {
        self.inner
            .secondary_structure
            .iter()
            .map(|s| format!("{s:?}"))
            .collect()
    }

    // todo: String for now
    #[getter]
    fn experimental_method(&self) -> Option<String> {
        match self.inner.experimental_method {
            Some(m) => Some(m.to_string()),
            None => None,
        }
    }

    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::MmCif::new(text)?,
        })
    }

    // todo: When implemented in rust.
    // fn save(&self, path: PathBuf) -> PyResult<()> {
    //     self.inner.save(&path)
    // }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: bio_files::MmCif::load(&path)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
