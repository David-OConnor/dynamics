//! Classical force-field parameterization.
//!
//! Load a backend, assign atom types, assign partial charges, then resolve the
//! bonded and nonbonded terms. Native dynamics and exporters consume the same result.

use std::{any::Any, io};

use amber::{AmberSet, ParamGeneralPaths};
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use charmm::CharmmSet;

use crate::{
    FfMolType,
    non_bonded::{LjCombiningRule, NbFix, Scale14},
    solvent::WaterModel,
};

pub mod amber;
pub mod charmm;
mod explicit;
pub mod opls;
mod parameterize;
mod table;

pub use explicit::ExplicitParams;
use opls::OplsSet;
pub(crate) use parameterize::ProvidedTopology;
pub use parameterize::*;
pub use table::TableForceField;

/// A family of force fields, with its conventions and recommended water model. We resolve this
/// to data once, with `FfParamSet::from_family`; the rest of the library uses that data, and
/// doesn't match on the family.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForceFieldFamily {
    /// ff19SB (proteins), OL24 (DNA), OL3 (RNA), lipid21, and GAFF2 (small molecules), with
    /// OPC water. Amber's recommendations as of Sept 2025.
    #[default]
    Amber,
    /// CHARMM36m for proteins, with CHARMM's TIP3P water (with LJ on H), and SOD and CLA ions.
    /// Includes Urey-Bradley terms, harmonic impropers, CMAP, special 1-4 LJ parameters, and
    /// NBFIX. We build proteins from CHARMM's residue topologies; they need
    /// `MolDynamics::residues`. Other molecule types need explicit parameters. (e.g. imported;
    /// see `import`) Use `MdConfig::for_family` for CHARMM's cutoffs and LJ force switch.
    Charmm36,
    /// OPLS-AA/L proteins (Kaminski et al., 2001), with TIP4P water and
    /// OPLS ions. Uses geometric LJ mixing and 0.5 scaling for both 1-4 terms.
    /// Proteins require `MolDynamics::residues` and explicit hydrogens.
    /// Hydrogenated ligands use Foyer's bundled OPLS-AA SMARTS rules, charges
    /// and bonded tables. Other ligand chemistry uses cached or local LigParGen/BOSS
    /// assignments, then the remote service if allowed. Other molecule classes
    /// require explicit OPLS parameters; GAFF2 is not used with this family.
    Opls,
    /// A caller-supplied classical force field. Construct its set with `FfParamSet::custom`.
    Custom,
}

/// 1-4 scale factors for each molecule type.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scale14ByMolType {
    pub peptide: Scale14,
    pub small_organic: Scale14,
    pub dna: Scale14,
    pub rna: Scale14,
    pub lipid: Scale14,
    /// Set this to `Scale14::GLYCAM` if loading GLYCAM parameters.
    pub carbohydrate: Scale14,
}

impl Scale14ByMolType {
    pub fn get(&self, mol_type: FfMolType) -> Scale14 {
        match mol_type {
            FfMolType::Peptide => self.peptide,
            FfMolType::SmallOrganic => self.small_organic,
            FfMolType::Dna => self.dna,
            FfMolType::Rna => self.rna,
            FfMolType::Lipid => self.lipid,
            FfMolType::Carbohydrate => self.carbohydrate,
        }
    }
}

/// Non-bonded conventions that are part of a force field's definition, rather than of individual
/// atom types. These must match the parameters they're used with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NonBondedRules {
    /// Applies to all pairs in the system, so all molecules' parameters must use the same rule.
    pub lj_combining: LjCombiningRule,
    pub scale_14: Scale14ByMolType,
    /// Pair-specific LJ parameters by force field type, which override the combining rule. (e.g.
    /// CHARMM's NBFIX)
    pub nbfix: NbFix,
}

/// A classical force field consists of a parameterization backend and system conventions.
/// The engine consumes resolved terms, independently of the backend or file formats.
#[derive(Debug)]
pub struct FfParamSet {
    pub family: ForceFieldFamily,
    pub nonbonded_rules: NonBondedRules,
    pub default_water: WaterModel,
    pub backend: Box<dyn ForceFieldBackend>,
}

impl Default for FfParamSet {
    fn default() -> Self {
        Self {
            family: ForceFieldFamily::Amber,
            nonbonded_rules: NonBondedRules::default(),
            default_water: WaterModel::default(),
            backend: Box::new(TableForceField::default()),
        }
    }
}

impl FfParamSet {
    /// All three built-in backends load their tables from compile-time embedded data.
    pub fn from_family(family: ForceFieldFamily) -> io::Result<Self> {
        match family {
            ForceFieldFamily::Amber => Self::new_amber(),
            ForceFieldFamily::Charmm36 => Self::new_charmm(),
            ForceFieldFamily::Opls => Self::new_opls(),
            ForceFieldFamily::Custom => Err(io::Error::other(
                "Custom force fields require a backend, nonbonded rules and water model",
            )),
        }
    }

    pub fn new_amber() -> io::Result<Self> {
        Ok(Self {
            backend: Box::new(AmberSet::bundled()?),
            ..Self::default()
        })
    }

    /// Load caller-supplied Amber DAT/FRCMOD/LIB files.
    pub fn new(paths: &ParamGeneralPaths) -> io::Result<Self> {
        Self::from_amber_paths(paths)
    }

    pub fn from_amber_paths(paths: &ParamGeneralPaths) -> io::Result<Self> {
        Ok(Self {
            backend: Box::new(AmberSet::from_paths(paths)?),
            ..Self::default()
        })
    }

    pub fn new_charmm() -> io::Result<Self> {
        let backend = CharmmSet::bundled()?;
        let rules = NonBondedRules {
            lj_combining: LjCombiningRule::LorentzBerthelot,
            scale_14: Scale14ByMolType::uniform(Scale14 {
                lj: 1.,
                coulomb: 1.,
            }),
            nbfix: backend.nbfix(),
        };
        Ok(Self {
            family: ForceFieldFamily::Charmm36,
            backend: Box::new(backend),
            nonbonded_rules: rules,
            default_water: WaterModel::TIP3P_CHARMM,
        })
    }

    pub fn new_opls() -> io::Result<Self> {
        Ok(Self {
            family: ForceFieldFamily::Opls,
            backend: Box::new(OplsSet::bundled()?),
            nonbonded_rules: NonBondedRules {
                lj_combining: LjCombiningRule::Geometric,
                scale_14: Scale14ByMolType::uniform(Scale14::OPLS),
                nbfix: NbFix::default(),
            },
            default_water: WaterModel::TIP4P_OPLS,
        })
    }

    /// Add a force field without changing the simulation or export engines.
    /// Implement `ForceFieldBackend` for typing and `TypedTopology` for charges and terms,
    /// or use `TableForceField` with already named/charged atoms and classical parameter tables.
    pub fn custom(
        backend: impl ForceFieldBackend + 'static,
        nonbonded_rules: NonBondedRules,
        default_water: WaterModel,
    ) -> Self {
        Self {
            family: ForceFieldFamily::Custom,
            backend: Box::new(backend),
            nonbonded_rules,
            default_water,
        }
    }

    pub fn amber(&self) -> Option<&AmberSet> {
        (&*self.backend as &dyn Any).downcast_ref()
    }

    pub fn amber_mut(&mut self) -> Option<&mut AmberSet> {
        (&mut *self.backend as &mut dyn Any).downcast_mut()
    }

    pub fn charmm(&self) -> Option<&CharmmSet> {
        (&*self.backend as &dyn Any).downcast_ref()
    }

    pub fn charmm_mut(&mut self) -> Option<&mut CharmmSet> {
        (&mut *self.backend as &mut dyn Any).downcast_mut()
    }

    pub fn opls(&self) -> Option<&OplsSet> {
        (&*self.backend as &dyn Any).downcast_ref()
    }

    pub fn opls_mut(&mut self) -> Option<&mut OplsSet> {
        (&mut *self.backend as &mut dyn Any).downcast_mut()
    }

    /// Type-keyed tables, where exposed by the backend. Template-based force fields
    /// expose their resolved terms through `parameterize` instead.
    pub fn parameters(
        &self,
        molecule_type: FfMolType,
    ) -> Option<&bio_files::md_params::ForceFieldParams> {
        if let Some(amber) = self.amber() {
            return amber.parameters(molecule_type);
        }
        (&*self.backend as &dyn Any)
            .downcast_ref::<TableForceField>()
            .map(|table| &table.parameters)
    }
}

impl Scale14ByMolType {
    pub fn uniform(scale: Scale14) -> Self {
        Self {
            peptide: scale,
            small_organic: scale,
            dna: scale,
            rna: scale,
            lipid: scale,
            carbohydrate: scale,
        }
    }
}
