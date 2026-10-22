//! The three stages of classical parameterization, shared by every backend.

use std::{any::Any, fmt::Debug};

use bio_files::{AtomGeneric, md_params::ForceFieldParams};

use super::{ExplicitParams, FfParamSet, NonBondedRules};
use crate::{MolDynamics, ParamError};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NetworkPolicy {
    LocalOnly,
    /// Permit an optional provider when embedded tables and cached results are insufficient.
    #[default]
    AllowRemote,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ParameterizationOptions {
    pub network: NetworkPolicy,
    /// Recompute names even when the input has compatible assignments.
    pub reassign_atom_types: bool,
    pub charge_policy: ChargePolicy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChargePolicy {
    /// The backend's compatible input charges, residue charges or default charge model.
    #[default]
    BackendDefault,
    /// Recompute charges independently of whether type names are already present.
    Recompute,
    /// Use the caller's complete charge vector, independently of atom typing.
    Supplied,
}

impl ParameterizationOptions {
    pub fn local_only() -> Self {
        Self {
            network: NetworkPolicy::LocalOnly,
            ..Self::default()
        }
    }
}

/// A charge assignment is separate from both type names and energy parameters.
#[derive(Clone, Debug)]
pub struct PartialCharges {
    /// Elementary charge, in atom-slice order.
    pub values: Vec<f32>,
    /// E.g. Amber residue templates, GeoStd ML, Foyer or LigParGen.
    pub method: String,
}

#[derive(Clone, Debug)]
pub struct ParameterAssignment {
    pub explicit: ExplicitParams,
    /// Optional type-keyed overrides, for Amber FRCMOD consumers.
    pub overrides: Option<ForceFieldParams>,
}

impl From<ExplicitParams> for ParameterAssignment {
    fn from(explicit: ExplicitParams) -> Self {
        Self {
            explicit,
            overrides: None,
        }
    }
}

/// A backend selects atom names and retains the topology needed for subsequent stages.
/// These operations must not modify the input molecule. Providers that return names,
/// charges and terms together may retain that complete response in their typed topology.
pub trait ForceFieldBackend: Any + Debug + Send + Sync {
    fn name(&self) -> &str;

    fn assign_atom_types<'a>(
        &'a self,
        molecule: &'a MolDynamics,
        options: &ParameterizationOptions,
    ) -> Result<Box<dyn TypedTopology + 'a>, ParamError>;
}

/// Result of the naming stage. Charge assignment and parameter resolution are independent
/// operations: resolving bonded/LJ terms never requires running charge inference.
pub trait TypedTopology: Debug {
    fn atom_types(&self) -> &[String];
    fn source(&self) -> &str;
    fn assign_partial_charges(&self) -> Result<PartialCharges, ParamError>;
    fn assign_parameters(&self, rules: &NonBondedRules) -> Result<ParameterAssignment, ParamError>;
}

/// All three assignments, ready to apply atomically to the original molecule.
#[derive(Clone, Debug)]
pub struct Parameterization {
    pub atom_types: Vec<String>,
    pub type_source: String,
    pub partial_charges: PartialCharges,
    pub parameters: ParameterAssignment,
}

impl FfParamSet {
    pub fn assign_atom_types<'a>(
        &'a self,
        molecule: &'a MolDynamics,
        options: &ParameterizationOptions,
    ) -> Result<Box<dyn TypedTopology + 'a>, ParamError> {
        let adjacency = crate::util::build_adjacency_list(&molecule.atoms, &molecule.bonds)?;
        if let Some(cached) = &molecule.adjacency_list {
            let mut cached = cached.clone();
            for neighbors in &mut cached {
                neighbors.sort_unstable();
            }
            if cached != adjacency {
                return Err(ParamError::new(
                    "Adjacency cache does not match this molecule's bonds",
                ));
            }
        }
        let typed = self.backend.assign_atom_types(molecule, options)?;
        if options.charge_policy == ChargePolicy::Supplied {
            Ok(Box::new(WithSuppliedCharges {
                typed,
                atoms: &molecule.atoms,
            }))
        } else {
            Ok(typed)
        }
    }

    /// Run the three stages on an immutable molecule. No partial mutation on failure.
    pub fn assign(
        &self,
        molecule: &MolDynamics,
        options: &ParameterizationOptions,
    ) -> Result<Parameterization, ParamError> {
        if let Some(parameters) = &molecule.explicit_params {
            validate_parameters(parameters, molecule.atoms.len(), &self.nonbonded_rules)?;
            let charges = input_charges(&molecule.atoms)?;
            return Ok(Parameterization {
                atom_types: molecule
                    .atoms
                    .iter()
                    .map(|a| {
                        a.force_field_type
                            .clone()
                            .unwrap_or_else(|| "imported".into())
                    })
                    .collect(),
                type_source: "supplied topology".into(),
                partial_charges: PartialCharges {
                    values: charges,
                    method: "supplied topology".into(),
                },
                parameters: ParameterAssignment {
                    explicit: parameters.clone(),
                    overrides: molecule.mol_specific_params.clone(),
                },
            });
        }

        let typed = self.assign_atom_types(molecule, options)?;
        if typed.atom_types().len() != molecule.atoms.len()
            || typed.atom_types().iter().any(|t| t.is_empty())
        {
            return Err(ParamError::new("Invalid force-field atom type assignment"));
        }
        let partial_charges = typed.assign_partial_charges()?;
        if partial_charges.values.len() != molecule.atoms.len()
            || partial_charges.values.iter().any(|q| !q.is_finite())
        {
            return Err(ParamError::new("Invalid partial charge assignment"));
        }
        let parameters = typed.assign_parameters(&self.nonbonded_rules)?;
        validate_parameters(
            &parameters.explicit,
            molecule.atoms.len(),
            &self.nonbonded_rules,
        )?;
        Ok(Parameterization {
            atom_types: typed.atom_types().to_vec(),
            type_source: typed.source().into(),
            partial_charges,
            parameters,
        })
    }

    pub fn parameterize(&self, molecule: &MolDynamics) -> Result<MolDynamics, ParamError> {
        self.parameterize_with_options(molecule, &ParameterizationOptions::default())
    }

    pub fn parameterize_with_options(
        &self,
        molecule: &MolDynamics,
        options: &ParameterizationOptions,
    ) -> Result<MolDynamics, ParamError> {
        let assignment = self.assign(molecule, options)?;
        let mut result = molecule.clone();
        for ((atom, name), charge) in result
            .atoms
            .iter_mut()
            .zip(assignment.atom_types)
            .zip(assignment.partial_charges.values)
        {
            atom.force_field_type = Some(name);
            atom.partial_charge = Some(charge);
        }
        result.explicit_params = Some(assignment.parameters.explicit);
        result.mol_specific_params = assignment.parameters.overrides;
        Ok(result)
    }
}

#[derive(Debug)]
struct WithSuppliedCharges<'a> {
    typed: Box<dyn TypedTopology + 'a>,
    atoms: &'a [AtomGeneric],
}

impl TypedTopology for WithSuppliedCharges<'_> {
    fn atom_types(&self) -> &[String] {
        self.typed.atom_types()
    }
    fn source(&self) -> &str {
        self.typed.source()
    }
    fn assign_partial_charges(&self) -> Result<PartialCharges, ParamError> {
        Ok(PartialCharges {
            values: input_charges(self.atoms)?,
            method: "caller-supplied charges".into(),
        })
    }
    fn assign_parameters(&self, rules: &NonBondedRules) -> Result<ParameterAssignment, ParamError> {
        self.typed.assign_parameters(rules)
    }
}

pub(crate) fn input_charges(atoms: &[AtomGeneric]) -> Result<Vec<f32>, ParamError> {
    atoms
        .iter()
        .map(|a| {
            a.partial_charge.filter(|q| q.is_finite()).ok_or_else(|| {
                ParamError::new(&format!(
                    "Missing or invalid charge for atom {}",
                    a.serial_number
                ))
            })
        })
        .collect()
}

pub(crate) fn validate_parameters(
    parameters: &ExplicitParams,
    atom_count: usize,
    rules: &NonBondedRules,
) -> Result<(), ParamError> {
    if parameters.lj_combining != rules.lj_combining {
        return Err(ParamError::new(
            "Molecule and parameter set use different LJ combining rules",
        ));
    }
    parameters.validate(atom_count)
}

/// Joint provider result, used when an imported/cached/generated topology supplies all stages.
#[derive(Debug)]
pub(crate) struct ProvidedTopology {
    pub names: Vec<String>,
    pub charges: Vec<f32>,
    pub parameters: ExplicitParams,
    pub source: String,
}

impl ProvidedTopology {
    pub fn new(
        atoms: &[AtomGeneric],
        parameters: ExplicitParams,
        source: &str,
    ) -> Result<Self, ParamError> {
        Ok(Self {
            names: atoms
                .iter()
                .map(|a| {
                    a.force_field_type.clone().ok_or_else(|| {
                        ParamError::new("Provider returned an atom without a force-field type")
                    })
                })
                .collect::<Result<_, _>>()?,
            charges: input_charges(atoms)?,
            parameters,
            source: source.into(),
        })
    }
}

impl TypedTopology for ProvidedTopology {
    fn atom_types(&self) -> &[String] {
        &self.names
    }
    fn source(&self) -> &str {
        &self.source
    }
    fn assign_partial_charges(&self) -> Result<PartialCharges, ParamError> {
        Ok(PartialCharges {
            values: self.charges.clone(),
            method: self.source.clone(),
        })
    }
    fn assign_parameters(
        &self,
        _rules: &NonBondedRules,
    ) -> Result<ParameterAssignment, ParamError> {
        Ok(self.parameters.clone().into())
    }
}
