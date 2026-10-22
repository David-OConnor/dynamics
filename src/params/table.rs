//! A backend for any classical table set with already assigned type names and charges.
//! It uses strict lookup: Amber aliases and estimated substitutions are not applied.

use bio_files::md_params::{ForceFieldParams, ForceFieldParamsIndexed};

use super::{
    ExplicitParams, ForceFieldBackend, NonBondedRules, ParameterAssignment,
    ParameterizationOptions, PartialCharges, TypedTopology, parameterize::input_charges,
};
use crate::{MolDynamics, ParamError, merge_params, util::build_adjacency_list};

#[derive(Debug, Default)]
pub struct TableForceField {
    pub name: String,
    pub parameters: ForceFieldParams,
}

impl ForceFieldBackend for TableForceField {
    fn name(&self) -> &str {
        &self.name
    }

    fn assign_atom_types<'a>(
        &'a self,
        molecule: &'a MolDynamics,
        _options: &ParameterizationOptions,
    ) -> Result<Box<dyn TypedTopology + 'a>, ParamError> {
        let names = molecule
            .atoms
            .iter()
            .map(|a| {
                a.force_field_type.clone().ok_or_else(|| {
                    ParamError::new("Table force fields require assigned atom types")
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Box::new(TypedTable {
            set: self,
            molecule,
            names,
        }))
    }
}

#[derive(Debug)]
struct TypedTable<'a> {
    set: &'a TableForceField,
    molecule: &'a MolDynamics,
    names: Vec<String>,
}

impl TypedTopology for TypedTable<'_> {
    fn atom_types(&self) -> &[String] {
        &self.names
    }
    fn source(&self) -> &str {
        &self.set.name
    }

    fn assign_partial_charges(&self) -> Result<PartialCharges, ParamError> {
        Ok(PartialCharges {
            values: input_charges(&self.molecule.atoms)?,
            method: "supplied charges".into(),
        })
    }

    fn assign_parameters(&self, rules: &NonBondedRules) -> Result<ParameterAssignment, ParamError> {
        let adjacency = build_adjacency_list(&self.molecule.atoms, &self.molecule.bonds)?;
        let parameters = match &self.molecule.mol_specific_params {
            Some(overrides) => merge_params(&self.set.parameters, overrides),
            None => self.set.parameters.clone(),
        };
        let indexed = ForceFieldParamsIndexed::from_types(
            &parameters,
            &self.molecule.atoms,
            &adjacency,
            false,
        )?;
        Ok(ParameterAssignment {
            explicit: ExplicitParams::from_indexed(
                &indexed,
                self.names.len(),
                rules,
                self.molecule.ff_mol_type,
            )?,
            overrides: self.molecule.mol_specific_params.clone(),
        })
    }
}
