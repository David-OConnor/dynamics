//! Amber implementation of the common typing, charge and parameter interfaces.

use bio_files::{
    AtomGeneric,
    md_params::{ForceFieldParams, ForceFieldParamsIndexed},
};

use super::{AmberSet, charges, populate_peptide_ff_and_q, typing};
use crate::{
    FfMolType, MolDynamics, ParamError, merge_params,
    params::{
        ChargePolicy, ExplicitParams, ForceFieldBackend, NonBondedRules, ParameterAssignment,
        ParameterizationOptions, PartialCharges, TypedTopology,
    },
    util::build_adjacency_list,
};

impl AmberSet {
    pub fn parameters(&self, kind: FfMolType) -> Option<&ForceFieldParams> {
        match kind {
            FfMolType::Peptide => self.peptide.as_ref(),
            FfMolType::SmallOrganic => self.small_mol.as_ref(),
            FfMolType::Dna => self.dna.as_ref(),
            FfMolType::Rna => self.rna.as_ref(),
            FfMolType::Lipid => self.lipids.as_ref(),
            FfMolType::Carbohydrate => self.carbohydrates.as_ref(),
        }
    }
}

impl ForceFieldBackend for AmberSet {
    fn name(&self) -> &str {
        "Amber"
    }

    fn assign_atom_types<'a>(
        &'a self,
        molecule: &'a MolDynamics,
        options: &ParameterizationOptions,
    ) -> Result<Box<dyn TypedTopology + 'a>, ParamError> {
        let general = self
            .parameters(molecule.ff_mol_type)
            .ok_or_else(|| ParamError::new("Missing Amber parameters for this molecule class"))?;
        let mut atoms = molecule.atoms.clone();
        let compatible = atoms.iter().all(|a| {
            a.force_field_type.as_ref().is_some_and(|name| {
                general.mass.contains_key(name)
                    || molecule
                        .mol_specific_params
                        .as_ref()
                        .is_some_and(|p| p.mass.contains_key(name))
            })
        });
        let retype = options.reassign_atom_types || !compatible;
        let mut template_charges = None;
        let source = match molecule.ff_mol_type {
            FfMolType::SmallOrganic => {
                if retype {
                    let names = typing::assign_atom_types(&atoms, &molecule.bonds)?;
                    for (atom, name) in atoms.iter_mut().zip(names) {
                        if name.eq_ignore_ascii_case("du") {
                            return Err(ParamError::new(&format!(
                                "No GAFF2 type for atom {}",
                                atom.serial_number
                            )));
                        }
                        atom.force_field_type = Some(name);
                    }
                }
                if retype {
                    "GAFF2 rules"
                } else {
                    "supplied Amber types"
                }
            }
            FfMolType::Peptide => {
                if retype
                    || options.charge_policy == ChargePolicy::Recompute
                    || atoms.iter().any(|a| a.partial_charge.is_none())
                {
                    let mut templated = atoms.clone();
                    populate_peptide_ff_and_q(
                        &mut templated,
                        molecule.residues.as_deref().ok_or_else(|| {
                            ParamError::new("Amber protein assignment requires residues")
                        })?,
                        self.peptide_ff_q_map
                            .as_ref()
                            .ok_or_else(|| ParamError::new("Missing Amber residue templates"))?,
                    )?;
                    template_charges = Some(
                        templated
                            .iter()
                            .map(|a| {
                                a.partial_charge.ok_or_else(|| {
                                    ParamError::new("Incomplete Amber residue charge assignment")
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    if retype {
                        atoms = templated;
                    }
                }
                "Amber residue templates"
            }
            _ => {
                if retype {
                    return Err(ParamError::new(
                        "This Amber molecule class requires assigned template types",
                    ));
                }
                "supplied Amber template types"
            }
        };
        let names = atoms
            .iter()
            .map(|a| {
                a.force_field_type
                    .clone()
                    .ok_or_else(|| ParamError::new("Incomplete Amber atom type assignment"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Box::new(TypedAmber {
            set: self,
            molecule,
            atoms,
            names,
            source,
            template_charges,
            compatible_input: compatible,
            options: *options,
        }))
    }
}

#[derive(Debug)]
struct TypedAmber<'a> {
    set: &'a AmberSet,
    molecule: &'a MolDynamics,
    atoms: Vec<AtomGeneric>,
    names: Vec<String>,
    source: &'static str,
    template_charges: Option<Vec<f32>>,
    compatible_input: bool,
    options: ParameterizationOptions,
}

impl TypedTopology for TypedAmber<'_> {
    fn atom_types(&self) -> &[String] {
        &self.names
    }
    fn source(&self) -> &str {
        self.source
    }

    fn assign_partial_charges(&self) -> Result<PartialCharges, ParamError> {
        if self.options.charge_policy != ChargePolicy::Recompute
            && self.compatible_input
            && self
                .molecule
                .atoms
                .iter()
                .all(|a| a.partial_charge.is_some_and(|q| q.is_finite()))
        {
            return Ok(PartialCharges {
                values: self
                    .molecule
                    .atoms
                    .iter()
                    .map(|a| a.partial_charge.unwrap())
                    .collect(),
                method: "supplied Amber charges".into(),
            });
        }
        let (values, method) = if self.molecule.ff_mol_type == FfMolType::SmallOrganic {
            (
                charges::infer_charge(&self.atoms, &self.molecule.bonds)
                    .map_err(|e| ParamError::new(&format!("Amber GeoStd charge inference: {e}")))?,
                "Amber GeoStd ML",
            )
        } else if let Some(values) = &self.template_charges {
            (values.clone(), "Amber residue templates")
        } else {
            return Err(ParamError::new(
                "This Amber molecule class requires supplied template charges",
            ));
        };
        Ok(PartialCharges {
            values,
            method: method.into(),
        })
    }

    fn assign_parameters(&self, rules: &NonBondedRules) -> Result<ParameterAssignment, ParamError> {
        let general = self.set.parameters(self.molecule.ff_mol_type).unwrap();
        let adjacency = build_adjacency_list(&self.atoms, &self.molecule.bonds)?;
        let supplied = self.molecule.mol_specific_params.as_ref();
        let known = supplied.map_or_else(|| general.clone(), |p| merge_params(general, p));
        let overrides = if self.molecule.ff_mol_type == FfMolType::SmallOrganic {
            let missing = typing::assign_missing_params(&self.atoms, &adjacency, &known)?;
            Some(supplied.map_or(missing.clone(), |p| merge_params(&missing, p)))
        } else {
            supplied.cloned()
        };
        let parameters = overrides
            .as_ref()
            .map_or_else(|| general.clone(), |p| merge_params(general, p));
        // This compatibility resolver contains Amber-specific aliases and fallbacks.
        // Other table backends use strict `from_types` instead.
        let indexed = ForceFieldParamsIndexed::new(&parameters, &self.atoms, &adjacency, false)?;
        Ok(ParameterAssignment {
            explicit: ExplicitParams::from_indexed(
                &indexed,
                self.atoms.len(),
                rules,
                self.molecule.ff_mol_type,
            )?,
            overrides,
        })
    }
}
