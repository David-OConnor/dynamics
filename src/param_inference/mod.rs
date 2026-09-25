//! GAFF2 atom typing and estimation of missing bonded parameters.
//!
//! Atom types are assigned by evaluating the bundled ATOMTYPE_GFF2.DEF in order,
//! then applying Amber's conjugated-type pairing. See Wang et al., Antechamber,
//! sections 2.2-2.3: https://ambermd.org/antechamber/antechamber.pdf, and
//! https://github.com/Amber-MD/AmberClassic/tree/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber.
//!
//! Inputs must include hydrogens and chemically meaningful bond orders. This is
//! not a replacement for Antechamber's bond-order/valence search or AM1-BCC:
//! the high-level helper uses this crate's separately trained charge model.
//! Explicit Kekule bonds are recommended for charged aromatic systems because
//! AtomGeneric lacks formal charges. See `topology` for aromatic input handling.

mod chem_env;
pub(crate) mod frcmod;
mod frcmod_missing_params;
mod parmchk_parse;
mod post_process;
#[cfg(test)]
mod tests;
mod topology;

use std::{borrow::Cow, collections::HashMap, io, sync::LazyLock};

use bio_files::{
    AtomGeneric, BondGeneric,
    amber_typedef::{AmberDef, AtomTypeDef, WildAtom},
    md_params::ForceFieldParams,
};
use chem_env::{ChemEnvPattern, Properties};
pub use frcmod::assign_missing_params;
use na_seq::Element;
use topology::Topology;

use crate::partial_charge_inference::infer_charge;

const DEF_ABCG2: &str = include_str!("../../param_data/antechamber_defs/ATOMTYPE_ABCG2.DEF");
const DEF_GFF2: &str = include_str!("../../param_data/antechamber_defs/ATOMTYPE_GFF2.DEF");

/// Bundled Amber rule sets. ABCG2 is a charge-typing definition, not a source of
/// bonded FRCMOD overrides; the latter use PARMCHK.DAT and the GAFF2 parameters.
pub struct AmberDefSet {
    pub abcg2: AmberDef,
    pub gff2: AmberDef,
}
impl AmberDefSet {
    pub fn new() -> io::Result<Self> {
        // The bio_files DEF parser maps unsupported atomic numbers to Zinc. Exclude those
        // rows instead of accidentally assigning a metal the type of another element.
        fn parse(text: &str) -> io::Result<AmberDef> {
            let supported = text
                .lines()
                .filter(|line| {
                    let cols: Vec<_> = line.split_whitespace().collect();
                    cols.first() != Some(&"ATD")
                        || cols
                            .get(3)
                            .and_then(|s| s.parse::<u8>().ok())
                            .is_none_or(|n| Element::from_atomic_number(n).is_ok())
                })
                .collect::<Vec<_>>()
                .join("\n");
            AmberDef::new(&supported)
        }
        Ok(Self {
            abcg2: parse(DEF_ABCG2)?,
            gff2: parse(DEF_GFF2)?,
        })
    }
}
static DEFAULT_DEFS: LazyLock<AmberDefSet> =
    LazyLock::new(|| AmberDefSet::new().expect("bundled Amber definitions"));

// Cache by source text rather than a pointer to AmberDefSet: its public fields
// can be edited by callers. Custom rules still compile normally, while repeated
// calls with the bundled rules reuse immutable syntax trees across threads.
static PROPERTIES: LazyLock<HashMap<&'static str, Properties>> = LazyLock::new(|| {
    DEFAULT_DEFS
        .gff2
        .atomtypes
        .iter()
        .map(|def| def.atomic_property.as_deref().unwrap_or("*"))
        .map(|text| {
            (
                text,
                Properties::parse(text).expect("bundled property syntax"),
            )
        })
        .collect()
});

static ENVIRONMENTS: LazyLock<HashMap<&'static str, ChemEnvPattern>> = LazyLock::new(|| {
    DEFAULT_DEFS
        .gff2
        .atomtypes
        .iter()
        .map(|def| def.chem_env.as_deref().unwrap_or("*"))
        .map(|text| {
            (
                text,
                ChemEnvPattern::parse(text).expect("bundled environment syntax"),
            )
        })
        .collect()
});

/// Precomputed Amber atom properties. Ring and aromatic fields are *counts*:
/// a fused atom may belong to several rings and to more than one AR class.
#[derive(Debug)]
pub struct AtomEnvData {
    degree: usize,
    num_attached_h: usize,
    rings: [usize; 11],
    aromatic: [usize; 5],
    bonds: Vec<(usize, topology::BondKind)>,
}

fn is_elec_withdrawing_element(el: Element) -> bool {
    use Element::*;
    matches!(
        el,
        Oxygen | Nitrogen | Sulfur | Fluorine | Chlorine | Bromine | Iodine
    )
}

struct CompiledDef<'a> {
    def: &'a AtomTypeDef,
    properties: Cow<'static, Properties>,
    environment: Cow<'static, ChemEnvPattern>,
}

impl<'a> CompiledDef<'a> {
    fn new(def: &'a AtomTypeDef) -> io::Result<Self> {
        let prop_text = def.atomic_property.as_deref().unwrap_or("*");
        let env_text = def.chem_env.as_deref().unwrap_or("*");
        let properties = PROPERTIES
            .get(prop_text)
            .map(Cow::Borrowed)
            .or_else(|| Properties::parse(prop_text).map(Cow::Owned));
        let environment = ENVIRONMENTS
            .get(env_text)
            .map(Cow::Borrowed)
            .or_else(|| ChemEnvPattern::parse(env_text).map(Cow::Owned));
        match (properties, environment) {
            (Some(properties), Some(environment)) => Ok(Self {
                def,
                properties,
                environment,
            }),
            _ => Err(topology::invalid(format!(
                "Unsupported DEF syntax for {}",
                def.name
            ))),
        }
    }

    fn matches(
        &self,
        idx: usize,
        atoms: &[AtomGeneric],
        env: &[AtomEnvData],
        wild: &[WildAtom],
    ) -> bool {
        let def = self.def;
        let data = &env[idx];
        if def.element.is_some_and(|el| atoms[idx].element != el)
            || def
                .attached_atoms
                .is_some_and(|n| data.degree != n as usize)
            || def
                .attached_h
                .is_some_and(|n| data.num_attached_h != n as usize)
            || def.residue.as_deref().is_some_and(|res| res != "&")
        {
            return false;
        }
        if let Some(required) = def.electron_withdrawal_count {
            let Some(&(parent, _)) = data.bonds.first() else {
                return false;
            };
            let count = env[parent]
                .bonds
                .iter()
                .filter(|&&(nb, _)| is_elec_withdrawing_element(atoms[nb].element))
                .count();
            if count != required as usize {
                return false;
            }
        }
        self.properties.matches(idx, None, env) && self.environment.matches(idx, atoms, env, wild)
    }
}

/// Match a single GAFF definition against precomputed atom environments.
/// The bundled GAFF wildcard element groups are used by this compatibility API.
pub fn matches_def(
    def: &AtomTypeDef,
    idx: usize,
    atoms: &[AtomGeneric],
    env_all: &[AtomEnvData],
    _bonds: &[BondGeneric],
    _adj: &[Vec<usize>],
) -> bool {
    idx < atoms.len()
        && env_all.len() == atoms.len()
        && CompiledDef::new(def)
            .is_ok_and(|rule| rule.matches(idx, atoms, env_all, &DEFAULT_DEFS.gff2.wildatoms))
}

fn assign_types(
    atoms: &[AtomGeneric],
    topology: &Topology,
    defs: &AmberDefSet,
) -> io::Result<Vec<String>> {
    // Compile once per molecule, not for every atom/DEF combination. Retain
    // file order: Amber uses the first matching rule, not a specificity score.
    let rules = defs
        .gff2
        .atomtypes
        .iter()
        .map(CompiledDef::new)
        .collect::<io::Result<Vec<_>>>()?;
    let mut result = (0..atoms.len())
        .map(|idx| {
            rules
                .iter()
                .find(|rule| rule.matches(idx, atoms, &topology.env, &defs.gff2.wildatoms))
                .map_or_else(|| "du".to_owned(), |rule| rule.def.name.clone())
        })
        .collect::<Vec<_>>();
    post_process::adjust_conjugated_types(&topology.env, &mut result);
    Ok(result)
}

/// Checked GAFF2 typing. Serial numbers may be arbitrary unique identifiers.
/// Invalid graphs, unsupported bond types, and unresolved aromatic orders return
/// an error. An unmatched atom receives Amber's DU type; no bonded parameter is implied.
pub fn try_find_ff_types(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    defs: &AmberDefSet,
) -> io::Result<Vec<String>> {
    let topology = Topology::new(atoms, bonds)?;
    assign_types(atoms, &topology, defs)
}

/// Compatibility interface. Prefer `try_find_ff_types` to obtain input diagnostics.
/// Invalid input returns one dummy type per atom instead of panicking or indexing
/// unrelated atoms. Valid inputs use exactly the same rules as the checked API.
pub fn find_ff_types(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    defs: &AmberDefSet,
) -> Vec<String> {
    try_find_ff_types(atoms, bonds, defs).unwrap_or_else(|_| vec!["du".to_owned(); atoms.len()])
}

/// Assign GAFF2 types, infer charges using the trained model, and estimate missing
/// bonded parameters. Changes to atoms are committed only when all steps succeed.
/// A supplied adjacency cache must describe the same graph in atom-slice order.
pub fn update_small_mol_params(
    atoms: &mut [AtomGeneric],
    bonds: &[BondGeneric],
    adjacency_list: Option<&[Vec<usize>]>,
    gaff2: &ForceFieldParams,
) -> io::Result<ForceFieldParams> {
    let topology = Topology::new(atoms, bonds)?;
    if let Some(adj) = adjacency_list
        && (adj.len() != atoms.len()
            || adj.iter().zip(&topology.adj).any(|(given, expected)| {
                let mut given = given.clone();
                given.sort_unstable();
                given != *expected
            }))
    {
        return Err(topology::invalid("Adjacency cache does not match bonds"));
    }
    let types = assign_types(atoms, &topology, &DEFAULT_DEFS)?;
    let mut staged = atoms.to_vec();
    for (atom, ty) in staged.iter_mut().zip(types) {
        if ty.eq_ignore_ascii_case("du") {
            return Err(topology::invalid(format!(
                "No GAFF2 type for atom {}",
                atom.serial_number
            )));
        }
        atom.force_field_type = Some(ty);
    }

    let params = assign_missing_params(&staged, &topology.adj, gaff2)?;
    // The charge model uses endpoint - 1 internally; normalize only its copy.
    let charge = infer_charge(&staged, &topology.normalized_bonds).map_err(io::Error::other)?;
    if charge.len() != atoms.len() || charge.iter().any(|v| !v.is_finite()) {
        return Err(io::Error::other("Invalid inferred charge vector"));
    }

    for ((atom, staged), charge) in atoms.iter_mut().zip(staged).zip(charge) {
        atom.force_field_type = staged.force_field_type;
        atom.partial_charge = Some(charge);
    }

    Ok(params)
}
