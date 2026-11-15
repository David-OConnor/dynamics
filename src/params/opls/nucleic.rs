//! OPLS-AA/RESP nucleotide templates from the GROMACS ffoplsaanr contribution.

use std::{
    collections::{BTreeSet, HashMap},
    io,
};

use bio_files::gromacs::Rtp;

use super::{TypedOplsPolymer, error};
use crate::{
    FfMolType, MolDynamics, ParamError,
    params::{ForceFieldBackend, ParameterizationOptions, TypedTopology},
};

const TABLES: &str = include_str!("../../../param_data/opls/nucleotide_params.itp");
const RESIDUES: &str = include_str!("../../../param_data/opls/nucleotides.rtp");
const SOURCE: &str = "OPLS-AA/RESP nucleotide templates";

#[derive(Debug)]
pub(crate) struct NucleotideSet {
    templates: Rtp,
}

impl NucleotideSet {
    pub(crate) fn bundled() -> io::Result<Self> {
        Ok(Self {
            templates: Rtp::new(RESIDUES)?,
        })
    }

    fn type_nucleotides(&self, mol: &MolDynamics) -> Result<TypedOplsPolymer, ParamError> {
        let dna = match mol.ff_mol_type {
            FfMolType::Dna => true,
            FfMolType::Rna => false,
            _ => return Err(error("OPLS-AA/RESP requires a DNA or RNA molecule")),
        };
        let n = mol.atoms.len();
        let indices: HashMap<_, _> = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| (atom.serial_number, i))
            .collect();
        if n == 0 || indices.len() != n {
            return Err(error(
                "OPLS nucleotide assignment requires unique atom serial numbers",
            ));
        }

        let names: Vec<_> = mol
            .atoms
            .iter()
            .map(|a| {
                let label = a
                    .type_in_res_general
                    .clone()
                    .unwrap_or_else(|| super::name(a));
                atom_name(&label, dna)
            })
            .collect();

        // Cutting the phosphodiester bonds isolates nucleotides, even with grouped or absent
        // display residues. The same bonds identify each strand's termini.
        let mut within = vec![Vec::new(); n];
        let mut links = Vec::new();
        for bond in &mol.bonds {
            let (&a, &b) = indices
                .get(&bond.atom_0_sn)
                .zip(indices.get(&bond.atom_1_sn))
                .ok_or_else(|| error("OPLS nucleotide bond refers to an absent atom"))?;
            match (names[a].as_str(), names[b].as_str()) {
                ("O3'", "P") => links.push((a, b)),
                ("P", "O3'") => links.push((b, a)),
                _ => {
                    within[a].push(b);
                    within[b].push(a);
                }
            }
        }

        let mut owner = vec![usize::MAX; n];
        let mut groups = Vec::new();
        for start in 0..n {
            if owner[start] != usize::MAX {
                continue;
            }

            let mut group = HashMap::new();
            let mut stack = vec![start];
            owner[start] = groups.len();
            while let Some(i) = stack.pop() {
                if group.insert(names[i].as_str(), i).is_some() {
                    return Err(error(format!(
                        "Duplicate OPLS nucleotide atom {} near atom {}",
                        names[i], mol.atoms[start].serial_number
                    )));
                }

                for &j in &within[i] {
                    if owner[j] == usize::MAX {
                        owner[j] = groups.len();
                        stack.push(j);
                    }
                }
            }
            groups.push(group);
        }

        let mut has_previous = vec![false; groups.len()];
        let mut has_next = vec![false; groups.len()];
        let mut adjacency = vec![BTreeSet::new(); n];
        for (o3, p) in links {
            let (from, to) = (owner[o3], owner[p]);
            if has_next[from] || has_previous[to] {
                return Err(error(
                    "OPLS nucleotide chains cannot have multiple O3'-P bonds at an end",
                ));
            }

            has_next[from] = true;
            has_previous[to] = true;
            adjacency[o3].insert(p);
            adjacency[p].insert(o3);
        }

        let mut types = vec![String::new(); n];
        let mut charges = vec![0.; n];
        for (ri, group) in groups.iter().enumerate() {
            let base = base_name(group, dna).ok_or_else(|| {
                error(format!(
                    "Unsupported or incomplete OPLS nucleotide {}",
                    ri + 1
                ))
            })?;
            let suffix = match (has_previous[ri], has_next[ri]) {
                (false, false) => "N",
                (false, true) => "5",
                (true, false) => "3",
                (true, true) => "",
            };
            let key = format!("{base}{suffix}");
            let template = &self.templates.residues[&key];
            let at = |name: &str| {
                group.get(name).copied().ok_or_else(|| {
                    error(format!(
                        "OPLS nucleotide {key} {} is missing {name}",
                        ri + 1
                    ))
                })
            };

            for atom in &template.atoms {
                let i = at(&atom.name)?;
                types[i].clone_from(&atom.ff_type);
                charges[i] = atom.charge;
            }
            for (a, b) in &template.bonds {
                let (a, b) = (at(a)?, at(b)?);
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }

        if let Some(i) = types.iter().position(String::is_empty) {
            return Err(error(format!(
                "Unexpected atom {} in OPLS nucleotide {}",
                names[i],
                owner[i] + 1
            )));
        }

        Ok(TypedOplsPolymer {
            tables: TABLES.to_owned(),
            source: SOURCE,
            comb_rule: 2,
            types,
            charges,
            previous: Vec::new(),
            next: Vec::new(),
            mapped: HashMap::new(),
            instances: Vec::new(),
            adjacency,
            owner,
        })
    }
}

impl ForceFieldBackend for NucleotideSet {
    fn name(&self) -> &str {
        SOURCE
    }

    fn assign_atom_types<'a>(
        &'a self,
        molecule: &'a MolDynamics,
        _options: &ParameterizationOptions,
    ) -> Result<Box<dyn TypedTopology + 'a>, ParamError> {
        Ok(Box::new(self.type_nucleotides(molecule)?))
    }
}

fn base_name(atoms: &HashMap<&str, usize>, dna: bool) -> Option<&'static str> {
    let has = |name| atoms.contains_key(name);
    let name = match (dna, has("N9"), has("N6"), has("N4"), has("C7")) {
        (true, true, true, false, false) => "DA",
        (true, true, false, false, false) => "DG",
        (true, false, false, true, false) => "DC",
        (true, false, false, false, true) => "DT",
        (false, true, true, false, false) => "A",
        (false, true, false, false, false) => "G",
        (false, false, false, true, false) => "C",
        (false, false, false, false, false) if has("O4") => "U",
        _ => return None,
    };

    Some(name)
}

fn atom_name(label: &str, dna: bool) -> String {
    let name = label.trim().to_ascii_uppercase();
    let name = match name.as_str() {
        "H2`" => "HO2'".into(),
        "H3`" => "HO3'".into(),
        "H5`" => "HO5'".into(),
        _ => name.replace(['*', '`'], "'"),
    };
    let canonical = match name.as_str() {
        "O1P" => "OP1",
        "O2P" => "OP2",
        "H5'1" | "1H5'" => "H5'",
        "H5'2" | "2H5'" => "H5''",
        "H2'1" | "1H2'" => "H2'",
        "H2'2" | "2H2'" if dna => "H2''",
        "H2''" | "H2'2" | "2H2'" if !dna => "HO2'",
        "H3T" => "HO3'",
        "H5T" => "HO5'",
        "C5M" => "C7",
        "H5M1" => "H71",
        "H5M2" => "H72",
        "H5M3" => "H73",
        _ => &name,
    };

    canonical.to_owned()
}
