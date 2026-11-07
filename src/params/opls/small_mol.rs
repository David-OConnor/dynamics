//! Native OPLS-AA small-molecule typing using the bundled Foyer rule set.
//! Atom types, charges and bonded parameters all come from that same set.
//! No GAFF2 parameters or charge inference are used here.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fmt::Write,
};

use bio_files::{
    AtomGeneric, BondGeneric,
    gromacs::{
        GromacsFfParams, GromacsTopology,
        ff_params::{ItpAtomType, ItpTypeEntry},
    },
};
use na_seq::Element;

use super::smarts::{Graph, Pattern};
use crate::{
    ParamError,
    import::{ExplicitParams, gromacs_molecule_params},
};

const DATA: &str = include_str!("../../../param_data/opls/foyer-ligands.tsv");

#[derive(Debug)]
struct Rule {
    name: String,
    pattern: Pattern,
    overrides: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct SmallMolSet {
    /// Atom types, by name, with their class as bond type. Harmonic bonds and angles (funct 1),
    /// and Ryckaert-Bellemans torsions (funct 3), by class. "X" is a wildcard. Foyer stores these
    /// in GROMACS units, except for angles, which we convert from radians.
    pub(crate) tables: GromacsFfParams,
    rules: Vec<Rule>,
    known: super::known::Ligand,
    generator: super::ligpargen::Generator,
}

fn err(message: impl AsRef<str>) -> ParamError {
    ParamError::new(message.as_ref())
}

impl SmallMolSet {
    /// Cached/generated topologies supply all three assignments together. Retain the
    /// response so the separate charge and parameter stages never repeat a request.
    pub(crate) fn prepare_provider(
        &self,
        atoms: &[AtomGeneric],
        bonds: &[BondGeneric],
        network: crate::params::NetworkPolicy,
    ) -> Result<crate::params::ProvidedTopology, ParamError> {
        use crate::params::ProvidedTopology;
        let mut staged = atoms.to_vec();
        if let Ok(Some(parameters)) = self.generator.cached(&mut staged, bonds) {
            return ProvidedTopology::new(&staged, parameters, "cached LigParGen topology");
        }
        let bundled_error = match self.build_bundled(&mut staged, bonds) {
            Ok(parameters) => {
                return ProvidedTopology::new(
                    &staged,
                    parameters,
                    "bundled OPLS-AA templates/Foyer",
                );
            }
            Err(error) => error,
        };
        let (parameters, source) = self
            .generator
            .generate_with_policy(&mut staged, bonds, network)
            .map_err(|e| err(format!("{}; {}", bundled_error.descrip, e.descrip)))?;
        ProvidedTopology::new(&staged, parameters, source)
    }

    pub fn new() -> Result<Self, ParamError> {
        let mut result = Self {
            tables: GromacsFfParams::default(),
            rules: Vec::new(),
            known: super::known::Ligand::creatine()?,
            generator: Default::default(),
        };
        for line in DATA
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let cols: Vec<_> = line.split('\t').collect();
            let number = |i: usize| {
                cols.get(i)
                    .ok_or_else(|| err("Invalid bundled OPLS data"))?
                    .parse::<f64>()
                    .map_err(|_| err("Invalid bundled OPLS number"))
            };
            if cols[0] == "T" && cols.len() == 10 {
                let name = cols[1].to_owned();
                let atom_type = ItpAtomType {
                    name: name.clone(),
                    bond_type: cols[2].to_owned(),
                    atomic_number: Element::from_letter(cols[3])
                        .ok()
                        .map(|e| e.atomic_number() as u32),
                    mass: number(4)?,
                    charge: number(5)?,
                    ptype: "A".to_owned(),
                    v: number(6)?,
                    w: number(7)?,
                    comment: None,
                };

                let types = &mut result.tables.atom_types;
                match types.iter_mut().find(|t| t.name == name) {
                    Some(existing) => *existing = atom_type,
                    None => types.push(atom_type),
                }

                if !cols[9].is_empty() {
                    result.rules.push(Rule {
                        name,
                        pattern: Pattern::parse(cols[9])?,
                        overrides: cols[8]
                            .split(',')
                            .filter(|v| !v.is_empty())
                            .map(str::to_owned)
                            .collect(),
                    });
                }
                continue;
            }
            let arity = match cols[0] {
                "B" => 2,
                "A" => 3,
                "R" => 4,
                _ => return Err(err("Invalid bundled OPLS record")),
            };
            let expected = 1 + arity + if arity == 4 { 6 } else { 2 };
            if cols.len() != expected {
                return Err(err("Invalid bundled OPLS bonded record"));
            }
            let mut params: Vec<f64> = (1 + arity..cols.len())
                .map(number)
                .collect::<Result<_, _>>()?;
            if arity == 3 {
                params[0] = params[0].to_degrees();
            }

            let term = ItpTypeEntry {
                // Foyer leaves wildcards blank.
                types: cols[1..=arity]
                    .iter()
                    .map(|s| if s.is_empty() { "X" } else { s }.to_owned())
                    .collect(),
                funct: if arity == 4 { 3 } else { 1 },
                params,
                comment: None,
            };
            match arity {
                2 => result.tables.bond_types.push(term),
                3 => result.tables.angle_types.push(term),
                _ => result.tables.dihedral_types.push(term),
            }
        }
        Ok(result)
    }

    fn type_atoms(&self, atoms: &[AtomGeneric], graph: &Graph) -> Result<Vec<String>, ParamError> {
        let mut labels = vec![HashSet::<String>::new(); atoms.len()];
        let mut excluded = labels.clone();
        let mut converged = false;
        // Foyer's ordered, accumulating whitelist/blacklist iteration. Type
        // references see all matches, including subsequently overridden types.
        for _ in 0..10 {
            let mut changed = false;
            for rule in &self.rules {
                for i in 0..atoms.len() {
                    if !labels[i].contains(&rule.name) && rule.pattern.matches(i, graph, &labels) {
                        labels[i].insert(rule.name.clone());
                        excluded[i].extend(rule.overrides.iter().cloned());
                        changed = true;
                    }
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(err("OPLS ligand typing did not converge"));
        }
        let mut names = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            let candidates: BTreeSet<_> = labels[i].difference(&excluded[i]).cloned().collect();
            if candidates.len() != 1 {
                let reason = if candidates.is_empty() {
                    "no matching type".to_owned()
                } else {
                    format!(
                        "ambiguous types: {}",
                        candidates.into_iter().collect::<Vec<_>>().join(", ")
                    )
                };
                return Err(err(format!(
                    "OPLS-AA cannot type ligand atom {} ({}): {reason}. The bundled Foyer rules cover a subset of OPLS chemistry. Include all hydrogens; unsupported chemistry needs an explicit OPLS topology.",
                    atom.serial_number,
                    atom.element.to_letter(),
                )));
            }
            names.push(candidates.into_iter().next().unwrap());
        }
        Ok(names)
    }

    pub fn remember(&self, mol: &crate::MolDynamics) {
        self.generator.remember(mol);
    }

    pub fn build_cached(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        if let Ok(Some(params)) = self.generator.cached(atoms, bonds) {
            return Ok(params);
        }
        self.build_bundled(atoms, bonds)
    }

    pub fn build(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        if let Ok(params) = self.build_cached(atoms, bonds) {
            return Ok(params);
        }
        self.generator.generate(atoms, bonds)
    }

    fn build_bundled(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        if atoms.is_empty() {
            return Err(err("Cannot parameterize an empty OPLS ligand"));
        }
        let graph = Graph::new(atoms, bonds)?;
        if let Some(params) = self.known.build(atoms, bonds)? {
            return Ok(params);
        }
        let names = self.type_atoms(atoms, &graph)?;

        let by_name: HashMap<_, _> = self
            .tables
            .atom_types
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();
        let types = names
            .iter()
            .map(|n| {
                by_name
                    .get(n.as_str())
                    .copied()
                    .ok_or_else(|| err(format!("OPLS-AA atom type {n} has no parameters")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total_charge: f64 = types.iter().map(|a| a.charge).sum();
        if (total_charge - total_charge.round()).abs() > 0.005 {
            return Err(err(format!(
                "OPLS ligand typing gives a non-integral charge ({total_charge:.5} e); this chemical environment needs explicit OPLS parameters"
            )));
        }
        let mut text = String::from("[ defaults ]\n1 3 yes 0.5 0.5\n[ atomtypes ]\n");
        let mut seen = HashSet::new();
        for (i, t) in types.iter().enumerate() {
            if seen.insert(&names[i]) {
                writeln!(
                    text,
                    "{} {:.12} 0 A {:.12e} {:.12e}",
                    names[i], t.mass, t.v, t.w
                )
                .unwrap();
            }
        }
        text.push_str("[ moleculetype ]\nOPLS_LIGAND 3\n[ atoms ]\n");
        for (i, t) in types.iter().enumerate() {
            writeln!(
                text,
                "{} {} 1 LIG A{} {} {:.12} {:.12}",
                i + 1,
                names[i],
                i + 1,
                i + 1,
                t.charge,
                t.mass
            )
            .unwrap();
        }
        // Terms of other forms, e.g. from user-loaded tables, don't apply.
        let lookup = |ids: &[usize],
                      terms: &[ItpTypeEntry],
                      funct: u8,
                      n_params: usize,
                      specific: bool|
         -> Result<Vec<f64>, ParamError> {
            let classes: Vec<_> = ids.iter().map(|&i| types[i].bond_type.as_str()).collect();
            let mut found = None;
            for term in terms.iter().filter(|t| {
                t.types.len() == ids.len() && t.funct == funct && t.params.len() >= n_params
            }) {
                let matches = |reverse: bool| {
                    term.types.iter().enumerate().all(|(i, c)| {
                        c == "X" || c == classes[if reverse { ids.len() - 1 - i } else { i }]
                    })
                };
                if matches(false) || matches(true) {
                    if found.is_none() {
                        found = Some(term);
                    }
                    if !specific || term.types.iter().all(|c| c != "X") {
                        found = Some(term);
                        break;
                    }
                }
            }
            found.map(|t| t.params[..n_params].to_vec()).ok_or_else(|| err(format!(
                "Missing OPLS-AA {}-atom bonded parameter for {}; an explicit OPLS topology is required for this ligand",
                ids.len(), ids.iter().map(|&i| names[i].as_str()).collect::<Vec<_>>().join("-"),
            )))
        };
        text.push_str("[ bonds ]\n");
        for (i, neighbors) in graph.adj.iter().enumerate() {
            for &j in neighbors.iter().filter(|&&j| j > i) {
                let v = lookup(&[i, j], &self.tables.bond_types, 1, 2, false)?;
                writeln!(text, "{} {} 1 {:.12e} {:.12e}", i + 1, j + 1, v[0], v[1]).unwrap();
            }
        }
        text.push_str("[ angles ]\n");
        for (j, neighbors) in graph.adj.iter().enumerate() {
            for (a, &i) in neighbors.iter().enumerate() {
                for &k in &neighbors[a + 1..] {
                    let v = lookup(&[i, j, k], &self.tables.angle_types, 1, 2, false)?;
                    writeln!(
                        text,
                        "{} {} {} 1 {:.12e} {:.12e}",
                        i + 1,
                        j + 1,
                        k + 1,
                        v[0],
                        v[1]
                    )
                    .unwrap();
                }
            }
        }
        text.push_str("[ dihedrals ]\n");
        for (j, neighbors) in graph.adj.iter().enumerate() {
            for &k in neighbors.iter().filter(|&&k| k > j) {
                for &i in neighbors.iter().filter(|&&i| i != k) {
                    for &l in graph.adj[k].iter().filter(|&&l| l != j && l != i) {
                        let v = lookup(&[i, j, k, l], &self.tables.dihedral_types, 3, 6, true)?;
                        writeln!(
                            text,
                            "{} {} {} {} 3 {}",
                            i + 1,
                            j + 1,
                            k + 1,
                            l + 1,
                            v.iter()
                                .map(|c| format!("{c:.12e}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        )
                        .unwrap();
                    }
                }
            }
        }
        // Shortest-path exclusions: retain one scaled pair per 1-4 pair and
        // keep ring 1-2/1-3 pairs excluded even if a 3-bond path also exists.
        text.push_str("[ pairs ]\n");
        for start in 0..atoms.len() {
            let mut depth = vec![usize::MAX; atoms.len()];
            depth[start] = 0;
            let mut queue = VecDeque::from([start]);
            while let Some(i) = queue.pop_front() {
                if depth[i] == 3 {
                    continue;
                }
                for &j in &graph.adj[i] {
                    if depth[j] == usize::MAX {
                        depth[j] = depth[i] + 1;
                        queue.push_back(j);
                    }
                }
            }
            for (j, &d) in depth.iter().enumerate().skip(start + 1) {
                if d == 3 {
                    writeln!(text, "{} {} 1", start + 1, j + 1).unwrap();
                }
            }
        }
        let top =
            GromacsTopology::new(&text).map_err(|e| err(format!("OPLS ligand topology: {e}")))?;
        let params = gromacs_molecule_params(&top.molecule_types[0], &top)?;
        // Commit typing only after every atom and bonded term has parameters.
        for ((a, name), t) in atoms.iter_mut().zip(names).zip(types) {
            a.force_field_type = Some(name);
            a.partial_charge = Some(t.charge as f32);
        }
        Ok(params)
    }
}
