//! The SMARTS subset used by the bundled Foyer OPLS-AA definitions: elements,
//! coordination, chordless rings, type labels, logical operators and branches.
//! Patterns use induced subgraph matching, as in Foyer's SMARTSGraph.

use std::collections::{HashMap, HashSet};

use bio_files::{AtomGeneric, BondGeneric};
use na_seq::Element;

use crate::ParamError;

#[derive(Debug)]
enum Expr {
    Any,
    Element(u8),
    Degree(usize),
    RingSize(usize),
    RingCount(usize),
    Label(String),
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
}

impl Expr {
    fn parse(text: &str) -> Result<Self, ParamError> {
        // Weak AND has lower precedence than OR in SMARTS.
        for (delimiter, conjunction) in [(';', true), (',', false), ('&', true)] {
            if text.contains(delimiter) {
                let expressions = text
                    .split(delimiter)
                    .map(Self::parse)
                    .collect::<Result<_, _>>()?;
                return Ok(if conjunction {
                    Self::And(expressions)
                } else {
                    Self::Or(expressions)
                });
            }
        }
        if let Some(rest) = text.strip_prefix('!') {
            return Ok(Self::Not(Box::new(Self::parse(rest)?)));
        }
        if text == "*" {
            return Ok(Self::Any);
        }
        if let Some(label) = text.strip_prefix('%') {
            return Ok(Self::Label(label.to_owned()));
        }
        let number = |s: &str| s.parse::<usize>().map_err(|_| invalid(text));
        if let Some(n) = text.strip_prefix('X') {
            return Ok(Self::Degree(number(n)?));
        }
        if let Some(n) = text.strip_prefix('r') {
            return Ok(Self::RingSize(number(n)?));
        }
        if let Some(n) = text.strip_prefix('R') {
            return Ok(Self::RingCount(number(n)?));
        }
        let element = if let Some(n) = text.strip_prefix('#') {
            u8::try_from(number(n)?).map_err(|_| invalid(text))?
        } else if text == "Li" {
            // The bundled rules include lithium, which AtomGeneric's Element
            // enum cannot currently represent. Retain the rule as non-matching
            // for supported input elements rather than failing to load OPLS.
            3
        } else {
            Element::from_letter(text)
                .map_err(|_| invalid(text))?
                .atomic_number()
        };
        Ok(Self::Element(element))
    }

    fn matches(&self, i: usize, graph: &Graph, labels: &[HashSet<String>]) -> bool {
        match self {
            Self::Any => true,
            Self::Element(e) => graph.elements[i].atomic_number() == *e,
            Self::Degree(n) => graph.adj[i].len() == *n,
            Self::RingSize(n) => graph.rings[i].contains(n),
            Self::RingCount(n) => graph.rings[i].len() == *n,
            Self::Label(label) => labels[i].contains(label),
            Self::Not(e) => !e.matches(i, graph, labels),
            Self::And(v) => v.iter().all(|e| e.matches(i, graph, labels)),
            Self::Or(v) => v.iter().any(|e| e.matches(i, graph, labels)),
        }
    }
}

fn invalid(text: &str) -> ParamError {
    ParamError::new(&format!("Unsupported OPLS typing SMARTS: {text}"))
}

#[derive(Debug)]
pub(crate) struct Pattern {
    nodes: Vec<Expr>,
    adj: Vec<Vec<usize>>,
}

impl Pattern {
    pub fn parse(text: &str) -> Result<Self, ParamError> {
        let mut result = Self {
            nodes: Vec::new(),
            adj: Vec::new(),
        };
        let bytes = text.as_bytes();
        let mut pos = 0;
        let mut current: Option<usize> = None;
        let mut branches = Vec::new();
        let mut closures: HashMap<u8, usize> = HashMap::new();
        while pos < bytes.len() {
            match bytes[pos] {
                b'(' => {
                    branches.push(current.ok_or_else(|| invalid(text))?);
                    pos += 1;
                    continue;
                }
                b')' => {
                    current = Some(branches.pop().ok_or_else(|| invalid(text))?);
                    pos += 1;
                    continue;
                }
                b'0'..=b'9' => {
                    let atom = current.ok_or_else(|| invalid(text))?;
                    if let Some(other) = closures.remove(&bytes[pos]) {
                        result.adj[atom].push(other);
                        result.adj[other].push(atom);
                    } else {
                        closures.insert(bytes[pos], atom);
                    }
                    pos += 1;
                    continue;
                }
                _ => (),
            }
            let start = pos;
            let expression = if bytes[pos] == b'[' {
                pos += 1;
                let end = text[pos..].find(']').ok_or_else(|| invalid(text))? + pos;
                let expression = Expr::parse(&text[pos..end])?;
                pos = end + 1;
                expression
            } else {
                pos += 1;
                if pos < bytes.len() && bytes[pos].is_ascii_lowercase() {
                    pos += 1;
                }
                Expr::parse(&text[start..pos])?
            };
            let node = result.nodes.len();
            result.nodes.push(expression);
            result.adj.push(Vec::new());
            if let Some(parent) = current {
                result.adj[node].push(parent);
                result.adj[parent].push(node);
            }
            current = Some(node);
        }
        if !branches.is_empty() || !closures.is_empty() || result.nodes.is_empty() {
            return Err(invalid(text));
        }
        Ok(result)
    }

    pub fn matches(&self, root: usize, graph: &Graph, labels: &[HashSet<String>]) -> bool {
        if self.nodes.len() > graph.adj.len() || !self.nodes[0].matches(root, graph, labels) {
            return false;
        }

        let mut mapped = vec![None; self.nodes.len()];
        mapped[0] = Some(root);
        self.search(graph, labels, &mut mapped)
    }

    fn search(
        &self,
        graph: &Graph,
        labels: &[HashSet<String>],
        mapped: &mut [Option<usize>],
    ) -> bool {
        let next = (0..self.nodes.len())
            .filter(|&p| mapped[p].is_none())
            .max_by_key(|&p| self.adj[p].iter().filter(|&&q| mapped[q].is_some()).count());
        let Some(p) = next else {
            return true;
        };

        let parent = self.adj[p].iter().find_map(|&q| mapped[q]);
        let Some(parent) = parent else {
            return false;
        };

        for &i in &graph.adj[parent] {
            if mapped.contains(&Some(i))
                || graph.adj[i].len() < self.adj[p].len()
                || !self.nodes[p].matches(i, graph, labels)
            {
                continue;
            }
            // Induced matching: extra bonds between mapped vertices disqualify
            // a match, including chords in a SMARTS ring.
            if mapped.iter().enumerate().any(|(q, m)| {
                m.is_some_and(|j| self.adj[p].contains(&q) != graph.adj[i].contains(&j))
            }) {
                continue;
            }

            mapped[p] = Some(i);
            if self.search(graph, labels, mapped) {
                return true;
            }
            mapped[p] = None;
        }
        false
    }
}

#[derive(Debug)]
pub(crate) struct Graph {
    pub adj: Vec<Vec<usize>>,
    elements: Vec<Element>,
    // One entry per chordless cycle containing each atom; duplicates in size
    // represent distinct cycles. Foyer evaluates cycles of up to eight atoms.
    rings: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> Result<Self, ParamError> {
        let mut indices = HashMap::new();
        for (i, a) in atoms.iter().enumerate() {
            if indices.insert(a.serial_number, i).is_some() {
                return Err(ParamError::new(
                    "Duplicate atom serial number in OPLS ligand",
                ));
            }
        }
        let mut adj = vec![Vec::new(); atoms.len()];
        for bond in bonds {
            let (&i, &j) = indices
                .get(&bond.atom_0_sn)
                .zip(indices.get(&bond.atom_1_sn))
                .ok_or_else(|| ParamError::new("OPLS ligand bond references an absent atom"))?;
            if i == j || adj[i].contains(&j) {
                return Err(ParamError::new("Duplicate or self bond in OPLS ligand"));
            }
            adj[i].push(j);
            adj[j].push(i);
        }

        for neighbors in &mut adj {
            neighbors.sort_unstable();
        }

        let mut rings = vec![Vec::new(); atoms.len()];
        for start in 0..atoms.len() {
            for &next in &adj[start] {
                if next > start {
                    visit(&mut vec![start, next], &adj, &mut rings);
                }
            }
        }

        Ok(Self {
            adj,
            elements: atoms.iter().map(|a| a.element).collect(),
            rings,
        })
    }
}

fn visit(path: &mut Vec<usize>, adj: &[Vec<usize>], rings: &mut [Vec<usize>]) {
    let start = path[0];
    let last = *path.last().unwrap();
    for &next in &adj[last] {
        if next <= start || path.contains(&next) {
            continue;
        }
        if path[1..path.len() - 1]
            .iter()
            .any(|v| adj[next].contains(v))
        {
            continue;
        }
        if adj[next].contains(&start) {
            if path[1] < next {
                for &i in path.iter() {
                    rings[i].push(path.len() + 1);
                }
                rings[next].push(path.len() + 1);
            }
        } else if path.len() < 7 {
            path.push(next);
            visit(path, adj, rings);
            path.pop();
        }
    }
}
