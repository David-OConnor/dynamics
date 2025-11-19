//! For using and parsing the DEF f9 column, which specifies chemical environment.
//! This is used to handle many "edge" cases using DEF data, without bespoke logic.

use bio_files::{AtomGeneric, BondGeneric, BondType};
use na_seq::Element;

use crate::param_inference::AtomEnvData;

/// For DEF col f9 constraints
#[derive(Debug, Clone, PartialEq, Eq)]
enum NeighborBase {
    AnyXX,
    Code(String), // e.g. "C3", "O1", "N2", "XB2", "XD3", ...
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SecondHopPattern {
    base: NeighborBase,
}

impl SecondHopPattern {
    fn matches_second_hop(
        &self,
        center: usize,
        nb: usize,
        atoms: &[AtomGeneric],
        _env_all: &[AtomEnvData],
        bonds: &[BondGeneric],
        adj: &[Vec<usize>],
    ) -> bool {
        match &self.base {
            NeighborBase::Code(code) if code == "XA1" => {
                // C3(XA1): neighbor is a carbonyl carbon
                is_xa1_like_carbon(nb, atoms, bonds)
            }
            NeighborBase::Code(code) if code == "C3" || code == "C2" || code == "XB2" => {
                // For C3(C3), C3(C2), XB2(C3), XB2(C2), XB2(XB2) we approximate:
                // neighbor-of-neighbor (excluding center) must be a heavy atom, usually C.
                adj[nb]
                    .iter()
                    .copied()
                    .filter(|&j| j != center)
                    .any(|j| atoms[j].element != Element::Hydrogen)
            }
            NeighborBase::AnyXX => {
                // Any neighbor-of-neighbor (excluding center)
                adj[nb].iter().any(|&j| j != center)
            }
            _ => false,
        }
    }
}

/// For DEF col f9 constraints
#[derive(Debug, Clone)]
pub(super) struct NeighborPattern {
    base: NeighborBase,
    ring_size: Option<u8>,    // RGn on the neighbor, if present
    requires_aromatic: bool,  // AR1/AR2/AR3
    requires_db: bool,        // DB
    requires_tb: bool,        // TB
    other_flags: Vec<String>, // sb, sb', DL, XA1, etc (for future extension)\
    second_hop: Option<SecondHopPattern>,
}

impl NeighborPattern {
    pub(super) fn matches_neighbor(
        &self,
        center: usize,
        nb: usize,
        atoms: &[AtomGeneric],
        env_all: &[AtomEnvData],
        bonds: &[BondGeneric],
        adj: &[Vec<usize>],
    ) -> bool {
        let nb_atom = &atoms[nb];

        match &self.base {
            NeighborBase::AnyXX => {}
            NeighborBase::Code(code) => {
                match code.as_str() {
                    "C" | "C2" | "C3" | "C4" => {
                        if nb_atom.element != Element::Carbon {
                            return false;
                        }
                    }
                    "O" | "O1" => {
                        if nb_atom.element != Element::Oxygen {
                            return false;
                        }
                    }
                    "N" | "N1" | "N2" | "N3" => {
                        if nb_atom.element != Element::Nitrogen {
                            return false;
                        }
                    }
                    "P" | "P2" | "P3" | "P4" => {
                        if nb_atom.element != Element::Phosphorus {
                            return false;
                        }
                    }
                    // XB2, XD3, XD4, XA1, etc are handled either via second_hop
                    // or in other_flags; for now we don't accept them as bare bases.
                    _ => return false,
                }
            }
        }

        if self.requires_aromatic {
            let env = &env_all[nb];
            if !env.is_aromatic && env.ring_sizes.is_empty() {
                return false;
            }
        }

        if let Some(rs) = self.ring_size {
            if !env_all[nb].ring_sizes.contains(&rs) {
                return false;
            }
        }

        if self.requires_db || self.requires_tb {
            let mut has_db = false;
            let mut has_tb = false;

            for b in bonds {
                let i = b.atom_0_sn as usize - 1;
                let j = b.atom_1_sn as usize - 1;
                if (i == center && j == nb) || (i == nb && j == center) {
                    match b.bond_type {
                        BondType::Double => has_db = true,
                        BondType::Triple => has_tb = true,
                        _ => {}
                    }
                }
            }

            if self.requires_db && !has_db {
                return false;
            }
            if self.requires_tb && !has_tb {
                return false;
            }
        }

        if let Some(ref hop) = self.second_hop {
            if !hop.matches_second_hop(center, nb, atoms, env_all, bonds, adj) {
                return false;
            }
        }

        true
    }
}

/// For DEF col f9 constraints. Corresponds to a line from a Def file.
#[derive(Debug, Clone)]
pub(super) struct ChemEnvPattern {
    neighbors: Vec<NeighborPattern>,
}

impl ChemEnvPattern {
    pub(super) fn matches(
        &self,
        idx: usize,
        atoms: &[AtomGeneric],
        env_all: &[AtomEnvData],
        bonds: &[BondGeneric],
        adj: &[Vec<usize>],
    ) -> bool {
        if self.neighbors.is_empty() {
            return true;
        }

        let heavy_neighbors: Vec<usize> = adj[idx]
            .iter()
            .copied()
            .filter(|&j| atoms[j].element != Element::Hydrogen)
            .collect();

        if heavy_neighbors.len() < self.neighbors.len() {
            return false;
        }

        fn backtrack(
            pat_idx: usize,
            pattern: &[NeighborPattern],
            center: usize,
            heavy_neighbors: &[usize],
            used: &mut [bool],
            atoms: &[AtomGeneric],
            env_all: &[AtomEnvData],
            bonds: &[BondGeneric],
            adj: &[Vec<usize>],
        ) -> bool {
            if pat_idx == pattern.len() {
                return true;
            }

            let pat = &pattern[pat_idx];

            for (n_i, &nb) in heavy_neighbors.iter().enumerate() {
                if used[n_i] {
                    continue;
                }

                if pat.matches_neighbor(center, nb, atoms, env_all, bonds, adj) {
                    used[n_i] = true;
                    if backtrack(
                        pat_idx + 1,
                        pattern,
                        center,
                        heavy_neighbors,
                        used,
                        atoms,
                        env_all,
                        bonds,
                        adj,
                    ) {
                        return true;
                    }
                    used[n_i] = false;
                }
            }

            false
        }

        let mut used = vec![false; heavy_neighbors.len()];
        backtrack(
            0,
            &self.neighbors,
            idx,
            &heavy_neighbors,
            &mut used,
            atoms,
            env_all,
            bonds,
            adj,
        )
    }
}

impl From<&str> for ChemEnvPattern {
    /// Parse a line from a DEF file.
    fn from(s: &str) -> Self {
        let s = s.trim();
        if s.is_empty() || s == "&" {
            return Self {
                neighbors: Vec::new(),
            };
        }

        let inner = if s.starts_with('(') && s.ends_with(')') && s.len() >= 2 {
            &s[1..s.len() - 1]
        } else {
            s
        };

        let mut neighbors = Vec::new();

        for raw in split_env_neighbors(inner) {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }

            // First split off [flags] if present.
            let (main, flags_str) = if let Some(pos) = raw.find('[') {
                (&raw[..pos], Some(&raw[pos + 1..raw.len() - 1])) // strip '[' and ']'
            } else {
                (raw, None)
            };

            // Then split off nested "(...)" if present, e.g. C3(XA1), C3(C3), XB2(C2)
            let (base_token, nested_token) = if let Some(pos) = main.find('(') {
                (&main[..pos], Some(&main[pos + 1..main.len() - 1])) // strip '(' and ')'
            } else {
                (main, None)
            };

            let base = match base_token {
                "XX" => NeighborBase::AnyXX,
                _ => NeighborBase::Code(base_token.to_string()),
            };

            let second_hop = nested_token.map(|t| SecondHopPattern {
                base: if t == "XX" {
                    NeighborBase::AnyXX
                } else {
                    NeighborBase::Code(t.to_string())
                },
            });

            let mut ring_size = None;
            let mut requires_aromatic = false;
            let mut requires_db = false;
            let mut requires_tb = false;
            let mut other_flags = Vec::new();

            if let Some(flags) = flags_str {
                for tok in flags.split(|c| c == '.' || c == ',') {
                    let tok = tok.trim();
                    if tok.is_empty() {
                        continue;
                    }
                    match tok {
                        t if t.starts_with("RG") => {
                            if let Ok(n) = t[2..].parse::<u8>() {
                                ring_size = Some(n);
                            } else {
                                other_flags.push(t.to_string());
                            }
                        }
                        t if t.starts_with("AR") => {
                            requires_aromatic = true;
                            other_flags.push(t.to_string());
                        }
                        "DB" => requires_db = true,
                        "TB" => requires_tb = true,
                        other => other_flags.push(other.to_string()),
                    }
                }
            }

            neighbors.push(NeighborPattern {
                base,
                ring_size,
                requires_aromatic,
                requires_db,
                requires_tb,
                other_flags,
                second_hop,
            });
        }

        Self { neighbors }
    }
}

impl From<String> for ChemEnvPattern {
    fn from(s: String) -> Self {
        ChemEnvPattern::from(s.as_str())
    }
}

fn split_env_neighbors(inner: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth_bracket = 0u8;

    for c in inner.chars() {
        match c {
            '[' => {
                depth_bracket += 1;
                current.push(c);
            }
            ']' => {
                if depth_bracket > 0 {
                    depth_bracket -= 1;
                }
                current.push(c);
            }
            ',' if depth_bracket == 0 => {
                if !current.trim().is_empty() {
                    result.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }

    if !current.trim().is_empty() {
        result.push(current.trim().to_string());
    }

    result
}

fn is_xa1_like_carbon(idx: usize, atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> bool {
    let atom = &atoms[idx];
    if atom.element != Element::Carbon {
        return false;
    }

    for b in bonds {
        let i = b.atom_0_sn as usize - 1;
        let j = b.atom_1_sn as usize - 1;
        if i == idx || j == idx {
            let other = if i == idx { j } else { i };
            if atoms[other].element == Element::Oxygen && matches!(b.bond_type, BondType::Double) {
                return true;
            }
        }
    }

    false
}

pub(super) fn is_carbonyl_carbon(idx: usize, atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> bool {
    use na_seq::Element::Oxygen;

    let mut neighbors: Vec<usize> = Vec::new();
    let mut has_co_double = false;

    for b in bonds {
        let i = b.atom_0_sn as usize - 1;
        let j = b.atom_1_sn as usize - 1;

        if i == idx || j == idx {
            let other = if i == idx { j } else { i };

            if !neighbors.contains(&other) {
                neighbors.push(other);
            }

            if atoms[other].element == Oxygen && matches!(b.bond_type, BondType::Double) {
                has_co_double = true;
            }
        }
    }

    // Standard carbonyl: explicit C=O double bond.
    if has_co_double {
        return true;
    }

    // Carboxylate / carboxylic-type: trigonal carbon with two O neighbors.
    let o_neighbors = neighbors
        .iter()
        .filter(|&&nb| atoms[nb].element == Oxygen)
        .count();

    if neighbors.len() == 3 && o_neighbors == 2 {
        return true;
    }

    false
}
