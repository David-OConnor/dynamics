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

/// For DEF col f9 constraints
#[derive(Debug, Clone)]
pub(super) struct NeighborPattern {
    base: NeighborBase,
    ring_size: Option<u8>,    // RGn on the neighbor, if present
    requires_aromatic: bool,  // AR1/AR2/AR3
    requires_db: bool,        // DB
    requires_tb: bool,        // TB
    other_flags: Vec<String>, // sb, sb', DL, XA1, etc (for future extension)
}

impl NeighborPattern {
    pub(super) fn matches_neighbor(
        &self,
        center: usize,
        nb: usize,
        atoms: &[AtomGeneric],
        env_all: &[AtomEnvData],
        bonds: &[BondGeneric],
    ) -> bool {
        let nb_atom = &atoms[nb];

        match &self.base {
            NeighborBase::AnyXX => {}
            NeighborBase::Code(code) => {
                match code.as_str() {
                    "C" | "C2" | "C3" => {
                        if nb_atom.element != Element::Carbon {
                            return false;
                        }
                    }
                    "O" | "O1" => {
                        if nb_atom.element != Element::Oxygen {
                            return false;
                        }
                    }
                    "N" | "N2" => {
                        if nb_atom.element != Element::Nitrogen {
                            return false;
                        }
                    }
                    "P" | "P2" => {
                        if nb_atom.element != Element::Phosphorus {
                            return false;
                        }
                    }
                    _ => {
                        // Unknown base code (XB2, XD3, XA1, ...) – be conservative.
                        return false;
                    }
                }
            }
        }

        if self.requires_aromatic && !env_all[nb].is_aromatic {
            return false;
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

        true
    }
}

/// For DEF col f9 constraints
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
        ) -> bool {
            if pat_idx == pattern.len() {
                return true;
            }

            let pat = &pattern[pat_idx];

            for (n_i, &nb) in heavy_neighbors.iter().enumerate() {
                if used[n_i] {
                    continue;
                }

                if pat.matches_neighbor(center, nb, atoms, env_all, bonds) {
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
        )
    }
}

impl From<&str> for ChemEnvPattern {
    fn from(s: &str) -> Self {
        let s = s.trim();
        if s.is_empty() || s == "&" {
            return Self { neighbors: Vec::new() };
        }

        let inner = if s.starts_with('(') && s.ends_with(')') && s.len() >= 2 {
            &s[1..s.len() - 1]
        } else {
            s
        };

        let mut neighbors = Vec::new();

        // No nested []/() in these patterns, so a simple split is fine.
        for raw in inner.split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }

            // Split into base and [flags]
            let (base_str, flags_str) = if let Some(pos) = raw.find('[') {
                // assume last char is ']' if '[' exists; DEF format is well-formed
                (&raw[..pos], Some(&raw[pos + 1..raw.len() - 1]))
            } else {
                (raw, None)
            };

            let base = match base_str {
                "XX" => NeighborBase::AnyXX,
                _ => NeighborBase::Code(base_str.to_string()),
            };

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
                        "DB" => {
                            requires_db = true;
                        }
                        "TB" => {
                            requires_tb = true;
                        }
                        other => {
                            other_flags.push(other.to_string());
                        }
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