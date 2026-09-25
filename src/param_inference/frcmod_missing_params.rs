//! Enumerate unique bonded type combinations. Improper centers follow the
//! PARMCHK improper flag and must have exactly three neighbors (parmchk2).

use std::{collections::BTreeSet, io};

use bio_files::{AtomGeneric, md_params::ForceFieldParams};

use super::{parmchk_parse::PARAMETERS, topology::invalid};

#[derive(Default)]
pub(super) struct MissingParams {
    pub bond: BTreeSet<(String, String)>,
    pub angle: BTreeSet<(String, String, String)>,
    pub dihedral: BTreeSet<(String, String, String, String)>,
    pub improper: BTreeSet<(String, String, String, String)>,
}
impl MissingParams {
    pub fn new(
        atoms: &[AtomGeneric],
        adj: &[Vec<usize>],
        params: &ForceFieldParams,
    ) -> io::Result<Self> {
        if adj.len() != atoms.len() {
            return Err(invalid("Adjacency length differs from atom count"));
        }

        let types = atoms
            .iter()
            .map(|a| {
                a.force_field_type
                    .as_deref()
                    .filter(|t| !t.is_empty())
                    .ok_or_else(|| {
                        invalid(format!(
                            "Missing force field type for atom {}",
                            a.serial_number
                        ))
                    })
            })
            .collect::<io::Result<Vec<_>>>()?;

        for (i, neighbors) in adj.iter().enumerate() {
            let unique: BTreeSet<_> = neighbors.iter().copied().collect();
            if unique.len() != neighbors.len()
                || neighbors
                    .iter()
                    .any(|&j| j >= atoms.len() || j == i || !adj[j].contains(&i))
            {
                return Err(invalid(
                    "Invalid adjacency: expected symmetric, distinct, in-range neighbors",
                ));
            }
        }

        let mut result = Self::default();

        for (i, neighbors) in adj.iter().enumerate() {
            for &j in neighbors {
                if i >= j {
                    continue;
                }
                let key = if types[i] < types[j] {
                    (types[i].into(), types[j].into())
                } else {
                    (types[j].into(), types[i].into())
                };
                if params.get_bond(&key, false).is_none() {
                    result.bond.insert(key);
                }
                for &a in neighbors.iter().filter(|&&a| a != j) {
                    for &d in adj[j].iter().filter(|&&d| d != i && d != a) {
                        let forward = (
                            types[a].into(),
                            types[i].into(),
                            types[j].into(),
                            types[d].into(),
                        );
                        let reverse = (
                            types[d].into(),
                            types[j].into(),
                            types[i].into(),
                            types[a].into(),
                        );
                        let key = forward.min(reverse);
                        // Exact only: parmchk2 tries equivalent specific torsions
                        // before the generic X-b-c-X term for the original types.
                        if params.get_dihedral(&key, true, false).is_none() {
                            result.dihedral.insert(key);
                        }
                    }
                }
            }

            for a in 0..neighbors.len() {
                for b in a + 1..neighbors.len() {
                    let (a, b) = (types[neighbors[a]], types[neighbors[b]]);
                    let key = (a.min(b).into(), types[i].into(), a.max(b).into());
                    if params.get_valence_angle(&key, false).is_none() {
                        result.angle.insert(key);
                    }
                }
            }

            if neighbors.len() == 3 && PARAMETERS.improper(types[i]) {
                let mut outer = [
                    types[neighbors[0]],
                    types[neighbors[1]],
                    types[neighbors[2]],
                ];
                outer.sort_unstable();
                let key = (
                    outer[0].into(),
                    outer[1].into(),
                    types[i].into(),
                    outer[2].into(),
                );
                // Do not reverse an improper: its central atom is always third.
                // Matching all satellite permutations is done in frcmod.
                result.improper.insert(key);
            }
        }

        Ok(result)
    }
}
