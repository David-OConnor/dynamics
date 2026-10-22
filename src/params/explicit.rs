//! Resolved classical force-field terms in the engine's native units.

use bio_files::md_params::{
    AngleBendingParams, BondStretchingParams, DihedralParams, ForceFieldParamsIndexed, LjParams,
    MassParams,
};
use na_seq::Element;

use crate::{
    AtomDynamics, ParamError,
    non_bonded::{LjCombiningRule, NbFix, Pair14},
    params::charmm::cmap::CmapGrid,
};

/// Force field parameters for a molecule, per atom and per term, instead of by force field type.
/// Indices are into the molecule's atoms. Units are Amber's: Å, kcal/mol, radians, amu. Bond
/// energy is k_b (r − r_0)², and angle energy k (θ − θ_0)².
#[derive(Clone, Debug, Default)]
pub struct ExplicitParams {
    /// Per atom.
    pub masses: Vec<f32>,
    /// (σ, ε), per atom.
    pub lj: Vec<(f32, f32)>,
    pub bonds: Vec<((usize, usize), BondStretchingParams)>,
    pub angles: Vec<((usize, usize, usize), AngleBendingParams)>,
    /// Several terms may share the same atoms. `barrier_height` is per term; `divider` is ignored.
    pub dihedrals: Vec<((usize, usize, usize, usize), DihedralParams)>,
    pub impropers: Vec<((usize, usize, usize, usize), DihedralParams)>,
    /// Pairs with no non-bonded interaction: 1-2 and 1-3 pairs, and any others the source
    /// specifies. Not 1-4 pairs.
    pub exclusions: Vec<(usize, usize)>,
    /// 1-4 pairs, their scale factors, and LJ parameters if they differ from the atoms'.
    pub pairs_14: Vec<((usize, usize), Pair14)>,
    /// The combining rule these LJ parameters are for. It must match the rest of the system.
    pub lj_combining: LjCombiningRule,
    /// Urey-Bradley terms, between the outer atoms of angles: (k kcal/mol/Å², r_0 Å), with
    /// energy k (r − r_0)².
    pub urey_bradley: Vec<((usize, usize), (f32, f32))>,
    /// Harmonic impropers, e.g. CHARMM's: (k kcal/mol/rad², ψ_0 radians), with energy
    /// k (ψ − ψ_0)². The first atom is the central one.
    pub harmonic_impropers: Vec<((usize, usize, usize, usize), (f32, f32))>,
    /// CMAP terms: The atoms of φ, then ψ, and an index into `cmap_grids`.
    pub cmaps: Vec<([usize; 8], usize)>,
    pub cmap_grids: Vec<CmapGrid>,
    /// Pair-specific LJ parameters by force field type, (σ Å, ε kcal/mol), which override the
    /// combining rule. These apply to pairs across the system, e.g. between an ion and a
    /// protein; `MdState::new` combines them with the parameter set's.
    pub nbfix: NbFix,
}

impl ExplicitParams {
    /// Convert resolved type-keyed terms to the same representation used by all backends.
    pub fn from_indexed(
        indexed: &ForceFieldParamsIndexed,
        atom_count: usize,
        rules: &super::NonBondedRules,
        molecule_type: crate::FfMolType,
    ) -> Result<Self, ParamError> {
        use std::collections::VecDeque;
        let mut result = Self {
            lj_combining: rules.lj_combining,
            ..Self::default()
        };
        for i in 0..atom_count {
            result.masses.push(
                indexed
                    .mass
                    .get(&i)
                    .ok_or_else(|| ParamError::new("Missing resolved atom mass"))?
                    .mass,
            );
            let lj = indexed
                .lennard_jones
                .get(&i)
                .ok_or_else(|| ParamError::new("Missing resolved LJ parameters"))?;
            result.lj.push((lj.sigma, lj.eps));
        }
        result.bonds = indexed
            .bond_stretching
            .iter()
            .map(|(i, p)| (*i, p.clone()))
            .collect();
        result.angles = indexed.angle.iter().map(|(i, p)| (*i, p.clone())).collect();
        for (source, target) in [
            (&indexed.dihedral, &mut result.dihedrals),
            (&indexed.improper, &mut result.impropers),
        ] {
            for (indices, terms) in source {
                target.extend(terms.iter().map(|term| (*indices, term.clone())));
            }
        }
        let mut adjacency = vec![Vec::new(); atom_count];
        for &(i, j) in &indexed.bonds_topology {
            if i >= atom_count || j >= atom_count {
                return Err(ParamError::new("Resolved bond index out of range"));
            }
            adjacency[i].push(j);
            adjacency[j].push(i);
        }
        // Exclusions and scaled pairs follow shortest bond distance, including rings.
        for start in 0..atom_count {
            let mut depth = vec![usize::MAX; atom_count];
            depth[start] = 0;
            let mut queue = VecDeque::from([start]);
            while let Some(i) = queue.pop_front() {
                if depth[i] == 3 {
                    continue;
                }
                for &j in &adjacency[i] {
                    if depth[j] == usize::MAX {
                        depth[j] = depth[i] + 1;
                        queue.push_back(j);
                    }
                }
            }
            for (j, &distance) in depth.iter().enumerate().skip(start + 1) {
                match distance {
                    1 | 2 => result.exclusions.push((start, j)),
                    3 => result
                        .pairs_14
                        .push(((start, j), rules.scale_14.get(molecule_type).into())),
                    _ => (),
                }
            }
        }
        Ok(result)
    }

    /// Validate the common representation before an engine or exporter consumes it.
    pub fn validate(&self, atom_count: usize) -> Result<(), ParamError> {
        if self.masses.len() != atom_count
            || self.lj.len() != atom_count
            || self.masses.iter().any(|m| !m.is_finite() || *m <= 0.)
            || self
                .lj
                .iter()
                .any(|(s, e)| !s.is_finite() || !e.is_finite() || *s < 0. || *e < 0.)
        {
            return Err(ParamError::new(
                "Invalid resolved per-atom masses or LJ parameters",
            ));
        }
        let check = |ids: &[usize]| {
            if ids.iter().any(|i| *i >= atom_count) {
                Err(ParamError::new(
                    "Resolved parameter atom index out of range",
                ))
            } else {
                Ok(())
            }
        };
        for &((i, j), ref p) in &self.bonds {
            check(&[i, j])?;
            if !p.k_b.is_finite() || !p.r_0.is_finite() {
                return Err(ParamError::new("Invalid resolved bond parameter"));
            }
        }
        for &((i, j, k), ref p) in &self.angles {
            check(&[i, j, k])?;
            if !p.k.is_finite() || !p.theta_0.is_finite() {
                return Err(ParamError::new("Invalid resolved angle parameter"));
            }
        }
        for &((i, j, k, l), ref p) in self.dihedrals.iter().chain(&self.impropers) {
            check(&[i, j, k, l])?;
            if !p.barrier_height.is_finite() || !p.phase.is_finite() {
                return Err(ParamError::new("Invalid resolved torsion parameter"));
            }
        }
        for &(i, j) in &self.exclusions {
            check(&[i, j])?;
        }
        for &((i, j), pair) in &self.pairs_14 {
            check(&[i, j])?;
            if !pair.scale.lj.is_finite()
                || !pair.scale.coulomb.is_finite()
                || pair
                    .lj
                    .is_some_and(|(s, e)| !s.is_finite() || !e.is_finite() || s < 0. || e < 0.)
            {
                return Err(ParamError::new("Invalid resolved 1-4 parameters"));
            }
        }
        for &((i, j), (k, r)) in &self.urey_bradley {
            check(&[i, j])?;
            if !k.is_finite() || !r.is_finite() {
                return Err(ParamError::new("Invalid Urey-Bradley parameter"));
            }
        }
        for &((i, j, k, l), (force, angle)) in &self.harmonic_impropers {
            check(&[i, j, k, l])?;
            if !force.is_finite() || !angle.is_finite() {
                return Err(ParamError::new("Invalid harmonic improper"));
            }
        }
        for &(indices, grid) in &self.cmaps {
            check(&indices)?;
            if grid >= self.cmap_grids.len() {
                return Err(ParamError::new("Invalid CMAP grid index"));
            }
        }
        Ok(())
    }

    /// Index these parameters, as we do for parameters assigned by type. `atoms` are this
    /// molecule's atoms.
    pub(crate) fn to_indexed(
        &self,
        atoms: &[AtomDynamics],
        h_constrained: bool,
    ) -> Result<ForceFieldParamsIndexed, ParamError> {
        let n = atoms.len();
        if self.masses.len() != n || self.lj.len() != n {
            return Err(ParamError::new(&format!(
                "Explicit parameters have {} masses and {} LJ entries, for {n} atoms",
                self.masses.len(),
                self.lj.len()
            )));
        }
        let check = |indices: &[usize]| -> Result<(), ParamError> {
            match indices.iter().find(|i| **i >= n) {
                Some(i) => Err(ParamError::new(&format!(
                    "Explicit parameter atom index {i} is out of range for {n} atoms"
                ))),
                None => Ok(()),
            }
        };

        let mut result = ForceFieldParamsIndexed::default();

        for (i, atom) in atoms.iter().enumerate() {
            let atom_type = atom.force_field_type.clone();
            result.mass.insert(
                i,
                MassParams {
                    atom_type: atom_type.clone(),
                    mass: self.masses[i],
                    comment: None,
                },
            );
            result.lennard_jones.insert(
                i,
                LjParams {
                    atom_type,
                    sigma: self.lj[i].0,
                    eps: self.lj[i].1,
                },
            );
        }

        for &((i, j), ref params) in &self.bonds {
            check(&[i, j])?;
            let key = (i.min(j), i.max(j));
            result.bonds_topology.insert(key);

            // As for parameters assigned by type, constrained bonds to H are rigid.
            if h_constrained
                && (atoms[i].element == Element::Hydrogen || atoms[j].element == Element::Hydrogen)
            {
                let inv_mass = 1. / self.masses[i] + 1. / self.masses[j];
                result
                    .bond_rigid_constraints
                    .insert(key, (params.r_0.powi(2), inv_mass));
            } else {
                result.bond_stretching.insert(key, params.clone());
            }
        }

        for &((i, ctr, k), ref params) in &self.angles {
            check(&[i, ctr, k])?;
            // Canonical order, as for parameters assigned by type.
            let key = (i.min(k), ctr, i.max(k));
            if result.angle.insert(key, params.clone()).is_some() {
                return Err(ParamError::new(&format!(
                    "Multiple angle terms for atoms {i}-{ctr}-{k}"
                )));
            }
        }

        for (map, terms) in [
            (&mut result.dihedral, &self.dihedrals),
            (&mut result.improper, &self.impropers),
        ] {
            for &((a, b, c, d), ref params) in terms {
                check(&[a, b, c, d])?;
                map.entry((a, b, c, d)).or_default().push(DihedralParams {
                    divider: 1,
                    ..params.clone()
                });
            }
        }

        Ok(result)
    }
}
