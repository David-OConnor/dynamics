//! Import systems parameterized outside this library: Amber topologies (`prmtop`), as written by
//! tleap, ParmEd, CHARMM-GUI, and OpenFF Interchange, and GROMACS topologies (`.top`), as written by
//! `pdb2gmx`, ParmEd, ACPYPE, and Interchange. We use their parameters per atom and per term, so
//! we don't assign force field types to these molecules.
//!
//! We support Amber-style functional forms: Harmonic bonds and angles, periodic proper and
//! improper dihedrals, LJ with a combining rule, and 1-4 pairs with scaled Coulomb and LJ. This
//! covers Amber force fields without CMAP, GAFF, OPLS, and OpenFF. Other terms, e.g. CHARMM's
//! Urey-Bradley terms, harmonic impropers, CMAP, and pair-specific LJ (NBFIX), cause an error
//! that lists them.
//!
//! Water: This library simulates water as rigid bodies, placed by `MdConfig::solvent`. We build a
//! `WaterModel` from the file's water (and ions, if present); use it with `MdConfig::water_model`,
//! or `ImportedSystem::param_set`. We don't include the file's water molecules in `mols`, but return
//! their positions.

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fs,
    path::Path,
};

use bio_files::{
    AtomGeneric, BondGeneric, BondType,
    gromacs::{
        gro::Gro,
        top_parse::{GromacsTopology, TopAtomType, TopMoleculeType, combine},
    },
    inpcrd::AmberCoords,
    md_params::{
        AngleBendingParams, BondStretchingParams, DihedralParams, ForceFieldParamsIndexed,
        LjParams, MassParams,
    },
    prmtop::AmberPrmtop,
};
use lin_alg::f64::Vec3 as Vec3F64;
use na_seq::Element;

use crate::{
    AtomDynamics, FfMolType, MolDynamics, ParamError, SimBoxInit,
    non_bonded::{LjCombiningRule, Scale14},
    params::FfParamSet,
    solvent::{Ion, IonParams, WaterModel},
};

const KJ_TO_KCAL: f64 = 1. / 4.184;
const NM_TO_ANGSTROM: f64 = 10.;

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
    /// 1-4 pairs, and their scale factors.
    pub pairs_14: Vec<((usize, usize), Scale14)>,
    /// The combining rule these LJ parameters are for. It must match the rest of the system.
    pub lj_combining: LjCombiningRule,
}

impl ExplicitParams {
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

/// A system imported from topology and coordinate files. See the module documentation.
#[derive(Clone, Debug)]
pub struct ImportedSystem {
    /// Everything except water. Each carries `ExplicitParams`. Ions are single-atom molecules.
    pub mols: Vec<MolDynamics>,
    /// Built from the file's water molecules, and its ions if present. None if there's no water.
    pub water_model: Option<WaterModel>,
    /// Positions of the file's water molecules: (O, H, H), in Å.
    pub water_posits: Vec<[Vec3F64; 3]>,
    /// The file's periodic box, if it's rectangular.
    pub sim_box: Option<SimBoxInit>,
    pub lj_combining: LjCombiningRule,
}

impl ImportedSystem {
    /// A parameter set to run this system with: The imported molecules carry their own parameters,
    /// so this only sets the conventions they share with the rest of the system, and the water
    /// model.
    pub fn param_set(&self) -> FfParamSet {
        let mut result = FfParamSet::default();
        result.nonbonded_rules.lj_combining = self.lj_combining;
        if let Some(w) = self.water_model {
            result.default_water = w;
        }
        result
    }

    /// Load an Amber topology (`prmtop` or `parm7`), and coordinates (`inpcrd` or ASCII `rst7`).
    pub fn from_amber_files(prmtop: &Path, coords: &Path) -> Result<Self, ParamError> {
        let prmtop = AmberPrmtop::load(prmtop)?;
        let coords = AmberCoords::load(coords)?;
        Self::from_amber(&prmtop, &coords)
    }

    pub fn from_amber(prmtop: &AmberPrmtop, coords: &AmberCoords) -> Result<Self, ParamError> {
        let n = prmtop.n_atoms();
        if coords.posits.len() != n {
            return Err(ParamError::new(&format!(
                "The coordinates have {} atoms; the topology has {n}",
                coords.posits.len()
            )));
        }

        let mut unsupported = prmtop.unsupported_terms.clone();

        let elements: Vec<Element> = (0..n)
            .map(|i| {
                guess_element(
                    prmtop.atomic_numbers.as_ref().map(|z| z[i]),
                    prmtop.masses[i] as f64,
                    &prmtop.atom_names[i],
                )
            })
            .collect();

        // Per-atom LJ from the type table's diagonal, and the combining rule that reproduces the
        // off-diagonal entries.
        let type_lj: Vec<(f64, f64)> = (0..prmtop.n_lj_types)
            .map(|t| {
                let i = prmtop.nonbonded_parm_index[t * prmtop.n_lj_types + t];
                if i <= 0 {
                    return (0., 0.);
                }
                let i = (i - 1) as usize;
                sigma_eps_from_ab(prmtop.lj_acoef[i], prmtop.lj_bcoef[i])
            })
            .collect();

        let used_types: BTreeSet<usize> = prmtop.lj_type_index.iter().copied().collect();
        let lj_combining = match combining_rule(&used_types, &type_lj, |t0, t1| {
            let i = prmtop.nonbonded_parm_index[t0 * prmtop.n_lj_types + t1];
            (i > 0).then(|| {
                let i = (i - 1) as usize;
                (prmtop.lj_acoef[i], prmtop.lj_bcoef[i])
            })
        }) {
            Some(rule) => rule,
            None => {
                unsupported.push("Pair-specific LJ parameters (NBFIX)".to_owned());
                LjCombiningRule::LorentzBerthelot
            }
        };

        let atom_lj = |i: usize| -> (f32, f32) {
            let (s, e) = type_lj[prmtop.lj_type_index[i]];
            (s as f32, e as f32)
        };

        let components = connected_components(n, prmtop.bonds.iter().map(|b| b.atoms));

        // Classify each component.
        let mut mols = Vec::new();
        let mut water_mols = Vec::new();
        let mut ions = Vec::new();

        for comp in &components {
            if let Some(w) = water_sites(comp, &elements, &prmtop.masses) {
                water_mols.push(w);
                continue;
            }
            if comp.len() == 1 {
                ions.push(comp[0]);
            }

            let local: HashMap<usize, usize> =
                comp.iter().enumerate().map(|(li, &gi)| (gi, li)).collect();
            let loc = |gi: usize| local[&gi];

            let atoms: Vec<AtomGeneric> = comp
                .iter()
                .enumerate()
                .map(|(li, &gi)| AtomGeneric {
                    serial_number: li as u32 + 1,
                    posit: coords.posits[gi],
                    element: elements[gi],
                    force_field_type: Some(prmtop.atom_types[gi].clone()),
                    partial_charge: Some(prmtop.charges[gi]),
                    type_in_res_general: Some(prmtop.atom_names[gi].clone()),
                    ..Default::default()
                })
                .collect();

            if comp.iter().any(|&gi| prmtop.masses[gi] == 0.) {
                unsupported.push("Extra points outside water".to_owned());
            }

            let mut params = ExplicitParams {
                masses: comp.iter().map(|&gi| prmtop.masses[gi]).collect(),
                lj: comp.iter().map(|&gi| atom_lj(gi)).collect(),
                lj_combining,
                ..Default::default()
            };
            let mut bonds = Vec::new();

            let types = |gi: &[usize]| -> Vec<String> {
                gi.iter().map(|i| prmtop.atom_types[*i].clone()).collect()
            };

            for b in prmtop
                .bonds
                .iter()
                .filter(|b| local.contains_key(&b.atoms.0))
            {
                let t = types(&[b.atoms.0, b.atoms.1]);
                params.bonds.push((
                    (loc(b.atoms.0), loc(b.atoms.1)),
                    BondStretchingParams {
                        atom_types: (t[0].clone(), t[1].clone()),
                        k_b: b.k,
                        r_0: b.r0,
                        comment: None,
                    },
                ));
                bonds.push(bond_generic(loc(b.atoms.0), loc(b.atoms.1)));
            }

            for a in prmtop
                .angles
                .iter()
                .filter(|a| local.contains_key(&a.atoms.0))
            {
                let t = types(&[a.atoms.0, a.atoms.1, a.atoms.2]);
                params.angles.push((
                    (loc(a.atoms.0), loc(a.atoms.1), loc(a.atoms.2)),
                    AngleBendingParams {
                        atom_types: (t[0].clone(), t[1].clone(), t[2].clone()),
                        k: a.k,
                        theta_0: a.theta0,
                        comment: None,
                    },
                ));
            }

            let mut pairs_14: Vec<((usize, usize), Scale14)> = Vec::new();
            for d in prmtop
                .dihedrals
                .iter()
                .filter(|d| local.contains_key(&d.atoms[0]))
            {
                let [a, b, c, e] = d.atoms;
                let t = types(&d.atoms);
                let key = (loc(a), loc(b), loc(c), loc(e));
                let p = DihedralParams {
                    atom_types: (t[0].clone(), t[1].clone(), t[2].clone(), t[3].clone()),
                    divider: 1,
                    barrier_height: d.k,
                    phase: d.phase,
                    periodicity: d.periodicity.round() as u8,
                    comment: None,
                };
                if d.improper {
                    params.impropers.push((key, p));
                } else {
                    params.dihedrals.push((key, p));
                }

                if !d.improper && !d.skip_14 {
                    let pair = (key.0.min(key.3), key.0.max(key.3));
                    if !pairs_14.iter().any(|(p, _)| *p == pair) {
                        pairs_14.push((
                            pair,
                            Scale14 {
                                lj: 1. / d.scnb,
                                coulomb: 1. / d.scee,
                            },
                        ));
                    }
                }
            }

            let pair_set: BTreeSet<_> = pairs_14.iter().map(|(p, _)| *p).collect();
            params.exclusions = prmtop
                .excluded_pairs
                .iter()
                .filter(|(i, j)| local.contains_key(i) && local.contains_key(j))
                .map(|&(i, j)| (loc(i).min(loc(j)), loc(i).max(loc(j))))
                .filter(|p| !pair_set.contains(p))
                .collect();
            params.pairs_14 = pairs_14;

            mols.push(explicit_mol(atoms, bonds, params));
        }

        unsupported.sort();
        unsupported.dedup();
        if !unsupported.is_empty() {
            return Err(unsupported_error(&unsupported));
        }

        let ions_found: Vec<_> = ions
            .iter()
            .filter_map(|&i| {
                ion_params(elements[i], prmtop.charges[i], prmtop.masses[i], atom_lj(i))
            })
            .collect();

        let water_model = match water_mols.first() {
            Some(w) => Some(amber_water_model(
                prmtop,
                w,
                &coords.posits,
                &atom_lj,
                &ions_found,
            )?),
            None => None,
        };

        let water_posits = water_mols
            .iter()
            .map(|w| [coords.posits[w.o], coords.posits[w.h0], coords.posits[w.h1]])
            .collect();

        let box_lens = coords
            .box_dims
            .filter(|b| {
                (b[3] - 90.).abs() < 1e-3 && (b[4] - 90.).abs() < 1e-3 && (b[5] - 90.).abs() < 1e-3
            })
            .map(|b| [b[0], b[1], b[2]])
            .or_else(|| {
                prmtop
                    .box_dims
                    .filter(|b| (b.beta - 90.).abs() < 1e-3)
                    .map(|b| b.lengths.map(|v| v as f64))
            });

        Ok(Self {
            mols,
            water_model,
            water_posits,
            sim_box: box_lens.map(box_from_lengths),
            lj_combining,
        })
    }

    /// Load a GROMACS topology (`.top`) and coordinates (`.gro`). `include_dirs` are searched for
    /// `#include`d files after the including file's directory, e.g. force field directories.
    pub fn from_gromacs_files(
        top: &Path,
        gro: &Path,
        include_dirs: &[std::path::PathBuf],
    ) -> Result<Self, ParamError> {
        let top = GromacsTopology::load_with(top, include_dirs, &[])?;
        let gro = Gro::new(&fs::read_to_string(gro)?)?;
        Self::from_gromacs(&top, &gro)
    }

    pub fn from_gromacs(top: &GromacsTopology, gro: &Gro) -> Result<Self, ParamError> {
        let mut unsupported = Vec::new();
        let defaults = &top.defaults;

        if defaults.nbfunc != 1 {
            unsupported.push("Buckingham non-bonded interactions".to_owned());
        }
        let lj_combining = match defaults.comb_rule {
            2 => LjCombiningRule::LorentzBerthelot,
            _ => LjCombiningRule::Geometric,
        };
        if top
            .other_sections
            .iter()
            .any(|s| s == "intermolecular_interactions")
        {
            unsupported.push("Intermolecular interactions".to_owned());
        }

        // Pair-specific LJ only matters if both types are in use.
        let used_types: BTreeSet<&str> = top
            .molecules
            .iter()
            .filter_map(|(name, _)| top.molecule_type(name))
            .flat_map(|m| m.atoms.iter().map(|a| a.atom_type.as_str()))
            .collect();
        if top
            .nonbond_param_types
            .iter()
            .any(|(a, b)| used_types.contains(a.as_str()) && used_types.contains(b.as_str()))
        {
            unsupported.push("Pair-specific LJ parameters (NBFIX)".to_owned());
        }

        let expected: usize = top
            .molecules
            .iter()
            .map(|(name, count)| {
                top.molecule_type(name)
                    .map(|m| m.atoms.len() * count)
                    .unwrap_or(0)
            })
            .sum();
        if gro.atoms.len() != expected {
            return Err(ParamError::new(&format!(
                "The coordinates have {} atoms; the topology has {expected}",
                gro.atoms.len()
            )));
        }

        let mut mols = Vec::new();
        let mut water_posits = Vec::new();
        let mut water_model = None;
        let mut ions_found = Vec::new();
        let mut cursor = 0;

        // Convert each molecule type once, and reuse for its copies.
        let mut converted: HashMap<&str, Converted> = HashMap::new();

        for (name, count) in &top.molecules {
            let mt = top
                .molecule_type(name)
                .ok_or_else(|| ParamError::new(&format!("Molecule type {name} isn't defined")))?;

            if !converted.contains_key(name.as_str()) {
                let c = convert_gmx_molecule(mt, top, lj_combining, &mut unsupported)?;
                converted.insert(name.as_str(), c);
            }
            let c = &converted[name.as_str()];

            for _ in 0..*count {
                let posits: Vec<Vec3F64> = gro.atoms[cursor..cursor + mt.atoms.len()]
                    .iter()
                    .map(|a| a.posit * NM_TO_ANGSTROM)
                    .collect();
                cursor += mt.atoms.len();

                match c {
                    Converted::Water(w) => {
                        water_posits.push([posits[w.o], posits[w.h0], posits[w.h1]]);
                    }
                    Converted::Mol(mol) => {
                        let mut mol = mol.clone();
                        for (atom, p) in mol.atoms.iter_mut().zip(&posits) {
                            atom.posit = *p;
                        }
                        mols.push(mol);
                    }
                }
            }

            if let Converted::Mol(mol) = c
                && mol.atoms.len() == 1
                && let Some(params) = &mol.explicit_params
            {
                let a = &mol.atoms[0];
                if let Some(ion) = ion_params(
                    a.element,
                    a.partial_charge.unwrap_or_default(),
                    params.masses[0],
                    params.lj[0],
                ) {
                    ions_found.push(ion);
                }
            }
        }

        unsupported.sort();
        unsupported.dedup();
        if !unsupported.is_empty() {
            return Err(unsupported_error(&unsupported));
        }

        for (name, _) in &top.molecules {
            if let Converted::Water(w) = &converted[name.as_str()] {
                let mt = top.molecule_type(name).unwrap();
                water_model = Some(gmx_water_model(mt, top, w, &ions_found)?);
                break;
            }
        }

        let b = gro.box_vec * NM_TO_ANGSTROM;
        let sim_box = (b.x > 0. && b.y > 0. && b.z > 0.).then(|| box_from_lengths([b.x, b.y, b.z]));

        Ok(Self {
            mols,
            water_model,
            water_posits,
            sim_box,
            lj_combining,
        })
    }
}

fn unsupported_error(terms: &[String]) -> ParamError {
    ParamError::new(&format!(
        "This system uses terms we don't support yet: {}",
        terms.join(", ")
    ))
}

fn box_from_lengths(l: [f64; 3]) -> SimBoxInit {
    SimBoxInit::Fixed((
        lin_alg::f32::Vec3::new_zero(),
        lin_alg::f32::Vec3::new(l[0] as f32, l[1] as f32, l[2] as f32),
    ))
}

fn bond_generic(i: usize, j: usize) -> BondGeneric {
    BondGeneric {
        bond_type: BondType::Single,
        atom_0_sn: i as u32 + 1,
        atom_1_sn: j as u32 + 1,
    }
}

fn explicit_mol(
    atoms: Vec<AtomGeneric>,
    bonds: Vec<BondGeneric>,
    params: ExplicitParams,
) -> MolDynamics {
    MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms,
        bonds,
        explicit_params: Some(params),
        ..Default::default()
    }
}

/// σ and ε from E = A/r¹² − B/r⁶.
fn sigma_eps_from_ab(a: f64, b: f64) -> (f64, f64) {
    if a <= 0. || b <= 0. {
        (0., 0.)
    } else {
        ((a / b).powf(1. / 6.), b * b / (4. * a))
    }
}

/// Find the combining rule that reproduces every pair's LJ coefficients from per-type
/// parameters. None if neither does, i.e. the file has pair-specific parameters.
fn combining_rule(
    types: &BTreeSet<usize>,
    type_lj: &[(f64, f64)],
    pair_ab: impl Fn(usize, usize) -> Option<(f64, f64)>,
) -> Option<LjCombiningRule> {
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-4 * x.abs().max(y.abs()) + 1e-8;

    [
        LjCombiningRule::LorentzBerthelot,
        LjCombiningRule::Geometric,
    ]
    .into_iter()
    .find(|rule| {
        types.iter().all(|&t0| {
            types.iter().filter(|t1| **t1 >= t0).all(|&t1| {
                let (s0, e0) = type_lj[t0];
                let (s1, e1) = type_lj[t1];
                let (s, e) = rule.combine(s0 as f32, e0 as f32, s1 as f32, e1 as f32);
                let (s, e) = (s as f64, e as f64);
                let a = 4. * e * s.powi(12);
                let b = 4. * e * s.powi(6);
                let (fa, fb) = pair_ab(t0, t1).unwrap_or((0., 0.));
                close(a, fa) && close(b, fb)
            })
        })
    })
}

/// Connected components of a bond graph, ordered by their first atom, with atoms in ascending
/// order.
fn connected_components(n: usize, bonds: impl Iterator<Item = (usize, usize)>) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for (i, j) in bonds {
        adj[i].push(j);
        adj[j].push(i);
    }

    let mut comp_of = vec![usize::MAX; n];
    let mut result: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if comp_of[start] != usize::MAX {
            continue;
        }
        let c = result.len();
        let mut members = vec![start];
        comp_of[start] = c;
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            for &j in &adj[i] {
                if comp_of[j] == usize::MAX {
                    comp_of[j] = c;
                    members.push(j);
                    queue.push_back(j);
                }
            }
        }
        members.sort_unstable();
        result.push(members);
    }
    result
}

/// Indices of a water molecule's sites.
#[derive(Clone, Copy, Debug)]
struct WaterSites {
    o: usize,
    h0: usize,
    h1: usize,
    /// A massless charge site, for 4-site models.
    m: Option<usize>,
}

/// If these atoms are a water molecule (one O, two H, and optionally one massless site), its
/// sites.
fn water_sites(atoms: &[usize], elements: &[Element], masses: &[f32]) -> Option<WaterSites> {
    if !(3..=4).contains(&atoms.len()) {
        return None;
    }
    let (mut o, mut h, mut m) = (Vec::new(), Vec::new(), Vec::new());
    for &i in atoms {
        if masses[i] == 0. {
            m.push(i);
        } else if elements[i] == Element::Oxygen {
            o.push(i);
        } else if elements[i] == Element::Hydrogen {
            h.push(i);
        } else {
            return None;
        }
    }
    (o.len() == 1 && h.len() == 2 && m.len() == atoms.len() - 3).then(|| WaterSites {
        o: o[0],
        h0: h[0],
        h1: h[1],
        m: m.first().copied(),
    })
}

/// If this single atom is a monovalent ion we use for neutralizing, its parameters.
fn ion_params(element: Element, charge: f32, mass: f32, lj: (f32, f32)) -> Option<IonParams> {
    let ion = match element {
        Element::Sodium => Ion::Sodium,
        Element::Potassium => Ion::Potassium,
        Element::Chlorine => Ion::Chloride,
        _ => return None,
    };
    ((charge - ion.charge()).abs() < 0.01).then_some(IonParams {
        ion,
        mass,
        lj_sigma: lj.0,
        lj_eps: lj.1,
    })
}

/// A water model with the file's parameters. Its ions are the file's, if present; otherwise OPC's.
fn water_model(
    masses: (f64, f64),
    charges: (f64, f64, f64, f64), // O, H0, H1, M (0 if 3-site)
    lj_o: (f64, f64),
    lj_h: (f64, f64),
    o_h_dist: f64,
    h_o_h_angle: f64,
    o_m_dist: f64,
    has_m: bool,
    ions: &[IonParams],
) -> Result<WaterModel, ParamError> {
    let (q_o, q_h0, q_h1, q_m) = charges;
    if lj_h.1.abs() > 1e-6 {
        return Err(ParamError::new(
            "LJ on water hydrogens (e.g. CHARMM's TIP3P) isn't supported yet",
        ));
    }
    if (q_h0 - q_h1).abs() > 1e-4 || (q_o + q_h0 + q_h1 + q_m).abs() > 1e-3 {
        return Err(ParamError::new(
            "Unsupported water charges: They must be symmetric and neutral",
        ));
    }
    if has_m && q_o.abs() > 1e-4 {
        return Err(ParamError::new(
            "Unsupported water model: Charge on both O and a virtual site",
        ));
    }

    let base = WaterModel::OPC;
    let cation = ions
        .iter()
        .find(|i| i.ion.charge() > 0.)
        .copied()
        .unwrap_or(base.cation);
    let anion = ions
        .iter()
        .find(|i| i.ion.charge() < 0.)
        .copied()
        .unwrap_or(base.anion);

    Ok(WaterModel {
        mass_o: masses.0 as f32,
        mass_h: masses.1 as f32,
        o_h_dist: o_h_dist as f32,
        o_m_dist: if has_m { o_m_dist as f32 } else { 0. },
        h_o_h_angle: h_o_h_angle as f32,
        q_h: q_h0 as f32,
        lj_sigma_o: lj_o.0 as f32,
        lj_eps_o: lj_o.1 as f32,
        cation,
        anion,
    })
}

fn amber_water_model(
    prmtop: &AmberPrmtop,
    w: &WaterSites,
    posits: &[Vec3F64],
    atom_lj: &impl Fn(usize) -> (f32, f32),
    ions: &[IonParams],
) -> Result<WaterModel, ParamError> {
    let pair = |a: usize, b: usize| (a.min(b), a.max(b));
    let bond_r0 = |a: usize, b: usize| {
        prmtop
            .bonds
            .iter()
            .find(|bd| pair(bd.atoms.0, bd.atoms.1) == pair(a, b))
            .map(|bd| bd.r0 as f64)
    };
    let dist = |a: usize, b: usize| (posits[a] - posits[b]).magnitude();

    let o_h = bond_r0(w.o, w.h0).unwrap_or_else(|| dist(w.o, w.h0));
    let angle = match prmtop
        .angles
        .iter()
        .find(|a| a.atoms.1 == w.o && pair(a.atoms.0, a.atoms.2) == pair(w.h0, w.h1))
    {
        Some(a) => a.theta0 as f64,
        None => {
            let h_h = bond_r0(w.h0, w.h1).unwrap_or_else(|| dist(w.h0, w.h1));
            2. * (h_h / (2. * o_h)).asin()
        }
    };
    let o_m =
        w.m.map(|m| bond_r0(w.o, m).unwrap_or_else(|| dist(w.o, m)))
            .unwrap_or(0.);

    let lj = |i: usize| {
        let (s, e) = atom_lj(i);
        (s as f64, e as f64)
    };
    let q = |i: usize| prmtop.charges[i] as f64;

    water_model(
        (prmtop.masses[w.o] as f64, prmtop.masses[w.h0] as f64),
        (q(w.o), q(w.h0), q(w.h1), w.m.map(q).unwrap_or(0.)),
        lj(w.o),
        lj(w.h0),
        o_h,
        angle,
        o_m,
        w.m.is_some(),
        ions,
    )
}

/// A GROMACS molecule type, converted once and reused for each copy.
enum Converted {
    Water(WaterSites),
    Mol(MolDynamics),
}

/// Per-atom (σ Å, ε kcal/mol) from an atom type.
fn gmx_type_lj(t: &TopAtomType, comb_rule: u8) -> (f64, f64) {
    gmx_sigma_eps(t.v, t.w, comb_rule)
}

/// (σ Å, ε kcal/mol) from GROMACS V and W: σ and ε, or C6 and C12 for combination rule 1.
fn gmx_sigma_eps(v: f64, w: f64, comb_rule: u8) -> (f64, f64) {
    let (sigma_nm, eps_kj) = if comb_rule == 1 {
        if v <= 0. || w <= 0. {
            (0., 0.)
        } else {
            ((w / v).powf(1. / 6.), v * v / (4. * w))
        }
    } else {
        (v, w)
    };
    (sigma_nm * NM_TO_ANGSTROM, eps_kj * KJ_TO_KCAL)
}

fn convert_gmx_molecule(
    mt: &TopMoleculeType,
    top: &GromacsTopology,
    lj_combining: LjCombiningRule,
    unsupported: &mut Vec<String>,
) -> Result<Converted, ParamError> {
    let comb = top.defaults.comb_rule;
    let n = mt.atoms.len();

    let atom_type = |i: usize| -> Result<&TopAtomType, ParamError> {
        top.atom_types
            .get(&mt.atoms[i].atom_type)
            .ok_or_else(|| ParamError::new(&format!("Unknown atom type {}", mt.atoms[i].atom_type)))
    };

    let mut elements = Vec::with_capacity(n);
    let mut masses = Vec::with_capacity(n);
    for (i, a) in mt.atoms.iter().enumerate() {
        let t = atom_type(i)?;
        elements.push(guess_element(
            t.atomic_number.map(|z| z as i32),
            a.mass,
            &a.name,
        ));
        masses.push(a.mass as f32);
    }

    let all: Vec<usize> = (0..n).collect();
    if let Some(w) = water_sites(&all, &elements, &masses) {
        return Ok(Converted::Water(w));
    }

    for s in &mt.other_sections {
        unsupported.push(format!("[ {s} ]"));
    }
    if mt.n_cmap > 0 {
        unsupported.push("CMAP backbone corrections".to_owned());
    }
    if !mt.virtual_sites.is_empty() {
        unsupported.push("Virtual sites outside water".to_owned());
    }
    if !mt.settles.is_empty() {
        unsupported.push("SETTLE outside water".to_owned());
    }
    if !mt.constraints.is_empty() {
        unsupported.push("Constraints in the topology (outside water)".to_owned());
    }

    let lj: Vec<(f64, f64)> = (0..n)
        .map(|i| atom_type(i).map(|t| gmx_type_lj(t, comb)))
        .collect::<Result<_, _>>()?;

    let atoms: Vec<AtomGeneric> = mt
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| AtomGeneric {
            serial_number: i as u32 + 1,
            element: elements[i],
            force_field_type: Some(a.atom_type.clone()),
            partial_charge: Some(a.charge as f32),
            type_in_res_general: Some(a.name.clone()),
            ..Default::default()
        })
        .collect();

    let type_name = |i: usize| mt.atoms[i].atom_type.clone();

    let mut params = ExplicitParams {
        masses,
        lj: lj.iter().map(|(s, e)| (*s as f32, *e as f32)).collect(),
        lj_combining,
        ..Default::default()
    };
    let mut bonds = Vec::new();
    // Bonds that count toward exclusions.
    let mut exclusion_graph = vec![Vec::new(); n];

    for b in &mt.bonds {
        let (i, j) = (b.atoms[0], b.atoms[1]);
        match b.funct {
            1 => {
                // V = ½ kb (b − b0)², kJ/mol/nm²  →  k_b (r − r0)², kcal/mol/Å²
                params.bonds.push((
                    (i, j),
                    BondStretchingParams {
                        atom_types: (type_name(i), type_name(j)),
                        k_b: (b.params[1] * 0.5 * KJ_TO_KCAL / (NM_TO_ANGSTROM * NM_TO_ANGSTROM))
                            as f32,
                        r_0: (b.params[0] * NM_TO_ANGSTROM) as f32,
                        comment: None,
                    },
                ));
                bonds.push(bond_generic(i, j));
            }
            // A connection: Exclusions, but no force.
            5 => bonds.push(bond_generic(i, j)),
            f => unsupported.push(format!("Bond function {f}")),
        }
        if !matches!(b.funct, 6 | 7) {
            exclusion_graph[i].push(j);
            exclusion_graph[j].push(i);
        }
    }

    for a in &mt.angles {
        let (i, ctr, k) = (a.atoms[0], a.atoms[1], a.atoms[2]);
        match a.funct {
            // V = ½ k (θ − θ0)², kJ/mol/rad², θ0 in degrees
            1 => params.angles.push((
                (i, ctr, k),
                AngleBendingParams {
                    atom_types: (type_name(i), type_name(ctr), type_name(k)),
                    k: (a.params[1] * 0.5 * KJ_TO_KCAL) as f32,
                    theta_0: a.params[0].to_radians() as f32,
                    comment: None,
                },
            )),
            5 => unsupported.push("Urey-Bradley angle terms".to_owned()),
            f => unsupported.push(format!("Angle function {f}")),
        }
    }

    for d in &mt.dihedrals {
        let (a, b, c, e) = (d.atoms[0], d.atoms[1], d.atoms[2], d.atoms[3]);
        let periodic = || DihedralParams {
            atom_types: (type_name(a), type_name(b), type_name(c), type_name(e)),
            divider: 1,
            barrier_height: (d.params[1] * KJ_TO_KCAL) as f32,
            phase: d.params[0].to_radians() as f32,
            periodicity: d.params[2].round() as u8,
            comment: None,
        };
        match d.funct {
            1 | 9 => params.dihedrals.push(((a, b, c, e), periodic())),
            4 => params.impropers.push(((a, b, c, e), periodic())),
            2 => unsupported.push("Harmonic impropers".to_owned()),
            3 => unsupported.push("Ryckaert-Bellemans dihedrals".to_owned()),
            f => unsupported.push(format!("Dihedral function {f}")),
        }
    }

    // Exclusions: Pairs within `nrexcl` bonds, and explicit ones.
    let mut excluded: BTreeSet<(usize, usize)> = BTreeSet::new();
    for start in 0..n {
        let mut depth = vec![usize::MAX; n];
        depth[start] = 0;
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            if depth[i] == mt.nrexcl {
                continue;
            }
            for &j in &exclusion_graph[i] {
                if depth[j] == usize::MAX {
                    depth[j] = depth[i] + 1;
                    queue.push_back(j);
                }
            }
        }
        for (j, dpt) in depth.iter().enumerate() {
            if j > start && *dpt != usize::MAX {
                excluded.insert((start, j));
            }
        }
    }
    excluded.extend(mt.exclusions.iter().copied());

    // 1-4 pairs. We express each as a scale on the normal interaction, so its LJ σ must match.
    let normal_lj = |i: usize, j: usize| -> (f64, f64) {
        let (ti, tj) = (atom_type(i).unwrap(), atom_type(j).unwrap());
        let (v, w) = combine(comb, ti.v, ti.w, tj.v, tj.w);
        gmx_sigma_eps(v, w, comb)
    };
    for p in &mt.pairs {
        let (i, j) = (p.atoms[0].min(p.atoms[1]), p.atoms[0].max(p.atoms[1]));
        if p.funct != 1 {
            unsupported.push(format!("Pair function {}", p.funct));
            continue;
        }
        if !excluded.contains(&(i, j)) {
            unsupported.push("1-4 pairs that also have normal non-bonded interactions".to_owned());
            continue;
        }

        let (s14, e14) = gmx_sigma_eps(p.params[0], p.params[1], comb);
        let (s, e) = normal_lj(i, j);
        let lj_scale = if e < 1e-12 {
            if e14 > 1e-12 {
                unsupported.push("Separate 1-4 LJ parameters".to_owned());
            }
            top.defaults.fudge_lj
        } else {
            if (s14 - s).abs() > 1e-4 * s.max(1e-6) {
                unsupported.push("Separate 1-4 LJ parameters (e.g. CHARMM)".to_owned());
            }
            e14 / e
        };

        excluded.remove(&(i, j));
        params.pairs_14.push((
            (i, j),
            Scale14 {
                lj: lj_scale as f32,
                coulomb: top.defaults.fudge_qq as f32,
            },
        ));
    }
    params.exclusions = excluded.into_iter().collect();

    Ok(Converted::Mol(explicit_mol(atoms, bonds, params)))
}

fn gmx_water_model(
    mt: &TopMoleculeType,
    top: &GromacsTopology,
    w: &WaterSites,
    ions: &[IonParams],
) -> Result<WaterModel, ParamError> {
    let comb = top.defaults.comb_rule;
    let lj = |i: usize| {
        top.atom_types
            .get(&mt.atoms[i].atom_type)
            .map(|t| gmx_type_lj(t, comb))
            .unwrap_or((0., 0.))
    };

    let pair = |a: usize, b: usize| (a.min(b), a.max(b));
    let bond_b0 = |a: usize, b: usize| {
        mt.bonds
            .iter()
            .chain(&mt.constraints)
            .find(|bd| pair(bd.atoms[0], bd.atoms[1]) == pair(a, b))
            .and_then(|bd| bd.params.first().copied())
    };

    let (o_h, angle) = match mt.settles.first() {
        Some(s) => (s.d_oh, 2. * (s.d_hh / (2. * s.d_oh)).asin()),
        None => {
            let o_h = bond_b0(w.o, w.h0)
                .ok_or_else(|| ParamError::new("Water has no O-H bond or SETTLE"))?;
            let angle = match mt.angles.iter().find(|a| a.atoms[1] == w.o && a.funct == 1) {
                Some(a) => a.params[0].to_radians(),
                None => {
                    let h_h = bond_b0(w.h0, w.h1)
                        .ok_or_else(|| ParamError::new("Water has no H-O-H angle"))?;
                    2. * (h_h / (2. * o_h)).asin()
                }
            };
            (o_h, angle)
        }
    };

    // A bisector virtual site: r_M = r_O + a (r_H0 − r_O) + b (r_H1 − r_O), with a = b.
    let o_m = match (w.m, mt.virtual_sites.first()) {
        (Some(_), Some(vs)) if vs.funct == 1 && vs.params.len() >= 2 => {
            let (a, b) = (vs.params[0], vs.params[1]);
            if (a - b).abs() > 1e-6 {
                return Err(ParamError::new(
                    "Unsupported water virtual site: Not on the bisector",
                ));
            }
            2. * a * o_h * (angle / 2.).cos()
        }
        (Some(_), _) => {
            return Err(ParamError::new(
                "Unsupported water virtual site construction",
            ));
        }
        (None, _) => 0.,
    };

    let q = |i: usize| mt.atoms[i].charge;
    water_model(
        (mt.atoms[w.o].mass, mt.atoms[w.h0].mass),
        (q(w.o), q(w.h0), q(w.h1), w.m.map(q).unwrap_or(0.)),
        lj(w.o),
        lj(w.h0),
        o_h * NM_TO_ANGSTROM,
        angle,
        o_m * NM_TO_ANGSTROM,
        w.m.is_some(),
        ions,
    )
}

/// The element of an atom from a topology: From its atomic number if present, otherwise its
/// mass, otherwise its name.
fn guess_element(atomic_number: Option<i32>, mass: f64, name: &str) -> Element {
    if let Some(z) = atomic_number
        && z > 0
        && let Ok(el) = Element::from_atomic_number(z as u8)
    {
        return el;
    }

    // Includes hydrogens with repartitioned mass.
    if mass > 0.5 && mass < 4.5 {
        return Element::Hydrogen;
    }

    // (mass, atomic number) for elements common in biomolecular systems.
    const COMMON: [(f64, u8); 15] = [
        (12.011, 6),
        (14.007, 7),
        (15.999, 8),
        (18.998, 9),
        (22.990, 11),
        (24.305, 12),
        (30.974, 15),
        (32.06, 16),
        (35.45, 17),
        (39.098, 19),
        (40.078, 20),
        (55.845, 26),
        (65.38, 30),
        (79.904, 35),
        (126.904, 53),
    ];
    if mass > 0.5
        && let Some((_, z)) = COMMON.iter().find(|(m, _)| (m - mass).abs() < 0.6)
        && let Ok(el) = Element::from_atomic_number(*z)
    {
        return el;
    }

    let letters: String = name.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    for len in [2, 1] {
        if letters.len() >= len
            && let Ok(el) = Element::from_letter(&letters[..len])
        {
            return el;
        }
    }
    Element::Carbon
}
