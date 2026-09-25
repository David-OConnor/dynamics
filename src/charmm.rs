//! CHARMM36m for proteins. We build each residue from CHARMM's topology (`top_all36_prot.rtf`),
//! apply patches for termini, protonation states, and disulfide bonds, generate angles and
//! dihedrals from bonds, and assign parameters per term, as CHARMM and psfgen do. The result is
//! `ExplicitParams`.
//!
//! Atoms keep their order, names, and positions; we set their force field types and partial
//! charges. We match heavy atoms to CHARMM's by name, and hydrogens by the heavy atom they're
//! bonded to, so PDB and Amber hydrogen names work. (e.g. HB2 and HB3, vs CHARMM's HB1 and HB2)
//!
//! We choose protonation states from the hydrogens present: His (HSD, HSE, or HSP), Asp and Glu
//! (ASPP, GLUP), Lys (LSN), Cys (CYS, CYM, or a disulfide with DISU), and charged or neutral
//! termini. A residue with no bonded neighbor that isn't a terminus (e.g. at a gap in a structure)
//! gets no patch; terms that span the gap are left out.

use std::{
    collections::{BTreeSet, HashMap},
    io,
};

use bio_files::{
    AtomGeneric, BondGeneric, ResidueGeneric, ResidueType,
    charmm::{CharmmParams, CharmmTopology, RtfAtom, RtfResidue, split_stream},
    md_params::{AngleBendingParams, BondStretchingParams, DihedralParams},
};
use na_seq::{AminoAcid, Element};

use crate::{
    ExplicitParams, ParamError,
    cmap::CmapGrid,
    non_bonded::{LjCombiningRule, NbFix, Pair14, Scale14},
};

// CHARMM36m (proteins), and CHARMM's water and ions, from toppar_c36_jul24.
const PAR_PROT: &str = include_str!("../param_data/charmm/par_all36m_prot.prm");
const TOP_PROT: &str = include_str!("../param_data/charmm/top_all36_prot.rtf");
const WATER_IONS: &str = include_str!("../param_data/charmm/toppar_water_ions.str");

/// 2 / 2^(1/6). Converts CHARMM's Rmin/2 to σ.
const RMIN_HALF_TO_SIGMA: f32 = 1.781_797_4;
/// 1 / 2^(1/6). Converts Rmin to σ.
const RMIN_TO_SIGMA: f32 = 0.890_898_7;

/// CHARMM topology (residues and patches) and parameters.
#[derive(Clone, Debug, Default)]
pub struct CharmmSet {
    pub topology: CharmmTopology,
    pub params: CharmmParams,
}

impl CharmmSet {
    /// CHARMM36m for proteins, with CHARMM's TIP3P water and ions. Uses files included with this
    /// library, from toppar_c36_jul24.
    pub fn new_c36m() -> io::Result<Self> {
        let mut topology = CharmmTopology::new(TOP_PROT)?;
        let mut params = CharmmParams::new(PAR_PROT)?;

        // The stream file sets default patches to NONE, for water and ions; we keep the protein
        // file's defaults (NTER and CTER) for chains.
        let defaults = (
            topology.default_first.clone(),
            topology.default_last.clone(),
        );
        let (rtf_blocks, prm_blocks) = split_stream(WATER_IONS);
        for block in &rtf_blocks {
            topology.add_rtf(block)?;
        }
        (topology.default_first, topology.default_last) = defaults;
        for block in &prm_blocks {
            params.add_prm(block)?;
        }

        Ok(Self { topology, params })
    }

    /// Pair-specific LJ parameters, (σ Å, ε kcal/mol), by type pair. From NBFIX entries.
    pub fn nbfix(&self) -> NbFix {
        self.params
            .nbfix
            .iter()
            .map(|(types, &(eps, rmin))| (types.clone(), (rmin * RMIN_TO_SIGMA, eps)))
            .collect()
    }

    fn mass(&self, atom_type: &str) -> Option<f32> {
        self.topology
            .masses
            .get(atom_type)
            .map(|m| m.0)
            .or_else(|| self.params.masses.get(atom_type).copied())
    }

    fn is_hydrogen_type(&self, atom_type: &str) -> bool {
        match self.topology.masses.get(atom_type) {
            Some((_, element)) if !element.is_empty() => element.eq_ignore_ascii_case("H"),
            // Fall back to the mass.
            _ => self.mass(atom_type).is_some_and(|m| m < 1.5),
        }
    }

    fn residue(&self, name: &str) -> Result<&RtfResidue, ParamError> {
        self.topology.residue(name).ok_or_else(|| {
            ParamError::new(&format!(
                "CHARMM topology is missing residue or patch {name}"
            ))
        })
    }
}

/// A residue built from CHARMM's topology, with patches applied. Names in bonded terms may have
/// `-` and `+` prefixes, for the previous and next residues.
#[derive(Clone, Debug)]
struct ResInstance {
    /// E.g. "ASP+ASPP+CTER", for messages.
    label: String,
    atoms: Vec<RtfAtom>,
    bonds: Vec<[String; 2]>,
    impropers: Vec<[String; 4]>,
    cmaps: Vec<[String; 8]>,
}

impl ResInstance {
    fn new(res: &RtfResidue) -> Self {
        Self {
            label: res.name.clone(),
            atoms: res.atoms.clone(),
            bonds: res.bonds.clone(),
            impropers: res.impropers.clone(),
            cmaps: res.cmaps.clone(),
        }
    }

    fn set_atom(&mut self, atom: RtfAtom) {
        match self.atoms.iter_mut().find(|a| a.name == atom.name) {
            Some(a) => *a = atom,
            None => self.atoms.push(atom),
        }
    }

    /// Remove an atom, and terms in this residue that include it.
    fn delete_atom(&mut self, name: &str) {
        self.atoms.retain(|a| a.name != name);
        self.bonds.retain(|t| !t.iter().any(|n| n == name));
        self.impropers.retain(|t| !t.iter().any(|n| n == name));
        self.cmaps.retain(|t| !t.iter().any(|n| n == name));
    }

    fn apply_patch(&mut self, patch: &RtfResidue) {
        self.label = format!("{}+{}", self.label, patch.name);
        for name in &patch.delete_atoms {
            self.delete_atom(name);
        }
        for atom in &patch.atoms {
            self.set_atom(atom.clone());
        }
        self.bonds.extend(patch.bonds.iter().cloned());
        self.impropers.extend(patch.impropers.iter().cloned());
        self.cmaps.extend(patch.cmaps.iter().cloned());
    }

    fn atom(&self, name: &str) -> Option<&RtfAtom> {
        self.atoms.iter().find(|a| a.name == name)
    }
}

/// Split a two-residue patch's atom name, e.g. "1SG", into its residue (0 or 1), and name.
fn split_patch_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.get(1..)?;
    match name.as_bytes().first()? {
        b'1' => Some((0, rest)),
        b'2' => Some((1, rest)),
        _ => None,
    }
}

/// Apply a patch that spans two residues, e.g. DISU, whose atom names have "1" and "2" prefixes.
/// Returns bonds between the two residues.
fn apply_two_residue_patch(
    patch: &RtfResidue,
    mut res: [&mut ResInstance; 2],
) -> Result<Vec<[String; 2]>, ParamError> {
    let bad_name =
        |n: &str| ParamError::new(&format!("Unexpected atom name {n} in patch {}", patch.name));

    for r in res.iter_mut() {
        r.label = format!("{}+{}", r.label, patch.name);
    }
    for name in &patch.delete_atoms {
        let (i, n) = split_patch_name(name).ok_or_else(|| bad_name(name))?;
        res[i].delete_atom(n);
    }
    for atom in &patch.atoms {
        let (i, n) = split_patch_name(&atom.name).ok_or_else(|| bad_name(&atom.name))?;
        res[i].set_atom(RtfAtom {
            name: n.to_owned(),
            ..atom.clone()
        });
    }

    let mut cross = Vec::new();
    for [a, b] in &patch.bonds {
        let (ia, na) = split_patch_name(a).ok_or_else(|| bad_name(a))?;
        let (ib, nb) = split_patch_name(b).ok_or_else(|| bad_name(b))?;
        if ia == ib {
            res[ia].bonds.push([na.to_owned(), nb.to_owned()]);
        } else if ia == 0 {
            cross.push([na.to_owned(), nb.to_owned()]);
        } else {
            cross.push([nb.to_owned(), na.to_owned()]);
        }
    }
    if !patch.impropers.is_empty() || !patch.cmaps.is_empty() {
        return Err(ParamError::new(&format!(
            "Impropers and CMAP in two-residue patch {} aren't supported",
            patch.name
        )));
    }

    Ok(cross)
}

fn residue_name(aa: AminoAcid) -> Option<&'static str> {
    use AminoAcid::*;
    Some(match aa {
        Ala => "ALA",
        Arg => "ARG",
        Asn => "ASN",
        Asp => "ASP",
        Cys => "CYS",
        Gln => "GLN",
        Glu => "GLU",
        Gly => "GLY",
        His => "HSE", // We choose the tautomer from hydrogens.
        Ile => "ILE",
        Leu => "LEU",
        Lys => "LYS",
        Met => "MET",
        Phe => "PHE",
        Pro => "PRO",
        Ser => "SER",
        Thr => "THR",
        Trp => "TRP",
        Tyr => "TYR",
        Val => "VAL",
        _ => return None,
    })
}

fn atom_name(atom: &AtomGeneric) -> Option<String> {
    atom.type_in_res
        .as_ref()
        .map(|t| t.to_string())
        .or_else(|| atom.type_in_res_general.clone())
        .map(|n| n.trim().to_uppercase())
}

/// A molecule's residue, as indices into its atoms.
struct MolResidue {
    aa: AminoAcid,
    serial_number: u32,
    atoms: Vec<usize>,
}

/// Assign CHARMM36m types and charges to a peptide's atoms, and build its parameters. `atoms`
/// must include hydrogens. Bonds and residues refer to atoms by serial number.
pub fn build_peptide(
    atoms: &mut [AtomGeneric],
    bonds: &[BondGeneric],
    residues: &[ResidueGeneric],
    set: &CharmmSet,
) -> Result<ExplicitParams, ParamError> {
    let n = atoms.len();
    let index_of: HashMap<u32, usize> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial_number, i))
        .collect();

    // Bonds in the input, for finding hydrogens' heavy atoms, links between residues, and
    // disulfides. We generate CHARMM's bonds from its topology.
    let mut nbrs_in = vec![Vec::new(); n];
    for b in bonds {
        if let (Some(&i), Some(&j)) = (index_of.get(&b.atom_0_sn), index_of.get(&b.atom_1_sn)) {
            if i != j && !nbrs_in[i].contains(&j) {
                nbrs_in[i].push(j);
                nbrs_in[j].push(i);
            }
        }
    }

    let names: Vec<Option<String>> = atoms.iter().map(atom_name).collect();
    let is_h: Vec<bool> = atoms
        .iter()
        .map(|a| a.element == Element::Hydrogen)
        .collect();

    let mut res_of_atom = vec![None; n];
    let mut mol_res = Vec::new();
    for res in residues {
        let ResidueType::AminoAcid(aa) = res.res_type else {
            continue;
        };
        let res_atoms: Vec<usize> = res
            .atom_sns
            .iter()
            .filter_map(|sn| index_of.get(sn).copied())
            .collect();
        if res_atoms.is_empty() {
            continue;
        }
        for &i in &res_atoms {
            res_of_atom[i] = Some(mol_res.len());
        }
        mol_res.push(MolResidue {
            aa,
            serial_number: res.serial_number,
            atoms: res_atoms,
        });
    }

    if let Some(i) = res_of_atom.iter().position(|r| r.is_none()) {
        return Err(ParamError::new(&format!(
            "Atom {} isn't in an amino acid residue. CHARMM peptides must consist of standard \
             amino acids.",
            atoms[i]
        )));
    }
    let res_of_atom: Vec<usize> = res_of_atom.into_iter().map(|r| r.unwrap()).collect();

    // Atom in a residue by (input) name.
    let find = |ri: usize, name: &str| -> Option<usize> {
        mol_res[ri]
            .atoms
            .iter()
            .copied()
            .find(|&i| names[i].as_deref() == Some(name))
    };
    let h_count = |i: usize| nbrs_in[i].iter().filter(|&&j| is_h[j]).count();
    let h_count_named = |ri: usize, name: &str| find(ri, name).map(h_count).unwrap_or(0);

    // Links between residues: C of the previous residue bonded to this residue's N.
    let n_res = mol_res.len();
    let mut prev = vec![None; n_res];
    let mut next = vec![None; n_res];
    for ri in 0..n_res {
        let Some(n_atom) = find(ri, "N") else {
            continue;
        };
        for &j in &nbrs_in[n_atom] {
            let rj = res_of_atom[j];
            if rj != ri && names[j].as_deref() == Some("C") {
                prev[ri] = Some(rj);
                next[rj] = Some(ri);
            }
        }
    }

    // Disulfides: SG bonded to SG in another residue.
    let mut disulfides = Vec::new();
    for ri in 0..n_res {
        if mol_res[ri].aa != AminoAcid::Cys {
            continue;
        }
        let Some(sg) = find(ri, "SG") else {
            continue;
        };
        for &j in &nbrs_in[sg] {
            let rj = res_of_atom[j];
            if rj > ri && names[j].as_deref() == Some("SG") {
                disulfides.push((ri, rj));
            }
        }
    }
    let in_disulfide = |ri: usize| disulfides.iter().any(|&(a, b)| a == ri || b == ri);

    // Build residues, and choose how input names map to CHARMM's, where they differ.
    let mut instances = Vec::with_capacity(n_res);
    let mut renames: Vec<HashMap<&'static str, &'static str>> = vec![HashMap::new(); n_res];

    for ri in 0..n_res {
        let aa = mol_res[ri].aa;
        let res_label = || format!("{:?} {}", aa, mol_res[ri].serial_number);

        let mut name = residue_name(aa).ok_or_else(|| {
            ParamError::new(&format!(
                "CHARMM36m doesn't include residue {}",
                res_label()
            ))
        })?;
        let mut side_patch = None;

        match aa {
            AminoAcid::His => {
                name = match (h_count_named(ri, "ND1"), h_count_named(ri, "NE2")) {
                    (1, 1) => "HSP",
                    (1, 0) => "HSD",
                    _ => "HSE",
                };
            }
            AminoAcid::Asp | AminoAcid::Glu => {
                let (o1, o2, patch) = if aa == AminoAcid::Asp {
                    ("OD1", "OD2", "ASPP")
                } else {
                    ("OE1", "OE2", "GLUP")
                };
                let (h1, h2) = (h_count_named(ri, o1), h_count_named(ri, o2));
                if h1 + h2 == 1 {
                    side_patch = Some(patch);
                    // CHARMM puts the proton on the second O.
                    if h1 == 1 {
                        renames[ri].insert(o1, o2);
                        renames[ri].insert(o2, o1);
                    }
                }
            }
            AminoAcid::Lys => {
                if h_count_named(ri, "NZ") == 2 {
                    side_patch = Some("LSN");
                }
            }
            AminoAcid::Cys => {
                if !in_disulfide(ri) && h_count_named(ri, "SG") == 0 {
                    name = "CYM";
                }
            }
            AminoAcid::Ile => {
                renames[ri].insert("CD1", "CD");
            }
            _ => {}
        }

        let base = set.residue(name)?;
        let mut inst = ResInstance::new(base);
        if let Some(p) = side_patch {
            inst.apply_patch(set.residue(p)?);
        }

        // N-terminus
        if prev[ri].is_none()
            && let Some(n_atom) = find(ri, "N")
        {
            let default_first = base
                .first_patch
                .clone()
                .or_else(|| set.topology.default_first.clone())
                .unwrap_or_else(|| "NTER".to_owned());

            let patch = match (aa, h_count(n_atom)) {
                (AminoAcid::Pro, 2) | (_, 3) => Some(default_first),
                (AminoAcid::Gly, 2) => {
                    return Err(ParamError::new(&format!(
                        "Neutral N-terminal glycine ({}) isn't supported with CHARMM",
                        res_label()
                    )));
                }
                (AminoAcid::Pro, _) => None,
                (_, 2) => Some("NNEU".to_owned()),
                // E.g. at a gap in the structure.
                _ => None,
            };
            if let Some(p) = patch
                && p != "NONE"
            {
                inst.apply_patch(set.residue(&p)?);
            }
        }

        // C-terminus
        if next[ri].is_none()
            && let Some(oxt) = find(ri, "OXT")
        {
            let o_h = h_count_named(ri, "O");
            let patch = if h_count(oxt) + o_h == 1 {
                "CNEU"
            } else {
                base.last_patch
                    .as_deref()
                    .or(set.topology.default_last.as_deref())
                    .unwrap_or("CTER")
            };
            if patch != "NONE" {
                inst.apply_patch(set.residue(patch)?);
            }
            // CNEU puts the proton on OT2.
            if o_h == 1 {
                renames[ri].insert("O", "OT2");
                renames[ri].insert("OXT", "OT1");
            } else {
                renames[ri].insert("O", "OT1");
                renames[ri].insert("OXT", "OT2");
            }
        }

        instances.push(inst);
    }

    let mut cross_bonds = Vec::new();
    if !disulfides.is_empty() {
        let patch = set.residue("DISU")?;
        for &(ra, rb) in &disulfides {
            let (lo, hi) = instances.split_at_mut(rb);
            let bonds = apply_two_residue_patch(patch, [&mut lo[ra], &mut hi[0]])?;
            for [a, b] in bonds {
                cross_bonds.push(((ra, a), (rb, b)));
            }
        }
    }

    // Match input atoms to CHARMM's.
    let mut charmm_atom: Vec<Option<(usize, String)>> = vec![None; n]; // (residue, CHARMM name)
    let mut by_name: HashMap<(usize, String), usize> = HashMap::new();
    let mut errors = Vec::new();

    for ri in 0..n_res {
        let inst = &instances[ri];
        let res_label = format!(
            "{:?} {} (CHARMM {})",
            mol_res[ri].aa, mol_res[ri].serial_number, inst.label
        );

        // Heavy atoms, by name.
        for &i in mol_res[ri].atoms.iter().filter(|&&i| !is_h[i]) {
            let Some(name) = &names[i] else {
                errors.push(format!("An atom in {res_label} has no name"));
                continue;
            };
            let name = renames[ri]
                .get(name.as_str())
                .map(|n| n.to_string())
                .unwrap_or_else(|| name.clone());

            match inst.atom(&name) {
                Some(a) if !set.is_hydrogen_type(&a.atom_type) => {
                    if by_name.insert((ri, name.clone()), i).is_some() {
                        errors.push(format!("{res_label} has more than one {name}"));
                    }
                    charmm_atom[i] = Some((ri, name));
                }
                _ => errors.push(format!("{res_label} has unexpected atom {name}")),
            }
        }
        for a in &inst.atoms {
            if !set.is_hydrogen_type(&a.atom_type) && !by_name.contains_key(&(ri, a.name.clone())) {
                errors.push(format!("{res_label} is missing atom {}", a.name));
            }
        }

        // Hydrogens, by the heavy atom they're bonded to. Where names agree, we keep them.
        let mut hs_by_heavy: HashMap<String, Vec<usize>> = HashMap::new();
        for &i in mol_res[ri].atoms.iter().filter(|&&i| is_h[i]) {
            let heavy = match nbrs_in[i].as_slice() {
                [j] => charmm_atom[*j].as_ref().filter(|(r, _)| *r == ri),
                _ => None,
            };
            match heavy {
                Some((_, heavy_name)) => hs_by_heavy.entry(heavy_name.clone()).or_default().push(i),
                None => errors.push(format!(
                    "Hydrogen {} in {res_label} must be bonded to one heavy atom in its residue",
                    names[i].as_deref().unwrap_or("?")
                )),
            }
        }

        let charmm_hs = |heavy: &str| -> Vec<String> {
            inst.bonds
                .iter()
                .filter_map(|[a, b]| {
                    let other = if a == heavy {
                        b
                    } else if b == heavy {
                        a
                    } else {
                        return None;
                    };
                    inst.atom(other)
                        .filter(|at| set.is_hydrogen_type(&at.atom_type))
                        .map(|at| at.name.clone())
                })
                .collect()
        };

        for a in inst
            .atoms
            .iter()
            .filter(|a| !set.is_hydrogen_type(&a.atom_type))
        {
            let expected = charmm_hs(&a.name);
            let mut mol_hs = hs_by_heavy.remove(&a.name).unwrap_or_default();
            if mol_hs.len() != expected.len() {
                errors.push(format!(
                    "{res_label}: {} has {} hydrogens; CHARMM expects {} ({})",
                    a.name,
                    mol_hs.len(),
                    expected.len(),
                    expected.join(", ")
                ));
                continue;
            }
            mol_hs.sort_by(|&x, &y| names[x].cmp(&names[y]));

            let mut unassigned: Vec<&String> = Vec::new();
            for h_name in &expected {
                match mol_hs
                    .iter()
                    .position(|&i| names[i].as_deref() == Some(h_name.as_str()))
                {
                    Some(k) => {
                        let i = mol_hs.remove(k);
                        charmm_atom[i] = Some((ri, h_name.clone()));
                        by_name.insert((ri, h_name.clone()), i);
                    }
                    None => unassigned.push(h_name),
                }
            }
            for (i, h_name) in mol_hs.into_iter().zip(unassigned) {
                charmm_atom[i] = Some((ri, h_name.clone()));
                by_name.insert((ri, h_name.clone()), i);
            }
        }
        // Hydrogens on heavy atoms CHARMM doesn't have are reported as unexpected heavy atoms.
    }

    if !errors.is_empty() {
        errors.dedup();
        return Err(ParamError::new(&format!(
            "Unable to build the peptide with CHARMM36m: {}",
            summarize(&errors)
        )));
    }

    // Types and charges.
    let mut types = Vec::with_capacity(n);
    for (i, atom) in atoms.iter_mut().enumerate() {
        let (ri, name) = charmm_atom[i].as_ref().unwrap();
        let a = instances[*ri].atom(name).unwrap();
        atom.force_field_type = Some(a.atom_type.clone());
        atom.partial_charge = Some(a.charge);
        types.push(a.atom_type.clone());
    }

    // Resolve a name in a residue's terms, with `-` and `+` prefixes, to an atom index.
    let resolve = |ri: usize, name: &str| -> Option<usize> {
        let (r, n) = if let Some(n) = name.strip_prefix('-') {
            (prev[ri]?, n)
        } else if let Some(n) = name.strip_prefix('+') {
            (next[ri]?, n)
        } else {
            (ri, name)
        };
        by_name.get(&(r, n.to_owned())).copied()
    };

    let mut bond_set = BTreeSet::new();
    for (ri, inst) in instances.iter().enumerate() {
        for [a, b] in &inst.bonds {
            if let (Some(i), Some(j)) = (resolve(ri, a), resolve(ri, b))
                && i != j
            {
                bond_set.insert((i.min(j), i.max(j)));
            }
        }
    }
    for ((ra, a), (rb, b)) in &cross_bonds {
        if let (Some(&i), Some(&j)) = (
            by_name.get(&(*ra, a.clone())),
            by_name.get(&(*rb, b.clone())),
        ) {
            bond_set.insert((i.min(j), i.max(j)));
        }
    }

    let mut nbrs = vec![Vec::new(); n];
    for &(i, j) in &bond_set {
        nbrs[i].push(j);
        nbrs[j].push(i);
    }

    let mut result = ExplicitParams {
        lj_combining: LjCombiningRule::LorentzBerthelot,
        ..Default::default()
    };
    let mut missing = BTreeSet::new();

    for t in &types {
        match (set.mass(t), set.params.nonbonded.get(t)) {
            (Some(mass), Some(nb)) => {
                result.masses.push(mass);
                result.lj.push((nb.rmin_half * RMIN_HALF_TO_SIGMA, nb.eps));
            }
            _ => {
                missing.insert(format!("mass or LJ for type {t}"));
                result.masses.push(1.);
                result.lj.push((0., 0.));
            }
        }
    }

    for &(i, j) in &bond_set {
        let (ti, tj) = (&types[i], &types[j]);
        match set.params.bond(ti, tj) {
            Some((k_b, r_0)) => result.bonds.push((
                (i, j),
                BondStretchingParams {
                    atom_types: (ti.clone(), tj.clone()),
                    k_b,
                    r_0,
                    comment: None,
                },
            )),
            None => {
                missing.insert(format!("bond {ti}-{tj}"));
            }
        }
    }

    let mut excluded = BTreeSet::new();
    for &pair in &bond_set {
        excluded.insert(pair);
    }

    for ctr in 0..n {
        for (a, &i) in nbrs[ctr].iter().enumerate() {
            for &k in &nbrs[ctr][a + 1..] {
                let (i, k) = (i.min(k), i.max(k));
                excluded.insert((i, k));

                let (ti, tc, tk) = (&types[i], &types[ctr], &types[k]);
                match set.params.angle(ti, tc, tk) {
                    Some(p) => {
                        result.angles.push((
                            (i, ctr, k),
                            AngleBendingParams {
                                atom_types: (ti.clone(), tc.clone(), tk.clone()),
                                k: p.k,
                                theta_0: p.theta0.to_radians(),
                                comment: None,
                            },
                        ));
                        if let Some(ub) = p.urey_bradley
                            && ub.0 != 0.
                        {
                            result.urey_bradley.push(((i, k), ub));
                        }
                    }
                    None => {
                        missing.insert(format!("angle {ti}-{tc}-{tk}"));
                    }
                }
            }
        }
    }

    let mut pairs_14 = BTreeSet::new();
    for &(j, k) in &bond_set {
        for &i in nbrs[j].iter().filter(|&&i| i != k) {
            for &l in nbrs[k].iter().filter(|&&l| l != j && l != i) {
                let t = [&types[i], &types[j], &types[k], &types[l]];
                let terms = set.params.dihedral(t.map(|s| s.as_str()));
                if terms.is_empty() {
                    missing.insert(format!("dihedral {}-{}-{}-{}", t[0], t[1], t[2], t[3]));
                }
                for term in terms {
                    result.dihedrals.push((
                        (i, j, k, l),
                        DihedralParams {
                            atom_types: (t[0].clone(), t[1].clone(), t[2].clone(), t[3].clone()),
                            divider: 1,
                            barrier_height: term.k,
                            phase: term.delta.to_radians(),
                            periodicity: term.n,
                            comment: None,
                        },
                    ));
                }
                pairs_14.insert((i.min(l), i.max(l)));
            }
        }
    }

    let mut cmap_grid_index: HashMap<[String; 8], usize> = HashMap::new();

    for (ri, inst) in instances.iter().enumerate() {
        for imp in &inst.impropers {
            let Some(idx) = resolve_all(imp, |nm| resolve(ri, nm)) else {
                continue; // E.g. at a terminus.
            };
            let t: [&str; 4] = idx.map(|i| types[i].as_str());
            match set.params.improper(t) {
                Some(p) => result
                    .harmonic_impropers
                    .push(((idx[0], idx[1], idx[2], idx[3]), (p.k, p.psi0.to_radians()))),
                None => {
                    missing.insert(format!("improper {}", t.join("-")));
                }
            }
        }

        for cmap in &inst.cmaps {
            let Some(idx) = resolve_all(cmap, |nm| resolve(ri, nm)) else {
                continue;
            };
            let t: [&str; 8] = idx.map(|i| types[i].as_str());
            let Some(c) = set.params.cmap(t) else {
                missing.insert(format!("CMAP {}", t.join("-")));
                continue;
            };
            let grid = *cmap_grid_index.entry(c.types.clone()).or_insert_with(|| {
                result
                    .cmap_grids
                    .push(CmapGrid::new(c.size, c.energies.clone()));
                result.cmap_grids.len() - 1
            });
            result.cmaps.push((idx, grid));
        }
    }

    if !missing.is_empty() {
        let missing: Vec<_> = missing.into_iter().collect();
        return Err(ParamError::new(&format!(
            "Missing CHARMM parameters: {}",
            summarize(&missing)
        )));
    }

    // 1-4 pairs, with CHARMM's special 1-4 LJ parameters where present. Coulomb and LJ are
    // unscaled. (e14fac = 1)
    for pair in pairs_14 {
        if excluded.contains(&pair) {
            continue; // e.g. in 5-membered rings
        }
        let special = |i: usize| {
            let nb = &set.params.nonbonded[&types[i]];
            (nb.eps_14.is_some() || nb.rmin_half_14.is_some()).then(|| {
                (
                    nb.rmin_half_14.unwrap_or(nb.rmin_half) * RMIN_HALF_TO_SIGMA,
                    nb.eps_14.unwrap_or(nb.eps),
                )
            })
        };
        let (i, j) = pair;
        let lj = if special(i).is_some() || special(j).is_some() {
            let (s_i, e_i) = special(i).unwrap_or(result.lj[i]);
            let (s_j, e_j) = special(j).unwrap_or(result.lj[j]);
            Some(LjCombiningRule::LorentzBerthelot.combine(s_i, e_i, s_j, e_j))
        } else {
            None
        };
        result.pairs_14.push((
            pair,
            Pair14 {
                scale: Scale14 {
                    lj: 1.,
                    coulomb: 1.,
                },
                lj,
            },
        ));
    }
    result.exclusions = excluded.into_iter().collect();

    Ok(result)
}

/// Resolve each name to an atom index; None if any is missing.
fn resolve_all<const N: usize>(
    names: &[String; N],
    resolve: impl Fn(&str) -> Option<usize>,
) -> Option<[usize; N]> {
    let mut result = [0; N];
    for (r, name) in result.iter_mut().zip(names) {
        *r = resolve(name)?;
    }
    Some(result)
}

fn summarize(items: &[String]) -> String {
    const MAX: usize = 12;
    let mut s = items
        .iter()
        .take(MAX)
        .cloned()
        .collect::<Vec<_>>()
        .join("; ");
    if items.len() > MAX {
        s += &format!("; and {} more", items.len() - MAX);
    }
    s
}
