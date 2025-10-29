//! For Amber and other parameters.

use std::{
    collections::{HashMap, HashSet},
    io,
    path::PathBuf,
};

use bio_files::{
    AtomGeneric, BondGeneric, ChainGeneric, LipidStandard, MmCif, MolType, ResidueEnd,
    ResidueGeneric, ResidueType, create_bonds,
    md_params::{
        AngleBendingParams, BondStretchingParams, ChargeParams, ChargeParamsLipid, DihedralParams,
        ForceFieldParams, LjParams, MassParams, load_amino_charges, parse_amino_charges,
        parse_lipid_charges,
    },
};
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes, Element};

use crate::{Dihedral, ParamError, merge_params, populate_hydrogens_dihedrals};

pub type ProtFfChargeMap = HashMap<AminoAcidGeneral, Vec<ChargeParams>>;
pub type LipidFfChargeMap = HashMap<LipidStandard, Vec<ChargeParamsLipid>>;

// We include Amber parameter files with this package.
// Proteins and amino acids:
const PARM_19: &str = include_str!("../param_data/parm19.dat"); // Bonded, and LJ
const FRCMOD_FF19SB: &str = include_str!("../param_data/frcmod.ff19SB"); // Bonded, and LJ: overrides and new types
const AMINO_19: &str = include_str!("../param_data/amino19.lib"); // Charge; internal residues
const AMINO_NT12: &str = include_str!("../param_data/aminont12.lib"); // Charge; protonated N-terminus residues
const AMINO_CT12: &str = include_str!("../param_data/aminoct12.lib"); // Charge; protonated C-terminus residues

// Ligands/small organic molecules: *General Amber Force Fields*.
const GAFF2: &str = include_str!("../param_data/gaff2.dat");
// Lipids
const LIPID_21: &str = include_str!("../param_data/lipid21.dat"); // Bonded and LJ

// Public, so we can use it for lipid templates.
pub const LIPID_21_LIB: &str = include_str!("../param_data/lipid21.lib"); // Charge and FF names

// DNA (OL24) and RNA (OL3)
const OL24_LIB: &str = include_str!("../param_data/ff-nucleic-OL24.lib");
const OL24_FRCMOD: &str = include_str!("../param_data/ff-nucleic-OL24.frcmod");
// todo: frcmod.protonated_nucleic?
// RNA (I believe this is the OL3 Amber's FF page recommends?)
const RNA_LIB: &str = include_str!("../param_data/RNA.lib");
// todo: RNA.YIL.lib? RNA_CI.lib? RNA_Shaw.lib? These are, I believe, "alternative" libraries,
// todo, and not required. YIL: Yildirim torsion refit. CI: Legacy Cornell-style. SHAW: incomplete,
// todo from a person named Shaw.

#[derive(Default, Debug)]
/// A set of general parameters that aren't molecule-specific. E.g. from GAFF2, OL3, RNA, or amino19.
/// These are used as a baseline, and in some cases, overridden by molecule-specific parameters.
pub struct FfParamSet {
    pub peptide: Option<ForceFieldParams>,
    pub small_mol: Option<ForceFieldParams>,
    pub dna: Option<ForceFieldParams>,
    pub rna: Option<ForceFieldParams>,
    pub lipids: Option<ForceFieldParams>,
    pub carbohydrates: Option<ForceFieldParams>,
    /// In addition to charge, this also contains the mapping of res type to FF type; required to map
    /// other parameters to protein atoms. E.g. from `amino19.lib`, and its N and C-terminus variants.
    pub peptide_ff_q_map: Option<ProtFfChargeMapSet>,
    pub lipid_ff_q_map: Option<LipidFfChargeMap>,
}

/// Paths for to general parameter files. Used to create a FfParamSet.
#[derive(Clone, Debug, Default)]
pub struct ParamGeneralPaths {
    /// E.g. parm19.dat
    pub peptide: Option<PathBuf>,
    /// E.g. ff19sb.dat
    pub peptide_mod: Option<PathBuf>,
    /// E.g. amino19.lib
    pub peptide_ff_q: Option<PathBuf>,
    /// E.g. aminoct12.lib
    pub peptide_ff_q_c: Option<PathBuf>,
    /// E.g. aminont12.lib
    pub peptide_ff_q_n: Option<PathBuf>,
    /// e.g. gaff2.dat
    pub small_organic: Option<PathBuf>,
    /// e.g. ff-nucleic-OL24.lib
    pub dna: Option<PathBuf>,
    /// e.g. ff-nucleic-OL24.frcmod
    pub dna_mod: Option<PathBuf>,
    /// e.g. RNA.lib
    pub rna: Option<PathBuf>,
    pub lipid: Option<PathBuf>,
    pub carbohydrate: Option<PathBuf>,
}

impl FfParamSet {
    /// Load general parameter files for proteins, and small organic molecules.
    /// This also populates ff type and charge for protein atoms.
    pub fn new(paths: &ParamGeneralPaths) -> io::Result<Self> {
        let mut result = FfParamSet::default();

        if let Some(p) = &paths.peptide {
            let peptide = ForceFieldParams::load_dat(p)?;

            if let Some(p_mod) = &paths.peptide_mod {
                let frcmod = ForceFieldParams::load_frcmod(p_mod)?;
                result.peptide = Some(merge_params(&peptide, &frcmod));
            } else {
                result.peptide = Some(peptide);
            }
        }

        let mut ff_map = ProtFfChargeMapSet::default();
        if let Some(p) = &paths.peptide_ff_q {
            ff_map.internal = load_amino_charges(p)?;
        }
        if let Some(p) = &paths.peptide_ff_q_c {
            ff_map.internal = load_amino_charges(p)?;
        }
        if let Some(p) = &paths.peptide_ff_q_n {
            ff_map.internal = load_amino_charges(p)?;
        }

        result.peptide_ff_q_map = Some(ff_map);

        if let Some(p) = &paths.small_organic {
            result.small_mol = Some(ForceFieldParams::load_dat(p)?);
        }

        if let Some(p) = &paths.dna {
            let peptide = ForceFieldParams::load_dat(p)?;

            if let Some(p_mod) = &paths.dna_mod {
                let frcmod = ForceFieldParams::load_frcmod(p_mod)?;
                result.dna = Some(merge_params(&peptide, &frcmod));
            } else {
                result.dna = Some(peptide);
            }
        }

        if let Some(p) = &paths.rna {
            result.rna = Some(ForceFieldParams::load_dat(p)?);
        }

        if let Some(p) = &paths.lipid {
            result.lipids = Some(ForceFieldParams::load_dat(p)?);
        }

        if let Some(p) = &paths.carbohydrate {
            result.carbohydrates = Some(ForceFieldParams::load_dat(p)?);
        }

        Ok(result)
    }

    /// Create a parameter set using Amber parameters included with this library. This uses
    /// the param sets recommended by Amber, CAO Sept 2025: ff19SB, OL24, OL3, GLYCAM_06j, lipids21,
    /// and gaff2.
    pub fn new_amber() -> io::Result<Self> {
        let mut result = FfParamSet::default();

        let peptide = ForceFieldParams::from_dat(PARM_19)?;
        let peptide_frcmod = ForceFieldParams::from_frcmod(FRCMOD_FF19SB)?;
        result.peptide = Some(merge_params(&peptide, &peptide_frcmod));

        {
            let internal = parse_amino_charges(AMINO_19)?;
            let n_terminus = parse_amino_charges(AMINO_NT12)?;
            let c_terminus = parse_amino_charges(AMINO_CT12)?;

            result.peptide_ff_q_map = Some(ProtFfChargeMapSet {
                internal,
                n_terminus,
                c_terminus,
            });
        }

        let lipid_dat = ForceFieldParams::from_dat(LIPID_21)?;
        result.lipids = Some(lipid_dat);

        let lipid_charges = parse_lipid_charges(LIPID_21_LIB)?;
        result.lipid_ff_q_map = Some(lipid_charges);

        result.small_mol = Some(ForceFieldParams::from_dat(GAFF2)?);

        // todo: Load these, and get them working. They currently trigger a mass-parsing error.
        // todo: You must update your Lib parser in bio_files to handle this variant.

        // let dna = ForceFieldParams::from_dat(OL24_LIB)?;
        // let dna_frcmod = ForceFieldParams::from_frcmod(OL24_FRCMOD)?;
        // result.dna = Some(merge_params(&dna, Some(&dna_frcmod)));
        // result.rna = Some(ForceFieldParams::from_dat(RNA_LIB)?);

        Ok(result)
    }
}

/// This variant of forcefield parameters offers the fastest lookups. Unlike the Vec and Hashmap
/// based parameter structs, this is specific to the atom in our docking setup: The indices are provincial
/// to specific sets of atoms. For a description of fields, see `ForceFieldParams`, or the individual
/// param-type structs here.
///
/// Note: The single-atom fields of `mass` and `partial_charges` are omitted: They're part of our
/// `AtomDynamics` struct.`
#[derive(Clone, Debug, Default)]
pub(crate) struct ForceFieldParamsIndexed {
    pub mass: HashMap<usize, MassParams>,
    pub bond_stretching: HashMap<(usize, usize), BondStretchingParams>,
    /// Any bond to Hydrogen if configured as constrained. (Distance^2 in Å, 1 / mass in Daltons)
    pub bond_rigid_constraints: HashMap<(usize, usize), (f32, f32)>,
    pub angle: HashMap<(usize, usize, usize), AngleBendingParams>,
    pub dihedral: HashMap<(usize, usize, usize, usize), Vec<DihedralParams>>,
    pub improper: HashMap<(usize, usize, usize, usize), Vec<DihedralParams>>,
    /// We use this to determine which 1-2 exclusions to apply for non-bonded forces. We use this
    /// instead of `bond_stretching`, because `bond_stretching` omits bonds to Hydrogen, which we need
    /// to account when applying exclusions.
    pub bonds_topology: HashSet<(usize, usize)>,
    pub lennard_jones: HashMap<usize, LjParams>,
}

#[derive(Clone, Default, Debug)]
/// Maps type-in-residue (found in, e.g. mmCIF and PDB files) to Amber FF type, and partial charge.
/// We assume that if one of these is loaded, so are the others. So, these aren't `Options`s, but
/// the field that holds this struct should be one.
pub struct ProtFfChargeMapSet {
    pub internal: ProtFfChargeMap,
    pub n_terminus: ProtFfChargeMap,
    pub c_terminus: ProtFfChargeMap,
}

/// Populate forcefield type, and partial charge on atoms. This should be run on mmCIF
/// files prior to running molecular dynamics on them. These files from RCSB PDB do not
/// natively have this data.
///
/// `residues` must be the full set; this is relevant to how we index it.
pub fn populate_peptide_ff_and_q(
    atoms: &mut [AtomGeneric],
    residues: &[ResidueGeneric],
    ff_type_charge: &ProtFfChargeMapSet,
) -> Result<(), ParamError> {
    // Tis is slower than if we had an index map already.
    let mut index_map = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        index_map.insert(atom.serial_number, i);
    }

    for res in residues {
        for sn in &res.atom_sns {
            let atom = match atoms.get_mut(index_map[sn]) {
                Some(a) => a,
                None => {
                    return Err(ParamError::new(&format!(
                        "Unable to populate Charge or FF type for atom {sn}"
                    )));
                }
            };

            if atom.hetero {
                continue;
            }

            let Some(type_in_res) = &atom.type_in_res else {
                return Err(ParamError::new(&format!(
                    "MD failure: Missing type in residue for atom: {atom}"
                )));
            };

            let ResidueType::AminoAcid(aa) = &res.res_type else {
                // e.g. water or other hetero atoms; skip.
                continue;
            };

            // todo: Eventually, determine how to load non-standard AA variants from files; set up your
            // todo state to use those labels. They are available in the params.
            let aa_gen = AminoAcidGeneral::Standard(*aa);

            let charge_map = match res.end {
                ResidueEnd::Internal => &ff_type_charge.internal,
                ResidueEnd::NTerminus => &ff_type_charge.n_terminus,
                ResidueEnd::CTerminus => &ff_type_charge.c_terminus,
                ResidueEnd::Hetero => {
                    return Err(ParamError::new(&format!(
                        "Error: Encountered hetero atom when parsing amino acid FF types: {atom}"
                    )));
                }
            };

            let charges = match charge_map.get(&aa_gen) {
                Some(c) => c,
                // A specific workaround to plain "HIS" being absent from amino19.lib (2025.
                // Choose one of "HID", "HIE", "HIP arbitrarily.
                // todo: Re-evaluate this, e.g. which one of the three to load.
                None if aa_gen == AminoAcidGeneral::Standard(AminoAcid::His) => charge_map
                    .get(&AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Hid))
                    .ok_or_else(|| ParamError::new("Unable to find AA mapping"))?,
                None => return Err(ParamError::new("Unable to find AA mapping")),
            };

            let mut found = false;

            for charge in charges {
                // todo: Note that we have multiple branches in some case, due to Amber names like
                // todo: "HYP" for variants on AAs for different protenation states. Handle this.
                if charge.type_in_res == *type_in_res {
                    atom.force_field_type = Some(charge.ff_type.clone());
                    atom.partial_charge = Some(charge.charge);

                    found = true;
                    break;
                }
            }

            // Code below is mainly for the case of missing data; otherwise, the logic for this operation
            // is complete.

            if !found {
                match type_in_res {
                    // todo: This is a workaround for having trouble with H types. LIkely
                    // todo when we create them. For now, this meets the intent.
                    AtomTypeInRes::H(_) => {
                        // todo: This is a workaround for the above; try other HIS variants.
                        if aa_gen == AminoAcidGeneral::Standard(AminoAcid::His) {
                            let charges = charge_map
                                .get(&AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Hie))
                                .ok_or_else(|| {
                                    ParamError::new("Unable to find AA mapping for HIE")
                                })?;

                            // todo: You may need HIP too, even with this workaround.
                            // todo: DRY

                            for charge in charges {
                                if charge.type_in_res == *type_in_res {
                                    atom.force_field_type = Some(charge.ff_type.clone());
                                    atom.partial_charge = Some(charge.charge);

                                    found = true;
                                    break;
                                }
                            }
                            if found {
                                break;
                            }
                        }

                        // Note: We've witnessed this due to errors in the mmCIF file, e.g. on ASP #88 on 9GLS.
                        eprintln!(
                            "Error assigning FF type and q based on atom type in res: Failed to match H type. Res #{}, Atom #{}, {type_in_res}, {aa_gen:?}. \
                         Falling back to a generic H",
                            res.serial_number, atom.serial_number,
                        );

                        for charge in charges {
                            if charge.type_in_res == AtomTypeInRes::H("H".to_string())
                                || charge.type_in_res == AtomTypeInRes::H("HA".to_string())
                            {
                                atom.force_field_type = Some("HB2".to_string());
                                atom.partial_charge = Some(charge.charge);

                                found = true;
                                break;
                            }
                        }
                    }
                    _ => (),
                }

                // i.e. if still not found after our specific workarounds above.
                if !found {
                    eprintln!("Problem populating FF or Q: {}", atom);
                    continue;
                }
            }
        }
    }

    Ok(())
}

/// Combines several functions that should be run after loading protein files from PDB. Add hydrogens,
/// load force field parameters and partial charge, and add bonds.
pub fn prepare_peptide(
    atoms: &mut Vec<AtomGeneric>,
    bonds: &mut Vec<BondGeneric>,
    residues: &mut Vec<ResidueGeneric>,
    chains: &mut [ChainGeneric],
    ff_map: &ProtFfChargeMapSet,
    ph: f32, // todo: Implement.
) -> Result<Vec<Dihedral>, ParamError> {
    let mut dihedrals = Vec::new();

    let h_count = atoms
        .iter()
        .filter(|a| a.element == Element::Hydrogen)
        .count();
    if h_count < 10 {
        dihedrals = populate_hydrogens_dihedrals(atoms, residues, chains, ff_map, ph)?;
    }

    // todo: Similar checks for empty etc.
    populate_peptide_ff_and_q(atoms, residues, ff_map)?;

    if bonds.is_empty() {
        *bonds = create_bonds(atoms);
    }

    Ok(dihedrals)
}

/// See docs on `prepare_peptide`. This is a convenience variant that uses an `MmCif` file.
pub fn prepare_peptide_mmcif(
    mol: &mut MmCif,
    ff_map: &ProtFfChargeMapSet,
    ph: f32, // todo: Implement.
) -> Result<(Vec<BondGeneric>, Vec<Dihedral>), ParamError> {
    let mut dihedrals = Vec::new();

    let h_count = mol
        .atoms
        .iter()
        .filter(|a| a.element == Element::Hydrogen)
        .count();
    if h_count < 10 {
        dihedrals = populate_hydrogens_dihedrals(
            &mut mol.atoms,
            &mut mol.residues,
            &mut mol.chains,
            ff_map,
            ph,
        )?;
    }

    // todo: Similar checks for empty etc.
    populate_peptide_ff_and_q(&mut mol.atoms, &mol.residues, ff_map)?;

    let bonds = create_bonds(&mol.atoms);

    Ok((bonds, dihedrals))
}
