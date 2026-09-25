//! Data functionality for Forcefield params. Includes Amber parameters built in to binaries
//! which use this library, and can load params for other sets as required.
//!
//! Uses `bio_files` for the base data structures.

use std::{collections::HashMap, io, path::PathBuf};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, ChainGeneric, LipidStandard, MmCif, ResidueEnd, ResidueGeneric,
    ResidueType, create_bonds,
    md_params::{
        ChargeParams, ChargeParamsProtein, ForceFieldParams, NucleotideTemplate,
        load_amino_charges, parse_lib_lipid, parse_lib_nucleic_acid, parse_lib_peptide,
    },
};
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes, Element};

use crate::{
    Dihedral, FfMolType, ParamError,
    charmm::CharmmSet,
    merge_params,
    non_bonded::{LjCombiningRule, NbFix, Scale14},
    param_inference::update_small_mol_params,
    populate_hydrogens_dihedrals,
    solvent::WaterModel,
};

pub type ProtFfChargeMap = HashMap<AminoAcidGeneral, Vec<ChargeParamsProtein>>;
pub type LipidFfChargeMap = HashMap<LipidStandard, Vec<ChargeParams>>;
pub type NucleicAcidFfChargeMap = HashMap<NucleotideTemplate, Vec<ChargeParams>>;

// We include Amber parameter files with this package.
// Proteins and amino acids:
const PARM_19: &str = include_str!("../param_data/parm19.dat"); // Bonded, and LJ
const FRCMOD_FF19SB: &str = include_str!("../param_data/frcmod.ff19SB"); // Bonded, and LJ: overrides and new types
pub const AMINO_19: &str = include_str!("../param_data/amino19.lib"); // Charge; internal residues
const AMINO_NT12: &str = include_str!("../param_data/aminont12.lib"); // Charge; protonated N-terminus residues
const AMINO_CT12: &str = include_str!("../param_data/aminoct12.lib"); // Charge; protonated C-terminus residues

// Ligands/small organic molecules: *General Amber Force Fields*.
const GAFF2: &str = include_str!("../param_data/gaff2.dat");
// Lipids
const LIPID_21: &str = include_str!("../param_data/lipid21.dat"); // Bonded and LJ

// Public, so we can use it for lipid templates.
pub const LIPID_21_LIB: &str = include_str!("../param_data/lipid21.lib"); // Charge and FF names

// DNA (OL24) and RNA (OL3)
pub const OL24_LIB: &str = include_str!("../param_data/ff-nucleic-OL24.lib");
const OL24_FRCMOD: &str = include_str!("../param_data/ff-nucleic-OL24.frcmod");
// todo: frcmod.protonated_nucleic?
// RNA (I believe this is the OL3 Amber's FF page recommends?)
pub const RNA_LIB: &str = include_str!("../param_data/RNA.lib");
// todo: RNA.YIL.lib? RNA_CI.lib? RNA_Shaw.lib? These are, I believe, "alternative" libraries,
// todo, and not required. YIL: Yildirim torsion refit. CI: Legacy Cornell-style. SHAW: incomplete,
// todo from a person named Shaw.

// Note: Water parameters are concise; we store them directly. See `WaterModel`.

/// A family of force fields, with its conventions and recommended water model. We resolve this
/// to data once, with `FfParamSet::from_family`; the rest of the library uses that data, and
/// doesn't match on the family.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForceFieldFamily {
    /// ff19SB (proteins), OL24 (DNA), OL3 (RNA), lipid21, and GAFF2 (small molecules), with
    /// OPC water. Amber's recommendations as of Sept 2025.
    #[default]
    Amber,
    /// CHARMM36m for proteins, with CHARMM's TIP3P water (with LJ on H), and SOD and CLA ions.
    /// Includes Urey-Bradley terms, harmonic impropers, CMAP, special 1-4 LJ parameters, and
    /// NBFIX. We build proteins from CHARMM's residue topologies; they need
    /// `MolDynamics::residues`. Other molecule types need explicit parameters. (e.g. imported;
    /// see `import`) Use `MdConfig::for_family` for CHARMM's cutoffs and LJ force switch.
    Charmm36,
}

/// How we assign force field types, partial charges, and missing bonded parameters to small
/// molecules that don't have them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SmallMolTyper {
    /// GAFF2 types using Antechamber's rules, partial charges from this library's trained model,
    /// and parmchk-style estimates for missing parameters. See `param_inference`.
    #[default]
    Gaff2,
}

impl SmallMolTyper {
    /// Assigns force field types and partial charges to `atoms` in place, and returns
    /// molecule-specific parameters. `general_params` is the family's small-molecule set.
    pub fn assign(
        self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
        adjacency_list: Option<&[Vec<usize>]>,
        general_params: Option<&ForceFieldParams>,
    ) -> io::Result<ForceFieldParams> {
        let Some(general_params) = general_params else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "The parameter set has no small-molecule parameters to type this molecule with.                  Pass explicit parameters, e.g. from a topology file.",
            ));
        };
        match self {
            Self::Gaff2 => update_small_mol_params(atoms, bonds, adjacency_list, general_params),
        }
    }
}

/// 1-4 scale factors for each molecule type.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scale14ByMolType {
    pub peptide: Scale14,
    pub small_organic: Scale14,
    pub dna: Scale14,
    pub rna: Scale14,
    pub lipid: Scale14,
    /// Set this to `Scale14::GLYCAM` if loading GLYCAM parameters.
    pub carbohydrate: Scale14,
}

impl Scale14ByMolType {
    pub fn get(&self, mol_type: FfMolType) -> Scale14 {
        match mol_type {
            FfMolType::Peptide => self.peptide,
            FfMolType::SmallOrganic => self.small_organic,
            FfMolType::Dna => self.dna,
            FfMolType::Rna => self.rna,
            FfMolType::Lipid => self.lipid,
            FfMolType::Carbohydrate => self.carbohydrate,
        }
    }
}

/// Non-bonded conventions that are part of a force field's definition, rather than of individual
/// atom types. These must match the parameters they're used with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NonBondedRules {
    /// Applies to all pairs in the system, so all molecules' parameters must use the same rule.
    pub lj_combining: LjCombiningRule,
    pub scale_14: Scale14ByMolType,
    /// Pair-specific LJ parameters by force field type, which override the combining rule. (e.g.
    /// CHARMM's NBFIX)
    pub nbfix: NbFix,
}

#[derive(Default, Debug)]
/// A set of general parameters that aren't molecule-specific. E.g. from GAFF2, OL3, RNA, or amino19.
/// These are used as a baseline, and in some cases, overridden by molecule-specific parameters.
///
/// This also holds the rest of what defines a force field family: its non-bonded conventions,
/// recommended water model, and small-molecule typing.
pub struct FfParamSet {
    /// The family these parameters belong to. `MdState::new` requires this to match
    /// `MdConfig::ff_family`.
    pub family: ForceFieldFamily,
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
    // todo: QC these types; lipid as place holder. See how they parse.
    pub dna_ff_q_map: Option<NucleicAcidFfChargeMap>,
    pub rna_ff_q_map: Option<NucleicAcidFfChargeMap>,
    /// Defaults to Amber's conventions, which also cover GAFF2, lipid21, and the nucleic acid sets.
    pub nonbonded_rules: NonBondedRules,
    /// The water model this family recommends, including its counter-ions. `MdConfig::water_model`
    /// overrides this.
    pub default_water: WaterModel,
    pub small_mol_typer: SmallMolTyper,
    /// CHARMM's topology and parameters. If present, we build peptides from these, instead of
    /// assigning parameters by type from `peptide`.
    pub charmm: Option<CharmmSet>,
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
    /// Load the parameter set for a force field family, using parameter files included with this
    /// library.
    pub fn from_family(family: ForceFieldFamily) -> io::Result<Self> {
        match family {
            ForceFieldFamily::Amber => Self::new_amber(),
            ForceFieldFamily::Charmm36 => Self::new_charmm(),
        }
    }

    /// CHARMM36m for proteins, with CHARMM's TIP3P water and ions, using files included with this
    /// library. (toppar_c36_jul24) See `ForceFieldFamily::Charmm36`.
    ///
    /// To add hydrogens to proteins, use the Amber set's `peptide_ff_q_map`, e.g. with
    /// `prepare_peptide`; `MdState::new` then assigns CHARMM types and charges.
    pub fn new_charmm() -> io::Result<Self> {
        let charmm = CharmmSet::new_c36m()?;

        let scale_14 = Scale14 {
            lj: 1.,
            coulomb: 1.,
        };

        Ok(Self {
            family: ForceFieldFamily::Charmm36,
            nonbonded_rules: NonBondedRules {
                lj_combining: LjCombiningRule::LorentzBerthelot,
                // CHARMM's e14fac is 1. Special 1-4 LJ parameters are per pair.
                scale_14: Scale14ByMolType {
                    peptide: scale_14,
                    small_organic: scale_14,
                    dna: scale_14,
                    rna: scale_14,
                    lipid: scale_14,
                    carbohydrate: scale_14,
                },
                nbfix: charmm.nbfix(),
            },
            default_water: WaterModel::TIP3P_CHARMM,
            // No CGenFF yet; small molecules need explicit parameters.
            small_mol_typer: SmallMolTyper::Gaff2,
            charmm: Some(charmm),
            ..Default::default()
        })
    }

    /// Load general parameter files for the most common classes of organic molecules.
    /// This also populates ff type and charge for protein atoms; these are provided by molecule-specific
    /// formats for small molecules.
    ///
    /// These are Amber-format files (.dat, .frcmod, .lib), so we use Amber's conventions.
    pub fn new(paths: &ParamGeneralPaths) -> io::Result<Self> {
        let mut result = Self::amber_conventions();

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
        let mut result = Self::amber_conventions();

        // We use parm19 for both peptides, and nucleic acids.
        let parm19 = ForceFieldParams::from_dat(PARM_19)?;

        let peptide_frcmod = ForceFieldParams::from_frcmod(FRCMOD_FF19SB)?;
        result.peptide = Some(merge_params(&parm19, &peptide_frcmod));

        {
            let internal = parse_lib_peptide(AMINO_19)?;
            let n_terminus = parse_lib_peptide(AMINO_NT12)?;
            let c_terminus = parse_lib_peptide(AMINO_CT12)?;

            result.peptide_ff_q_map = Some(ProtFfChargeMapSet {
                internal,
                n_terminus,
                c_terminus,
            });
        }

        let lipid_dat = ForceFieldParams::from_dat(LIPID_21)?;
        result.lipids = Some(lipid_dat);

        let lipid_charges = parse_lib_lipid(LIPID_21_LIB)?;
        result.lipid_ff_q_map = Some(lipid_charges);

        result.small_mol = Some(ForceFieldParams::from_dat(GAFF2)?);

        // todo: Load these, and get them working. They currently trigger a mass-parsing error.
        // todo: You must update your Lib parser in bio_files to handle this variant.

        let dna_frcmod = ForceFieldParams::from_frcmod(OL24_FRCMOD)?;
        result.dna = Some(merge_params(&parm19, &dna_frcmod));

        // todo: A/R
        result.rna = Some(parm19.clone());

        // todo: Currently hardcoded peptide/lipid versions for this lib parsing. Generalize?
        let dna_charges = parse_lib_nucleic_acid(OL24_LIB)?;
        result.dna_ff_q_map = Some(dna_charges);

        // todo: Currently hardcoded peptide/lipid versions for this lib parsing. Generalize?
        let rna_charges = parse_lib_nucleic_acid(RNA_LIB)?;
        result.rna_ff_q_map = Some(rna_charges);

        Ok(result)
    }

    /// An empty set with Amber's family tag, non-bonded rules, water model, and typing.
    fn amber_conventions() -> Self {
        Self {
            family: ForceFieldFamily::Amber,
            nonbonded_rules: NonBondedRules {
                lj_combining: LjCombiningRule::LorentzBerthelot,
                scale_14: Scale14ByMolType {
                    peptide: Scale14::AMBER,
                    small_organic: Scale14::AMBER,
                    dna: Scale14::AMBER,
                    rna: Scale14::AMBER,
                    lipid: Scale14::AMBER,
                    // todo: GLYCAM (Amber's recommendation) uses `Scale14::GLYCAM`, but we don't
                    // todo: include its parameters yet. We use this for all molecules, as before.
                    carbohydrate: Scale14::AMBER,
                },
                nbfix: Default::default(),
            },
            default_water: WaterModel::OPC,
            small_mol_typer: SmallMolTyper::Gaff2,
            ..Default::default()
        }
    }
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
                // e.g. solvent or other hetero atoms; skip.
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

                        // The amber template doesn't have HH23; only 2 Hs on that. I believe
                        // this may be an omission.
                        if aa_gen == AminoAcidGeneral::Standard(AminoAcid::Arg)
                            && *type_in_res == AtomTypeInRes::H("HH23".to_owned())
                        {
                            for charge in charges {
                                if charge.type_in_res == AtomTypeInRes::H("HH22".to_string()) {
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
