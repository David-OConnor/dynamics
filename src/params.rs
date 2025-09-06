//! For Amber and other parameters.

use std::collections::{HashMap, HashSet};

use bio_files::amber_params::{
    AngleBendingParams, BondStretchingParams, ChargeParams, DihedralParams, ForceFieldParamsKeyed,
    LjParams, MassParams, VdwParams,
};
use na_seq::AminoAcidGeneral;

pub type ProtFfMap = HashMap<AminoAcidGeneral, Vec<ChargeParams>>;

#[derive(Default, Debug)]
/// A set of general parameters that aren't molecule-specific. E.g. from GAFF2, OL3, RNA, or amino19.
/// These are used as a baseline, and in some cases, overridden by molecule-specific parameters.
pub struct FfParamSet {
    pub peptide: Option<ForceFieldParamsKeyed>,
    pub small_mol: Option<ForceFieldParamsKeyed>,
    pub dna: Option<ForceFieldParamsKeyed>,
    pub rna: Option<ForceFieldParamsKeyed>,
    pub lipids: Option<ForceFieldParamsKeyed>,
    pub carbohydrates: Option<ForceFieldParamsKeyed>,
    /// In addition to charge, this also contains the mapping of res type to FF type; required to map
    /// other parameters to protein atoms. From `amino19.lib`, and its N and C-terminus variants.
    // todo: Should this go here? Look into how it's used.
    pub prot_ff_q_map: Option<ProtFFTypeChargeMap>,
}

/// This variant of forcefield parameters offers the fastest lookups. Unlike the Vec and Hashmap
/// based parameter structs, this is specific to the atom in our docking setup: The indices are provincial
/// to specific sets of atoms.
///
/// Note: The single-atom fields of `mass` and `partial_charges` are omitted: They're part of our
/// `AtomDynamics` struct.`
#[derive(Clone, Debug, Default)]
pub(crate) struct ForceFieldParamsIndexed {
    pub mass: HashMap<usize, MassParams>,
    pub bond_stretching: HashMap<(usize, usize), BondStretchingParams>,
    pub angle: HashMap<(usize, usize, usize), AngleBendingParams>,
    pub dihedral: HashMap<(usize, usize, usize, usize), DihedralParams>,
    /// Generally only for planar hub and spoke arrangements, and always hold a planar dihedral shape.
    /// (e.g. τ/2 with symmetry 2)
    pub improper: HashMap<(usize, usize, usize, usize), DihedralParams>,
    /// We use this to determine which 1-2 exclusions to apply for non-bonded forces. We use this
    /// instead of `bond_stretching`, because `bond_stretching` omits bonds to Hydrogen, which we need
    /// to account when applying exclusions.
    pub bonds_topology: HashSet<(usize, usize)>,

    // Dihedrals are represented in Amber params as a fourier series; this Vec indlues all matches.
    // e.g. X-ca-ca-X may be present multiple times in gaff2.dat. (Although seems to be uncommon)
    //
    // X -nh-sx-X    4    3.000         0.000          -2.000
    // X -nh-sx-X    4    0.400       180.000           3.000
    pub lennard_jones: HashMap<usize, LjParams>,
}

#[derive(Default, Debug)]
/// Maps type-in-residue (found in, e.g. mmCIF and PDB files) to Amber FF type, and partial charge.
/// We assume that if one of these is loaded, so are the others. So, these aren't `Options`s, but
/// the field that holds this struct should be one.
pub struct ProtFFTypeChargeMap {
    pub internal: ProtFfMap,
    pub n_terminus: ProtFfMap,
    pub c_terminus: ProtFfMap,
}
