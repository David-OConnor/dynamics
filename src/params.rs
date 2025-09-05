//! For Amber and other parameters.

use std::collections::HashMap;

use bio_files::amber_params::{ChargeParams, ForceFieldParamsKeyed};
use na_seq::AminoAcidGeneral;

pub type ProtFfMap = HashMap<AminoAcidGeneral, Vec<ChargeParams>>;

/// Maps type-in-residue (found in, e.g. mmCIF and PDB files) to Amber FF type, and partial charge.
/// We assume that if one of these is loaded, so are the others. So, these aren't `Options`s, but
/// the field that holds this struct should be one.
pub struct ProtFFTypeChargeMap {
    pub internal: ProtFfMap,
    pub n_terminus: ProtFfMap,
    pub c_terminus: ProtFfMap,
}

#[derive(Default)]
/// Force field parameters (e.g. Amber) for molecular dynamics.
pub struct FfParamSet {
    /// E.g. parsed from Amber `gaff2.dat`.
    pub lig_general: Option<ForceFieldParamsKeyed>,
    /// E.g. ff19SB. Loaded at init.
    pub prot_general: Option<ForceFieldParamsKeyed>,
    /// In addition to charge, this also contains the mapping of res type to FF type; required to map
    /// other parameters to protein atoms. From `amino19.lib`, and its N and C-terminus variants.
    pub prot_ff_q_map: Option<ProtFFTypeChargeMap>,
    /// Key: A unique identifier for the molecule. (e.g. ligand)
    pub lig_specific: HashMap<String, ForceFieldParamsKeyed>,
}
