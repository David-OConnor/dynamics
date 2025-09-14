//! Logic for pH-related adjustment to the DigitMap.
//!
//! How this affects protenation based on pH:
//! At low pH:
//! - HIP provides both HD* and HE* hydrogens on His.
//! - ASH, GLH hydrogens become valid; standard ASP/GLU are excluded.
//! - CYS stays protonated; CYM excluded.
//! - LYS stays protonated; LYN excluded.
//!
//! At neutral pH (~7):
//! - His becomes neutral (we pick HIE by default here). Only HE* hydrogens will be “valid”; HD* are not,
//! unless we later swap to HID by an environment rule.
//! - ASP/GLU are deprotonated; ASH/GLH excluded.
//! - CYS is protonated; CYM excluded.
//! - LYS remains protonated; LYN excluded.
//!
//! At high pH:
//! - LYN appears (no LYS hydrogens).
//! - CYM may appear if pH is sufficiently high.
//! - His neutral (HIE) unless you push very low pH again.

// todo: pick HID vs HIE by local geometry:
// Compute ND1/NE2 distances to nearby acceptors (O, carboxylate O, backbone carbonyl O, etc.) and
// choose the proton on the ring nitrogen farther from a strong acceptor (so the closer one can
// accept H-bond). If it picks ND1 → use HID; if NE2 → HIE. Then pass that choice into
// make_h_digit_map (e.g. extra arg) or cache a per-run his_selected before you call it.
// That’s still a localized change.

use na_seq::AminoAcidProtenationVariant;

// Intrinsic pKas (typical, solvent-exposed). These are deliberately simple.
// todo: Make them better?
// todo: Add Arg, Tyr and Termini on AminoAcidGeneric? Are they in the Amber data?
const PKA_ASP: f32 = 3.9;
const PKA_GLU: f32 = 4.2;
const PKA_HIS: f32 = 6.0;
const PKA_CYS: f32 = 8.3;
const PKA_LYS: f32 = 10.5;

const WINDOW: f32 = 0.8; // buffer to avoid edge flipping

pub(crate) fn his_choice(ph: f32) -> Option<AminoAcidProtenationVariant> {
    // HIP if clearly below pKa, neutral otherwise (choose HIE by default).
    if ph <= PKA_HIS - WINDOW {
        Some(AminoAcidProtenationVariant::Hip)
    } else if ph >= PKA_HIS + WINDOW {
        Some(AminoAcidProtenationVariant::Hie) // neutral tautomer default
    } else {
        // ambiguous band -> prefer neutral HIE (keeps one consistent choice)
        Some(AminoAcidProtenationVariant::Hie)
    }
}

pub(crate) fn variant_allowed_at_ph(aa_var: AminoAcidProtenationVariant, ph: f32) -> bool {
    match aa_var {
        // Histidine: HIP at low pH; HID/HIE at high pH. We pick a single tautomer upstream.
        AminoAcidProtenationVariant::Hip => ph <= PKA_HIS - WINDOW,
        AminoAcidProtenationVariant::Hid | AminoAcidProtenationVariant::Hie => {
            ph >= PKA_HIS - WINDOW
        } // allow neutral above/borderline

        // Acids: protonated variants (ASH/GLH) only well below their pKa
        AminoAcidProtenationVariant::Ash => ph <= PKA_ASP - WINDOW,
        AminoAcidProtenationVariant::Glh => ph <= PKA_GLU - WINDOW,

        // Cys: thiolate (CYM) only well above pKa
        AminoAcidProtenationVariant::Cym => ph >= PKA_CYS + WINDOW,

        // Disulfide cystine (CYX) is not pH-driven; don’t include here via pH.
        AminoAcidProtenationVariant::Cyx => false,

        // Lys: neutral LYN only well above pKa
        AminoAcidProtenationVariant::Lyn => ph >= PKA_LYS + WINDOW,

        // Capping or special forms (not pH-driven)
        AminoAcidProtenationVariant::Ace
        | AminoAcidProtenationVariant::Nhe
        | AminoAcidProtenationVariant::Nme
        | AminoAcidProtenationVariant::Hyp => false,
    }
}

pub(crate) fn standard_allowed_at_ph(aa: na_seq::AminoAcid, ph: f32) -> bool {
    // Only restrict standards when you actually have alt protonation variants in Amber.
    match aa {
        // For acids, Amber "ASP"/"GLU" are deprotonated; allow them above pKa
        na_seq::AminoAcid::Asp => ph >= PKA_ASP + WINDOW,
        na_seq::AminoAcid::Glu => ph >= PKA_GLU + WINDOW,

        // For Lys, Amber "LYS" is protonated; allow it below pKa
        na_seq::AminoAcid::Lys => ph <= PKA_LYS - WINDOW,

        // For Cys, Amber "CYS" is protonated; allow it below/around pKa
        na_seq::AminoAcid::Cys => ph <= PKA_CYS + WINDOW,

        // Histidine: you typically don’t use "HIS" in Amber; variants are used instead.
        na_seq::AminoAcid::His => false,

        // Others unaffected
        _ => true,
    }
}
