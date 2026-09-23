//! Amber's general paired-type adjustment, replacing molecule-specific overrides.
//! See `atomtype.c::atadjust` and `cpadjust`:
//! https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber/atomtype.c

use std::collections::VecDeque;

use super::{AtomEnvData, topology::BondKind};

pub(super) fn adjust_conjugated_types(env: &[AtomEnvData], types: &mut [String]) {
    fn partner(name: &str) -> Option<&'static str> {
        match name {
            "cc" => Some("cd"),
            "ce" => Some("cf"),
            "cg" => Some("ch"),
            "nc" => Some("nd"),
            "ne" => Some("nf"),
            "pc" => Some("pd"),
            "pe" => Some("pf"),
            "cp" => Some("cq"),
            _ => None,
        }
    }
    // A connected conjugated component has two equivalent label orientations.
    // Seed with its first atom, as Amber does. Single bonds preserve the label;
    // double/triple bonds flip it. cp/cq use a separate component graph.
    let mut signs = vec![None; types.len()];
    for start in 0..types.len() {
        if signs[start].is_some() || partner(&types[start]).is_none() {
            continue;
        }
        let cp = types[start] == "cp";
        signs[start] = Some(false);
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            for &(j, kind) in &env[i].bonds {
                if signs[j].is_some() || partner(&types[j]).is_none() || (types[j] == "cp") != cp {
                    continue;
                }
                let flip = if cp {
                    kind != BondKind::Single
                } else {
                    match kind {
                        BondKind::Single | BondKind::AromaticSingle => false,
                        BondKind::Double | BondKind::AromaticDouble | BondKind::Triple => true,
                        BondKind::Delocalized => continue,
                    }
                };
                signs[j] = Some(signs[i].unwrap() ^ flip);
                queue.push_back(j);
            }
        }
    }
    for (ty, sign) in types.iter_mut().zip(signs) {
        if sign == Some(true) {
            *ty = partner(ty).unwrap().to_owned();
        }
    }
}
