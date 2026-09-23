//! Directed atom-type substitutions and scores from Amber PARMCHK.DAT.
//! See `parmchk2.c::read_parmchk_parm`, `chk_bond`, `chk_angle`, and
//! `chk_torsion` at AmberClassic revision 8e55e97ada48b96eefaec2e6a3fa849018aaeea5.
//! EQUA is a list of allowed replacements for one PARM, not a global alias or
//! equivalence relation. In particular, n7 -> n3 does not imply n3 -> n7.

use std::{collections::HashMap, io, sync::LazyLock};
const PARMCHK: &str = include_str!("../../param_data/antechamber_defs/PARMCHK.DAT");
pub(super) static PARAMETERS: LazyLock<ParmChk> =
    LazyLock::new(|| ParmChk::parse(PARMCHK).expect("bundled PARMCHK.DAT"));

#[derive(Debug)]
pub(super) struct Parm {
    pub improper: bool,
    pub group: i32,
    pub equivalent_type: i32,
    pub replacements: HashMap<String, Replacement>,
}
#[derive(Debug)]
pub(super) struct Replacement {
    pub equivalent: bool,
    pub order: usize,
    /// bl, blf, central ba, central baf, outer ba, outer baf, central torsion,
    /// outer torsion, general similarity (also used for improper torsions).
    values: [f32; 9],
}
#[derive(Clone, Copy)]
pub(super) enum Position {
    Bond,
    AngleCenter,
    AngleOuter,
    TorsionCenter,
    TorsionOuter,
    ImproperCenter,
    ImproperOuter,
}

pub(super) struct ParmChk {
    pub parms: HashMap<String, Parm>,
    settings: HashMap<String, f32>,
}
impl ParmChk {
    pub fn parse(text: &str) -> io::Result<Self> {
        let mut result = Self {
            parms: HashMap::new(),
            settings: HashMap::new(),
        };
        let mut current = None;
        for line in text.lines() {
            let cols: Vec<_> = line.split_whitespace().collect();
            let Some(&tag) = cols.first() else {
                continue;
            };
            let invalid = || {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid PARMCHK record: {line}"),
                )
            };
            match tag {
                "PARM" => {
                    if cols.len() < 7 {
                        return Err(invalid());
                    }
                    let improper: u8 = cols[2].parse().map_err(|_| invalid())?;
                    result.parms.insert(
                        cols[1].into(),
                        Parm {
                            improper: improper == 1,
                            group: cols[3].parse().map_err(|_| invalid())?,
                            equivalent_type: cols[5].parse().map_err(|_| invalid())?,
                            replacements: HashMap::new(),
                        },
                    );
                    current = Some(cols[1].to_owned());
                }
                "EQUA" | "CORR" => {
                    if cols.len() < 2 {
                        return Err(invalid());
                    }
                    let parm = result
                        .parms
                        .get_mut(current.as_deref().ok_or_else(invalid)?)
                        .ok_or_else(invalid)?;
                    let mut values = [0.; 9];
                    if tag == "CORR" {
                        // Amber initializes unspecified CORR scores to zero.
                        for (i, token) in cols.iter().skip(2).take(9).enumerate() {
                            values[i] = token.parse().map_err(|_| invalid())?;
                        }
                    }
                    let order = parm.replacements.len() + 1;
                    parm.replacements.insert(
                        cols[1].into(),
                        Replacement {
                            equivalent: tag == "EQUA",
                            order,
                            values,
                        },
                    );
                }
                tag if tag.starts_with("WEIGHT_")
                    || tag.starts_with("DEFAULT_")
                    || tag == "THRESHOLD_BA" =>
                {
                    let value = cols
                        .get(1)
                        .ok_or_else(invalid)?
                        .parse()
                        .map_err(|_| invalid())?;
                    result.settings.insert(tag.into(), value);
                }
                _ => {}
            }
        }
        for name in [
            "WEIGHT_BL",
            "WEIGHT_BLF",
            "WEIGHT_BA",
            "WEIGHT_BAF",
            "WEIGHT_X",
            "WEIGHT_X3",
            "WEIGHT_BA_CTR",
            "WEIGHT_TOR_CTR",
            "WEIGHT_IMPROPER",
            "WEIGHT_GROUP",
            "WEIGHT_EQUTYPE",
            "DEFAULT_BL",
            "DEFAULT_BLF",
            "DEFAULT_BA_CTR",
            "DEFAULT_BAF_CTR",
            "DEFAULT_BA",
            "DEFAULT_BAF",
            "DEFAULT_TOR_CTR",
            "DEFAULT_TOR",
            "DEFAULT_FRACT1",
            "DEFAULT_FRACT2",
        ] {
            if !result.settings.contains_key(name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Missing PARMCHK setting {name}"),
                ));
            }
        }
        Ok(result)
    }
    pub fn weight(&self, key: &str) -> f32 {
        self.settings[key]
    }
    pub fn improper(&self, name: &str) -> bool {
        self.parms.get(name).is_some_and(|p| p.improper)
    }

    pub fn replacement_order(&self, from: &str, to: &str) -> usize {
        if from == to || to == "X" {
            return 0;
        }
        self.parms
            .get(from)
            .and_then(|p| p.replacements.get(to))
            .map_or(usize::MAX, |r| r.order)
    }

    /// Return whether this is an EQUA replacement and its position-specific score.
    /// Missing relationships are disallowed, not assigned an arbitrary large score.
    pub fn penalty(&self, from: &str, to: &str, position: Position) -> Option<(bool, f32)> {
        if from == to {
            return Some((true, 0.));
        }
        let replacement = self.parms.get(from)?.replacements.get(to)?;
        if replacement.equivalent {
            return Some((true, 0.));
        }
        let defaults = [
            "DEFAULT_BL",
            "DEFAULT_BLF",
            "DEFAULT_BA_CTR",
            "DEFAULT_BAF_CTR",
            "DEFAULT_BA",
            "DEFAULT_BAF",
            "DEFAULT_TOR_CTR",
            "DEFAULT_TOR",
        ];
        let mut v = replacement.values;
        for (i, key) in defaults.iter().enumerate() {
            if v[i] < 0. {
                v[i] = self.settings[*key];
            }
        }
        // Amber blends the central torsion score with overall similarity for
        // all CORR records, after replacing missing (-1) values with defaults.
        v[6] = v[6] * self.weight("DEFAULT_FRACT1") + v[8] * self.weight("DEFAULT_FRACT2");
        use Position::*;
        let score = match position {
            Bond => v[0] * self.weight("WEIGHT_BL") + v[1] * self.weight("WEIGHT_BLF"),
            AngleCenter => {
                (v[2] * self.weight("WEIGHT_BA") + v[3] * self.weight("WEIGHT_BAF"))
                    * self.weight("WEIGHT_BA_CTR")
            }
            AngleOuter => v[4] * self.weight("WEIGHT_BA") + v[5] * self.weight("WEIGHT_BAF"),
            TorsionCenter => v[6] * self.weight("WEIGHT_TOR_CTR"),
            TorsionOuter => v[7],
            ImproperCenter => v[8] * self.weight("WEIGHT_IMPROPER"),
            ImproperOuter => v[8],
        };
        Some((false, score))
    }
    pub fn group_penalty(&self, types: &[&str]) -> f32 {
        let groups: Vec<_> = types
            .iter()
            .filter(|&&t| t != "X")
            .filter_map(|t| self.parms.get(*t).map(|p| p.group))
            .collect();
        if groups.windows(2).any(|pair| pair[0] != pair[1]) {
            self.weight("WEIGHT_GROUP")
        } else {
            0.
        }
    }
    pub fn pair_penalty(&self, from: [&str; 2], to: [&str; 2]) -> f32 {
        let flag = |ty| self.parms.get(ty).map_or(0, |p| p.equivalent_type);
        let (a, b, c, d) = (flag(from[0]), flag(from[1]), flag(to[0]), flag(to[1]));
        if a == 0 && b == 0 {
            return 0.;
        }
        if (a.abs() + b.abs() == 3) != (c.abs() + d.abs() == 3) {
            return self.weight("WEIGHT_EQUTYPE");
        }
        if a + b == 0 && c < 0 && d < 0 {
            return 0.5 * self.weight("WEIGHT_EQUTYPE");
        }
        0.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn directed_equivalences_and_all_parm_rows_are_retained() {
        let p = &*PARAMETERS;
        assert_eq!(p.penalty("n7", "n3", Position::Bond), Some((true, 0.)));
        assert_eq!(p.penalty("n3", "n7", Position::Bond), None);
        assert_eq!(p.penalty("nk", "ny", Position::Bond), Some((true, 0.)));
        assert!(p.parms.contains_key("c3")); // PARM with no following EQUA
        assert_eq!(p.parms["cc"].equivalent_type, -1);
        assert_eq!(p.parms["cd"].equivalent_type, -2);
        assert!(!p.improper("c6")); // saturated cyclohexane carbon
    }
    #[test]
    fn amber_penalty_columns_and_defaults() {
        let p = &*PARAMETERS;
        let v = p.parms["c2"].replacements["ca"].values;
        assert_eq!(
            p.penalty("c2", "ca", Position::Bond),
            Some((false, (v[0] + v[1]) * 0.5))
        );
        assert_eq!(
            p.penalty("c2", "ca", Position::AngleCenter),
            Some((false, (v[2] + v[3]) * 5.))
        );
        assert_eq!(
            p.penalty("c2", "ca", Position::AngleOuter),
            Some((false, (v[4] + v[5]) * 0.5))
        );
        assert_eq!(
            p.penalty("c2", "ca", Position::ImproperCenter),
            Some((false, v[8] * 10.))
        );
        assert_eq!(
            p.penalty("c2", "ca", Position::TorsionOuter),
            Some((false, 87.))
        );
    }
}
