use std::{collections::HashMap, io, io::ErrorKind};

const PARMCHK: &str = include_str!("../../param_data/antechamber_defs/PARMCHK.DAT");
const ATCOR: &str = include_str!("../../param_data/antechamber_defs/ATCOR.DAT");

pub(in crate::param_inference) type AtCor = HashMap<String, (u8, Vec<String>)>;

/// From `PARM` and `EQUA` lines in `PARMCHK.DAT`.
/// Example:
/// PARM 	C	1		0	12.01	0	6
/// EQUA 	c
#[derive(Debug)]
pub(in crate::param_inference) struct Parm {
    /// PARM type, e.g. "C"
    pub atom_type: String,
    /// 0, 1, or 2 I believe
    pub improper_flag: u8,
    pub group_id: u8,
    pub mass: f32,
    pub equivalent_flag: bool,
    pub atomic_num: u8,
    /// GAFF type, e.g. "c3"
    pub equa: String,
}

impl Parm {
    /// Parse from a pair of lines.
    pub fn parse(parm: &str, equa: &str) -> io::Result<Self> {
        let cols_parm: Vec<&str> = parm.split_whitespace().collect();
        if cols_parm.len() < 7 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Invalid PARMCHK PARM data; not enough columns: {cols_parm:?}"),
            ));
        }

        let group_id = cols_parm[3]
            .parse()
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
        let atomic_num = cols_parm[6]
            .parse()
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        let cols_equa: Vec<&str> = equa.split_whitespace().collect();
        if cols_equa.len() < 2 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid PARMCHK EQUA data; not enough columns.",
            ));
        }

        let improper_flag = cols_parm[2]
            .parse()
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;

        Ok(Self {
            atom_type: cols_parm[1].to_owned(),
            improper_flag,
            group_id,
            mass: parse_float(cols_parm[4])?,
            equivalent_flag: cols_parm[5] == "1",
            atomic_num,
            equa: cols_equa[1].to_owned(),
        })
    }
}

/// From `CORR` lines in `PARMCHK.DAT`. Note that `ATCOR.DAT` also prefixes lines with `CORR`,
/// but these have a different format.
/// Example:
/// CORR    c2   4.9  21.9   4.3   2.9   2.9   2.3  96.7  -1.0  36.1
#[derive(Debug)]
pub(in crate::param_inference) struct Corr {
    pub base: String,
    pub other: String,
    pub vals: [f32; 9],
}

impl Corr {
    pub fn parse(line: &str, base: &str) -> io::Result<Self> {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 11 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Invalid PARMCHK COR data; not enough columns: {cols:?}"),
            ));
        }

        Ok(Self {
            base: base.to_owned(),
            other: cols[1].to_owned(),
            vals: [
                parse_float(cols[2])?,
                parse_float(cols[3])?,
                parse_float(cols[4])?,
                parse_float(cols[5])?,
                parse_float(cols[6])?,
                parse_float(cols[7])?,
                parse_float(cols[8])?,
                parse_float(cols[9])?,
                parse_float(cols[10])?,
            ],
        })
    }
}

/// All data from PARMCHK.DAT. This is used to map force field types from arbitrary small molecules
/// to those in GAFF2.dat. (Mabye broader as well?) We use this to determine the closest GAFF2 dihedrals (etc)
/// to mape to each missing one.
#[derive(Debug)]
pub(in crate::param_inference) struct ParmChk {
    pub parms: Vec<Parm>,
    pub corr: Vec<Corr>,
    pub corr_map: HashMap<(String, String), [f32; 9]>,
    /// PARM type -> canonical type for CORR
    pub aliases: HashMap<String, String>,
    /// GAFF type -> PARM type
    pub equa_to_parm: HashMap<String, String>,
}

impl ParmChk {
    pub fn new() -> io::Result<Self> {
        let mut parms = Vec::new();
        let mut corr = Vec::new();
        let mut corr_map = HashMap::new();
        let mut aliases = HashMap::new();
        let mut equa_to_parm = HashMap::new();

        let lines: Vec<&str> = PARMCHK.lines().collect();
        let mut current_parm: Option<String> = None;

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            if line.starts_with("PARM") {
                let cols_parm: Vec<&str> = line.split_whitespace().collect();
                if cols_parm.len() >= 2 {
                    current_parm = Some(cols_parm[1].to_owned());
                }

                if i + 1 < lines.len() && lines[i + 1].trim_start().starts_with("EQUA") {
                    let equa_line = lines[i + 1];
                    let cols_equa: Vec<&str> = equa_line.split_whitespace().collect();
                    let base = cols_parm[1];

                    // Parse Parm as before (we still want the metadata)
                    let p = Parm::parse(lines[i], equa_line)?;
                    parms.push(p);

                    // GAFF type mapping (optional; useful if you ever need it)
                    if cols_equa.len() >= 2 {
                        equa_to_parm.insert(cols_equa[1].to_owned(), base.to_owned());
                    }

                    if cols_equa.len() == 2 {
                        // EQUA <type> : pure alias, no numbers
                        // Example: PARM OS ... / EQUA os
                        aliases.insert(base.to_owned(), cols_equa[1].to_owned());
                    } else if cols_equa.len() >= 11 {
                        // EQUA <type> followed by 9 floats -> treat like CORR(base, other)
                        let other = cols_equa[1];
                        let vals = [
                            parse_float(cols_equa[2])?,
                            parse_float(cols_equa[3])?,
                            parse_float(cols_equa[4])?,
                            parse_float(cols_equa[5])?,
                            parse_float(cols_equa[6])?,
                            parse_float(cols_equa[7])?,
                            parse_float(cols_equa[8])?,
                            parse_float(cols_equa[9])?,
                            parse_float(cols_equa[10])?,
                        ];
                        corr_map.insert((base.to_owned(), other.to_owned()), vals);
                        corr_map.insert((other.to_owned(), base.to_owned()), vals);
                    }

                    i += 2;
                    continue;
                } else {
                    i += 1;
                    continue;
                }
            } else if line.starts_with("CORR") {
                let cols: Vec<&str> = line.split_whitespace().collect();

                if let Some(base) = &current_parm {
                    if cols.len() <= 2 {
                        // short "CORR <type>" alias form
                        let target = cols.get(1).ok_or_else(|| {
                            io::Error::new(
                                ErrorKind::InvalidData,
                                format!("Invalid short PARMCHK CORR line: {cols:?}"),
                            )
                        })?;
                        aliases.insert(base.clone(), (*target).to_string());
                        i += 1;
                        continue;
                    }

                    // full numeric CORR line
                    let c = Corr::parse(lines[i], base)?;
                    corr_map.insert((c.base.clone(), c.other.clone()), c.vals);
                    corr_map.insert((c.other.clone(), c.base.clone()), c.vals);
                    corr.push(c);
                }

                i += 1;
                continue;
            }

            i += 1;
        }

        Ok(Self {
            parms,
            corr,
            corr_map,
            aliases,
            equa_to_parm,
        })
    }

    pub fn types_equivalent(&self, a: &str, b: &str) -> bool {
        self.canonical(a) == self.canonical(b)
    }

    fn parm_type_for<'a>(&'a self, gaff: &'a str) -> &'a str {
        self.equa_to_parm
            .get(gaff)
            .map(|s| s.as_str())
            .unwrap_or(gaff)
    }

    fn canonical(&self, t: &str) -> String {
        // Start from the PARM atom_type that corresponds to this GAFF type.
        let mut cur = self.parm_type_for(t);
        let mut depth = 0;

        // Follow alias chain like OS -> os, or via short CORR aliases.
        loop {
            if let Some(next) = self.aliases.get(cur) {
                cur = next;
                depth += 1;
                if depth > 10 {
                    break;
                }
            } else {
                break;
            }
        }

        cur.to_string()
    }

    fn corr_value(&self, a: &str, b: &str, idx: usize) -> f32 {
        let ca = self.canonical(a);
        let cb = self.canonical(b);

        // If they reduce to the same canonical PARM type, treat as identical.
        if ca == cb {
            return 0.0;
        }

        let val = self
            .corr_map
            .get(&(ca.clone(), cb.clone()))
            .or_else(|| self.corr_map.get(&(cb, ca))) // just in case
            .map(|v| v[idx])
            .unwrap_or(1.0e6);

        if val < 0.0 {
            // PARMCHK uses -1.0 for “no data”.
            1.0e6
        } else {
            val
        }
    }

    pub(in crate::param_inference) fn bond_penalty(&self, from: &str, to: &str) -> f32 {
        // heuristic: use column 0 as "bond similarity"
        self.corr_value(from, to, 0)
    }

    pub(in crate::param_inference) fn angle_center_penalty(&self, from: &str, to: &str) -> f32 {
        // heuristic: use column 2 for angle-center similarity
        self.corr_value(from, to, 2)
    }

    pub(in crate::param_inference) fn angle_outer_penalty(&self, from: &str, to: &str) -> f32 {
        // heuristic: use column 3 for angle-outer similarity
        self.corr_value(from, to, 3)
    }

    // index choices are a best guess; tweak if needed after inspecting PARMCHK.DAT docs
    pub(in crate::param_inference) fn dihe_inner_penalty(&self, from: &str, to: &str) -> f32 {
        self.corr_value(from, to, 4)
    }

    pub(in crate::param_inference) fn dihe_outer_penalty(&self, from: &str, to: &str) -> f32 {
        self.corr_value(from, to, 5)
    }

    pub(in crate::param_inference) fn improper_penalty(&self, from: &str, to: &str) -> f32 {
        self.corr_value(from, to, 6)
    }
}

/// Load data from Amber's ATCOR.dat, included in the application binary.
/// todo: Adjust this A/R once you understand it better and attempt to use it.
pub(in crate::param_inference) fn load_atcor() -> io::Result<AtCor> {
    let mut result = HashMap::new();

    for line in ATCOR.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();

        if cols.len() < 4 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid Atcor data; not enough columns.",
            ));
        }

        let num: u8 = cols[2].parse().unwrap();
        let mut remaining = Vec::new();

        for col in &cols[3..] {
            remaining.push(col.to_string());
        }

        result.insert(cols[1].to_owned(), (num, remaining));
    }

    Ok(result)
}

/// Helper
fn parse_float(v: &str) -> io::Result<f32> {
    v.parse()
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))
}
