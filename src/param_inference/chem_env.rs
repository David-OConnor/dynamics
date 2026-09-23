//! The GAFF DEF grammar, following `atomtype.c::apcheck` and `cematch`.
//! https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber/atomtype.c
//!
//! Commas mean AND, dots mean OR, a numeric prefix is an exact count (including
//! zero), and a prime constrains the bond to the preceding atom. `C3` means a
//! carbon with three neighbors, not three carbon neighbors or an sp3 carbon.

use bio_files::{AtomGeneric, amber_typedef::WildAtom};
use super::{AtomEnvData, topology::BondKind};

#[derive(Debug, Clone)]
pub(super) struct Properties(Vec<Vec<Property>>);
#[derive(Debug, Clone)]
struct Property { name: String, count: Option<usize>, primes: usize }

impl Properties {
    pub fn parse(text: &str) -> Option<Self> {
        if matches!(text, "*" | "&" | "") { return Some(Self(Vec::new())); }
        let inner = text.strip_prefix('[')?.strip_suffix(']')?;
        let mut groups = Vec::new();
        for group in inner.split(',') {
            let mut alternatives = Vec::new();
            for token in group.split('.') {
                let digits = token.bytes().take_while(u8::is_ascii_digit).count();
                let count = if digits == 0 { None } else { Some(token[..digits].parse().ok()?) };
                let rest = &token[digits..];
                let name = rest.trim_end_matches('\'');
                let primes = rest.len() - name.len();
                if primes > 2 || !matches!(name, "RG" | "NR" | "RG3" | "RG4" | "RG5" | "RG6" | "RG7" | "RG8" | "RG9" | "RG10" | "AR1" | "AR2" | "AR3" | "AR4" | "AR5" | "sb" | "SB" | "db" | "DB" | "tb" | "TB" | "AB" | "DL") { return None; }
                alternatives.push(Property { name: name.into(), count, primes });
            }
            groups.push(alternatives);
        }
        Some(Self(groups))
    }

    pub fn matches(&self, atom: usize, parent: Option<usize>, env: &[AtomEnvData]) -> bool {
        self.0.iter().all(|group| group.iter().any(|prop| {
            let data = &env[atom];
            let count = match prop.name.as_str() {
                "RG" => data.rings.iter().sum(),
                "NR" => usize::from(data.rings.iter().all(|&n| n == 0)),
                name if name.starts_with("RG") => data.rings[name[2..].parse::<usize>().unwrap()],
                name if name.starts_with("AR") => data.aromatic[name[2..].parse::<usize>().unwrap() - 1],
                name => data.bonds.iter().filter(|(_, kind)| kind.has_property(name)).count(),
            };
            if !prop.count.map_or(count > 0, |required| count == required) { return false; }
            if prop.primes == 0 { return true; }
            let Some(parent) = parent else { return false; };
            let has_bond = data.bonds.iter().any(|&(other, kind)| other == parent && kind.has_property(&prop.name));
            // jbond permits DL in a primed db/DB test; bondinfo does not count it as DB.
            let has_bond = has_bond || (matches!(prop.name.as_str(), "db" | "DB")
                && data.bonds.contains(&(parent, BondKind::Delocalized)));
            has_bond == (prop.primes == 1)
        }))
    }
}

#[derive(Debug, Clone)]
struct Neighbor { element: String, degree: Option<usize>, properties: Properties, children: Vec<Neighbor> }
#[derive(Debug, Clone)]
pub(super) struct ChemEnvPattern(Vec<Neighbor>);

impl ChemEnvPattern {
    pub fn parse(text: &str) -> Option<Self> {
        if matches!(text, "*" | "&" | "") { return Some(Self(Vec::new())); }
        let mut parser = Parser { text: text.as_bytes(), pos: 0 };
        let result = parser.group()?;
        (parser.pos == parser.text.len()).then_some(Self(result))
    }
    pub fn matches(&self, idx: usize, atoms: &[AtomGeneric], env: &[AtomEnvData], wild: &[WildAtom]) -> bool {
        // Backtrack over the entire tree, retaining occupied descendant vertices
        // while matching siblings. Greedy matching can reject a valid assignment.
        let mut used = vec![false; atoms.len()];
        used[idx] = true;
        let pending: Vec<_> = self.0.iter().map(|pattern| (idx, pattern)).collect();
        match_pending(&pending, &mut used, atoms, env, wild)
    }
}

fn element_matches(name: &str, idx: usize, atoms: &[AtomGeneric], env: &[AtomEnvData], wild: &[WildAtom]) -> bool {
    if name == "EW" { return super::is_elec_withdrawing_element(atoms[idx].element); }
    if let Some(group) = wild.iter().find(|group| group.name == name) {
        return group.elements.iter().any(|member| {
            let split = member.find(|c: char| c.is_ascii_digit()).unwrap_or(member.len());
            member[..split] == atoms[idx].element.to_letter()
                && (split == member.len() || member[split..].parse::<usize>().ok() == Some(env[idx].degree))
        });
    }
    name == atoms[idx].element.to_letter()
}

fn match_pending(pending: &[(usize, &Neighbor)], used: &mut [bool], atoms: &[AtomGeneric], env: &[AtomEnvData], wild: &[WildAtom]) -> bool {
    let Some((&(parent, pattern), rest)) = pending.split_first() else { return true; };
    for &(idx, _) in &env[parent].bonds {
        if used[idx] || pattern.degree.is_some_and(|degree| degree != env[idx].degree)
            || !element_matches(&pattern.element, idx, atoms, env, wild)
            || !pattern.properties.matches(idx, Some(parent), env) { continue; }
        used[idx] = true;
        let mut next: Vec<_> = pattern.children.iter().map(|child| (idx, child)).collect();
        next.extend_from_slice(rest);
        if match_pending(&next, used, atoms, env, wild) { return true; }
        used[idx] = false;
    }
    false
}

struct Parser<'a> { text: &'a [u8], pos: usize }
impl Parser<'_> {
    fn take(&mut self, ch: u8) -> bool {
        if self.text.get(self.pos) == Some(&ch) { self.pos += 1; true } else { false }
    }
    fn group(&mut self) -> Option<Vec<Neighbor>> {
        if !self.take(b'(') { return None; }
        let mut result = Vec::new();
        loop {
            let start = self.pos;
            while self.text.get(self.pos).is_some_and(u8::is_ascii_alphabetic) { self.pos += 1; }
            if start == self.pos { return None; }
            let element = std::str::from_utf8(&self.text[start..self.pos]).ok()?.to_owned();
            let start = self.pos;
            while self.text.get(self.pos).is_some_and(u8::is_ascii_digit) { self.pos += 1; }
            let degree = if start == self.pos { None } else { Some(std::str::from_utf8(&self.text[start..self.pos]).ok()?.parse().ok()?) };
            let properties = if self.text.get(self.pos) == Some(&b'[') {
                let start = self.pos;
                while self.text.get(self.pos).is_some_and(|&ch| ch != b']') { self.pos += 1; }
                if !self.take(b']') { return None; }
                Properties::parse(std::str::from_utf8(&self.text[start..self.pos]).ok()?)?
            } else { Properties(Vec::new()) };
            let children = if self.text.get(self.pos) == Some(&b'(') { self.group()? } else { Vec::new() };
            result.push(Neighbor { element, degree, properties, children });
            if self.take(b')') { return Some(result); }
            if !self.take(b',') { return None; }
        }
    }
}
