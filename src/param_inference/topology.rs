//! Molecular graph and Amber ring/bond properties, computed once per molecule.
//! The reference is AmberClassic `ring.c::{purify,aromatic}` and
//! `bondtype.c::finalize` (revision 8e55e97ada48b96eefaec2e6a3fa849018aaeea5).
//! Ring classes are Amber's rule-language properties, not a general aromaticity model.

use std::{collections::{HashMap, HashSet}, io};
use bio_files::{AtomGeneric, BondGeneric, BondType};
use na_seq::Element;
use super::AtomEnvData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BondKind { Single, Double, Triple, AromaticSingle, AromaticDouble, Delocalized }
impl BondKind {
    pub fn has_property(self, name: &str) -> bool {
        use BondKind::*;
        match name {
            "sb" => matches!(self, Single | AromaticSingle | Delocalized),
            "SB" => matches!(self, Single | Delocalized),
            "db" => matches!(self, Double | AromaticDouble),
            "DB" => self == Double,
            "tb" | "TB" => self == Triple,
            "AB" => matches!(self, AromaticSingle | AromaticDouble),
            "DL" => self == Delocalized,
            _ => false,
        }
    }
}

pub(super) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

pub(super) struct Topology {
    pub env: Vec<AtomEnvData>,
    pub adj: Vec<Vec<usize>>,
    // The charge model currently requires contiguous, one-based endpoints.
    pub normalized_bonds: Vec<BondGeneric>,
}

impl Topology {
    pub fn new(atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> io::Result<Self> {
        let mut indices = HashMap::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            if indices.insert(atom.serial_number, i).is_some() {
                return Err(invalid(format!("Duplicate atom serial number {}", atom.serial_number)));
            }
        }
        let mut adj = vec![Vec::new(); atoms.len()];
        let mut edges = Vec::with_capacity(bonds.len());
        let mut normalized_bonds = Vec::with_capacity(bonds.len());
        let mut seen = HashSet::new();
        let mut aromatic_edges = Vec::new();
        for bond in bonds {
            let (&a, &b) = indices.get(&bond.atom_0_sn).zip(indices.get(&bond.atom_1_sn))
                .ok_or_else(|| invalid("Bond references an absent atom serial number"))?;
            if a == b || !seen.insert((a.min(b), a.max(b))) { return Err(invalid("Self bond or duplicate bond")); }
            let kind = match bond.bond_type {
                BondType::Single | BondType::Amide => BondKind::Single,
                BondType::Double => BondKind::Double,
                BondType::Triple => BondKind::Triple,
                BondType::Delocalized => BondKind::Delocalized,
                BondType::Aromatic => { aromatic_edges.push(edges.len()); BondKind::AromaticSingle }
                _ => return Err(invalid(format!("Unsupported bond type: {}", bond.bond_type))),
            };
            adj[a].push(b);
            adj[b].push(a);
            edges.push((a, b, kind));
            normalized_bonds.push(BondGeneric { atom_0_sn: a as u32 + 1, atom_1_sn: b as u32 + 1, bond_type: bond.bond_type });
        }
        for neighbors in &mut adj { neighbors.sort_unstable(); }
        kekulize(atoms, &adj, &mut edges, &aromatic_edges)?;
        let rings = find_rings(atoms, &adj);
        let mut env: Vec<_> = adj.iter().map(|neighbors| AtomEnvData {
            degree: neighbors.len(),
            num_attached_h: neighbors.iter().filter(|&&i| atoms[i].element == Element::Hydrogen).count(),
            rings: [0; 11], aromatic: [0; 5], bonds: Vec::new(),
        }).collect();
        for ring in &rings { for &i in ring { env[i].rings[ring.len()] += 1; } }
        let planar: Vec<i32> = atoms.iter().enumerate().map(|(i, atom)| {
            use Element::*;
            match (atom.element, adj[i].len()) {
                (Carbon, 3) | (Nitrogen, 1..=3) | (Phosphorus, 2) => 2,
                (Oxygen | Sulfur, 2) | (Phosphorus, 3) => 1,
                (Carbon, 4) => -2,
                (Phosphorus, 4..) | (Sulfur, 3..) => -1,
                _ => 0,
            }
        }).collect();
        for ring in &rings {
            let sum: i32 = ring.iter().map(|&i| planar[i]).sum();
            let n = ring.len() as i32;
            let outside_double = edges.iter().any(|&(a, b, kind)| kind.has_property("db")
                && ((ring.contains(&a) && env[b].rings.iter().sum::<usize>() == 0)
                    || (ring.contains(&b) && env[a].rings.iter().sum::<usize>() == 0)));
            let class = if sum == -2 * n { 4 }
                else if ring.iter().any(|&i| planar[i] < 0) { 3 }
                else if (n..=2*n).contains(&sum) && outside_double { 2 }
                else if n == 6 && sum == 12 && ring.iter().all(|&i| {
                    !matches!(atoms[i].element, Element::Nitrogen | Element::Phosphorus)
                        || edges.iter().any(|&(a,b,k)| (a == i || b == i) && k.has_property("db"))
                }) { 0 }
                else if sum >= n + 3 { 1 }
                else { 3 };
            for &i in ring { env[i].aromatic[class] += 1; }
        }
        for &edge in &aromatic_edges {
            let (a,b,_) = edges[edge];
            if !rings.iter().any(|r| r.contains(&a) && r.contains(&b)
                && r.iter().all(|&i| env[i].aromatic[..3].iter().sum::<usize>() > 0)) {
                return Err(invalid("Aromatic bond is not in a supported planar ring"));
            }
        }
        // bondtype::finalize marks single/double bonds within planar five- and
        // six-membered rings as aromatic single/double. Keeping both categories
        // is essential: lowercase sb/db include them; uppercase SB/DB do not.
        for (a, b, kind) in &mut edges {
            if rings.iter().any(|r| (5..=6).contains(&r.len()) && r.contains(a) && r.contains(b)
                && r.iter().all(|&i| env[i].aromatic[0] + env[i].aromatic[1] > 0)) {
                *kind = match *kind { BondKind::Single => BondKind::AromaticSingle, BondKind::Double => BondKind::AromaticDouble, other => other };
            }
        }
        for (a, b, kind) in edges {
            env[a].bonds.push((b, kind)); env[b].bonds.push((a, kind));
        }
        for data in &mut env { data.bonds.sort_unstable_by_key(|&(i, _)| i); }
        Ok(Self { env, adj, normalized_bonds })
    }
}

/// Enumerate chordless rings of size 3..=10, as in current Amber ring.c.
/// Restrict each search to vertices >= its root and one direction to avoid the
/// old repeated traversals from every ring member. Prune chords during traversal.
fn find_rings(atoms: &[AtomGeneric], adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    use Element::*;
    let eligible: Vec<_> = atoms.iter().enumerate().map(|(i,a)| match a.element {
        Carbon => adj[i].len() > 2,
        Oxygen | Sulfur => adj[i].len() > 1,
        Nitrogen | Phosphorus => true,
        _ => false,
    }).collect();
    fn visit(path: &mut Vec<usize>, adj: &[Vec<usize>], eligible: &[bool], rings: &mut Vec<Vec<usize>>) {
        let start = path[0];
        let last = *path.last().unwrap();
        for &next in &adj[last] {
            if next <= start || !eligible[next] || path.contains(&next) { continue; }
            let closes = path.len() >= 2 && adj[next].contains(&start);
            if path[1..path.len()-1].iter().any(|v| adj[next].contains(v)) { continue; }
            if closes {
                if path[1] < next { let mut ring = path.clone(); ring.push(next); rings.push(ring); }
            } else if path.len() < 9 {
                path.push(next); visit(path, adj, eligible, rings); path.pop();
            }
        }
    }
    let mut rings = Vec::new();
    for start in 0..atoms.len() {
        if !eligible[start] { continue; }
        // Starting with an edge also avoids the empty interior slice at depth 1.
        for &next in &adj[start] {
            if next > start && eligible[next] { visit(&mut vec![start,next], adj, &eligible, &mut rings); }
        }
    }
    rings
}

/// Recover the single/double distinction lost in a MOL2 `ar` bond. This is a
/// constrained matching, not Amber's full valence/charge bond-order search.
/// Explicit Kekule bonds are preferable for charged or unusual aromatic systems:
/// AtomGeneric does not carry formal charges, so some cases are ambiguous.
fn kekulize(atoms: &[AtomGeneric], adj: &[Vec<usize>], edges: &mut [(usize,usize,BondKind)], aromatic: &[usize]) -> io::Result<()> {
    if aromatic.is_empty() { return Ok(()); }
    let mut required = vec![false; atoms.len()];
    for &edge in aromatic {
        let (a,b,_) = edges[edge];
        for i in [a,b] {
            required[i] = match atoms[i].element {
                Element::Carbon => true,
                Element::Nitrogen | Element::Phosphorus => adj[i].len() == 2,
                Element::Oxygen | Element::Sulfur => false,
                _ => return Err(invalid("Unsupported element in aromatic bond")),
            };
            if edges.iter().any(|&(x,y,k)| (x == i || y == i) && matches!(k, BondKind::Double | BondKind::Triple)) { required[i] = false; }
        }
    }
    fn solve(required: &mut [bool], edges: &[(usize,usize,BondKind)], aromatic: &[usize], chosen: &mut Vec<usize>) -> bool {
        let Some(i) = required.iter().position(|&r| r) else { return true; };
        required[i] = false;
        for &e in aromatic {
            let (a,b,_) = edges[e];
            let j = if a == i { b } else if b == i { a } else { continue; };
            if !required[j] { continue; }
            required[j] = false; chosen.push(e);
            if solve(required, edges, aromatic, chosen) { return true; }
            chosen.pop(); required[j] = true;
        }
        required[i] = true;
        false
    }
    let mut chosen = Vec::new();
    if !solve(&mut required, edges, aromatic, &mut chosen) {
        return Err(invalid("Cannot resolve aromatic bond orders; supply explicit single/double (Kekule) bonds"));
    }
    for e in chosen { edges[e].2 = BondKind::AromaticDouble; }
    Ok(())
}
