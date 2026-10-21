//! Export explicitly parameterized molecules using the same parameters as native
//! dynamics. This preserves OPLS residue torsions and CHARMM's additional terms.

use std::{
    collections::{BTreeSet, HashMap},
    fmt::Write,
};

use bio_files::gromacs::forcefield::{ForceField, TopologyConfig, WaterModel as GmxWater};

use crate::{
    FfMolType, ForceFieldFamily, MolDynamics, ParamError, params::FfParamSet, solvent::WaterModel,
};

/// Water/ion values and conventions shared with native dynamics, converted to
/// the nm and kJ/mol units expected by Bio Files' GROMACS writer.
pub fn topology_config(set: &FfParamSet, water: Option<&WaterModel>) -> TopologyConfig {
    let w = water.unwrap_or(&set.default_water);
    TopologyConfig {
        force_field: match set.family {
            ForceFieldFamily::Amber => ForceField::Amber,
            ForceFieldFamily::Charmm36 => ForceField::Charmm36,
            ForceFieldFamily::Opls => ForceField::Opls,
        },
        water_model: Some(GmxWater {
            mass_o: w.mass_o as f64,
            mass_h: w.mass_h as f64,
            oh: w.o_h_dist as f64 / 10.,
            om: w.o_m_dist as f64 / 10.,
            angle: w.h_o_h_angle as f64,
            q_h: w.q_h as f64,
            sigma_o: w.lj_sigma_o as f64 / 10.,
            epsilon_o: w.lj_eps_o as f64 * 4.184,
            sigma_h: w.lj_sigma_h as f64 / 10.,
            epsilon_h: w.lj_eps_h as f64 * 4.184,
            sodium: (
                w.cation.mass as f64,
                w.cation.lj_sigma as f64 / 10.,
                w.cation.lj_eps as f64 * 4.184,
            ),
            chloride: (
                w.anion.mass as f64,
                w.anion.lj_sigma as f64 / 10.,
                w.anion.lj_eps as f64 * 4.184,
            ),
        }),
    }
}

/// Resolve protein and supported OPLS ligand parameters on a local copy. Supplied
/// parameters take precedence, as in `MdState::new`.
pub fn parameterize(mol: &MolDynamics, set: &FfParamSet) -> Result<MolDynamics, ParamError> {
    let mut result = mol.clone();
    if result.explicit_params.is_none() && set.family != ForceFieldFamily::Amber {
        let params = match result.ff_mol_type {
            FfMolType::SmallOrganic if set.opls.is_some() => set
                .opls
                .as_ref()
                .unwrap()
                .build_small_molecule(&mut result.atoms, &result.bonds)?,
            FfMolType::Peptide => {
                let residues = result.residues.as_deref().ok_or_else(|| {
                    ParamError::new("Protein force field selection requires MolDynamics::residues")
                })?;
                if let Some(opls) = &set.opls {
                    opls.build_peptide(&mut result.atoms, &result.bonds, residues)?
                } else if let Some(charmm) = &set.charmm {
                    crate::charmm::build_peptide(
                        &mut result.atoms,
                        &result.bonds,
                        residues,
                        charmm,
                    )?
                } else {
                    return Err(ParamError::new("Missing family parameters"));
                }
            }
            _ => {
                return Err(ParamError::new(
                    "This force field requires explicit parameters for this molecule class; import a matching topology",
                ));
            }
        };
        result.explicit_params = Some(params);
    }
    if let Some(p) = &result.explicit_params {
        if p.lj_combining != set.nonbonded_rules.lj_combining {
            return Err(ParamError::new(
                "Explicit molecule and selected force field use different LJ combining rules",
            ));
        }
    }
    Ok(result)
}

/// Self-contained topology for already parameterized molecules. `molecules`
/// contains (name, molecule, copy count); atom ordering matches the input.
pub fn make_topology(
    molecules: &[(String, MolDynamics, usize)],
    set: &FfParamSet,
    water: Option<&WaterModel>,
    include_water: bool,
) -> Result<String, ParamError> {
    let cfg = topology_config(set, water);
    let mut s = cfg.defaults_text();
    s.push_str("[ atomtypes ]\n");
    let mut type_map = HashMap::<String, Vec<String>>::new();
    let mut nbfix = set.nonbonded_rules.nbfix.clone();
    for (mi, (_, mol, _)) in molecules.iter().enumerate() {
        let p = mol
            .explicit_params
            .as_ref()
            .ok_or_else(|| ParamError::new("Export requires explicit molecule parameters"))?;
        if p.masses.len() != mol.atoms.len() || p.lj.len() != mol.atoms.len() {
            return Err(ParamError::new("Invalid explicit parameter lengths"));
        }
        nbfix.extend(p.nbfix.clone());
        for (i, a) in mol.atoms.iter().enumerate() {
            let t = format!("DYN_{mi}_{i}");
            if let Some(original) = &a.force_field_type {
                type_map
                    .entry(original.clone())
                    .or_default()
                    .push(t.clone());
            }
            writeln!(
                s,
                "{t} {:.8} 0 A {:.12e} {:.12e}",
                p.masses[i],
                p.lj[i].0 / 10.,
                p.lj[i].1 * 4.184
            )
            .unwrap();
        }
    }
    s.push_str(&cfg.solvent_atomtypes(include_water));
    let w = water.unwrap_or(&set.default_water);
    type_map
        .entry(w.cation.ff_type.to_string())
        .or_default()
        .push("BF_NA".into());
    type_map
        .entry(w.anion.ff_type.to_string())
        .or_default()
        .push("BF_CL".into());
    s.push_str("\n[ nonbond_params ]\n");
    for ((a, b), (sigma, epsilon)) in nbfix {
        if let (Some(aa), Some(bb)) = (type_map.get(&a), type_map.get(&b)) {
            for x in aa {
                for y in bb {
                    writeln!(s, "{x} {y} 1 {:.12e} {:.12e}", sigma / 10., epsilon * 4.184).unwrap();
                }
            }
        }
    }
    // GROMACS CMAP uses five consecutive atoms. Each atom has a unique type,
    // so residue-dependent maps can safely share the global cmaptypes section.
    let mut cmap_types = String::new();
    let mut blocks = String::new();
    for (mi, (name, mol, _)) in molecules.iter().enumerate() {
        let p = mol.explicit_params.as_ref().unwrap();
        writeln!(blocks, "\n[ moleculetype ]\n{name} 0\n[ atoms ]").unwrap();
        for (i, a) in mol.atoms.iter().enumerate() {
            writeln!(
                blocks,
                "{} DYN_{mi}_{i} 1 {name} A{} {} {:.12} {:.8}",
                i + 1,
                i + 1,
                i + 1,
                a.partial_charge
                    .ok_or_else(|| ParamError::new("Missing export charge"))?,
                p.masses[i]
            )
            .unwrap();
        }
        blocks.push_str("[ bonds ]\n");
        for ((i, j), b) in &p.bonds {
            writeln!(
                blocks,
                "{} {} 1 {:.12} {:.12}",
                i + 1,
                j + 1,
                b.r_0 / 10.,
                b.k_b * 2. * 418.4
            )
            .unwrap();
        }
        blocks.push_str("[ angles ]\n");
        let mut used_ub = BTreeSet::new();
        for ((i, j, k), a) in &p.angles {
            if let Some((ui, (_, (ku, ru)))) = p
                .urey_bradley
                .iter()
                .enumerate()
                .find(|(_, ((u, v), _))| (*u == *i && *v == *k) || (*u == *k && *v == *i))
            {
                used_ub.insert(ui);
                writeln!(
                    blocks,
                    "{} {} {} 5 {:.12} {:.12} {:.12} {:.12}",
                    i + 1,
                    j + 1,
                    k + 1,
                    a.theta_0.to_degrees(),
                    a.k * 2. * 4.184,
                    ru / 10.,
                    ku * 2. * 418.4
                )
                .unwrap();
            } else {
                writeln!(
                    blocks,
                    "{} {} {} 1 {:.12} {:.12}",
                    i + 1,
                    j + 1,
                    k + 1,
                    a.theta_0.to_degrees(),
                    a.k * 2. * 4.184
                )
                .unwrap();
            }
        }
        if used_ub.len() != p.urey_bradley.len() {
            return Err(ParamError::new("Urey-Bradley term has no matching angle"));
        }
        blocks.push_str("[ dihedrals ]\n");
        for (terms, funct) in [(&p.dihedrals, 9), (&p.impropers, 4)] {
            for ((a, b, c, d), v) in terms {
                writeln!(
                    blocks,
                    "{} {} {} {} {funct} {:.12} {:.12} {}",
                    a + 1,
                    b + 1,
                    c + 1,
                    d + 1,
                    v.phase.to_degrees(),
                    v.barrier_height * 4.184,
                    v.periodicity
                )
                .unwrap();
            }
        }
        for ((a, b, c, d), (k, phi)) in &p.harmonic_impropers {
            writeln!(
                blocks,
                "{} {} {} {} 2 {:.12} {:.12}",
                a + 1,
                b + 1,
                c + 1,
                d + 1,
                phi.to_degrees(),
                k * 2. * 4.184
            )
            .unwrap();
        }
        blocks.push_str("[ pairs ]\n");
        let mut excluded: BTreeSet<_> = p.exclusions.iter().copied().collect();
        for ((i, j), pair) in &p.pairs_14 {
            if (pair.scale.coulomb as f64 - cfg.defaults().2).abs() > 1e-6 {
                return Err(ParamError::new(
                    "Export pair has Coulomb scaling inconsistent with the selected family",
                ));
            }
            let (sigma, epsilon) = pair.lj.unwrap_or_else(|| {
                p.lj_combining
                    .combine(p.lj[*i].0, p.lj[*i].1, p.lj[*j].0, p.lj[*j].1)
            });
            writeln!(
                blocks,
                "{} {} 1 {:.12e} {:.12e}",
                i + 1,
                j + 1,
                sigma / 10.,
                epsilon * pair.scale.lj * 4.184
            )
            .unwrap();
            excluded.insert((*i, *j));
        }
        blocks.push_str("[ exclusions ]\n");
        for (i, j) in excluded {
            writeln!(blocks, "{} {}", i + 1, j + 1).unwrap();
        }
        if !p.cmaps.is_empty() {
            blocks.push_str("[ cmap ]\n");
        }
        for (ids, grid) in &p.cmaps {
            if ids[1..4] != ids[4..7] {
                return Err(ParamError::new(
                    "GROMACS export requires consecutive CMAP torsions",
                ));
            }
            let five = [ids[0], ids[1], ids[2], ids[3], ids[7]];
            for i in five {
                write!(blocks, "{} ", i + 1).unwrap();
                write!(cmap_types, "DYN_{mi}_{i} ").unwrap();
            }
            blocks.push_str("1\n");
            let grid = &p.cmap_grids[*grid];
            writeln!(cmap_types, "1 {} {} \\", grid.size(), grid.size()).unwrap();
            for row in grid.energies().chunks(grid.size()) {
                for value in row {
                    write!(cmap_types, "{:.12} ", value * 4.184).unwrap();
                }
                cmap_types.push_str("\\\n");
            }
            // End this cmaptypes record without a continuation.
            cmap_types.truncate(cmap_types.len() - 2);
            cmap_types.push('\n');
        }
    }
    if !cmap_types.is_empty() {
        s.push_str("\n[ cmaptypes ]\n");
        s.push_str(&cmap_types);
    }
    s.push_str(&blocks);
    s.push_str(&cfg.solvent_moleculetypes(include_water));
    s.push_str("[ system ]\nMolchanica MD\n\n[ molecules ]\n");
    for (name, _, count) in molecules {
        writeln!(s, "{name} {count}").unwrap();
    }
    Ok(s)
}
