//! CHARMM36m: CHARMM's files, CMAP interpolation, the forces of CHARMM's extra bonded terms, water
//! with LJ on H, and a protein built from CHARMM's topology, compared to OpenMM. Also, a CGenFF
//! ligand from a GROMACS topology in CHARMM-GUI's layout, compared to GROMACS, and simulated with a
//! protein built from CHARMM36m.

use std::{collections::HashMap, fs};

use bio_files::{
    AtomGeneric, BondGeneric, BondType, ResidueEnd, ResidueGeneric, ResidueType,
    gromacs::mdp::VdwModifier,
};
use lin_alg::f32::Vec3;
use na_seq::{AminoAcid, Element};

use super::gromacs_compare::{
    ScratchDir, assert_system_close, dynamics_config, pme_coulomb, reference_mdp,
    run_reference_files,
};
use crate::{
    ComputationDevice, FfMolType, ForceFieldFamily, HydrogenConstraint, ImportedSystem, MdConfig,
    MdOverrides, MdState, MolDynamics, SimBoxInit, Solvent, WaterModel,
    bonded::ExtraBondedTerms,
    charmm::CharmmSet,
    cmap::CmapGrid,
    non_bonded::{CHARGE_UNIT_SCALER, LjCombiningRule, LjTableIndices},
    params::FfParamSet,
};

const OPENMM_REFERENCE: &str = include_str!("fixtures/charmm_ubq_openmm.txt");

/// Ubiquitin with hydrogens, and OpenMM's charges and energy components. See
/// `fixtures/regenerate_charmm_reference.py`.
struct Reference {
    atoms: Vec<AtomGeneric>,
    bonds: Vec<BondGeneric>,
    residues: Vec<ResidueGeneric>,
    energies: HashMap<String, f64>,
    counts: HashMap<String, usize>,
    /// Atoms, and energy.
    impropers: Vec<([usize; 4], f64)>,
}

fn load_reference() -> Reference {
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut residues: Vec<ResidueGeneric> = Vec::new();
    let mut energies = HashMap::new();
    let mut counts = HashMap::new();
    let mut impropers = Vec::new();

    for line in OPENMM_REFERENCE.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        match cols.first() {
            Some(&"ATOM") => {
                let serial_number = cols[1].parse::<u32>().unwrap() + 1;
                let res_sn: u32 = cols[2].parse().unwrap();
                let v = |i: usize| cols[i].parse::<f64>().unwrap();

                atoms.push(AtomGeneric {
                    serial_number,
                    posit: lin_alg::f64::Vec3::new(v(6), v(7), v(8)),
                    element: Element::from_letter(cols[5]).unwrap(),
                    type_in_res_general: Some(cols[4].to_owned()),
                    partial_charge: Some(v(9) as f32),
                    ..Default::default()
                });

                if residues.last().map(|r| r.serial_number) != Some(res_sn) {
                    residues.push(ResidueGeneric {
                        serial_number: res_sn,
                        res_type: ResidueType::AminoAcid(cols[3].parse::<AminoAcid>().unwrap()),
                        atom_sns: Vec::new(),
                        end: ResidueEnd::Internal,
                    });
                }
                residues.last_mut().unwrap().atom_sns.push(serial_number);
            }
            Some(&"BOND") => bonds.push(BondGeneric {
                bond_type: BondType::Single,
                atom_0_sn: cols[1].parse::<u32>().unwrap() + 1,
                atom_1_sn: cols[2].parse::<u32>().unwrap() + 1,
            }),
            Some(&"ENERGY") => {
                energies.insert(cols[1].to_owned(), cols[2].parse().unwrap());
            }
            Some(&"COUNT") => {
                counts.insert(cols[1].to_owned(), cols[2].parse().unwrap());
            }
            Some(&"IMPROPER") => impropers.push((
                std::array::from_fn(|i| cols[i + 1].parse().unwrap()),
                cols[5].parse().unwrap(),
            )),
            _ => (),
        }
    }

    Reference {
        atoms,
        bonds,
        residues,
        energies,
        counts,
        impropers,
    }
}

fn charmm_config(solvent: Solvent) -> MdConfig {
    MdConfig {
        hydrogen_constraint: HydrogenConstraint::Flexible,
        solvent,
        barostat_cfg: None,
        max_init_relaxation_iters: None,
        sim_box: SimBoxInit::Pad(12.),
        overrides: MdOverrides {
            skip_water_relaxation: true,
            ..Default::default()
        },
        ..MdConfig::for_family(ForceFieldFamily::Charmm36)
    }
}

fn ubiquitin(reference: &Reference) -> MolDynamics {
    MolDynamics {
        ff_mol_type: FfMolType::Peptide,
        atoms: reference.atoms.clone(),
        bonds: reference.bonds.clone(),
        residues: Some(reference.residues.clone()),
        ..Default::default()
    }
}

fn ubiquitin_state(reference: &Reference) -> MdState {
    let param_set = FfParamSet::new_charmm().unwrap();
    let cfg = charmm_config(Solvent::None);
    MdState::new(
        &ComputationDevice::Cpu,
        &cfg,
        &[ubiquitin(reference)],
        &param_set,
    )
    .unwrap()
    .0
}

fn assert_energy(got: f64, expected: f64, rel_tol: f64, abs_tol: f64, label: &str) {
    let tol = abs_tol + rel_tol * expected.abs();
    assert!(
        (got - expected).abs() <= tol,
        "{label}: got {got:.4}, expected {expected:.4} (tolerance {tol:.4})"
    );
}

/// Potential energy from one of `MdState`'s bonded-force functions.
fn energy_of(state: &mut MdState, apply: impl Fn(&mut MdState)) -> f64 {
    state.potential_energy = 0.;
    for a in &mut state.atoms {
        a.force = Vec3::new_zero();
    }
    apply(state);
    state.potential_energy
}

/// Energy from only some of the extra bonded terms.
fn extra_energy(
    state: &mut MdState,
    select: impl Fn(&ExtraBondedTerms) -> ExtraBondedTerms,
) -> f64 {
    let all = std::mem::take(&mut state.extra_bonded);
    state.extra_bonded = select(&all);
    let e = energy_of(state, |s| s.apply_extra_bonded_forces());
    state.extra_bonded = all;
    e
}

fn only_ub(t: &ExtraBondedTerms) -> ExtraBondedTerms {
    ExtraBondedTerms {
        urey_bradley: t.urey_bradley.clone(),
        ..Default::default()
    }
}

fn only_cmap(t: &ExtraBondedTerms) -> ExtraBondedTerms {
    ExtraBondedTerms {
        cmaps: t.cmaps.clone(),
        cmap_grids: t.cmap_grids.clone(),
        ..Default::default()
    }
}

#[test]
fn charmm_files_parse() {
    let set = CharmmSet::new_c36m().unwrap();

    for name in [
        "ALA", "HSD", "HSE", "HSP", "CYM", "NTER", "CTER", "GLYP", "PROP", "DISU", "ASPP", "LSN",
        "TIP3", "SOD", "CLA",
    ] {
        assert!(set.topology.residue(name).is_some(), "Missing {name}");
    }
    // CHARMM36m's CMAP grids, e.g. for residues before proline, and for glycine.
    assert!(set.params.cmaps.len() >= 6);
    assert!(set.params.cmaps.iter().all(|c| c.size == 24));

    // Water and ion parameters, which come from the stream file, match our water model.
    let w = WaterModel::TIP3P_CHARMM;
    let sigma = |t: &str| set.params.nonbonded[t].rmin_half * 1.781_797_4;
    assert!((sigma("OT") - w.lj_sigma_o).abs() < 1e-4);
    assert!((set.params.nonbonded["OT"].eps - w.lj_eps_o).abs() < 1e-6);
    assert!((sigma("HT") - w.lj_sigma_h).abs() < 1e-4);
    assert!((set.params.nonbonded["HT"].eps - w.lj_eps_h).abs() < 1e-6);
    assert!((sigma("SOD") - w.cation.lj_sigma).abs() < 1e-4);
    assert!((sigma("CLA") - w.anion.lj_sigma).abs() < 1e-4);
    assert_eq!(set.params.bond("HT", "OT").unwrap().1, 0.9572);

    // NBFIX, from the protein file and the stream file. (σ = Rmin / 2^(1/6))
    let nbfix = set.nbfix();
    let sod_oc = nbfix
        .get(&("SOD".to_owned(), "OC".to_owned()))
        .or_else(|| nbfix.get(&("OC".to_owned(), "SOD".to_owned())))
        .unwrap();
    assert!((sod_oc.0 - 3.23 * 0.890_898_7).abs() < 1e-4);
    assert!((sod_oc.1 - 0.07502).abs() < 1e-6);
    assert!(
        nbfix.contains_key(&("NC2".to_owned(), "OC".to_owned()))
            || nbfix.contains_key(&("OC".to_owned(), "NC2".to_owned()))
    );
}

/// CMAP interpolation passes through the grid values, is smooth between them, and its derivatives
/// match its energy.
#[test]
fn cmap_interpolation() {
    use std::f64::consts::TAU;

    let size = 24;
    let f = |φ: f64, ψ: f64| φ.cos() + (2. * ψ).sin() + 0.5 * φ.cos() * ψ.sin();
    let spacing = TAU / size as f64;
    let angle = |i: usize| -std::f64::consts::PI + i as f64 * spacing;

    let mut energies = Vec::new();
    for i in 0..size {
        for j in 0..size {
            energies.push(f(angle(i), angle(j)));
        }
    }
    let grid = CmapGrid::new(size, energies);

    for (i, j) in [(0, 0), (3, 17), (23, 23), (12, 5)] {
        let (e, _, _) = grid.eval(angle(i), angle(j));
        assert!((e - f(angle(i), angle(j))).abs() < 1e-10);
    }

    // Periodic.
    let (e0, _, _) = grid.eval(0.3, -2.9);
    let (e1, _, _) = grid.eval(0.3 + TAU, -2.9 - TAU);
    assert!((e0 - e1).abs() < 1e-10);

    let h = 1e-6;
    for (φ, ψ) in [(0.31, -2.2), (-3.1, 3.1), (1.7, 0.05), (2.95, -0.8)] {
        let (e, de_dφ, de_dψ) = grid.eval(φ, ψ);
        // Close to the function it samples.
        assert!((e - f(φ, ψ)).abs() < 2e-3, "{e} vs {}", f(φ, ψ));

        let fd_φ = (grid.eval(φ + h, ψ).0 - grid.eval(φ - h, ψ).0) / (2. * h);
        let fd_ψ = (grid.eval(φ, ψ + h).0 - grid.eval(φ, ψ - h).0) / (2. * h);
        assert!((de_dφ - fd_φ).abs() < 1e-6, "dE/dφ {de_dφ} vs {fd_φ}");
        assert!((de_dψ - fd_ψ).abs() < 1e-6, "dE/dψ {de_dψ} vs {fd_ψ}");
    }
}

/// Forces from Urey-Bradley terms, harmonic impropers, and CMAP match their energy's gradient.
#[test]
fn charmm_extra_bonded_forces_match_energy_gradient() {
    let reference = load_reference();
    let mut state = ubiquitin_state(&reference);

    let terms = state.extra_bonded.clone();
    // A few of each: (label, one-term selection, atoms of the term)
    let mut cases: Vec<(&str, ExtraBondedTerms, Vec<usize>)> = Vec::new();
    for k in [0, 500, terms.urey_bradley.len() - 1] {
        let ((i, j), p) = terms.urey_bradley[k];
        cases.push((
            "Urey-Bradley",
            ExtraBondedTerms {
                urey_bradley: vec![((i, j), p)],
                ..Default::default()
            },
            vec![i, j],
        ));
    }
    for k in [0, 100, terms.harmonic_impropers.len() - 1] {
        let (atoms, p) = terms.harmonic_impropers[k];
        cases.push((
            "improper",
            ExtraBondedTerms {
                harmonic_impropers: vec![(atoms, p)],
                ..Default::default()
            },
            atoms.to_vec(),
        ));
    }
    for k in [0, 30, terms.cmaps.len() - 1] {
        let (atoms, grid) = terms.cmaps[k];
        cases.push((
            "CMAP",
            ExtraBondedTerms {
                cmaps: vec![(atoms, 0)],
                cmap_grids: vec![terms.cmap_grids[grid].clone()],
                ..Default::default()
            },
            atoms.to_vec(),
        ));
    }

    let h = 2e-3;
    for (label, term, atoms) in cases {
        state.extra_bonded = term;
        energy_of(&mut state, |s| s.apply_extra_bonded_forces());
        let forces: Vec<Vec3> = state.atoms.iter().map(|a| a.force).collect();

        let mut atoms = atoms;
        atoms.sort();
        atoms.dedup();
        for i in atoms {
            for axis in 0..3 {
                let shift = |d: f32| match axis {
                    0 => Vec3::new(d, 0., 0.),
                    1 => Vec3::new(0., d, 0.),
                    _ => Vec3::new(0., 0., d),
                };
                let p = state.atoms[i].posit;
                state.atoms[i].posit = p + shift(h);
                let e_plus = energy_of(&mut state, |s| s.apply_extra_bonded_forces());
                state.atoms[i].posit = p - shift(h);
                let e_minus = energy_of(&mut state, |s| s.apply_extra_bonded_forces());
                state.atoms[i].posit = p;

                let fd = -(e_plus - e_minus) / (2. * h as f64);
                let f = match axis {
                    0 => forces[i].x,
                    1 => forces[i].y,
                    _ => forces[i].z,
                } as f64;
                assert!(
                    (f - fd).abs() < 0.02 + 0.01 * fd.abs(),
                    "{label}, atom {i}, axis {axis}: force {f}, -dE/dx {fd}"
                );
            }
        }
    }
}

/// Ubiquitin, built from CHARMM36m's topology, compared to OpenMM with charmm36_2024.xml:
/// Charges, term counts, and each energy component. Non-bonded energies are without cutoffs,
/// from our LJ tables (including NBFIX) and 1-4 pairs.
#[test]
fn charmm_ubiquitin_matches_openmm() {
    let reference = load_reference();
    let mut state = ubiquitin_state(&reference);
    let n = state.atoms.len();
    assert_eq!(n, reference.atoms.len());

    for (i, atom) in state.atoms.iter().enumerate() {
        let expected = reference.atoms[i].partial_charge.unwrap();
        let q = atom.partial_charge / CHARGE_UNIT_SCALER;
        assert!(
            (q - expected).abs() < 1e-4,
            "Charge of atom {i} ({}): {q} vs {expected}",
            reference.atoms[i].type_in_res_general.as_ref().unwrap()
        );
    }

    let count = |k: &str| reference.counts[k];
    let n_bonds = state.force_field_params.bond_stretching.len();
    assert_eq!(n_bonds, reference.bonds.len());
    assert_eq!(
        n_bonds + state.extra_bonded.urey_bradley.len(),
        count("bond_ub")
    );
    assert_eq!(
        state.extra_bonded.harmonic_impropers.len(),
        count("improper")
    );
    assert_eq!(state.extra_bonded.cmaps.len(), count("cmap"));
    assert_eq!(state.pairs_14_scaled.len(), count("pairs_14"));
    assert_eq!(
        state.pairs_excluded_12_13.len() + state.pairs_14_scaled.len(),
        count("exceptions")
    );

    let energy = |k: &str| reference.energies[k];

    let bonds = energy_of(&mut state, |s| s.apply_bond_stretching_forces());
    let ub = extra_energy(&mut state, only_ub);
    assert_energy(bonds + ub, energy("bond_ub"), 1e-3, 0.05, "bonds and UB");

    let angles = energy_of(&mut state, |s| s.apply_angle_bending_forces());
    assert_energy(angles, energy("angle"), 1e-3, 0.05, "angles");

    let dihedrals = energy_of(&mut state, |s| s.apply_dihedral_forces(false));
    assert_energy(dihedrals, energy("dihedral"), 1e-3, 0.05, "dihedrals");
    let periodic_impropers = energy_of(&mut state, |s| s.apply_dihedral_forces(true));
    assert_eq!(periodic_impropers, 0.);

    // Impropers, term by term. OpenMM's CHARMM force field orders some among equivalent atoms
    // differently from CHARMM's residue topologies, which we follow: e.g. Asp's CG CB OD1 OD2, vs
    // CHARMM's CG CB OD2 OD1. The energies of these differ, so we check that they are only
    // reordered among atoms of the same type.
    let all = state.extra_bonded.clone();
    let mut reordered = 0;
    for (atoms, e_ref) in &reference.impropers {
        match all.harmonic_impropers.iter().find(|(a, _)| a == atoms) {
            Some(&term) => {
                state.extra_bonded = ExtraBondedTerms {
                    harmonic_impropers: vec![term],
                    ..Default::default()
                };
                let e = energy_of(&mut state, |s| s.apply_extra_bonded_forces());
                assert_energy(e, *e_ref, 1e-3, 1e-3, &format!("improper {atoms:?}"));
            }
            None => {
                let sorted = |a: &[usize]| {
                    let mut a = a.to_vec();
                    a.sort();
                    a
                };
                let (ours, _) = all
                    .harmonic_impropers
                    .iter()
                    .find(|(a, _)| a[0] == atoms[0] && sorted(&a[1..]) == sorted(&atoms[1..]))
                    .unwrap_or_else(|| panic!("Missing improper {atoms:?}"));
                for k in 1..4 {
                    let ff_type = |i: usize| &state.atoms[i].force_field_type;
                    assert_eq!(ff_type(ours[k]), ff_type(atoms[k]), "improper {atoms:?}");
                }
                reordered += 1;
            }
        }
    }
    state.extra_bonded = all;
    // Ubiquitin's Asp, Glu, and Arg residues.
    assert!(
        reordered <= 15,
        "{reordered} impropers in a different order"
    );

    let cmap = extra_energy(&mut state, only_cmap);
    assert_energy(cmap, energy("cmap"), 1e-3, 0.02, "CMAP");

    // Non-bonded, without cutoffs.
    let (mut lj, mut lj_14, mut coulomb, mut coulomb_14) = (0., 0., 0., 0.);
    for i in 0..n {
        for j in i + 1..n {
            let key = (i, j);
            if state.pairs_excluded_12_13.contains(&key) {
                continue;
            }
            let (a0, a1) = (&state.atoms[i], &state.atoms[j]);
            let r = (a0.posit - a1.posit).magnitude() as f64;
            let qq = (a0.partial_charge * a1.partial_charge) as f64 / r;

            let pair_14 = state.pairs_14_scaled.get(&key);
            let (σ, ε) = pair_14
                .and_then(|p| p.lj)
                .unwrap_or_else(|| state.lj_tables.lookup(&LjTableIndices::StdStd(key)));
            let sr6 = (σ as f64 / r).powi(6);
            let e_lj = 4. * ε as f64 * (sr6 * sr6 - sr6);

            match pair_14 {
                Some(p) => {
                    lj_14 += p.scale.lj as f64 * e_lj;
                    coulomb_14 += p.scale.coulomb as f64 * qq;
                }
                None => {
                    lj += e_lj;
                    coulomb += qq;
                }
            }
        }
    }

    assert_energy(lj, energy("lj"), 1e-4, 0.02, "LJ");
    assert_energy(lj_14, energy("lj_14"), 1e-4, 0.02, "1-4 LJ");
    // Our Coulomb constant is 332.0522 kcal Å/mol e²; OpenMM's is 332.0637.
    assert_energy(coulomb_14, energy("coulomb_14"), 1e-4, 0.02, "1-4 Coulomb");
    assert_energy(
        coulomb + coulomb_14,
        energy("coulomb"),
        1e-4,
        0.02,
        "Coulomb",
    );
}

/// Protonation states and termini from the hydrogens present, and errors for atoms CHARMM's
/// residues don't have.
#[test]
fn charmm_protonation_states_and_errors() {
    let reference = load_reference();
    let set = CharmmSet::new_c36m().unwrap();

    let build = |atoms: &[AtomGeneric], bonds: &[BondGeneric]| {
        let mut atoms = atoms.to_vec();
        let result =
            crate::charmm::build_peptide(&mut atoms, bonds, &reference.residues, &set).map(|_| ());
        (atoms, result)
    };

    let (atoms, result) = build(&reference.atoms, &reference.bonds);
    result.unwrap();
    let ff_type = |atoms: &[AtomGeneric], res: u32, name: &str| {
        let res = reference
            .residues
            .iter()
            .find(|r| r.serial_number == res)
            .unwrap();
        atoms
            .iter()
            .find(|a| {
                res.atom_sns.contains(&a.serial_number)
                    && a.type_in_res_general.as_deref() == Some(name)
            })
            .map(|a| a.force_field_type.clone().unwrap())
            .unwrap()
    };
    // Met 1 is a charged N-terminus (NTER), Gly 76 a charged C-terminus (CTER), and His 68 has H
    // on ND1. (HSD)
    assert_eq!(ff_type(&atoms, 1, "N"), "NH3");
    assert_eq!(ff_type(&atoms, 76, "OXT"), "OC");
    assert_eq!(ff_type(&atoms, 76, "O"), "OC");
    assert_eq!(ff_type(&atoms, 68, "ND1"), "NR1");
    assert_eq!(ff_type(&atoms, 68, "NE2"), "NR2");
    // Ile's CD1 is CHARMM's CD.
    assert_eq!(ff_type(&atoms, 3, "CD1"), "CT3");

    // Move His 68's proton from ND1 to NE2: HSE.
    let his = reference
        .residues
        .iter()
        .find(|r| r.serial_number == 68)
        .unwrap();
    let find = |name: &str| {
        reference
            .atoms
            .iter()
            .find(|a| {
                his.atom_sns.contains(&a.serial_number)
                    && a.type_in_res_general.as_deref() == Some(name)
            })
            .unwrap()
            .serial_number
    };
    let (hd1, nd1, ne2) = (find("HD1"), find("ND1"), find("NE2"));
    let bonds: Vec<BondGeneric> = reference
        .bonds
        .iter()
        .map(|b| {
            let mut b = b.clone();
            if b.atom_0_sn == hd1 || b.atom_1_sn == hd1 {
                for sn in [&mut b.atom_0_sn, &mut b.atom_1_sn] {
                    if *sn == nd1 {
                        *sn = ne2;
                    }
                }
            }
            b
        })
        .collect();
    let (atoms, result) = build(&reference.atoms, &bonds);
    result.unwrap();
    assert_eq!(ff_type(&atoms, 68, "ND1"), "NR2");
    assert_eq!(ff_type(&atoms, 68, "NE2"), "NR1");

    // A missing hydrogen is an error, not a silent change of protonation state.
    let hb2 = reference
        .atoms
        .iter()
        .position(|a| a.type_in_res_general.as_deref() == Some("HB2"))
        .unwrap();
    let mut atoms = reference.atoms.clone();
    let removed = atoms.remove(hb2).serial_number;
    let bonds: Vec<_> = reference
        .bonds
        .iter()
        .filter(|b| b.atom_0_sn != removed && b.atom_1_sn != removed)
        .cloned()
        .collect();
    let (_, result) = build(&atoms, &bonds);
    let err = result.unwrap_err();
    assert!(err.descrip.contains("hydrogens"), "{}", err.descrip);
}

/// CHARMM's TIP3P has LJ on its hydrogens, so water-water and water-solute pairs with H have LJ;
/// OPC's don't.
#[test]
fn charmm_water_hydrogen_lj() {
    let reference = load_reference();

    for (family, h_lj) in [
        (ForceFieldFamily::Charmm36, true),
        (ForceFieldFamily::Amber, false),
    ] {
        let param_set = FfParamSet::from_family(family).unwrap();
        let cfg = MdConfig {
            ff_family: family,
            sim_box: SimBoxInit::Pad(6.),
            ..charmm_config(Solvent::WaterOpcSpecifyMolCount(60))
        };
        let mols = match family {
            ForceFieldFamily::Charmm36 => vec![ubiquitin(&reference)],
            _ => Vec::new(),
        };
        let cfg = MdConfig {
            sim_box: if mols.is_empty() {
                SimBoxInit::new_cube(20.)
            } else {
                cfg.sim_box.clone()
            },
            ..cfg
        };
        let (state, _) = MdState::new(&ComputationDevice::Cpu, &cfg, &mols, &param_set).unwrap();
        assert!(!state.water.is_empty());

        let with_h_lj = state
            .nb_pairs
            .iter()
            .filter(|p| {
                p.calc_lj
                    && matches!(
                        p.lj_indices,
                        LjTableIndices::StdWaterH(_)
                            | LjTableIndices::WaterWaterOH
                            | LjTableIndices::WaterWaterHH
                    )
            })
            .count();
        assert_eq!(with_h_lj > 0, h_lj, "{family:?}");

        if h_lj {
            let w = &state.water_model;
            let (σ, ε) = state.lj_tables.lookup(&LjTableIndices::WaterWaterHH);
            assert_eq!((σ, ε), (w.lj_sigma_h, w.lj_eps_h));
            let (σ, ε) = state.lj_tables.lookup(&LjTableIndices::WaterWaterOH);
            assert!((σ - 0.5 * (w.lj_sigma_o + w.lj_sigma_h)).abs() < 1e-6);
            assert!((ε - (w.lj_eps_o * w.lj_eps_h).sqrt()).abs() < 1e-6);
        }
    }
}

/// Solvated ubiquitin with CHARMM36m, CHARMM TIP3P, and CHARMM's LJ force switch: The system
/// stays intact, and its temperature stays bounded. Slow; run with `--release`, ideally with CUDA.
#[test]
#[ignore]
fn charmm_solvated_ubiquitin_is_stable() {
    let reference = load_reference();
    let param_set = FfParamSet::new_charmm().unwrap();
    let cfg = MdConfig {
        sim_box: SimBoxInit::Pad(10.),
        ..MdConfig::for_family(ForceFieldFamily::Charmm36)
    };
    #[cfg(feature = "cuda")]
    let dev = super::force_components::cuda_device().unwrap_or(ComputationDevice::Cpu);
    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;
    let (mut state, _) = MdState::new(&dev, &cfg, &[ubiquitin(&reference)], &param_set).unwrap();
    println!(
        "Atoms: {}, waters: {}",
        state.atoms.len(),
        state.water.len()
    );

    let bond_lengths = |s: &MdState| -> f32 {
        s.force_field_params
            .bonds_topology
            .iter()
            .map(|&(i, j)| (s.atoms[i].posit - s.atoms[j].posit).magnitude())
            .fold(0., f32::max)
    };

    for block in 0..10 {
        for _ in 0..100 {
            state.step(&dev, 0.002, None);
        }
        let temp = state.measure_temperature();
        println!(
            "{} ps: T = {temp:.1} K, E_pot = {:.1}, longest bond {:.3} Å",
            (block + 1) as f32 * 0.2,
            state.potential_energy,
            bond_lengths(&state)
        );
        assert!(state.potential_energy.is_finite());
        assert!(bond_lengths(&state) < 2.2);
    }
    // The target is 310 K. Rigid water currently runs hot with every force field family (e.g.
    // ~400 K for CHARMM TIP3P, and more for OPC, after 2 ps), so this only catches blow-ups.
    let temp = state.measure_temperature();
    assert!((250. ..500.).contains(&temp), "Temperature: {temp} K");
}

/// The CUDA kernel matches the CPU for CHARMM's non-bonded features: NBFIX, special 1-4 LJ, and
/// LJ on water hydrogens, with the LJ force switch.
#[cfg(feature = "cuda")]
#[test]
fn charmm_gpu_matches_cpu() {
    let Some(gpu) = super::force_components::cuda_device() else {
        return;
    };
    let reference = load_reference();
    let param_set = FfParamSet::new_charmm().unwrap();
    let cfg = charmm_config(Solvent::WaterOpcSpecifyMolCount(400));

    let forces = |dev: &ComputationDevice| {
        let (mut state, _) = MdState::new(dev, &cfg, &[ubiquitin(&reference)], &param_set).unwrap();
        state.reset_f_acc_pe_virial();
        state.apply_all_forces(dev, &None);
        let mut f: Vec<Vec3> = state.atoms.iter().map(|a| a.force).collect();
        for w in &state.water {
            f.extend([w.o.force, w.h0.force, w.h1.force]);
        }
        (f, state.potential_energy)
    };

    let (f_cpu, e_cpu) = forces(&ComputationDevice::Cpu);
    let (f_gpu, e_gpu) = forces(&gpu);
    assert_eq!(f_cpu.len(), f_gpu.len());
    assert_energy(e_gpu, e_cpu, 1e-4, 0.1, "potential energy");
    for (i, (c, g)) in f_cpu.iter().zip(&f_gpu).enumerate() {
        let tol = 0.02 + 2e-3 * c.magnitude();
        assert!(
            (*c - *g).magnitude() < tol,
            "Atom {i}: CPU {c:?}, GPU {g:?}"
        );
    }
}

/// Paracetamol (CGenFF's PACP), with Na+ and Cl-, as a GROMACS topology in CHARMM-GUI's layout:
/// Parameters by type in `forcefield.itp`, including Urey-Bradley angles, harmonic impropers,
/// special 1-4 LJ (`[ pairtypes ]`), NBFIX (`[ nonbond_params ]`), and CMAP types. See
/// `fixtures/regenerate_cgenff_ligand.py`.
const CGENFF_FILES: [(&str, &str); 5] = [
    (
        "forcefield.itp",
        include_str!("fixtures/cgenff_pacp/forcefield.itp"),
    ),
    ("pacp.itp", include_str!("fixtures/cgenff_pacp/pacp.itp")),
    ("ions.itp", include_str!("fixtures/cgenff_pacp/ions.itp")),
    ("topol.top", include_str!("fixtures/cgenff_pacp/topol.top")),
    ("conf.gro", include_str!("fixtures/cgenff_pacp/conf.gro")),
];

fn import_cgenff_ligand() -> ImportedSystem {
    let dir = ScratchDir::new("cgenff_pacp_inputs");
    for (name, text) in CGENFF_FILES {
        fs::write(dir.path().join(name), text).unwrap();
    }
    ImportedSystem::from_gromacs_files(
        &dir.path().join("topol.top"),
        &dir.path().join("conf.gro"),
        &[],
    )
    .unwrap()
}

/// NBFIX for a type pair, in either order.
fn nbfix_lookup(nbfix: &crate::NbFix, t0: &str, t1: &str) -> Option<(f32, f32)> {
    nbfix
        .get(&(t0.to_owned(), t1.to_owned()))
        .or_else(|| nbfix.get(&(t1.to_owned(), t0.to_owned())))
        .copied()
}

/// A CGenFF ligand and ions from a CHARMM-GUI-style GROMACS topology, compared to GROMACS.
#[test]
fn cgenff_ligand_import_matches_gromacs() {
    let imported = import_cgenff_ligand();
    assert_eq!(imported.mols.len(), 3);

    let lig = imported.mols[0].explicit_params.as_ref().unwrap();
    assert_eq!(lig.masses.len(), 20);
    assert_eq!(lig.bonds.len(), 20);
    assert_eq!(lig.angles.len(), 31);
    assert!(!lig.urey_bradley.is_empty());
    assert_eq!(lig.harmonic_impropers.len(), 1);
    assert!(lig.impropers.is_empty());
    assert_eq!(lig.pairs_14.len(), 37);
    // CGenFF's special 1-4 LJ parameters, e.g. for the methyl carbon and the amide N and O.
    assert!(lig.pairs_14.iter().any(|(_, p)| p.lj.is_some()));
    assert!(
        lig.pairs_14
            .iter()
            .all(|(_, p)| p.scale.coulomb == 1. && (p.lj.is_some() || p.scale.lj == 1.))
    );

    // NBFIX between the ions, from the file. (σ = Rmin / 2^(1/6))
    let (sigma, eps) = nbfix_lookup(&lig.nbfix, "SOD", "CLA").unwrap();
    assert!((sigma - 3.731 * 0.890_898_7).abs() < 1e-3, "σ {sigma}");
    assert!((eps - 0.0839).abs() < 1e-4, "ε {eps}");
    // Only pairs of types the system uses.
    assert!(nbfix_lookup(&lig.nbfix, "SOD", "OC").is_none());

    let reference = run_reference_files(
        "cgenff_pacp",
        &CGENFF_FILES,
        reference_mdp(pme_coulomb(), VdwModifier::None),
    );

    let cfg = MdConfig {
        hydrogen_constraint: HydrogenConstraint::Flexible,
        ..dynamics_config(MdOverrides::default())
    };
    let dev = ComputationDevice::Cpu;
    let (mut state, _) = MdState::new(&dev, &cfg, &imported.mols, &imported.param_set()).unwrap();

    // The ions' LJ pair uses NBFIX, not the combining rule.
    let (na, cl) = (20, 21);
    assert_eq!(state.atoms[na].force_field_type, "SOD");
    assert_eq!(
        state.lj_tables.lookup(&LjTableIndices::StdStd((na, cl))),
        (sigma, eps)
    );

    state.reset_f_acc_pe_virial();
    state.apply_all_forces(&dev, &None);
    let forces: Vec<_> = state.atoms.iter().map(|a| a.force).collect();

    assert_system_close(
        &forces,
        state.potential_energy as f32,
        &reference,
        0.015,
        5e-3,
        0.015,
        5e-3,
        "CGenFF ligand and ions",
    );
}

/// Bonded energy: bonds, angles, dihedrals, and CHARMM's extra terms.
fn bonded_energy(state: &mut MdState) -> f64 {
    energy_of(state, |s| {
        s.apply_bond_stretching_forces();
        s.apply_angle_bending_forces();
        s.apply_dihedral_forces(false);
        s.apply_dihedral_forces(true);
        s.apply_extra_bonded_forces();
    })
}

/// The CGenFF ligand and ions, imported, with ubiquitin built from CHARMM36m, as a user would
/// combine them: The imported molecules keep their parameters, and non-bonded pairs between them
/// and the protein use the combining rule, or NBFIX between their types. (e.g. SOD with Asp and
/// Glu's carboxylate O)
#[test]
fn cgenff_ligand_with_charmm_protein() {
    let reference = load_reference();
    let imported = import_cgenff_ligand();
    let param_set = FfParamSet::new_charmm().unwrap();
    let set = param_set.charmm.as_ref().unwrap();

    // Place the ligand and ions next to the protein.
    let protein = ubiquitin(&reference);
    let n_prot = protein.atoms.len();
    let prot_max_x = protein
        .atoms
        .iter()
        .map(|a| a.posit.x)
        .fold(f64::MIN, f64::max);
    let prot_center = protein
        .atoms
        .iter()
        .fold(lin_alg::f64::Vec3::new_zero(), |acc, a| acc + a.posit)
        / n_prot as f64;
    let lig_atoms: Vec<_> = imported.mols.iter().flat_map(|m| &m.atoms).collect();
    let lig_min_x = lig_atoms.iter().map(|a| a.posit.x).fold(f64::MAX, f64::min);
    let lig_center = lig_atoms
        .iter()
        .fold(lin_alg::f64::Vec3::new_zero(), |acc, a| acc + a.posit)
        / lig_atoms.len() as f64;
    let shift = lin_alg::f64::Vec3::new(
        prot_max_x - lig_min_x + 2.5,
        prot_center.y - lig_center.y,
        prot_center.z - lig_center.z,
    );
    let mut others = imported.mols.clone();
    for mol in &mut others {
        for a in &mut mol.atoms {
            a.posit = a.posit + shift;
        }
    }

    let cfg = charmm_config(Solvent::None);
    let dev = ComputationDevice::Cpu;
    let mut mols = vec![protein];
    mols.extend(others.iter().cloned());
    let (mut state, _) = MdState::new(&dev, &cfg, &mols, &param_set).unwrap();
    assert_eq!(state.atoms.len(), n_prot + 22);

    // The ligand's parameters are its own: Its bonded energy is the same as on its own, and its
    // charges are the file's.
    let (mut prot_only, _) = MdState::new(&dev, &cfg, &mols[..1], &param_set).unwrap();
    let (mut others_only, _) = MdState::new(&dev, &cfg, &others, &param_set).unwrap();
    let e_combined = bonded_energy(&mut state);
    let e_parts = bonded_energy(&mut prot_only) + bonded_energy(&mut others_only);
    assert_energy(e_combined, e_parts, 1e-5, 1e-3, "bonded energy");
    assert!(others_only.extra_bonded.urey_bradley.len() > 0);
    for (i, a) in others.iter().flat_map(|m| &m.atoms).enumerate() {
        let q = state.atoms[n_prot + i].partial_charge / CHARGE_UNIT_SCALER;
        assert!((q - a.partial_charge.unwrap()).abs() < 1e-5);
    }

    // LJ between the protein and the imported atoms: NBFIX where CHARMM has it for their types,
    // and otherwise the combining rule, from each side's parameters.
    let lj_imported: Vec<(f32, f32)> = others
        .iter()
        .flat_map(|m| m.explicit_params.as_ref().unwrap().lj.clone())
        .collect();
    let nbfix = set.nbfix();
    let mut n_nbfix = 0;
    for (k, &(s_i, e_i)) in lj_imported.iter().enumerate() {
        let i = n_prot + k;
        for j in 0..n_prot {
            let (t_i, t_j) = (
                &state.atoms[i].force_field_type,
                &state.atoms[j].force_field_type,
            );
            let nb = &set.params.nonbonded[t_j.as_str()];
            let expected = match nbfix_lookup(&nbfix, t_i, t_j) {
                Some(v) => {
                    n_nbfix += 1;
                    v
                }
                None => LjCombiningRule::LorentzBerthelot.combine(
                    s_i,
                    e_i,
                    nb.rmin_half * 1.781_797_4,
                    nb.eps,
                ),
            };
            let got = state.lj_tables.lookup(&LjTableIndices::StdStd((j, i)));
            assert!(
                (got.0 - expected.0).abs() < 1e-4 && (got.1 - expected.1).abs() < 1e-6,
                "{t_i}-{t_j}: {got:?} vs {expected:?}"
            );
        }
    }
    // SOD with each carboxylate O of ubiquitin's Asp, Glu, and C-terminus.
    assert!(n_nbfix > 0);

    // A short run: The ligand stays intact.
    for _ in 0..50 {
        state.step(&dev, 0.001, None);
    }
    assert!(state.potential_energy.is_finite());
    for &(i, j) in &state.force_field_params.bonds_topology {
        if i >= n_prot {
            let r = (state.atoms[i].posit - state.atoms[j].posit).magnitude();
            assert!(r < 1.7, "Ligand bond {i}-{j}: {r} Å");
        }
    }
}
