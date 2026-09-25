//! Tests for importing systems parameterized elsewhere: Amber prmtop files (compared to OpenMM),
//! and GROMACS topologies (compared to GROMACS).

use std::fs;

use bio_files::{
    gromacs::{mdp::VdwModifier, top_parse::GromacsTopology},
    inpcrd::AmberCoords,
    prmtop::AmberPrmtop,
};
use lin_alg::f32::Vec3;

use super::gromacs_compare::{
    GmxReference, ScratchDir, assert_system_close, dynamics_config, pme_coulomb, reference_mdp,
    run_reference_files,
};
use crate::{
    ComputationDevice, HydrogenConstraint, ImportedSystem, MdConfig, MdOverrides, MdState, Scale14,
};

const PRMTOP: &str = include_str!("fixtures/ala_dipeptide.prmtop");
const INPCRD: &str = include_str!("fixtures/ala_dipeptide.inpcrd");
const OPENMM_REFERENCE: &str = include_str!("fixtures/ala_dipeptide_openmm.txt");

/// Evaluate every force for an imported system, once, with the GROMACS-comparison settings, and
/// without hydrogen constraints.
fn evaluate(imported: &ImportedSystem) -> MdState {
    let cfg = MdConfig {
        hydrogen_constraint: HydrogenConstraint::Flexible,
        ..dynamics_config(MdOverrides::default())
    };
    let dev = ComputationDevice::Cpu;
    let (mut state, _) = MdState::new(&dev, &cfg, &imported.mols, &imported.param_set()).unwrap();
    state.reset_f_acc_pe_virial();
    state.apply_all_forces(&dev, &None);
    state
}

#[test]
fn prmtop_parses_fixed_width_fields() {
    let prmtop = AmberPrmtop::new(PRMTOP).unwrap();

    assert_eq!(prmtop.n_atoms(), 22);
    // Names run together in the file, e.g. "HH31CH3 HH32HH33".
    assert_eq!(&prmtop.atom_names[..4], ["HH31", "CH3", "HH32", "HH33"]);
    assert_eq!(prmtop.residue_labels, ["ACE", "ALA", "NME"]);
    assert_eq!(prmtop.residue_starts, [0, 6, 16]);
    assert_eq!(prmtop.bonds.len(), 21);
    assert!(prmtop.unsupported_terms.is_empty());
    assert!((prmtop.charges.iter().sum::<f32>()).abs() < 1e-3);

    // Every excluded pair is within the molecule, and ordered.
    for (i, j) in &prmtop.excluded_pairs {
        assert!(i < j && *j < 22);
    }
}

/// Alanine dipeptide from a prmtop, compared to OpenMM's energy and forces for the same file.
#[test]
fn prmtop_import_matches_openmm() {
    let prmtop = AmberPrmtop::new(PRMTOP).unwrap();
    let coords = AmberCoords::new(INPCRD).unwrap();
    let imported = ImportedSystem::from_amber(&prmtop, &coords).unwrap();

    assert_eq!(imported.mols.len(), 1);
    assert!(imported.water_model.is_none());
    let params = imported.mols[0].explicit_params.as_ref().unwrap();
    // Amber's defaults, for a file without per-dihedral scale factors.
    assert!(
        params
            .pairs_14
            .iter()
            .all(|(_, p)| p.scale == Scale14::AMBER && p.lj.is_none())
    );

    let mut lines = OPENMM_REFERENCE.lines();
    let energy: f32 = lines.next().unwrap().parse().unwrap();
    let forces: Vec<Vec3> = lines
        .take(22)
        .map(|l| {
            let v: Vec<f32> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            Vec3::new(v[0], v[1], v[2])
        })
        .collect();

    let state = evaluate(&imported);
    let dynamics_forces: Vec<_> = state.atoms.iter().map(|a| a.force).collect();

    assert_system_close(
        &dynamics_forces,
        state.potential_energy as f32,
        &GmxReference {
            forces,
            potential_energy_kcal: energy,
        },
        0.01,
        0.02,
        2e-3,
        0.02,
        "alanine dipeptide vs OpenMM",
    );
}

const GMX_FF_ITP: &str = "\
; Force field parameters, looked up by type
#define CX_CX_BOND 0.1400 392459.2

[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
1 2 yes 0.5 0.8333333333

[ atomtypes ]
; name at.num mass charge ptype sigma epsilon
CX 6 12.011 0.0 A 0.339967 0.359824
HX 1 1.008 0.0 A 0.264953 0.065689
NA 11 22.990 0.0 A 0.243928 0.445605
CL 17 35.450 0.0 A 0.447766 0.030543

[ bondtypes ]
CX CX 1 CX_CX_BOND
CX HX 1 0.1080 307105.6

[ angletypes ]
CX CX CX 1 108.0 527.184
CX CX HX 1 120.0 292.88

[ dihedraltypes ]
X CX CX X 9 180.0 15.16700 2
; Specific terms take precedence over wildcards, and stack.
CX CX CX CX 9 0.0 1.0 3
CX CX CX CX 9 180.0 0.5 2
X X CX HX 4 180.0 4.60240 2
";

const GMX_TOP: &str = "\
#include \"ff.itp\"

[ moleculetype ]
; A 5-membered ring with a substituent
LIG 3

[ atoms ]
1 CX 1 LIG C1 1 -0.30 12.011
2 CX 1 LIG C2 1  0.20 12.011
3 CX 1 LIG C3 1 -0.15 12.011
4 CX 1 LIG C4 1  0.25 12.011
5 CX 1 LIG C5 1 -0.20 12.011
6 HX 1 LIG H6 1  0.20 1.008

[ bonds ]
1 2 1
2 3 1
3 4 1
4 5 1
5 1 1
1 6 1

[ pairs ]
6 3 1
6 4 1

[ angles ]
5 1 2 1
1 2 3 1
2 3 4 1
3 4 5 1
4 5 1 1
6 1 2 1
6 1 5 1

[ dihedrals ]
1 2 3 4 9
2 3 4 5 9
3 4 5 1 9
4 5 1 2 9
5 1 2 3 9
6 1 2 3 9
6 1 5 4 9
2 5 1 6 4

#ifdef POSRES
[ position_restraints ]
1 1 1000 1000 1000
#endif

[ moleculetype ]
NA 1
[ atoms ]
1 NA 1 NA NA 1 1.0 22.990

[ moleculetype ]
CL 1
[ atoms ]
1 CL 1 CL CL 1 -1.0 35.450

[ system ]
Import test

[ molecules ]
LIG 2
NA 1
CL 1
";

/// Coordinates (nm) for two ligand copies and two ions, in a 4 nm box.
fn gmx_gro() -> String {
    let mut posits = Vec::new();
    for (i, center) in [Vec3::new(1.6, 2.0, 2.0), Vec3::new(2.4, 2.0, 2.1)]
        .into_iter()
        .enumerate()
    {
        let radius = 0.14 / (2. * (std::f32::consts::PI / 5.).sin());
        for k in 0..5 {
            let θ = 2. * std::f32::consts::PI * k as f32 / 5. + i as f32;
            // Out of plane, so dihedrals have torques.
            let z = 0.02 * if k % 2 == 0 { 1. } else { -1. };
            posits.push(center + Vec3::new(radius * θ.cos(), radius * θ.sin(), z));
        }
        let out = (posits[posits.len() - 5] - center).to_normalized();
        posits.push(posits[posits.len() - 5] + out * 0.108 + Vec3::new(0., 0., 0.03));
    }
    posits.push(Vec3::new(2.0, 1.4, 2.0));
    posits.push(Vec3::new(2.0, 2.6, 1.9));

    let names = ["C1", "C2", "C3", "C4", "C5", "H6"];
    let mut text = format!("Import test\n{:>5}\n", posits.len());
    for (i, p) in posits.iter().enumerate() {
        let (res, resname, name) = match i {
            0..6 => (1, "LIG", names[i]),
            6..12 => (2, "LIG", names[i - 6]),
            12 => (3, "NA", "NA"),
            _ => (4, "CL", "CL"),
        };
        text.push_str(&format!(
            "{res:>5}{resname:<5}{name:>5}{:>5}{:>8.3}{:>8.3}{:>8.3}\n",
            i + 1,
            p.x,
            p.y,
            p.z
        ));
    }
    text.push_str("   4.00000   4.00000   4.00000\n");
    text
}

/// A GROMACS topology using includes, macros, type tables (including stacked and wildcard
/// dihedral types, and an improper), explicit 1-4 pairs in a 5-membered ring, several copies of a
/// molecule, and ions. Compared to GROMACS' energy and forces for the same files.
#[test]
fn gromacs_topology_import_matches_gromacs() {
    let gro = gmx_gro();

    let dir = ScratchDir::new("import_inputs");
    fs::write(dir.path().join("ff.itp"), GMX_FF_ITP).unwrap();
    fs::write(dir.path().join("topol.top"), GMX_TOP).unwrap();
    fs::write(dir.path().join("conf.gro"), &gro).unwrap();

    let imported = ImportedSystem::from_gromacs_files(
        &dir.path().join("topol.top"),
        &dir.path().join("conf.gro"),
        &[],
    )
    .unwrap();

    assert_eq!(imported.mols.len(), 4);
    let lig = imported.mols[0].explicit_params.as_ref().unwrap();
    assert_eq!(lig.bonds.len(), 6);
    // 5 single ring terms, 2 stacked terms each for 5 ring dihedrals, and 2 wildcard matches.
    assert_eq!(lig.dihedrals.len(), 12);
    assert_eq!(lig.impropers.len(), 1);
    assert_eq!(lig.pairs_14.len(), 2);
    // 6 bonds, 7 angles; ring 1-3 pairs are excluded, not 1-4.
    assert_eq!(lig.exclusions.len(), 6 + 7);

    let reference = run_reference_files(
        "import",
        &[
            ("ff.itp", GMX_FF_ITP),
            ("topol.top", GMX_TOP),
            ("conf.gro", &gro),
        ],
        reference_mdp(pme_coulomb(), VdwModifier::None),
    );

    let state = evaluate(&imported);
    let forces: Vec<_> = state.atoms.iter().map(|a| a.force).collect();

    assert_system_close(
        &forces,
        state.potential_energy as f32,
        &reference,
        0.015,
        5e-3,
        0.015,
        5e-3,
        "imported GROMACS topology",
    );
}

/// CHARMM's terms in a GROMACS topology: A Urey-Bradley angle, and a harmonic improper.
#[test]
fn gromacs_charmm_terms_match_gromacs() {
    let top = GMX_TOP
        .replace("1 2 3 1\n", "1 2 3 5 108.0 400.0 0.23 20000.0\n")
        .replace("2 5 1 6 4\n", "2 5 1 6 2 5.0 300.0\n");
    let gro = gmx_gro();

    let dir = ScratchDir::new("import_charmm_inputs");
    fs::write(dir.path().join("ff.itp"), GMX_FF_ITP).unwrap();
    fs::write(dir.path().join("topol.top"), &top).unwrap();
    fs::write(dir.path().join("conf.gro"), &gro).unwrap();

    let imported = ImportedSystem::from_gromacs_files(
        &dir.path().join("topol.top"),
        &dir.path().join("conf.gro"),
        &[],
    )
    .unwrap();

    let lig = imported.mols[0].explicit_params.as_ref().unwrap();
    assert_eq!(lig.urey_bradley.len(), 1);
    assert_eq!(lig.harmonic_impropers.len(), 1);
    assert!(lig.impropers.is_empty());

    let reference = run_reference_files(
        "import_charmm",
        &[
            ("ff.itp", GMX_FF_ITP),
            ("topol.top", &top),
            ("conf.gro", &gro),
        ],
        reference_mdp(pme_coulomb(), VdwModifier::None),
    );

    let state = evaluate(&imported);
    let forces: Vec<_> = state.atoms.iter().map(|a| a.force).collect();

    assert_system_close(
        &forces,
        state.potential_energy as f32,
        &reference,
        0.015,
        5e-3,
        0.015,
        5e-3,
        "GROMACS topology with CHARMM terms",
    );
}

/// Water models from GROMACS topologies: 3-site with SETTLE, and 4-site with a virtual site.
#[test]
fn gromacs_water_models() {
    let tip3p = "\
[ defaults ]
1 2 yes 0.5 0.8333
[ atomtypes ]
OW 8 15.9994 0.0 A 0.315061 0.636386
HW 1 1.008 0.0 A 0.0 0.0
[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW 1 -0.834 15.9994
2 HW 1 SOL HW1 1 0.417 1.008
3 HW 1 SOL HW2 1 0.417 1.008
[ settles ]
1 1 0.09572 0.15139
[ exclusions ]
1 2 3
2 1 3
3 1 2
[ system ]
Water
[ molecules ]
SOL 1
";
    let gro = "Water\n    3\n    1SOL     OW    1   1.000   1.000   1.000\n    1SOL    HW1    2   1.096   1.000   1.000\n    1SOL    HW2    3   0.976   1.093   1.000\n   3.0 3.0 3.0\n";

    let top = GromacsTopology::new(tip3p).unwrap();
    let gro = bio_files::gromacs::gro::Gro::new(gro).unwrap();
    let imported = ImportedSystem::from_gromacs(&top, &gro).unwrap();

    assert!(imported.mols.is_empty());
    assert_eq!(imported.water_posits.len(), 1);
    let w = imported.water_model.unwrap();
    assert!((w.o_h_dist - 0.9572).abs() < 1e-4);
    assert!((w.h_o_h_angle.to_degrees() - 104.52).abs() < 0.01);
    assert_eq!(w.o_m_dist, 0.);
    assert!((w.q_h - 0.417).abs() < 1e-6);
    assert!((w.lj_sigma_o - 3.15061).abs() < 1e-4);
    assert!((w.lj_eps_o - 0.1521).abs() < 1e-4);

    // TIP4P-Ew: M is a bisector virtual site; O is uncharged.
    let tip4p = tip3p
        .replace("HW 1 1.008 0.0 A 0.0 0.0", "HW 1 1.008 0.0 A 0.0 0.0\nMW 0 0.0 0.0 D 0.0 0.0")
        .replace("1 OW 1 SOL OW 1 -0.834 15.9994", "1 OW 1 SOL OW 1 0.0 15.9994")
        .replace("2 HW 1 SOL HW1 1 0.417 1.008", "2 HW 1 SOL HW1 1 0.52422 1.008")
        .replace(
            "3 HW 1 SOL HW2 1 0.417 1.008",
            "3 HW 1 SOL HW2 1 0.52422 1.008\n4 MW 1 SOL MW 1 -1.04844 0.0\n[ virtual_sites3 ]\n4 1 2 3 1 0.106676721 0.106676721",
        )
        .replace("1 2 3\n2 1 3\n3 1 2", "1 2 3 4\n2 1 3 4\n3 1 2 4\n4 1 2 3");
    let gro4 = "Water\n    4\n    1SOL     OW    1   1.000   1.000   1.000\n    1SOL    HW1    2   1.096   1.000   1.000\n    1SOL    HW2    3   0.976   1.093   1.000\n    1SOL     MW    4   1.010   1.012   1.000\n   3.0 3.0 3.0\n";

    let top = GromacsTopology::new(&tip4p).unwrap();
    let gro = bio_files::gromacs::gro::Gro::new(gro4).unwrap();
    let w = ImportedSystem::from_gromacs(&top, &gro)
        .unwrap()
        .water_model
        .unwrap();
    // TIP4P-Ew's O-M distance is 0.125 Å.
    assert!((w.o_m_dist - 0.125).abs() < 1e-3, "O-M: {}", w.o_m_dist);
    assert!((w.q_h - 0.52422).abs() < 1e-6);
}

/// Terms we can't compute yet are reported, rather than dropped.
#[test]
fn gromacs_unsupported_terms_are_errors() {
    let top = GMX_TOP.replace("1 2 3 4 9\n", "1 2 3 4 3 1 2 3 4 5 6\n");
    let dir = ScratchDir::new("import_unsupported");
    fs::write(dir.path().join("ff.itp"), GMX_FF_ITP).unwrap();
    fs::write(dir.path().join("topol.top"), top).unwrap();
    fs::write(dir.path().join("conf.gro"), gmx_gro()).unwrap();

    let err = ImportedSystem::from_gromacs_files(
        &dir.path().join("topol.top"),
        &dir.path().join("conf.gro"),
        &[],
    )
    .unwrap_err();
    assert!(
        err.descrip.contains("Ryckaert-Bellemans"),
        "{}",
        err.descrip
    );
}
