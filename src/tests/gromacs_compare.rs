//! Direct force and energy comparisons against GROMACS.
//!
//! These tests intentionally require gmx on PATH. GROMACS is treated as the
//! reference implementation. Each run uses a zero-step `mdrun`, so no
//! integration, thermostat, minimization, or random velocities can hide a
//! force-field discrepancy.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use bio_files::{
    AtomGeneric, BondGeneric, BondType, FrameSlice,
    gromacs::{
        MdpParams, OutputControl, OutputEnergy,
        mdp::{
            Barostat as GmxBarostat, Constraints, CoulombType, Integrator as GmxIntegrator, Pbc,
            PmeConfig, Thermostat, VdwModifier, VdwType,
        },
        run_gmx,
        trr::read_trr,
    },
    md_params::{AngleBendingParams, BondStretchingParams, DihedralParams},
};
use ewald::force_coulomb_short_range;
use lin_alg::f32::Vec3;
use na_seq::Element;

use crate::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MdState, MolDynamics,
    SimBoxInit, Solvent,
    barostat::SimBox,
    bonded_forces::{f_angle_bending, f_bond_stretching, f_dihedral},
    forces::force_e_lj,
    non_bonded::CHARGE_UNIT_SCALER,
    params::FfParamSet,
};

const BOX_LEN_A: f32 = 40.0;
const COULOMB_CUTOFF_A: f32 = 10.0;
const PME_ALPHA_A_INV: f32 = 0.35;
const PME_SPACING_A: f32 = 0.8;
const KCAL_TO_KJ: f32 = 4.184;
const GMX_FORCE_TO_DYNAMICS: f32 = 1.0 / (KCAL_TO_KJ * 10.0);

static SCRATCH_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct GmxAtom {
    atom_type: &'static str,
    charge_e: f32,
    mass_amu: f32,
    sigma_a: f32,
    epsilon_kcal: f32,
}

#[derive(Clone)]
struct GmxBond {
    atoms: (usize, usize),
    r0_a: f32,
    k_amber: f32,
}

#[derive(Clone)]
struct GmxAngle {
    atoms: (usize, usize, usize),
    theta0_rad: f32,
    k_amber: f32,
}

#[derive(Clone)]
struct GmxDihedral {
    atoms: (usize, usize, usize, usize),
    phase_rad: f32,
    barrier_kcal: f32,
    periodicity: u8,
}

#[derive(Clone)]
struct GmxMolecule {
    name: &'static str,
    atoms: Vec<GmxAtom>,
    bonds: Vec<GmxBond>,
    angles: Vec<GmxAngle>,
    dihedrals: Vec<GmxDihedral>,
}

struct GmxReference {
    forces: Vec<Vec3>,
    potential_energy_kcal: f32,
}

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(label: &str) -> Self {
        let sequence = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dynamics_gromacs_compare_{}_{}_{}",
            std::process::id(),
            sequence,
            label
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("Could not remove GROMACS comparison scratch directory: {error}");
        }
    }
}

fn assert_close(got: f32, expected: f32, rel_tol: f32, abs_tol: f32, label: &str) {
    let error = (got - expected).abs();
    let limit = abs_tol.max(rel_tol * expected.abs());
    assert!(
        error <= limit,
        "{label}: Dynamics={got:.7}, GROMACS={expected:.7}, error={error:.3e}, limit={limit:.3e}"
    );
}

fn assert_vec_close(got: Vec3, expected: Vec3, rel_tol: f32, abs_tol: f32, label: &str) {
    for (axis, got, expected) in [
        ("x", got.x, expected.x),
        ("y", got.y, expected.y),
        ("z", got.z, expected.z),
    ] {
        assert_close(got, expected, rel_tol, abs_tol, &format!("{label}.{axis}"));
    }
}

fn assert_system_close(
    dynamics_forces: &[Vec3],
    dynamics_energy: f32,
    gromacs: &GmxReference,
    force_rel_tol: f32,
    force_abs_tol: f32,
    energy_rel_tol: f32,
    energy_abs_tol: f32,
    label: &str,
) {
    assert_eq!(dynamics_forces.len(), gromacs.forces.len());
    for (i, (dynamics, reference)) in dynamics_forces.iter().zip(&gromacs.forces).enumerate() {
        assert_vec_close(
            *dynamics,
            *reference,
            force_rel_tol,
            force_abs_tol,
            &format!("{label} atom {i}"),
        );
    }
    assert_close(
        dynamics_energy,
        gromacs.potential_energy_kcal,
        energy_rel_tol,
        energy_abs_tol,
        &format!("{label} potential energy"),
    );
}

fn reference_mdp(coulomb: CoulombType, vdw_modifier: VdwModifier) -> MdpParams {
    MdpParams {
        integrator: GmxIntegrator::Md,
        nsteps: 0,
        dt: 0.001,
        output_control: OutputControl {
            nstxout: Some(1),
            nstvout: None,
            nstfout: Some(1),
            nstlog: Some(1),
            nstcalcenergy: Some(1),
            nstenergy: Some(1),
            nstxout_compressed: None,
            compressed_x_precision: 1_000,
        },
        coulombtype: coulomb,
        rcoulomb: COULOMB_CUTOFF_A * 0.1,
        vdwtype: VdwType::CutOff,
        vdw_modifier,
        rvdw: COULOMB_CUTOFF_A * 0.1,
        thermostat: Thermostat::No,
        tau_t: vec![1.0],
        ref_t: vec![300.0],
        pcoupl: GmxBarostat::No,
        pbc: Pbc::Xyz,
        deform: None,
        deform_init_flow: false,
        gen_vel: false,
        gen_temp: 300.0,
        gen_seed: Some(1),
        constraints: Constraints::None,
        free_energy_calculations: Default::default(),
    }
}

fn pme_coulomb() -> CoulombType {
    CoulombType::Pme(PmeConfig {
        fourierspacing: PME_SPACING_A * 0.1,
        order: 4,
        alpha: PME_ALPHA_A_INV * 10.0,
        rtol_lj: 1e-3,
        epsilon_surface: None,
    })
}

fn make_gro(posits_a: &[Vec3], molecule_sizes: &[usize]) -> String {
    assert_eq!(posits_a.len(), molecule_sizes.iter().sum::<usize>());
    let mut text = format!("Dynamics/GROMACS force comparison\n{:>5}\n", posits_a.len());
    let mut atom_i = 0;
    for (mol_i, count) in molecule_sizes.iter().enumerate() {
        for local_i in 0..*count {
            let p = posits_a[atom_i] * 0.1;
            text.push_str(&format!(
                "{:>5}{:<5}{:>5}{:>5}{:>8.3}{:>8.3}{:>8.3}\n",
                mol_i + 1,
                format!("M{}", mol_i + 1),
                format!("A{}", local_i + 1),
                atom_i + 1,
                p.x,
                p.y,
                p.z,
            ));
            atom_i += 1;
        }
    }
    text.push_str(&format!(
        "{:>10.5}{:>10.5}{:>10.5}\n",
        BOX_LEN_A * 0.1,
        BOX_LEN_A * 0.1,
        BOX_LEN_A * 0.1,
    ));
    text
}

fn make_topology(molecules: &[GmxMolecule]) -> String {
    let mut atom_types: BTreeMap<&str, (f32, f32, f32)> = BTreeMap::new();
    for molecule in molecules {
        for atom in &molecule.atoms {
            if let Some(previous) = atom_types.insert(
                atom.atom_type,
                (atom.mass_amu, atom.sigma_a, atom.epsilon_kcal),
            ) {
                assert_eq!(previous, (atom.mass_amu, atom.sigma_a, atom.epsilon_kcal));
            }
        }
    }

    let mut text = String::from(
        "; Reference topology generated by dynamics tests\n\n\
         [ defaults ]\n\
         ; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n\
         1 2 yes 0.5 0.8333333333\n\n\
         [ atomtypes ]\n\
         ; name at.num mass charge ptype sigma(nm) epsilon(kJ/mol)\n",
    );
    for (name, (mass, sigma_a, epsilon_kcal)) in atom_types {
        text.push_str(&format!(
            "{name:<8} 6 {mass:.8} 0.0 A {:.10e} {:.10e}\n",
            sigma_a * 0.1,
            epsilon_kcal * KCAL_TO_KJ,
        ));
    }
    text.push('\n');

    for molecule in molecules {
        text.push_str(&format!(
            "[ moleculetype ]\n; name nrexcl\n{} 3\n\n[ atoms ]\n",
            molecule.name
        ));
        for (i, atom) in molecule.atoms.iter().enumerate() {
            text.push_str(&format!(
                "{} {} 1 {} A{} {} {:.9} {:.8}\n",
                i + 1,
                atom.atom_type,
                molecule.name,
                i + 1,
                i + 1,
                atom.charge_e,
                atom.mass_amu,
            ));
        }
        text.push('\n');

        if !molecule.bonds.is_empty() {
            text.push_str("[ bonds ]\n; ai aj funct r0(nm) k(kJ/mol/nm^2)\n");
            for bond in &molecule.bonds {
                // GROMACS: V = 1/2 k dr^2. Amber/Dynamics: V = k_b dr^2.
                let k_gmx = 2.0 * bond.k_amber * KCAL_TO_KJ * 100.0;
                text.push_str(&format!(
                    "{} {} 1 {:.9} {:.9}\n",
                    bond.atoms.0,
                    bond.atoms.1,
                    bond.r0_a * 0.1,
                    k_gmx,
                ));
            }
            text.push('\n');
        }

        if !molecule.angles.is_empty() {
            text.push_str("[ angles ]\n; ai aj ak funct theta0(deg) k(kJ/mol/rad^2)\n");
            for angle in &molecule.angles {
                // GROMACS: V = 1/2 k dtheta^2. Amber/Dynamics: V = k dtheta^2.
                let k_gmx = 2.0 * angle.k_amber * KCAL_TO_KJ;
                text.push_str(&format!(
                    "{} {} {} 1 {:.9} {:.9}\n",
                    angle.atoms.0,
                    angle.atoms.1,
                    angle.atoms.2,
                    angle.theta0_rad.to_degrees(),
                    k_gmx,
                ));
            }
            text.push('\n');
        }

        if !molecule.dihedrals.is_empty() {
            text.push_str("[ dihedrals ]\n; ai aj ak al funct phase k mult\n");
            for dihedral in &molecule.dihedrals {
                text.push_str(&format!(
                    "{} {} {} {} 9 {:.9} {:.9} {}\n",
                    dihedral.atoms.0,
                    dihedral.atoms.1,
                    dihedral.atoms.2,
                    dihedral.atoms.3,
                    dihedral.phase_rad.to_degrees(),
                    dihedral.barrier_kcal * KCAL_TO_KJ,
                    dihedral.periodicity,
                ));
            }
            text.push('\n');
        }
    }

    text.push_str("[ system ]\nDynamics reference system\n\n[ molecules ]\n");
    for molecule in molecules {
        text.push_str(&format!("{} 1\n", molecule.name));
    }
    text
}

fn run_reference(
    label: &str,
    posits: &[Vec3],
    molecules: &[GmxMolecule],
    mdp: MdpParams,
) -> GmxReference {
    let scratch = ScratchDir::new(label);
    let dir = scratch.path();
    let molecule_sizes: Vec<_> = molecules.iter().map(|m| m.atoms.len()).collect();

    fs::write(dir.join("conf.gro"), make_gro(posits, &molecule_sizes)).unwrap();
    fs::write(dir.join("topol.top"), make_topology(molecules)).unwrap();
    fs::write(dir.join("reference.mdp"), mdp.to_mdp_str()).unwrap();

    run_gmx(
        dir,
        &[
            "grompp",
            "-f",
            "reference.mdp",
            "-c",
            "conf.gro",
            "-p",
            "topol.top",
            "-o",
            "reference.tpr",
            "-maxwarn",
            "1",
        ],
    )
    .unwrap();
    run_gmx(
        dir,
        &[
            "mdrun",
            "-s",
            "reference.tpr",
            "-o",
            "reference.trr",
            "-e",
            "reference.edr",
            "-g",
            "reference.log",
            "-ntmpi",
            "1",
        ],
    )
    .unwrap();

    let frames = read_trr(
        &dir.join("reference.trr"),
        FrameSlice::Index {
            start: None,
            end: None,
        },
    )
    .unwrap();
    let frame = frames.first().expect("GROMACS must write a force frame");
    assert_eq!(frame.atom_forces.len(), posits.len());
    let forces = frame
        .atom_forces
        .iter()
        .map(|f| {
            Vec3::new(
                f.x as f32 * GMX_FORCE_TO_DYNAMICS,
                f.y as f32 * GMX_FORCE_TO_DYNAMICS,
                f.z as f32 * GMX_FORCE_TO_DYNAMICS,
            )
        })
        .collect();

    let energies = OutputEnergy::from_edr(&dir.join("reference.edr")).unwrap();
    let potential_energy_kcal = energies
        .first()
        .and_then(|energy| energy.potential_energy)
        .expect("GROMACS must write potential energy")
        / KCAL_TO_KJ;

    GmxReference {
        forces,
        potential_energy_kcal,
    }
}

fn inert_atom(atom_type: &'static str) -> GmxAtom {
    GmxAtom {
        atom_type,
        charge_e: 0.0,
        mass_amu: 12.0,
        sigma_a: 3.4,
        epsilon_kcal: 0.0,
    }
}

#[test]
fn gromacs_bond_matches_dynamics() {
    let cell = SimBox::new(Vec3::new_zero(), Vec3::splat(BOX_LEN_A));
    let params = BondStretchingParams {
        atom_types: ("x".into(), "x".into()),
        k_b: 120.0,
        r_0: 1.5,
        comment: None,
    };
    let posits = [Vec3::new(19.1, 20.0, 20.0), Vec3::new(20.9, 20.0, 20.0)];
    let (force0, energy) = f_bond_stretching(posits[0], posits[1], &params, &cell);
    let molecule = GmxMolecule {
        name: "BOND",
        atoms: vec![inert_atom("XB"), inert_atom("XB")],
        bonds: vec![GmxBond {
            atoms: (1, 2),
            r0_a: params.r_0,
            k_amber: params.k_b,
        }],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };
    let reference = run_reference(
        "bond",
        &posits,
        &[molecule],
        reference_mdp(CoulombType::CutOff, VdwModifier::None),
    );

    assert_system_close(
        &[force0, -force0],
        energy,
        &reference,
        2e-4,
        2e-4,
        2e-4,
        2e-4,
        "bond",
    );
}

#[test]
fn gromacs_angle_matches_dynamics() {
    let cell = SimBox::new(Vec3::new_zero(), Vec3::splat(BOX_LEN_A));
    let params = AngleBendingParams {
        atom_types: ("x".into(), "x".into(), "x".into()),
        k: 55.0,
        theta_0: 110.0_f32.to_radians(),
        comment: None,
    };
    let posits = [
        Vec3::new(18.8, 20.1, 20.0),
        Vec3::new(20.0, 20.0, 20.0),
        Vec3::new(20.4, 21.3, 20.2),
    ];
    let ((f0, f1, f2), energy) = f_angle_bending(posits[0], posits[1], posits[2], &params, &cell);
    let molecule = GmxMolecule {
        name: "ANGLE",
        atoms: vec![inert_atom("XA"), inert_atom("XA"), inert_atom("XA")],
        bonds: Vec::new(),
        angles: vec![GmxAngle {
            atoms: (1, 2, 3),
            theta0_rad: params.theta_0,
            k_amber: params.k,
        }],
        dihedrals: Vec::new(),
    };
    let reference = run_reference(
        "angle",
        &posits,
        &[molecule],
        reference_mdp(CoulombType::CutOff, VdwModifier::None),
    );

    assert_system_close(
        &[f0, f1, f2],
        energy,
        &reference,
        8e-4,
        8e-4,
        8e-4,
        8e-4,
        "angle",
    );
}

#[test]
fn gromacs_dihedral_matches_dynamics() {
    let cell = SimBox::new(Vec3::new_zero(), Vec3::splat(BOX_LEN_A));
    let params = [DihedralParams {
        atom_types: ("x".into(), "x".into(), "x".into(), "x".into()),
        divider: 1,
        barrier_height: 2.5,
        phase: 0.4,
        periodicity: 3,
        comment: None,
    }];
    let posits = [
        Vec3::new(19.0, 20.2, 20.4),
        Vec3::new(20.0, 20.0, 20.0),
        Vec3::new(21.1, 20.3, 19.9),
        Vec3::new(21.8, 21.2, 20.7),
    ];
    let ((f0, f1, f2, f3), energy) =
        f_dihedral(posits[0], posits[1], posits[2], posits[3], &params, &cell);
    let molecule = GmxMolecule {
        name: "DIHED",
        atoms: vec![
            inert_atom("XD"),
            inert_atom("XD"),
            inert_atom("XD"),
            inert_atom("XD"),
        ],
        bonds: Vec::new(),
        angles: Vec::new(),
        dihedrals: vec![GmxDihedral {
            atoms: (1, 2, 3, 4),
            phase_rad: params[0].phase,
            barrier_kcal: params[0].barrier_height,
            periodicity: params[0].periodicity,
        }],
    };
    let reference = run_reference(
        "dihedral",
        &posits,
        &[molecule],
        reference_mdp(CoulombType::CutOff, VdwModifier::None),
    );

    assert_system_close(
        &[f0, f1, f2, f3],
        energy,
        &reference,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        "dihedral",
    );
}

#[test]
fn gromacs_short_range_lj_matches_dynamics() {
    let sigma = 3.4;
    let epsilon = 0.086;
    let distance = 4.5;
    let posits = [Vec3::new(17.75, 20.0, 20.0), Vec3::new(22.25, 20.0, 20.0)];
    let dir = Vec3::new(-1.0, 0.0, 0.0);
    let (force0, energy) = force_e_lj(dir, 1.0 / distance, sigma, epsilon);
    let atom = GmxAtom {
        atom_type: "XLJ",
        charge_e: 0.0,
        mass_amu: 12.0,
        sigma_a: sigma,
        epsilon_kcal: epsilon,
    };
    let molecules = [
        GmxMolecule {
            name: "LJA",
            atoms: vec![atom.clone()],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
        GmxMolecule {
            name: "LJB",
            atoms: vec![atom],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
    ];
    let reference = run_reference(
        "lj",
        &posits,
        &molecules,
        reference_mdp(CoulombType::CutOff, VdwModifier::None),
    );

    assert_system_close(
        &[force0, -force0],
        energy,
        &reference,
        3e-4,
        2e-5,
        3e-4,
        2e-5,
        "short-range LJ",
    );
}

#[test]
fn gromacs_cutoff_coulomb_matches_dynamics() {
    let distance = 5.0;
    let q0_e = 0.6;
    let q1_e = -0.6;
    let posits = [Vec3::new(17.5, 20.0, 20.0), Vec3::new(22.5, 20.0, 20.0)];
    let dir = Vec3::new(-1.0, 0.0, 0.0);
    let (force0, energy) = force_coulomb_short_range(
        dir,
        distance,
        1.0 / distance,
        q0_e * CHARGE_UNIT_SCALER,
        q1_e * CHARGE_UNIT_SCALER,
        COULOMB_CUTOFF_A,
        0.0,
    );
    let molecules = [
        GmxMolecule {
            name: "CQA",
            atoms: vec![GmxAtom {
                atom_type: "XQ",
                charge_e: q0_e,
                mass_amu: 12.0,
                sigma_a: 3.4,
                epsilon_kcal: 0.0,
            }],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
        GmxMolecule {
            name: "CQB",
            atoms: vec![GmxAtom {
                atom_type: "XQ",
                charge_e: q1_e,
                mass_amu: 12.0,
                sigma_a: 3.4,
                epsilon_kcal: 0.0,
            }],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
    ];
    let reference = run_reference(
        "cutoff_coulomb",
        &posits,
        &molecules,
        reference_mdp(CoulombType::CutOff, VdwModifier::None),
    );

    assert_system_close(
        &[force0, -force0],
        energy,
        &reference,
        3e-4,
        2e-4,
        3e-4,
        2e-4,
        "cutoff Coulomb",
    );
}

fn dynamics_atom(serial_number: u32, posit: Vec3, partial_charge: f32) -> AtomGeneric {
    AtomGeneric {
        serial_number,
        posit: posit.into(),
        force_field_type: Some("ca".to_string()),
        element: Element::Carbon,
        partial_charge: Some(partial_charge),
        ..Default::default()
    }
}

fn dynamics_config(overrides: MdOverrides) -> MdConfig {
    MdConfig {
        integrator: Integrator::VerletVelocity { thermostat: None },
        sim_box: SimBoxInit::Fixed((Vec3::new_zero(), Vec3::splat(BOX_LEN_A))),
        solvent: Solvent::None,
        barostat_cfg: None,
        max_init_relaxation_iters: None,
        recenter_sim_box: false,
        spme_mesh_spacing: PME_SPACING_A,
        spme_alpha: PME_ALPHA_A_INV,
        coulomb_cutoff: COULOMB_CUTOFF_A,
        lj_cutoff: COULOMB_CUTOFF_A,
        overrides: MdOverrides {
            skip_counterion_insertion: true,
            ..overrides
        },
        ..Default::default()
    }
}

fn evaluate_dynamics_pair(q0: f32, q1: f32, distance: f32) -> MdState {
    let center = BOX_LEN_A / 2.0;
    let mols = [
        MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: vec![dynamics_atom(
                1,
                Vec3::new(center - distance / 2.0, center, center),
                q0,
            )],
            ..Default::default()
        },
        MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: vec![dynamics_atom(
                2,
                Vec3::new(center + distance / 2.0, center, center),
                q1,
            )],
            ..Default::default()
        },
    ];
    let params = FfParamSet::new_amber().unwrap();
    let config = dynamics_config(MdOverrides {
        bonded_disabled: true,
        lj_disabled: true,
        ..Default::default()
    });
    let (mut state, _) = MdState::new(&ComputationDevice::Cpu, &config, &mols, &params).unwrap();
    state.reset_f_acc_pe_virial();
    state.apply_all_forces(&ComputationDevice::Cpu, &None);
    state
}

#[test]
fn gromacs_spme_matches_dynamics() {
    let distance = 5.0;
    let q0 = 1.0;
    let q1 = -1.0;
    let state = evaluate_dynamics_pair(q0, q1, distance);
    let posits: Vec<_> = state.atoms.iter().map(|atom| atom.posit).collect();
    let molecules = [
        GmxMolecule {
            name: "PMEA",
            atoms: vec![GmxAtom {
                atom_type: "XP",
                charge_e: q0,
                mass_amu: state.atoms[0].mass,
                sigma_a: state.atoms[0].lj_sigma,
                epsilon_kcal: 0.0,
            }],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
        GmxMolecule {
            name: "PMEB",
            atoms: vec![GmxAtom {
                atom_type: "XP",
                charge_e: q1,
                mass_amu: state.atoms[1].mass,
                sigma_a: state.atoms[1].lj_sigma,
                epsilon_kcal: 0.0,
            }],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
    ];
    let reference = run_reference(
        "spme",
        &posits,
        &molecules,
        reference_mdp(pme_coulomb(), VdwModifier::None),
    );
    let forces: Vec<_> = state.atoms.iter().map(|atom| atom.force).collect();

    assert_system_close(
        &forces,
        state.potential_energy_nonbonded as f32,
        &reference,
        0.012,
        3e-3,
        0.012,
        3e-3,
        "SPME",
    );
}

fn evaluate_combined_dynamics() -> MdState {
    let center = BOX_LEN_A / 2.0;
    let bonded = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![
            dynamics_atom(1, Vec3::new(center - 0.8, center, center), 0.4),
            dynamics_atom(2, Vec3::new(center + 0.8, center, center), 0.0),
        ],
        bonds: vec![BondGeneric {
            atom_0_sn: 1,
            atom_1_sn: 2,
            bond_type: BondType::Aromatic,
        }],
        ..Default::default()
    };
    let isolated = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![dynamics_atom(
            3,
            Vec3::new(center, center + 5.0, center),
            -0.4,
        )],
        ..Default::default()
    };
    let params = FfParamSet::new_amber().unwrap();
    let (mut state, _) = MdState::new(
        &ComputationDevice::Cpu,
        &dynamics_config(MdOverrides::default()),
        &[bonded, isolated],
        &params,
    )
    .unwrap();
    state.reset_f_acc_pe_virial();
    state.apply_all_forces(&ComputationDevice::Cpu, &None);
    state
}

#[test]
fn gromacs_combined_bonded_lj_and_spme_matches_dynamics() {
    let state = evaluate_combined_dynamics();
    let bond = state
        .force_field_params
        .bond_stretching
        .get(&(0, 1))
        .expect("combined fixture must have a bond");
    let gmx_atoms: Vec<_> = state
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| GmxAtom {
            atom_type: match i {
                0 => "XC0",
                1 => "XC1",
                _ => "XC2",
            },
            charge_e: atom.partial_charge / CHARGE_UNIT_SCALER,
            mass_amu: atom.mass,
            sigma_a: atom.lj_sigma,
            epsilon_kcal: atom.lj_eps,
        })
        .collect();
    let molecules = [
        GmxMolecule {
            name: "BONDED",
            atoms: gmx_atoms[0..2].to_vec(),
            bonds: vec![GmxBond {
                atoms: (1, 2),
                r0_a: bond.r_0,
                k_amber: bond.k_b,
            }],
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
        GmxMolecule {
            name: "ISOL",
            atoms: vec![gmx_atoms[2].clone()],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
    ];
    let posits: Vec<_> = state.atoms.iter().map(|atom| atom.posit).collect();
    let reference = run_reference(
        "combined",
        &posits,
        &molecules,
        reference_mdp(pme_coulomb(), VdwModifier::None),
    );
    let forces: Vec<_> = state.atoms.iter().map(|atom| atom.force).collect();

    assert_system_close(
        &forces,
        state.potential_energy as f32,
        &reference,
        0.015,
        5e-3,
        0.015,
        5e-3,
        "combined",
    );
}
