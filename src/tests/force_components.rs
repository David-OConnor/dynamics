//! Small, deterministic reference systems for every force component.
//!
//! The low-level tests compare bonded terms with their analytic potentials or
//! numerical energy gradients. The end-to-end tests build normal `MdState`s so
//! they also exercise parameter assignment, exclusions, neighbor lists, PME,
//! and (when enabled) the CUDA kernels.

use bio_files::{
    AtomGeneric, BondGeneric, BondType,
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
    params::FfParamSet,
};

const BOX_LEN: f32 = 40.0;
const ALPHA: f32 = 0.35;
const COULOMB_CUTOFF: f32 = 10.0;

fn assert_close(got: f32, expected: f32, rel_tol: f32, abs_tol: f32, label: &str) {
    let err = (got - expected).abs();
    let limit = abs_tol.max(rel_tol * expected.abs());
    assert!(
        err <= limit,
        "{label}: got {got:.7}, expected {expected:.7}, error {err:.3e} exceeds {limit:.3e}"
    );
}

fn assert_vec_close(got: Vec3, expected: Vec3, rel_tol: f32, abs_tol: f32, label: &str) {
    for (axis, g, e) in [
        ("x", got.x, expected.x),
        ("y", got.y, expected.y),
        ("z", got.z, expected.z),
    ] {
        assert_close(g, e, rel_tol, abs_tol, &format!("{label}.{axis}"));
    }
}

fn shift(mut p: Vec3, axis: usize, delta: f32) -> Vec3 {
    match axis {
        0 => p.x += delta,
        1 => p.y += delta,
        2 => p.z += delta,
        _ => unreachable!(),
    }
    p
}

fn component(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => unreachable!(),
    }
}

#[test]
fn bond_force_energy_matches_amber_harmonic() {
    let cell = SimBox::new(Vec3::new_zero(), Vec3::splat(20.0));
    let params = BondStretchingParams {
        atom_types: ("x".into(), "x".into()),
        k_b: 120.0,
        r_0: 1.5,
        comment: None,
    };

    for r in [1.2_f32, 1.5, 1.8] {
        let (force, energy) = f_bond_stretching(
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(5.0 + r, 5.0, 5.0),
            &params,
            &cell,
        );
        let dr = r - params.r_0;
        assert_close(force.x, 2.0 * params.k_b * dr, 1e-6, 1e-5, "bond force");
        assert_close(energy, params.k_b * dr * dr, 1e-5, 3e-5, "bond energy");
        assert_close(force.y, 0.0, 0.0, 1e-7, "bond force y");
        assert_close(force.z, 0.0, 0.0, 1e-7, "bond force z");
    }
}

#[test]
fn bond_force_uses_minimum_image() {
    let cell = SimBox::new(Vec3::new_zero(), Vec3::splat(10.0));
    let params = BondStretchingParams {
        atom_types: ("x".into(), "x".into()),
        k_b: 100.0,
        r_0: 1.0,
        comment: None,
    };
    let (force, energy) = f_bond_stretching(
        Vec3::new(0.2, 5.0, 5.0),
        Vec3::new(9.0, 5.0, 5.0),
        &params,
        &cell,
    );

    assert_close(force.x, -40.0, 1e-5, 1e-4, "PBC bond force");
    assert_close(energy, 4.0, 1e-5, 1e-4, "PBC bond energy");
}

#[test]
fn angle_forces_match_energy_gradient_and_conserve_force() {
    let cell = SimBox::new(Vec3::splat(-10.0), Vec3::splat(10.0));
    let params = AngleBendingParams {
        atom_types: ("x".into(), "x".into(), "x".into()),
        k: 55.0,
        theta_0: 110.0_f32.to_radians(),
        comment: None,
    };
    let p = [
        Vec3::new(-1.2, 0.1, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.4, 1.3, 0.2),
    ];
    let ((f0, f1, f2), _) = f_angle_bending(p[0], p[1], p[2], &params, &cell);
    let forces = [f0, f1, f2];
    assert_vec_close(f0 + f1 + f2, Vec3::new_zero(), 0.0, 2e-5, "angle net force");

    let delta = 1e-3;
    for atom in 0..3 {
        for axis in 0..3 {
            let mut plus = p;
            let mut minus = p;
            plus[atom] = shift(plus[atom], axis, delta);
            minus[atom] = shift(minus[atom], axis, -delta);
            let (_, e_plus) = f_angle_bending(plus[0], plus[1], plus[2], &params, &cell);
            let (_, e_minus) = f_angle_bending(minus[0], minus[1], minus[2], &params, &cell);
            let numerical = -(e_plus - e_minus) / (2.0 * delta);
            assert_close(
                component(forces[atom], axis),
                numerical,
                0.025,
                3e-3,
                &format!("angle force atom {atom} axis {axis}"),
            );
        }
    }
}

#[test]
fn dihedral_forces_match_energy_gradient_and_conserve_force() {
    let cell = SimBox::new(Vec3::splat(-10.0), Vec3::splat(10.0));
    let params = [DihedralParams {
        atom_types: ("x".into(), "x".into(), "x".into(), "x".into()),
        divider: 1,
        barrier_height: 2.5,
        phase: 0.4,
        periodicity: 3,
        comment: None,
    }];
    let p = [
        Vec3::new(-1.0, 0.2, 0.4),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.1, 0.3, -0.1),
        Vec3::new(1.8, 1.2, 0.7),
    ];
    let ((f0, f1, f2, f3), _) = f_dihedral(p[0], p[1], p[2], p[3], &params, &cell);
    let forces = [f0, f1, f2, f3];
    assert_vec_close(
        f0 + f1 + f2 + f3,
        Vec3::new_zero(),
        0.0,
        3e-5,
        "dihedral net force",
    );

    let delta = 5e-4;
    for atom in 0..4 {
        for axis in 0..3 {
            let mut plus = p;
            let mut minus = p;
            plus[atom] = shift(plus[atom], axis, delta);
            minus[atom] = shift(minus[atom], axis, -delta);
            let (_, e_plus) = f_dihedral(plus[0], plus[1], plus[2], plus[3], &params, &cell);
            let (_, e_minus) = f_dihedral(minus[0], minus[1], minus[2], minus[3], &params, &cell);
            let numerical = -(e_plus - e_minus) / (2.0 * delta);
            assert_close(
                component(forces[atom], axis),
                numerical,
                0.025,
                2e-3,
                &format!("dihedral force atom {atom} axis {axis}"),
            );
        }
    }
}

fn atom(serial_number: u32, posit: Vec3, partial_charge: f32) -> AtomGeneric {
    AtomGeneric {
        serial_number,
        posit: posit.into(),
        force_field_type: Some("ca".to_string()),
        element: Element::Carbon,
        partial_charge: Some(partial_charge),
        ..Default::default()
    }
}

fn base_config(overrides: MdOverrides) -> MdConfig {
    MdConfig {
        integrator: Integrator::VerletVelocity { thermostat: None },
        sim_box: SimBoxInit::Fixed((Vec3::new_zero(), Vec3::splat(BOX_LEN))),
        solvent: Solvent::None,
        barostat_cfg: None,
        max_init_relaxation_iters: None,
        recenter_sim_box: false,
        spme_mesh_spacing: 0.8,
        spme_alpha: ALPHA,
        coulomb_cutoff: COULOMB_CUTOFF,
        lj_cutoff: COULOMB_CUTOFF,
        overrides: MdOverrides {
            skip_counterion_insertion: true,
            ..overrides
        },
        ..Default::default()
    }
}

fn evaluate_pair(
    dev: &ComputationDevice,
    q0: f32,
    q1: f32,
    dist: f32,
    overrides: MdOverrides,
) -> MdState {
    let center = BOX_LEN / 2.0;
    let mols = [
        MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: vec![atom(1, Vec3::new(center - dist / 2.0, center, center), q0)],
            ..Default::default()
        },
        MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: vec![atom(2, Vec3::new(center + dist / 2.0, center, center), q1)],
            ..Default::default()
        },
    ];
    let params = FfParamSet::new_amber().unwrap();
    let (mut state, _) = MdState::new(dev, &base_config(overrides), &mols, &params).unwrap();
    state.reset_f_acc_pe_virial();
    state.apply_all_forces(dev, &None);
    state
}

#[test]
fn cpu_short_range_pair_matches_lj_and_screened_coulomb_reference() {
    let dist = 4.5;
    let state = evaluate_pair(
        &ComputationDevice::Cpu,
        0.6,
        -0.4,
        dist,
        MdOverrides {
            bonded_disabled: true,
            long_range_recip_disabled: true,
            ..Default::default()
        },
    );
    let a0 = &state.atoms[0];
    let a1 = &state.atoms[1];
    let dir = Vec3::new(-1.0, 0.0, 0.0);
    let sigma = 0.5 * (a0.lj_sigma + a1.lj_sigma);
    let eps = (a0.lj_eps * a1.lj_eps).sqrt();
    let (f_lj, e_lj) = force_e_lj(dir, 1.0 / dist, sigma, eps);
    let (f_coul, e_coul) = force_coulomb_short_range(
        dir,
        dist,
        1.0 / dist,
        a0.partial_charge,
        a1.partial_charge,
        COULOMB_CUTOFF,
        ALPHA,
    );

    assert_vec_close(
        state.atoms[0].force,
        f_lj + f_coul,
        2e-5,
        2e-5,
        "CPU SR force",
    );
    assert_vec_close(
        state.atoms[1].force,
        -(f_lj + f_coul),
        2e-5,
        2e-5,
        "CPU SR reaction",
    );
    assert_close(
        state.potential_energy_nonbonded as f32,
        e_lj + e_coul,
        2e-5,
        2e-5,
        "CPU SR energy",
    );
}

#[test]
fn cpu_spme_pair_has_correct_total_coulomb_force_and_energy() {
    let dist = 5.0;
    let state = evaluate_pair(
        &ComputationDevice::Cpu,
        1.0,
        -1.0,
        dist,
        MdOverrides {
            bonded_disabled: true,
            lj_disabled: true,
            ..Default::default()
        },
    );
    let k_elec = crate::non_bonded::CHARGE_UNIT_SCALER.powi(2);
    let expected_force = k_elec / dist.powi(2);
    let expected_energy = -k_elec / dist;

    assert_close(
        state.atoms[0].force.x,
        expected_force,
        0.015,
        0.02,
        "CPU PME force",
    );
    assert_close(
        state.potential_energy_nonbonded as f32,
        expected_energy,
        0.015,
        0.02,
        "CPU PME energy",
    );
    assert_vec_close(
        state.atoms[0].force + state.atoms[1].force,
        Vec3::new_zero(),
        0.0,
        1e-3,
        "CPU PME net force",
    );
}

#[derive(Clone, Copy)]
enum ForceSelection {
    Bonded,
    LennardJones,
    Electrostatic,
    All,
}

fn evaluate_combined(dev: &ComputationDevice, selection: ForceSelection) -> MdState {
    let c = BOX_LEN / 2.0;
    let bonded = MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: vec![
            atom(1, Vec3::new(c - 0.8, c, c), 0.4),
            atom(2, Vec3::new(c + 0.8, c, c), 0.0),
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
        atoms: vec![atom(3, Vec3::new(c, c + 5.0, c), -0.4)],
        ..Default::default()
    };

    let overrides = match selection {
        ForceSelection::Bonded => MdOverrides {
            coulomb_disabled: true,
            lj_disabled: true,
            long_range_recip_disabled: true,
            ..Default::default()
        },
        ForceSelection::LennardJones => MdOverrides {
            bonded_disabled: true,
            coulomb_disabled: true,
            long_range_recip_disabled: true,
            ..Default::default()
        },
        ForceSelection::Electrostatic => MdOverrides {
            bonded_disabled: true,
            lj_disabled: true,
            ..Default::default()
        },
        ForceSelection::All => MdOverrides::default(),
    };
    let params = FfParamSet::new_amber().unwrap();
    let (mut state, _) =
        MdState::new(dev, &base_config(overrides), &[bonded, isolated], &params).unwrap();
    state.reset_f_acc_pe_virial();
    state.apply_all_forces(dev, &None);
    state
}

fn assert_component_superposition(dev: &ComputationDevice, tolerance: f32) {
    let bonded = evaluate_combined(dev, ForceSelection::Bonded);
    let lj = evaluate_combined(dev, ForceSelection::LennardJones);
    let electrostatic = evaluate_combined(dev, ForceSelection::Electrostatic);
    let all = evaluate_combined(dev, ForceSelection::All);

    for i in 0..all.atoms.len() {
        let expected = bonded.atoms[i].force + lj.atoms[i].force + electrostatic.atoms[i].force;
        assert_vec_close(
            all.atoms[i].force,
            expected,
            tolerance,
            2e-3,
            &format!("combined force atom {i}"),
        );
    }
    assert_close(
        all.potential_energy as f32,
        (bonded.potential_energy + lj.potential_energy + electrostatic.potential_energy) as f32,
        tolerance,
        2e-3,
        "combined potential energy",
    );
    let net = all
        .atoms
        .iter()
        .fold(Vec3::new_zero(), |sum, atom| sum + atom.force);
    assert!(
        net.magnitude() < 1e-2,
        "combined system net force is {net:?}"
    );
}

#[test]
fn cpu_all_force_components_equal_sum_of_individual_components() {
    assert_component_superposition(&ComputationDevice::Cpu, 2e-4);
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Option<ComputationDevice> {
    use cudarc::driver::CudaContext;

    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            eprintln!("Skipping CUDA force test: no CUDA device is available ({error})");
            return None;
        }
    };
    Some(ComputationDevice::Gpu(context.default_stream()))
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_short_range_matches_cpu_force_and_energy() {
    let Some(gpu) = cuda_device() else {
        return;
    };
    let overrides = MdOverrides {
        bonded_disabled: true,
        long_range_recip_disabled: true,
        ..Default::default()
    };
    let cpu = evaluate_pair(&ComputationDevice::Cpu, 0.6, -0.4, 4.5, overrides.clone());
    let gpu = evaluate_pair(&gpu, 0.6, -0.4, 4.5, overrides);

    for i in 0..2 {
        assert_vec_close(
            gpu.atoms[i].force,
            cpu.atoms[i].force,
            2e-4,
            2e-4,
            "GPU SR force",
        );
    }
    assert_close(
        gpu.potential_energy_nonbonded as f32,
        cpu.potential_energy_nonbonded as f32,
        2e-4,
        2e-4,
        "GPU SR energy",
    );
}

#[cfg(all(feature = "cuda", any(feature = "cufft", feature = "vkfft")))]
#[test]
fn gpu_spme_and_combined_forces_match_cpu() {
    let Some(gpu) = cuda_device() else {
        return;
    };
    let cpu = evaluate_combined(&ComputationDevice::Cpu, ForceSelection::All);
    let gpu_state = evaluate_combined(&gpu, ForceSelection::All);

    for i in 0..cpu.atoms.len() {
        assert_vec_close(
            gpu_state.atoms[i].force,
            cpu.atoms[i].force,
            0.015,
            3e-3,
            &format!("GPU PME combined force atom {i}"),
        );
    }
    assert_close(
        gpu_state.potential_energy as f32,
        cpu.potential_energy as f32,
        0.01,
        3e-3,
        "GPU PME combined energy",
    );
    assert_component_superposition(&gpu, 0.015);
}
