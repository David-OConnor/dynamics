use dynamics::{
    ComputationDevice, MdConfig, MdOverrides, MdState, ShrinkingBoxCfg, SimBox, SimBoxInit,
    Solvent, SolventTemplateType, WaterInitTemplate, params::FfParamSet,
    water_mols_from_template_in_region,
};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};

fn assert_waters_span_cell(md: &MdState) {
    for axis in [|p: Vec3| p.x, |p: Vec3| p.y, |p: Vec3| p.z] {
        let min = md
            .water
            .iter()
            .map(|water| axis(water.o.posit))
            .reduce(f32::min)
            .unwrap();
        let max = md
            .water
            .iter()
            .map(|water| axis(water.o.posit))
            .reduce(f32::max)
            .unwrap();

        assert!(min < -5.0, "water grid minimum {min} did not span the cell");
        assert!(max > 5.0, "water grid maximum {max} did not span the cell");
    }
}

#[test]
fn shrinking_box_schedule_and_gromacs_rate_agree() {
    let target = SimBox::new(Vec3::splat(-5.0), Vec3::splat(5.0));
    let cfg = ShrinkingBoxCfg {
        initial_box_scale: 2.0,
        box_shrink_per_step: 2.0,
    };

    assert_eq!(cfg.initial_cell(target).extent, Vec3::splat(20.0));
    assert_eq!(cfg.shrink_step_count(target), 5);
    let rates = cfg.gromacs_deform_nm_ps(target, 0.002);
    for rate in &rates[..3] {
        assert!((*rate + 100.0).abs() < 1.0e-3);
    }
    assert_eq!(&rates[3..], &[0.0, 0.0, 0.0]);
}

#[test]
fn interleaved_waters_span_the_solvent_grid() {
    let dev = ComputationDevice::Cpu;
    let cfg = MdConfig {
        sim_box: SimBoxInit::Fixed((Vec3::splat(-30.0), Vec3::splat(30.0))),
        solvent: Solvent::WaterOpcSpecifyMolCount(944),
        recenter_sim_box: false,
        max_init_relaxation_iters: None,
        overrides: MdOverrides {
            skip_water_relaxation: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let (mut md, _) = MdState::new(&dev, &cfg, &[], &FfParamSet::default()).unwrap();
    assert_waters_span_cell(&md);

    let centers = [-20.0, 0.0, 20.0]
        .into_iter()
        .flat_map(|x| {
            [-20.0, 0.0, 20.0].into_iter().flat_map(move |y| {
                [-20.0, 0.0, 20.0]
                    .into_iter()
                    .map(move |z| Vec3F64::new(x, y, z))
            })
        })
        .take(24)
        .collect::<Vec<_>>();

    assert!(md.redistribute_interleaved_opc_waters(&dev, &centers, Vec3F64::splat(20.0)));
    assert_eq!(md.water.len(), 944);
    assert_waters_span_cell(&md);
}

#[test]
fn specified_template_count_spans_the_region() {
    let mut o_posits = Vec::new();

    for z in [-6., 6.] {
        for y in [-4., 4.] {
            for x in [-6., -2., 2., 6.] {
                o_posits.push(Vec3::new(x, y, z));
            }
        }
    }

    let h0_posits = o_posits
        .iter()
        .map(|posit| *posit + Vec3::new(0.4, 0., 0.))
        .collect();
    let h1_posits = o_posits
        .iter()
        .map(|posit| *posit + Vec3::new(-0.4, 0., 0.))
        .collect();
    let velocities = vec![Vec3::new_zero(); o_posits.len()];
    let cell = SimBox::new(Vec3::new(-10., -10., -10.), Vec3::new(10., 10., 10.));
    let template = WaterInitTemplate::from_parts(
        o_posits,
        h0_posits,
        h1_posits,
        velocities.clone(),
        velocities.clone(),
        velocities,
        cell,
    )
    .unwrap();
    let waters = water_mols_from_template_in_region(
        &cell,
        &cell,
        &[],
        Some(8),
        &SolventTemplateType::Custom(template),
        false,
    )
    .unwrap();

    assert_eq!(waters.len(), 8);
    assert!(waters.iter().any(|water| water.o.posit.z < 0.));
    assert!(waters.iter().any(|water| water.o.posit.z > 0.));
}
