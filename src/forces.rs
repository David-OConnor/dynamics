#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use lin_alg::f32::{vec3s_from_dev, vec3s_to_dev};
use lin_alg::{f32::Vec3};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::{
    f32::{Vec3x8, f32x8},
    f64::f64x4,
};
use crate::AtomDynamics;
use crate::non_bonded::{BodyRef, LjTables, NonBondedPair};
use crate::water_opc::{ForcesOnWaterMol, WaterMol, WaterSite};

/// Handles both LJ, and Coulomb (SPME short range) force.
/// Inputs are structured differently here from our other one; uses pre-paired inputs and outputs, and
/// a common index. Exclusions (e.g. Amber-style 1-2 adn 1-3) are handled upstream.
///
/// Returns force, virial sum, and potential energy.
///
/// todo: This class of function is just connecting, and I believe could be automated
/// todo with a macro or code gen. look into how to do that. Start with your ideal API.
#[cfg(feature = "cuda")]
pub fn force_nonbonded_gpu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    tgt_is: &[u32],
    src_is: &[u32],
    posits_tgt: &[Vec3],
    posits_src: &[Vec3],
    sigmas: &[f32],
    epss: &[f32],
    qs_tgt: &[f32],
    qs_src: &[f32],
    // We use these to determine which target array to accumulate
    // force on.
    atom_types_tgt: &[u8],  // 0 = dyn, 1 = water
    water_types_tgt: &[u8], // See WaterSite repr
    atom_types_src: &[u8],
    water_types_src: &[u8],
    scale_14: &[bool],
    calc_ljs: &[bool],
    calc_coulombs: &[bool],
    symmetric: &[bool],
    cutoff_ewald: f32,
    alpha_ewald: f32,
    cell_extent: Vec3,
    n_dyn: usize,
    n_water: usize,
) -> (Vec<Vec3>, Vec<ForcesOnWaterMol>, f64, f64) {
    let n = posits_tgt.len();

    assert_eq!(tgt_is.len(), n);
    assert_eq!(src_is.len(), n);

    assert_eq!(posits_src.len(), n);

    assert_eq!(sigmas.len(), n);
    assert_eq!(epss.len(), n);
    assert_eq!(qs_tgt.len(), n);
    assert_eq!(qs_src.len(), n);

    assert_eq!(atom_types_tgt.len(), n);
    assert_eq!(water_types_tgt.len(), n);
    assert_eq!(atom_types_src.len(), n);
    assert_eq!(water_types_src.len(), n);

    assert_eq!(scale_14.len(), n);
    assert_eq!(calc_ljs.len(), n);
    assert_eq!(calc_coulombs.len(), n);
    assert_eq!(symmetric.len(), n);

    // May be safer for the GPU to pass u8 instead of bool??
    let scale_14: Vec<_> = scale_14.iter().map(|v| *v as u8).collect();
    let calc_ljs: Vec<_> = calc_ljs.iter().map(|v| *v as u8).collect();
    let calc_coulombs: Vec<_> = calc_coulombs.iter().map(|v| *v as u8).collect();
    let symmetric: Vec<_> = symmetric.iter().map(|v| *v as u8).collect();

    // Set up empty device arrays the kernel will fill as output.
    let mut forces_on_dyn = {
        let v = vec![Vec3::new_zero(); n_dyn];
        vec3s_to_dev(stream, &v)
    };

    let mut forces_on_water_o = {
        let v = vec![Vec3::new_zero(); n_water];
        vec3s_to_dev(stream, &v)
    };
    let mut forces_on_water_m = {
        let v = vec![Vec3::new_zero(); n_water];
        vec3s_to_dev(stream, &v)
    };
    let mut forces_on_water_h0 = {
        let v = vec![Vec3::new_zero(); n_water];
        vec3s_to_dev(stream, &v)
    };
    let mut forces_on_water_h1 = {
        let v = vec![Vec3::new_zero(); n_water];
        vec3s_to_dev(stream, &v)
    };

    let mut virial_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();
    let mut energy_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();

    // Store immutable input arrays to the device.

    let tgt_is_gpu = stream.memcpy_stod(tgt_is).unwrap();
    let src_is_gpu = stream.memcpy_stod(src_is).unwrap();

    let posits_src_gpu = vec3s_to_dev(stream, posits_src);
    let posits_tgt_gpu = vec3s_to_dev(stream, posits_tgt);

    let sigmas_gpu = stream.memcpy_stod(sigmas).unwrap();
    let epss_gpu = stream.memcpy_stod(epss).unwrap();

    let qs_tgt_gpu = stream.memcpy_stod(qs_tgt).unwrap();
    let qs_src_gpu = stream.memcpy_stod(qs_src).unwrap();

    let atom_types_tgt_gpu = stream.memcpy_stod(atom_types_tgt).unwrap();
    let water_types_tgt_gpu = stream.memcpy_stod(water_types_tgt).unwrap();
    let atom_types_src_gpu = stream.memcpy_stod(atom_types_src).unwrap();
    let water_types_src_gpu = stream.memcpy_stod(water_types_src).unwrap();

    // For Amber-style 1-4 covalent bond scaling; not general LJ.
    let scale_14_gpu = stream.memcpy_stod(&scale_14).unwrap();
    let calc_ljs_gpu = stream.memcpy_stod(&calc_ljs).unwrap();
    let calc_coulombs_gpu = stream.memcpy_stod(&calc_coulombs).unwrap();
    let symmetric_gpu = stream.memcpy_stod(&symmetric).unwrap();

    // todo: Likely load these functions (kernels) at init and pass as a param.
    // todo: Seems to take only 4 μs (per time step), so should be fine here.
    let kernel = module.load_function("nonbonded_force_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&kernel);

    launch_args.arg(&mut forces_on_dyn);
    launch_args.arg(&mut forces_on_water_o);
    launch_args.arg(&mut forces_on_water_m);
    launch_args.arg(&mut forces_on_water_h0);
    launch_args.arg(&mut forces_on_water_h1);
    launch_args.arg(&mut virial_gpu);
    launch_args.arg(&mut energy_gpu);
    //
    launch_args.arg(&tgt_is_gpu);
    launch_args.arg(&src_is_gpu);
    launch_args.arg(&posits_tgt_gpu);
    launch_args.arg(&posits_src_gpu);
    launch_args.arg(&sigmas_gpu);
    launch_args.arg(&epss_gpu);
    launch_args.arg(&qs_tgt_gpu);
    launch_args.arg(&qs_src_gpu);
    launch_args.arg(&atom_types_tgt_gpu);
    launch_args.arg(&water_types_tgt_gpu);
    launch_args.arg(&atom_types_src_gpu);
    launch_args.arg(&water_types_src_gpu);
    launch_args.arg(&scale_14_gpu);
    launch_args.arg(&calc_ljs_gpu);
    launch_args.arg(&calc_coulombs_gpu);
    launch_args.arg(&symmetric_gpu);
    //
    launch_args.arg(&cell_extent);
    launch_args.arg(&cutoff_ewald);
    launch_args.arg(&alpha_ewald);
    launch_args.arg(&n);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
    let forces_on_dyn = vec3s_from_dev(stream, &forces_on_dyn);

    let forces_on_water_o = vec3s_from_dev(stream, &forces_on_water_o);
    let forces_on_water_m = vec3s_from_dev(stream, &forces_on_water_m);
    let forces_on_water_h0 = vec3s_from_dev(stream, &forces_on_water_h0);
    let forces_on_water_h1 = vec3s_from_dev(stream, &forces_on_water_h1);

    let mut forces_on_water = Vec::new();
    for i in 0..n_water {
        let f_o = forces_on_water_o[i].into();
        let f_m = forces_on_water_m[i].into();
        let f_h0 = forces_on_water_h0[i].into();
        let f_h1 = forces_on_water_h1[i].into();

        forces_on_water.push(ForcesOnWaterMol {
            f_o,
            f_m,
            f_h0,
            f_h1,
        });
    }

    let virial = stream.memcpy_dtov(&virial_gpu).unwrap()[0];
    let energy = stream.memcpy_dtov(&energy_gpu).unwrap()[0];

    (forces_on_dyn, forces_on_water, virial, energy)
}

/// See notes on `V_lj()`. We set up the dist params we do to share computation
/// with Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
/// This variant also computes energy.
pub fn force_e_lj(dir: Vec3, inv_dist: f32, sigma: f32, eps: f32) -> (Vec3, f32) {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    // todo: mul_add is unstable. add later
    let mag = 24. * eps * (2. * sr12 - sr6) * inv_dist;
    // let mag = 24. * eps * mul_add(2., sr12,  -sr6) * inv_dist;

    let energy = 4. * eps * (sr12 - sr6);
    (dir * mag, energy)
}

// todo: Experimenting with keeping persistent item son the GPU
/// Items here don't change until neighbors are rebuilt. Keep them on the GPU; don't pass
/// each time.
#[cfg(feature = "cuda")]
pub struct NonbondedGpuCtx {
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub func: cudarc::driver::Function,
    pub cfg: LaunchConfig,

    // Immutable until neighbor rebuild
    pub tgt_is_gpu: DevicePtr<u32>,
    pub src_is_gpu: DevicePtr<u32>,
    pub atom_types_tgt_gpu: DevicePtr<u8>,
    pub water_types_tgt_gpu: DevicePtr<u8>,
    pub atom_types_src_gpu: DevicePtr<u8>,
    pub water_types_src_gpu: DevicePtr<u8>,
    pub sigmas_gpu: DevicePtr<f32>,
    pub epss_gpu: DevicePtr<f32>,
    pub qs_tgt_gpu: DevicePtr<f32>,
    pub qs_src_gpu: DevicePtr<f32>,
    pub scale_14_gpu: DevicePtr<u8>,
    pub calc_ljs_gpu: DevicePtr<u8>,
    pub calc_coulombs_gpu: DevicePtr<u8>,
    pub symmetric_gpu: DevicePtr<u8>,

    // Outputs (reused every step)
    pub forces_on_dyn: DevicePtr<Vec3>,
    pub forces_on_water_o: DevicePtr<Vec3>,
    pub forces_on_water_m: DevicePtr<Vec3>,
    pub forces_on_water_h0: DevicePtr<Vec3>,
    pub forces_on_water_h1: DevicePtr<Vec3>,
    pub virial_gpu: DevicePtr<f64>,
    pub energy_gpu: DevicePtr<f64>,

    pub n_pairs: usize,
    pub n_dyn: usize,
    pub n_water: usize,
}

#[cfg(feature = "cuda")]
impl NonbondedGpuCtx {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        pairs: &[NonBondedPair],
        atoms_dyn: &[AtomDynamics],
        water: &[WaterMol],
        lj_tables: &LjTables,
    ) -> Self {
        let n_dyn = atoms_dyn.len();
        let n_water = water.len();
        let n = pairs.len();

        // Build once: indices, flags, pairwise params, charges.
        let mut tgt_is = Vec::with_capacity(n);
        let mut src_is = Vec::with_capacity(n);
        let mut atom_types_tgt = vec![0; n];
        let mut water_types_tgt = vec![0; n];
        let mut atom_types_src = vec![0; n];
        let mut water_types_src = vec![0; n];
        let mut sigmas = Vec::with_capacity(n);
        let mut epss = Vec::with_capacity(n);
        let mut qs_tgt = Vec::with_capacity(n);
        let mut qs_src = Vec::with_capacity(n);
        let mut scale_14 = Vec::with_capacity(n);
        let mut calc_ljs = Vec::with_capacity(n);
        let mut calc_coulombs = Vec::with_capacity(n);
        let mut symmetric = Vec::with_capacity(n);

        for (i, p) in pairs.iter().enumerate() {
            let (ai_tgt, at_tgt, wt_tgt, atom_tgt) = match p.tgt {
                BodyRef::Dyn(j) => (j as u32, 0u8, 0u8, &atoms_dyn[j]),
                BodyRef::Water { mol, site } => {
                    let site_u8 = site as u8;
                    (mol as u32, 1u8, site_u8, match site {
                        WaterSite::O => &water[mol].o,
                        WaterSite::M => &water[mol].m,
                        WaterSite::H0 => &water[mol].h0,
                        WaterSite::H1 => &water[mol].h1,
                    })
                }
            };
            let (ai_src, at_src, wt_src, atom_src) = match p.src {
                BodyRef::Dyn(j) => (j as u32, 0u8, 0u8, &atoms_dyn[j]),
                BodyRef::Water { mol, site } => {
                    let site_u8 = site as u8;
                    (mol as u32, 1u8, site_u8, match site {
                        WaterSite::O => &water[mol].o,
                        WaterSite::M => &water[mol].m,
                        WaterSite::H0 => &water[mol].h0,
                        WaterSite::H1 => &water[mol].h1,
                    })
                }
            };

            tgt_is.push(ai_tgt);
            src_is.push(ai_src);
            atom_types_tgt[i] = at_tgt;
            water_types_tgt[i] = wt_tgt;
            atom_types_src[i] = at_src;
            water_types_src[i] = wt_src;

            let (σ, ε) = lj_tables.lookup(&p.lj_indices);
            sigmas.push(σ);
            epss.push(ε);

            qs_tgt.push(atom_tgt.partial_charge);
            qs_src.push(atom_src.partial_charge);

            scale_14.push(p.scale_14 as u8);
            calc_ljs.push(p.calc_lj as u8);
            calc_coulombs.push(p.calc_coulomb as u8);
            symmetric.push(p.symmetric as u8);
        }

        // Upload once
        let tgt_is_gpu = stream.memcpy_stod(&tgt_is).unwrap();
        let src_is_gpu = stream.memcpy_stod(&src_is).unwrap();
        let atom_types_tgt_gpu = stream.memcpy_stod(&atom_types_tgt).unwrap();
        let water_types_tgt_gpu = stream.memcpy_stod(&water_types_tgt).unwrap();
        let atom_types_src_gpu = stream.memcpy_stod(&atom_types_src).unwrap();
        let water_types_src_gpu = stream.memcpy_stod(&water_types_src).unwrap();
        let sigmas_gpu = stream.memcpy_stod(&sigmas).unwrap();
        let epss_gpu = stream.memcpy_stod(&epss).unwrap();
        let qs_tgt_gpu = stream.memcpy_stod(&qs_tgt).unwrap();
        let qs_src_gpu = stream.memcpy_stod(&qs_src).unwrap();
        let scale_14_gpu = stream.memcpy_stod(&scale_14).unwrap();
        let calc_ljs_gpu = stream.memcpy_stod(&calc_ljs).unwrap();
        let calc_coulombs_gpu = stream.memcpy_stod(&calc_coulombs).unwrap();
        let symmetric_gpu = stream.memcpy_stod(&symmetric).unwrap();

        // Outputs (reused)
        let forces_on_dyn = vec3s_to_dev(&stream, &vec![Vec3::new_zero(); n_dyn]);
        let forces_on_water_o = vec3s_to_dev(&stream, &vec![Vec3::new_zero(); n_water]);
        let forces_on_water_m = vec3s_to_dev(&stream, &vec![Vec3::new_zero(); n_water]);
        let forces_on_water_h0 = vec3s_to_dev(&stream, &vec![Vec3::new_zero(); n_water]);
        let forces_on_water_h1 = vec3s_to_dev(&stream, &vec![Vec3::new_zero(); n_water]);
        let virial_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();
        let energy_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();

        let func = module.load_function("nonbonded_force_kernel").unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);

        Self {
            stream, module, func, cfg,
            tgt_is_gpu, src_is_gpu,
            atom_types_tgt_gpu, water_types_tgt_gpu,
            atom_types_src_gpu, water_types_src_gpu,
            sigmas_gpu, epss_gpu, qs_tgt_gpu, qs_src_gpu,
            scale_14_gpu, calc_ljs_gpu, calc_coulombs_gpu, symmetric_gpu,
            forces_on_dyn, forces_on_water_o, forces_on_water_m,
            forces_on_water_h0, forces_on_water_h1,
            virial_gpu, energy_gpu,
            n_pairs: n, n_dyn, n_water,
        }
    }

    pub fn zero_outputs(&mut self) {
        // Refill with zero; you can also cudaMemset via a tiny kernel.
        let _ = self.stream.memcpy_dtod(&self.virial_gpu, &mut self.stream.memcpy_stod(&[0.0f64]).unwrap());
        let _ = self.stream.memcpy_dtod(&self.energy_gpu, &mut self.stream.memcpy_stod(&[0.0f64]).unwrap());
        // For force arrays, either memset kernel or overwrite after readback; memset is better:
        // ... launch a small kernel to zero forces_on_* ...
    }
}
