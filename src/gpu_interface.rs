use std::sync::Arc;

use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use lin_alg::{
    f32::{Vec3, vec3s_from_dev, vec3s_to_dev},
    f64::Vec3 as Vec3F64,
};

use crate::{
    AtomDynamics, ForcesOnWaterMol,
    non_bonded::{BodyRef, LjTables, NonBondedPair},
    water_opc::{WaterMol, WaterSite},
};

/// Device buffers that persist across all steps. Mutated on the GPU.
/// We initialize these once at the start. These are all flattened.
/// We pass thisto the kernel each step, but don't transfer.
pub(crate) struct ForcesGpu {
    pub forces_on_dyn: CudaSlice<f32>,
    pub forces_on_water_o: CudaSlice<f32>,
    pub forces_on_water_m: CudaSlice<f32>,
    pub forces_on_water_h0: CudaSlice<f32>,
    pub forces_on_water_h1: CudaSlice<f32>,
    pub virial_gpu: CudaSlice<f64>,
    pub energy_gpu: CudaSlice<f64>,
    pub cutoff_ewald: f32,
    pub alpha_ewald: f32,
}

impl ForcesGpu {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        n_dyn: usize,
        n_water: usize,
        cutoff_ewald: f32,
        alpha_ewald: f32,
    ) -> Self {
        // Set up empty device arrays the kernel will fill as output.
        // let forces_on_dyn = {
        //     let v = vec![Vec3::new_zero(); n_dyn];
        //     vec3s_to_dev(stream, &v)
        // };
        // let forces_on_water_o = {
        //     let v = vec![Vec3::new_zero(); n_water];
        //     vec3s_to_dev(stream, &v)
        // };
        // let forces_on_water_m = {
        //     let v = vec![Vec3::new_zero(); n_water];
        //     vec3s_to_dev(stream, &v)
        // };
        // let forces_on_water_h0 = {
        //     let v = vec![Vec3::new_zero(); n_water];
        //     vec3s_to_dev(stream, &v)
        // };
        // let forces_on_water_h1 = {
        //     let v = vec![Vec3::new_zero(); n_water];
        //     vec3s_to_dev(stream, &v)
        // };

        // Each is a float3 on device, Vec3 on host.
        let forces_on_dyn = stream.alloc_zeros::<f32>(n_dyn * 3).unwrap();

        let forces_on_water_o = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let forces_on_water_m = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let forces_on_water_h0 = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let forces_on_water_h1 = stream.alloc_zeros::<f32>(n_water * 3).unwrap();

        let virial_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();
        let energy_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();

        Self {
            // todo: Make sure the kernel overwrites these each step. If not, zero them.
            forces_on_dyn,
            forces_on_water_o,
            forces_on_water_m,
            forces_on_water_h0,
            forces_on_water_h1,
            virial_gpu,
            energy_gpu,
            cutoff_ewald,
            alpha_ewald,
        }
    }
}

/// Device buffers that persist until the neighbor list is rebuilt (pair metadata).
/// Copy items from host to GPU ("device") that change when we rebuild the neighbors, but don't
/// change otherwise. Build this whenever we rebuild the neighbors list.
///
/// We pass thisto the kernel each step, but don't transfer.
pub(crate) struct PerNeighborGpu {
    pub tgt_is: CudaSlice<u32>,
    pub src_is: CudaSlice<u32>,
    pub sigmas: CudaSlice<f32>,
    pub epss: CudaSlice<f32>,
    pub qs_tgt: CudaSlice<f32>,
    pub qs_src: CudaSlice<f32>,
    pub atom_types_tgt: CudaSlice<u8>,
    pub water_types_tgt: CudaSlice<u8>,
    pub atom_types_src: CudaSlice<u8>,
    pub water_types_src: CudaSlice<u8>,
    pub scale_14: CudaSlice<u8>,
    pub calc_ljs: CudaSlice<u8>,
    pub calc_coulombs: CudaSlice<u8>,
    pub symmetric: CudaSlice<u8>,
}

impl PerNeighborGpu {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        pairs: &[NonBondedPair],
        atoms_dyn: &[AtomDynamics],
        water: &[WaterMol],
        lj_tables: &LjTables,
    ) -> Self {
        let n = pairs.len();

        // Start by setting up on the CPU.
        let mut sigmas = Vec::with_capacity(n);
        let mut epss = Vec::with_capacity(n);

        let mut qs_tgt = Vec::with_capacity(n);
        let mut qs_src = Vec::with_capacity(n);

        let mut scale_14s = Vec::with_capacity(n);

        let mut tgt_is: Vec<u32> = Vec::with_capacity(n);
        let mut src_is: Vec<u32> = Vec::with_capacity(n);

        let mut calc_ljs = Vec::with_capacity(n);
        let mut calc_coulombs = Vec::with_capacity(n);
        let mut symmetric = Vec::with_capacity(n);

        // Unpack BodyRef to fields. It doesn't map neatly to CUDA flattening primitives.

        // These atom and water types are so the Kernel can assign to the correct output arrays.
        // 0 means Dyn, 1 means Water.
        let mut atom_types_tgt = vec![0; n];
        // 0 for not-water or N/A. 1 = O, 2 = M, 3 = H0, 4 = H1.
        // Pre-allocated to 0, which we use for dyn atom targets.
        let mut water_types_tgt = vec![0; n];

        let mut atom_types_src = vec![0; n];
        let mut water_types_src = vec![0; n];

        for (i, pair) in pairs.iter().enumerate() {
            let q_tgt = match pair.tgt {
                BodyRef::Dyn(j) => {
                    tgt_is.push(j as u32);
                    &atoms_dyn[j]
                }
                BodyRef::Water { mol: j, site } => {
                    tgt_is.push(j as u32);

                    // Mark so the kernel will use the water output.
                    atom_types_tgt[i] = 1;
                    water_types_tgt[i] = site as u8;

                    match site {
                        WaterSite::O => &water[j].o,
                        WaterSite::M => &water[j].m,
                        WaterSite::H0 => &water[j].h0,
                        WaterSite::H1 => &water[j].h1,
                    }
                }
                _ => unreachable!(),
            }.partial_charge;

            let q_src = match pair.src {
                BodyRef::Dyn(j) => {
                    src_is.push(j as u32);
                    &atoms_dyn[j]
                }
                BodyRef::Water { mol: j, site } => {
                    src_is.push(j as u32);

                    // Mark so the kernel will use the water output. (In case of dyn/water symmetric)
                    atom_types_src[i] = 1;
                    water_types_src[i] = site as u8;
                    match site {
                        WaterSite::O => &water[j].o,
                        WaterSite::M => &water[j].m,
                        WaterSite::H0 => &water[j].h0,
                        WaterSite::H1 => &water[j].h1,
                    }
                }
            }.partial_charge;

            let (σ, ε) = lj_tables.lookup(&pair.lj_indices);

            sigmas.push(σ);
            epss.push(ε);

            qs_tgt.push(q_tgt);
            qs_src.push(q_src);

            scale_14s.push(pair.scale_14);

            calc_ljs.push(pair.calc_lj);
            calc_coulombs.push(pair.calc_coulomb);
            symmetric.push(pair.symmetric);
        }

        // Transfer to GPU.

        {
            assert_eq!(tgt_is.len(), n);
            assert_eq!(src_is.len(), n);

            assert_eq!(sigmas.len(), n);
            assert_eq!(epss.len(), n);
            assert_eq!(qs_tgt.len(), n);
            assert_eq!(qs_src.len(), n);

            assert_eq!(atom_types_tgt.len(), n);
            assert_eq!(water_types_tgt.len(), n);
            assert_eq!(atom_types_src.len(), n);
            assert_eq!(water_types_src.len(), n);

            assert_eq!(scale_14s.len(), n);
            assert_eq!(calc_ljs.len(), n);
            assert_eq!(calc_coulombs.len(), n);
            assert_eq!(symmetric.len(), n);
        }

        // May be safer for the GPU to pass u8 instead of bool??
        let scale_14: Vec<_> = scale_14s.iter().map(|v| *v as u8).collect();
        let calc_ljs: Vec<_> = calc_ljs.iter().map(|v| *v as u8).collect();
        let calc_coulombs: Vec<_> = calc_coulombs.iter().map(|v| *v as u8).collect();
        let symmetric: Vec<_> = symmetric.iter().map(|v| *v as u8).collect();

        let tgt_is = stream.memcpy_stod(&tgt_is).unwrap();
        let src_is = stream.memcpy_stod(&src_is).unwrap();

        let sigmas = stream.memcpy_stod(&sigmas).unwrap();
        let epss = stream.memcpy_stod(&epss).unwrap();

        let qs_tgt = stream.memcpy_stod(&qs_tgt).unwrap();
        let qs_src = stream.memcpy_stod(&qs_src).unwrap();

        let atom_types_tgt = stream.memcpy_stod(&atom_types_tgt).unwrap();
        let water_types_tgt = stream.memcpy_stod(&water_types_tgt).unwrap();
        let atom_types_src = stream.memcpy_stod(&atom_types_src).unwrap();
        let water_types_src = stream.memcpy_stod(&water_types_src).unwrap();

        // For Amber-style 1-4 covalent bond scaling; not general LJ.
        let scale_14 = stream.memcpy_stod(&scale_14).unwrap();
        let calc_ljs = stream.memcpy_stod(&calc_ljs).unwrap();
        let calc_coulombs = stream.memcpy_stod(&calc_coulombs).unwrap();
        let symmetric = stream.memcpy_stod(&symmetric).unwrap();

        Self {
            tgt_is,
            src_is,
            sigmas,
            epss,
            qs_tgt,
            qs_src,
            atom_types_tgt,
            water_types_tgt,
            atom_types_src,
            water_types_src,
            scale_14,
            calc_ljs,
            calc_coulombs,
            symmetric,
        }
    }
}

/// Handles both LJ, and Coulomb (SPME short range) force. Run this every step.
/// Inputs are structured differently here from our other one; uses pre-paired inputs and outputs, and
/// a common index. Exclusions (e.g. Amber-style 1-2 adn 1-3) are handled upstream.
///
/// Returns force, virial sum, and potential energy.
pub fn force_nonbonded_gpu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    pairs: &[NonBondedPair],
    atoms_dyn: &[AtomDynamics],
    water: &[WaterMol],
    // todo: Only copy cell_extent when it changes, e.g. due to the barostat.
    cell_extent: Vec3,
    forces: &mut ForcesGpu,
    per_neighbor: &PerNeighborGpu,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64) {
    // let n = posits_tgt.len();

    // assert_eq!(posits_src.len(), n);

    // let n_dyn = atoms_dyn.len();
    // let n_water = water.len();

    let n = pairs.len();

    // todo: Eventually, keep positions on the GPU too.
    let mut posits_tgt: Vec<Vec3> = Vec::with_capacity(n);
    let mut posits_src: Vec<Vec3> = Vec::with_capacity(n);

    for pair in pairs {
        let atom_tgt = match pair.tgt {
            BodyRef::Dyn(j) => &atoms_dyn[j],
            BodyRef::Water { mol: j, site } => match site {
                WaterSite::O => &water[j].o,
                WaterSite::M => &water[j].m,
                WaterSite::H0 => &water[j].h0,
                WaterSite::H1 => &water[j].h1,
            },
            _ => unreachable!(),
        };

        let atom_src = match pair.src {
            BodyRef::Dyn(j) => &atoms_dyn[j],
            BodyRef::Water { mol: j, site } => match site {
                WaterSite::O => &water[j].o,
                WaterSite::M => &water[j].m,
                WaterSite::H0 => &water[j].h0,
                WaterSite::H1 => &water[j].h1,
            },
        };

        posits_tgt.push(atom_tgt.posit);
        posits_src.push(atom_src.posit);
    }

    // 1-4 scaling, and the symmetric case handled in the kernel.

    // let cell_extent: Vec3 = cell.extent.into();

    // Store immutable input arrays to the device.

    let posits_src_gpu = vec3s_to_dev(stream, &posits_src);
    let posits_tgt_gpu = vec3s_to_dev(stream, &posits_tgt);

    // todo: Likely load these functions (kernels) at init and pass as a param.
    // todo: Seems to take only 4 μs (per time step), so should be fine here.
    let kernel = module.load_function("nonbonded_force_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&kernel);

    let n_u32 = n as u32;

    // todo: How do we store and pass references to thsee? A struct of CudaSlices?
    launch_args.arg(&mut forces.forces_on_dyn);
    launch_args.arg(&mut forces.forces_on_water_o);
    launch_args.arg(&mut forces.forces_on_water_m);
    launch_args.arg(&mut forces.forces_on_water_h0);
    launch_args.arg(&mut forces.forces_on_water_h1);
    launch_args.arg(&mut forces.virial_gpu);
    launch_args.arg(&mut forces.energy_gpu);
    //
    launch_args.arg(&per_neighbor.tgt_is);
    launch_args.arg(&per_neighbor.src_is);

    launch_args.arg(&posits_tgt_gpu);
    launch_args.arg(&posits_src_gpu);

    launch_args.arg(&per_neighbor.sigmas);
    launch_args.arg(&per_neighbor.epss);
    launch_args.arg(&per_neighbor.qs_tgt);
    launch_args.arg(&per_neighbor.qs_src);
    launch_args.arg(&per_neighbor.atom_types_tgt);
    launch_args.arg(&per_neighbor.water_types_tgt);
    launch_args.arg(&per_neighbor.atom_types_src);
    launch_args.arg(&per_neighbor.water_types_src);
    launch_args.arg(&per_neighbor.scale_14);
    launch_args.arg(&per_neighbor.calc_ljs);
    launch_args.arg(&per_neighbor.calc_coulombs);
    launch_args.arg(&per_neighbor.symmetric);
    //
    launch_args.arg(&cell_extent);
    launch_args.arg(&forces.cutoff_ewald);
    launch_args.arg(&forces.alpha_ewald);
    launch_args.arg(&n_u32);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: As above, how do we get a stream of these from the device, when we've pre-cached?

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
    let forces_on_dyn = vec3s_from_dev(stream, &forces.forces_on_dyn);

    let forces_on_water_o = vec3s_from_dev(stream, &forces.forces_on_water_o);
    let forces_on_water_m = vec3s_from_dev(stream, &forces.forces_on_water_m);
    let forces_on_water_h0 = vec3s_from_dev(stream, &forces.forces_on_water_h0);
    let forces_on_water_h1 = vec3s_from_dev(stream, &forces.forces_on_water_h1);

    let mut forces_on_water = Vec::new();
    for i in 0..water.len() {
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

    let virial = stream.memcpy_dtov(&forces.virial_gpu).unwrap()[0];
    let energy = stream.memcpy_dtov(&forces.energy_gpu).unwrap()[0];

    let forces_on_dyn = forces_on_dyn.into_iter().map(|f| f.into()).collect();

    (forces_on_dyn, forces_on_water, virial, energy)
}
