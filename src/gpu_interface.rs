use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use lin_alg::{
    f32::{Vec3, vec3s_from_dev},
    f64::Vec3 as Vec3F64,
};

use crate::{
    AtomDynamics, ForcesOnWaterMol, MdOverrides,
    non_bonded::{BodyRef, LjTables, NonBondedPair},
    water_opc::{WaterMol, WaterSite},
};

/// Device buffers that persist across all steps. Mutated on the GPU.
/// We initialize these once at the start. These are all flattened.
/// We pass thisto the kernel each step, but don't transfer.
///
/// Note: Forces and energies must be zeroed each step.
pub(crate) struct ForcesPositsGpu {
    pub forces_on_dyn: CudaSlice<f32>,
    pub forces_on_water_o: CudaSlice<f32>,
    pub forces_on_water_m: CudaSlice<f32>,
    pub forces_on_water_h0: CudaSlice<f32>,
    pub forces_on_water_h1: CudaSlice<f32>,

    pub virial_gpu: CudaSlice<f64>,
    pub energy_gpu: CudaSlice<f64>,

    pub cutoff_ewald: f32,
    pub alpha_ewald: f32,

    pub pos_dyn: CudaSlice<f32>,
    pub pos_w_o: CudaSlice<f32>,
    pub pos_w_m: CudaSlice<f32>,
    pub pos_w_h0: CudaSlice<f32>,
    pub pos_w_h1: CudaSlice<f32>,
}

impl ForcesPositsGpu {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        n_dyn: usize,
        n_water: usize,
        cutoff_ewald: f32,
        alpha_ewald: f32,
    ) -> Self {
        // Set up empty device arrays the kernel will fill as output.
        let forces_on_dyn = stream.alloc_zeros::<f32>(n_dyn * 3).unwrap();
        let forces_on_water_o = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let forces_on_water_m = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let forces_on_water_h0 = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let forces_on_water_h1 = stream.alloc_zeros::<f32>(n_water * 3).unwrap();

        let virial_gpu = stream.clone_htod(&[0.0f64]).unwrap();
        let energy_gpu = stream.clone_htod(&[0.0f64]).unwrap();

        let pos_dyn = stream.alloc_zeros::<f32>(n_dyn * 3).unwrap();
        let pos_w_o = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let pos_w_m = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let pos_w_h0 = stream.alloc_zeros::<f32>(n_water * 3).unwrap();
        let pos_w_h1 = stream.alloc_zeros::<f32>(n_water * 3).unwrap();

        Self {
            forces_on_dyn,
            forces_on_water_o,
            forces_on_water_m,
            forces_on_water_h0,
            forces_on_water_h1,
            virial_gpu,
            energy_gpu,
            cutoff_ewald,
            alpha_ewald,

            pos_dyn,
            pos_w_o,
            pos_w_m,
            pos_w_h0,
            pos_w_h1,
        }
    }
}

/// Handles to device buffers that persist until the neighbor list is rebuilt (pair metadata).
/// Copy items from host to GPU ("device") that change when we rebuild the neighbors, but don't
/// change otherwise. Build this whenever we rebuild the neighbors list.
///
/// We pass this to the kernel each step, but don't transfer.
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
    // These are booleans for potentially safer FFI.
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
                BodyRef::NonWater(j) => {
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
            }
            .partial_charge;

            let q_src = match pair.src {
                BodyRef::NonWater(j) => {
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
            }
            .partial_charge;

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

        let scale_14: Vec<_> = scale_14s.iter().map(|v| *v as u8).collect();
        let calc_ljs: Vec<_> = calc_ljs.iter().map(|v| *v as u8).collect();
        let calc_coulombs: Vec<_> = calc_coulombs.iter().map(|v| *v as u8).collect();
        let symmetric: Vec<_> = symmetric.iter().map(|v| *v as u8).collect();

        let tgt_is = stream.clone_htod(&tgt_is).unwrap();
        let src_is = stream.clone_htod(&src_is).unwrap();

        let sigmas = stream.clone_htod(&sigmas).unwrap();
        let epss = stream.clone_htod(&epss).unwrap();

        let qs_tgt = stream.clone_htod(&qs_tgt).unwrap();
        let qs_src = stream.clone_htod(&qs_src).unwrap();

        let atom_types_tgt = stream.clone_htod(&atom_types_tgt).unwrap();
        let water_types_tgt = stream.clone_htod(&water_types_tgt).unwrap();
        let atom_types_src = stream.clone_htod(&atom_types_src).unwrap();
        let water_types_src = stream.clone_htod(&water_types_src).unwrap();

        // For Amber-style 1-4 covalent bond scaling; not general LJ.
        let scale_14 = stream.clone_htod(&scale_14).unwrap();
        let calc_ljs = stream.clone_htod(&calc_ljs).unwrap();
        let calc_coulombs = stream.clone_htod(&calc_coulombs).unwrap();
        let symmetric = stream.clone_htod(&symmetric).unwrap();

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

/// Run this each step, at the start of each GPU step.
fn upload_positions(
    stream: &Arc<CudaStream>,
    forces: &mut ForcesPositsGpu,
    atoms_dyn: &[AtomDynamics],
    water: &[WaterMol],
) {
    // pack to flat f32 arrays (x,y,z per atom)
    let mut h_pos_dyn = Vec::with_capacity(atoms_dyn.len() * 3);
    for a in atoms_dyn {
        let [x, y, z] = a.posit.to_arr();
        h_pos_dyn.extend_from_slice(&[x, y, z]);
    }

    let mut h_pos_o = Vec::with_capacity(water.len() * 3);
    let mut h_pos_m = Vec::with_capacity(water.len() * 3);
    let mut h_pos_h0 = Vec::with_capacity(water.len() * 3);
    let mut h_pos_h1 = Vec::with_capacity(water.len() * 3);

    for w in water {
        h_pos_o.extend_from_slice(&w.o.posit.to_arr());
        h_pos_m.extend_from_slice(&w.m.posit.to_arr());
        h_pos_h0.extend_from_slice(&w.h0.posit.to_arr());
        h_pos_h1.extend_from_slice(&w.h1.posit.to_arr());
    }

    // `htod` here, as opposted to `stod`, copies into an existing array, instead of allocating
    // a new one.
    // Copy into existing device buffers (avoid reallocating)
    stream.memcpy_htod(&h_pos_dyn, &mut forces.pos_dyn).unwrap();
    stream.memcpy_htod(&h_pos_o, &mut forces.pos_w_o).unwrap();
    stream.memcpy_htod(&h_pos_m, &mut forces.pos_w_m).unwrap();
    stream.memcpy_htod(&h_pos_h0, &mut forces.pos_w_h0).unwrap();
    stream.memcpy_htod(&h_pos_h1, &mut forces.pos_w_h1).unwrap();
}

/// Handles both LJ, and Coulomb (SPME short range) force using a shared kernel. Run this every step.
/// Inputs are structured differently here from our other one; uses pre-paired inputs and outputs, and
/// a common index. Exclusions (e.g. Amber-style 1-2 adn 1-3) are handled upstream.
///
/// Returns (force on non-water, force on water, virial sum, potential energy total, per-mol-pair potential energy)
pub fn force_nonbonded_gpu(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    kernel_zero_f32: &CudaFunction,
    kernel_zero_f64: &CudaFunction,
    pairs: &[NonBondedPair],
    atoms_dyn: &[AtomDynamics],
    water: &[WaterMol],
    // todo: Only copy cell_extent when it changes, e.g. due to the barostat.
    cell_extent: Vec3,
    forces: &mut ForcesPositsGpu,
    per_neighbor: &PerNeighborGpu,
    overrides: &MdOverrides,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>) {
    upload_positions(stream, forces, atoms_dyn, water);

    let n = pairs.len();

    zero_forces_and_accums(
        stream,
        kernel_zero_f32,
        kernel_zero_f64,
        forces,
        atoms_dyn.len(),
        water.len(),
    );

    // 1-4 scaling, and the symmetric case handled in the kernel.
    // Store immutable input arrays to the device.

    let n_u32 = n as u32;
    let coulomb_disabled = overrides.coulomb_disabled as u8;
    let lj_disabled = overrides.lj_disabled as u8;

    let cfg = LaunchConfig::for_num_elems(n_u32);
    let mut launch_args = stream.launch_builder(kernel);

    // todo: How do we store and pass references to thsee? A struct of CudaSlices?
    // These forces and positions are per-atom; much smaller than the per-pair arrays.
    launch_args.arg(&mut forces.forces_on_dyn);
    launch_args.arg(&mut forces.forces_on_water_o);
    launch_args.arg(&mut forces.forces_on_water_m);
    launch_args.arg(&mut forces.forces_on_water_h0);
    launch_args.arg(&mut forces.forces_on_water_h1);
    //
    launch_args.arg(&mut forces.virial_gpu);
    launch_args.arg(&mut forces.energy_gpu);
    //
    launch_args.arg(&forces.pos_dyn);
    launch_args.arg(&forces.pos_w_o);
    launch_args.arg(&forces.pos_w_m);
    launch_args.arg(&forces.pos_w_h0);
    launch_args.arg(&forces.pos_w_h1);
    //
    launch_args.arg(&per_neighbor.tgt_is);
    launch_args.arg(&per_neighbor.src_is);
    // These params below are per-pair.
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
    launch_args.arg(&coulomb_disabled);
    launch_args.arg(&lj_disabled);

    unsafe { launch_args.launch(cfg) }.unwrap();

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

    let virial = stream.clone_dtoh(&forces.virial_gpu).unwrap()[0];
    let energy = stream.clone_dtoh(&forces.energy_gpu).unwrap()[0];

    let forces_on_dyn = forces_on_dyn.into_iter().map(|f| f.into()).collect();

    (forces_on_dyn, forces_on_water, virial, energy, Vec::new())
}

/// Zero forces and accumulators on the device. Run this each step.
fn zero_forces_and_accums(
    stream: &Arc<CudaStream>,
    zero_f32: &CudaFunction,
    zero_f64: &CudaFunction,
    forces: &mut ForcesPositsGpu,
    n_non_water: usize,
    n_water: usize,
) {
    // Non-water atoms: 3 floats per atom
    let std_len_u32 = (n_non_water * 3) as u32;

    // If 0, we get a panic when launching.
    if std_len_u32 > 0 {
        let cfg_dyn = LaunchConfig::for_num_elems(std_len_u32);
        let mut l0 = stream.launch_builder(&zero_f32);

        l0.arg(&mut forces.forces_on_dyn);
        l0.arg(&std_len_u32);
        unsafe { l0.launch(cfg_dyn) }.unwrap();
    }

    // water arrays: 3 floats per molecule for each site-buffer
    let wat_len_u32 = (n_water * 3) as u32;

    if wat_len_u32 > 0 {
        let cfg_w = LaunchConfig::for_num_elems(wat_len_u32);

        let mut l1 = stream.launch_builder(&zero_f32);
        l1.arg(&mut forces.forces_on_water_o);
        l1.arg(&wat_len_u32);
        unsafe { l1.launch(cfg_w) }.unwrap();

        let mut l2 = stream.launch_builder(&zero_f32);
        l2.arg(&mut forces.forces_on_water_m);
        l2.arg(&wat_len_u32);
        unsafe { l2.launch(cfg_w) }.unwrap();

        let mut l3 = stream.launch_builder(&zero_f32);
        l3.arg(&mut forces.forces_on_water_h0);
        l3.arg(&wat_len_u32);
        unsafe { l3.launch(cfg_w) }.unwrap();

        let mut l4 = stream.launch_builder(&zero_f32);
        l4.arg(&mut forces.forces_on_water_h1);
        l4.arg(&wat_len_u32);
        unsafe { l4.launch(cfg_w) }.unwrap();
    }

    // scalars
    let one: u32 = 1;
    let cfg1 = LaunchConfig::for_num_elems(1);

    let mut l5 = stream.launch_builder(&zero_f64);
    l5.arg(&mut forces.virial_gpu);
    l5.arg(&one);
    unsafe { l5.launch(cfg1) }.unwrap();

    let mut l6 = stream.launch_builder(&zero_f64);
    l6.arg(&mut forces.energy_gpu);
    l6.arg(&one);
    unsafe { l6.launch(cfg1) }.unwrap();
}
