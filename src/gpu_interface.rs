use std::{mem::size_of, sync::Arc};

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, HostSlice, LaunchConfig, PushKernelArg,
    SyncOnDrop, result,
};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};

use crate::{
    AtomDynamics, ForcesOnWaterMol, MdOverrides,
    non_bonded::{BodyRef, LjTables, NonBondedPair},
    solvent::{WaterMolOpc, WaterSite},
};

/// Page-locked host memory without the write-combined flag used by cudarc's
/// `PinnedHostSlice`. Write-combined memory is ideal for H2D staging but very
/// slow for the CPU reads required after D2H force copies.
struct PinnedBuffer<T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    stream: Arc<CudaStream>,
}

unsafe impl<T: DeviceRepr> Send for PinnedBuffer<T> {}
unsafe impl<T: DeviceRepr> Sync for PinnedBuffer<T> {}

impl<T: DeviceRepr> PinnedBuffer<T> {
    fn new(stream: &Arc<CudaStream>, len: usize) -> Self {
        let ptr = unsafe { result::malloc_host(len * size_of::<T>(), 0).unwrap() } as *mut T;
        assert!(!ptr.is_null());
        Self {
            ptr,
            len,
            stream: stream.clone(),
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = unsafe { result::free_host(self.ptr.cast()) };
    }
}

impl<T: DeviceRepr> HostSlice<T> for PinnedBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self.as_slice(), SyncOnDrop::Sync(None))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self.as_mut_slice(), SyncOnDrop::Sync(None))
    }
}

pub(crate) struct GpuKernels {
    pub primary: CudaFunction, // Option only due to not impling Default.
    pub alchemical: CudaFunction,
}

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
    pub alch_dh_dl_gpu: CudaSlice<f64>,

    pub cutoff_ewald: f32,
    pub alpha_ewald: f32,

    pub pos_dyn: CudaSlice<f32>,
    pub pos_w_o: CudaSlice<f32>,
    pub pos_w_m: CudaSlice<f32>,
    pub pos_w_h0: CudaSlice<f32>,
    pub pos_w_h1: CudaSlice<f32>,

    // Reused page-locked host buffers. In addition to avoiding per-step heap
    // allocations, pinned memory prevents cudarc from synchronizing after every
    // individual H2D/D2H copy.
    host_pos_dyn: PinnedBuffer<f32>,
    host_pos_w_o: PinnedBuffer<f32>,
    host_pos_w_m: PinnedBuffer<f32>,
    host_pos_w_h0: PinnedBuffer<f32>,
    host_pos_w_h1: PinnedBuffer<f32>,
    host_forces_dyn: PinnedBuffer<f32>,
    host_forces_w_o: PinnedBuffer<f32>,
    host_forces_w_m: PinnedBuffer<f32>,
    host_forces_w_h0: PinnedBuffer<f32>,
    host_forces_w_h1: PinnedBuffer<f32>,
    host_virial: PinnedBuffer<f64>,
    host_energy: PinnedBuffer<f64>,
    host_alch_dh_dl: PinnedBuffer<f64>,
}

impl ForcesPositsGpu {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        n_dyn: usize,
        n_water: usize,
        cutoff_ewald: f32,
        alpha_ewald: f32,
    ) -> Self {
        // CUDA host/device allocation APIs do not accept zero-length allocations.
        // Keep one unused element for systems with no solute or no water.
        let n_dyn_storage = n_dyn.max(1);
        let n_water_storage = n_water.max(1);

        // Set up empty device arrays the kernel will fill as output.
        let forces_on_dyn = stream.alloc_zeros::<f32>(n_dyn_storage * 3).unwrap();
        let forces_on_water_o = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let forces_on_water_m = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let forces_on_water_h0 = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let forces_on_water_h1 = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();

        let virial_gpu = stream.clone_htod(&[0.0f64]).unwrap();
        let energy_gpu = stream.clone_htod(&[0.0f64]).unwrap();
        let alch_dh_dl_gpu = stream.clone_htod(&[0.0f64]).unwrap();

        let pos_dyn = stream.alloc_zeros::<f32>(n_dyn_storage * 3).unwrap();
        let pos_w_o = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let pos_w_m = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let pos_w_h0 = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let pos_w_h1 = stream.alloc_zeros::<f32>(n_water_storage * 3).unwrap();
        let mut host_pos_dyn = PinnedBuffer::new(stream, n_dyn_storage * 3);
        let mut host_pos_w_o = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_pos_w_m = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_pos_w_h0 = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_pos_w_h1 = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_forces_dyn = PinnedBuffer::new(stream, n_dyn_storage * 3);
        let mut host_forces_w_o = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_forces_w_m = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_forces_w_h0 = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_forces_w_h1 = PinnedBuffer::new(stream, n_water_storage * 3);
        let mut host_virial = PinnedBuffer::new(stream, 1);
        let mut host_energy = PinnedBuffer::new(stream, 1);
        let mut host_alch_dh_dl = PinnedBuffer::new(stream, 1);

        for host in [
            &mut host_pos_dyn,
            &mut host_pos_w_o,
            &mut host_pos_w_m,
            &mut host_pos_w_h0,
            &mut host_pos_w_h1,
            &mut host_forces_dyn,
            &mut host_forces_w_o,
            &mut host_forces_w_m,
            &mut host_forces_w_h0,
            &mut host_forces_w_h1,
        ] {
            host.as_mut_slice().fill(0.0);
        }
        host_virial.as_mut_slice().fill(0.0);
        host_energy.as_mut_slice().fill(0.0);
        host_alch_dh_dl.as_mut_slice().fill(0.0);

        Self {
            forces_on_dyn,
            forces_on_water_o,
            forces_on_water_m,
            forces_on_water_h0,
            forces_on_water_h1,
            virial_gpu,
            energy_gpu,
            alch_dh_dl_gpu,
            cutoff_ewald,
            alpha_ewald,

            pos_dyn,
            pos_w_o,
            pos_w_m,
            pos_w_h0,
            pos_w_h1,
            host_pos_dyn,
            host_pos_w_o,
            host_pos_w_m,
            host_pos_w_h0,
            host_pos_w_h1,
            host_forces_dyn,
            host_forces_w_o,
            host_forces_w_m,
            host_forces_w_h0,
            host_forces_w_h1,
            host_virial,
            host_energy,
            host_alch_dh_dl,
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
    pub alch_interactions: CudaSlice<u8>,
    pub has_alchemical_interactions: bool,
}

impl PerNeighborGpu {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        pairs: &[NonBondedPair],
        atoms_dyn: &[AtomDynamics],
        water: &[WaterMolOpc],
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
        let mut alch_interactions = Vec::with_capacity(n);

        // Unpack BodyRef to fields. It doesn't map neatly to CUDA flattening primitives.

        // These atom and solvent types are so the Kernel can assign to the correct output arrays.
        // 0 means Dyn, 1 means Water.
        let mut atom_types_tgt = vec![0; n];
        // 0 for not-solvent or N/A. 1 = O, 2 = M, 3 = H0, 4 = H1.
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

                    // Mark so the kernel will use the solvent output.
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

                    // Mark so the kernel will use the solvent output. (In case of dyn/solvent symmetric)
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
            alch_interactions.push(pair.alch_interaction);
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
            assert_eq!(alch_interactions.len(), n);
        }

        let scale_14: Vec<_> = scale_14s.iter().map(|v| *v as u8).collect();
        let calc_ljs: Vec<_> = calc_ljs.iter().map(|v| *v as u8).collect();
        let calc_coulombs: Vec<_> = calc_coulombs.iter().map(|v| *v as u8).collect();
        let symmetric: Vec<_> = symmetric.iter().map(|v| *v as u8).collect();
        let has_alchemical_interactions = alch_interactions.iter().any(|v| *v);
        let alch_interactions: Vec<_> = alch_interactions.iter().map(|v| *v as u8).collect();

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
        let alch_interactions = stream.clone_htod(&alch_interactions).unwrap();

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
            alch_interactions,
            has_alchemical_interactions,
        }
    }
}

/// Run this each step, at the start of each GPU step.
fn upload_positions(
    stream: &Arc<CudaStream>,
    forces: &mut ForcesPositsGpu,
    atoms_dyn: &[AtomDynamics],
    water: &[WaterMolOpc],
) {
    {
        let host = forces.host_pos_dyn.as_mut_slice();
        for (dst, atom) in host.chunks_exact_mut(3).zip(atoms_dyn) {
            dst.copy_from_slice(&atom.posit.to_arr());
        }
    }
    {
        let host_o = forces.host_pos_w_o.as_mut_slice();
        let host_m = forces.host_pos_w_m.as_mut_slice();
        let host_h0 = forces.host_pos_w_h0.as_mut_slice();
        let host_h1 = forces.host_pos_w_h1.as_mut_slice();
        for (i, molecule) in water.iter().enumerate() {
            let offset = 3 * i;
            host_o[offset..offset + 3].copy_from_slice(&molecule.o.posit.to_arr());
            host_m[offset..offset + 3].copy_from_slice(&molecule.m.posit.to_arr());
            host_h0[offset..offset + 3].copy_from_slice(&molecule.h0.posit.to_arr());
            host_h1[offset..offset + 3].copy_from_slice(&molecule.h1.posit.to_arr());
        }
    }

    stream
        .memcpy_htod(&forces.host_pos_dyn, &mut forces.pos_dyn)
        .unwrap();
    stream
        .memcpy_htod(&forces.host_pos_w_o, &mut forces.pos_w_o)
        .unwrap();
    stream
        .memcpy_htod(&forces.host_pos_w_m, &mut forces.pos_w_m)
        .unwrap();
    stream
        .memcpy_htod(&forces.host_pos_w_h0, &mut forces.pos_w_h0)
        .unwrap();
    stream
        .memcpy_htod(&forces.host_pos_w_h1, &mut forces.pos_w_h1)
        .unwrap();
}

/// Handles both LJ, and Coulomb (SPME short range) force using a shared kernel. Run this every step.
/// Inputs are structured differently here from our other one; uses pre-paired inputs and outputs, and
/// a common index. Exclusions (e.g. Amber-style 1-2 adn 1-3) are handled upstream.
///
/// Returns (force on non-solvent, force on solvent, virial sum, potential energy total,
/// per-mol-pair potential energy, alchemical dH/dlambda)
pub fn force_nonbonded_gpu(
    stream: &Arc<CudaStream>,
    kernels: &GpuKernels,
    pairs: &[NonBondedPair],
    atoms_dyn: &[AtomDynamics],
    water: &[WaterMolOpc],
    // todo: Only copy cell_extent when it changes, e.g. due to the barostat.
    cell_extent: Vec3,
    forces: &mut ForcesPositsGpu,
    per_neighbor: &PerNeighborGpu,
    overrides: &MdOverrides,
    lambda_alch: f64,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    upload_positions(stream, forces, atoms_dyn, water);

    let n = pairs.len();

    zero_forces_and_accums(stream, forces);

    // 1-4 scaling, and the symmetric case handled in the kernel.
    // Store immutable input arrays to the device.

    let n_u32 = n as u32;
    let coulomb_disabled = overrides.coulomb_disabled as u8;
    let lj_disabled = overrides.lj_disabled as u8;
    let alchemical_enabled = per_neighbor.has_alchemical_interactions;
    let lambda_alch = lambda_alch as f32;
    let cell_inv_extent = Vec3::new(
        cell_extent.x.recip(),
        cell_extent.y.recip(),
        cell_extent.z.recip(),
    );

    let cfg = LaunchConfig::for_num_elems(n_u32);
    let kernel_to_launch = if alchemical_enabled {
        &kernels.alchemical
    } else {
        &kernels.primary
    };

    let mut launch_args = stream.launch_builder(kernel_to_launch);

    // These forces and positions are per-atom; much smaller than the per-pair arrays.
    launch_args.arg(&mut forces.forces_on_dyn);
    launch_args.arg(&mut forces.forces_on_water_o);
    launch_args.arg(&mut forces.forces_on_water_m);
    launch_args.arg(&mut forces.forces_on_water_h0);
    launch_args.arg(&mut forces.forces_on_water_h1);
    //
    launch_args.arg(&mut forces.virial_gpu);
    launch_args.arg(&mut forces.energy_gpu);
    if alchemical_enabled {
        launch_args.arg(&mut forces.alch_dh_dl_gpu);
    }
    //
    launch_args.arg(&forces.pos_dyn);
    launch_args.arg(&forces.pos_w_o);
    launch_args.arg(&forces.pos_w_m);
    launch_args.arg(&forces.pos_w_h0);
    launch_args.arg(&forces.pos_w_h1);

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

    if alchemical_enabled {
        launch_args.arg(&per_neighbor.alch_interactions);
    }

    launch_args.arg(&cell_extent);
    launch_args.arg(&cell_inv_extent);
    launch_args.arg(&forces.cutoff_ewald);
    launch_args.arg(&forces.alpha_ewald);
    launch_args.arg(&n_u32);
    launch_args.arg(&coulomb_disabled);
    launch_args.arg(&lj_disabled);

    if alchemical_enabled {
        launch_args.arg(&lambda_alch);
    }

    unsafe {
        if launch_args.launch(cfg).is_err() {
            eprintln!(
                "Error launching the non bonded GPU force kernel. (This can happen if there is one or\
                more NaNs in the system"
            );
        }
    }

    // Queue every result copy into persistent pinned buffers. Reading the final
    // scalar waits for the whole ordered stream once; the earlier copies then
    // require no additional device synchronization.
    stream
        .memcpy_dtoh(&forces.forces_on_dyn, &mut forces.host_forces_dyn)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.forces_on_water_o, &mut forces.host_forces_w_o)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.forces_on_water_m, &mut forces.host_forces_w_m)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.forces_on_water_h0, &mut forces.host_forces_w_h0)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.forces_on_water_h1, &mut forces.host_forces_w_h1)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.virial_gpu, &mut forces.host_virial)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.energy_gpu, &mut forces.host_energy)
        .unwrap();
    stream
        .memcpy_dtoh(&forces.alch_dh_dl_gpu, &mut forces.host_alch_dh_dl)
        .unwrap();

    stream.synchronize().unwrap();
    let alch_value = forces.host_alch_dh_dl.as_slice()[0];
    let forces_dyn_host = forces.host_forces_dyn.as_slice();
    let forces_o_host = forces.host_forces_w_o.as_slice();
    let forces_m_host = forces.host_forces_w_m.as_slice();
    let forces_h0_host = forces.host_forces_w_h0.as_slice();
    let forces_h1_host = forces.host_forces_w_h1.as_slice();

    let forces_on_dyn = forces_dyn_host
        .chunks_exact(3)
        .map(|f| Vec3F64::new(f[0] as f64, f[1] as f64, f[2] as f64))
        .collect();

    let mut forces_on_water = Vec::with_capacity(water.len());
    for i in 0..water.len() {
        let offset = 3 * i;
        let to_f64 =
            |f: &[f32]| Vec3F64::new(f[offset] as f64, f[offset + 1] as f64, f[offset + 2] as f64);
        forces_on_water.push(ForcesOnWaterMol {
            f_o: to_f64(forces_o_host),
            f_m: to_f64(forces_m_host),
            f_h0: to_f64(forces_h0_host),
            f_h1: to_f64(forces_h1_host),
        });
    }

    let virial = forces.host_virial.as_slice()[0];
    let energy = forces.host_energy.as_slice()[0];
    let alch_dh_dl = if alchemical_enabled { alch_value } else { 0.0 };

    (
        forces_on_dyn,
        forces_on_water,
        virial,
        energy,
        Vec::new(),
        alch_dh_dl,
    )
}

/// Zero forces and accumulators on the device. Run this each step.
fn zero_forces_and_accums(stream: &Arc<CudaStream>, forces: &mut ForcesPositsGpu) {
    // Driver-level async memsets avoid eight kernel launches per MD step.
    stream.memset_zeros(&mut forces.forces_on_dyn).unwrap();
    stream.memset_zeros(&mut forces.forces_on_water_o).unwrap();
    stream.memset_zeros(&mut forces.forces_on_water_m).unwrap();
    stream.memset_zeros(&mut forces.forces_on_water_h0).unwrap();
    stream.memset_zeros(&mut forces.forces_on_water_h1).unwrap();
    stream.memset_zeros(&mut forces.virial_gpu).unwrap();
    stream.memset_zeros(&mut forces.energy_gpu).unwrap();
    stream.memset_zeros(&mut forces.alch_dh_dl_gpu).unwrap();
}
