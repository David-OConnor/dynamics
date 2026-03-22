//! An MD sim which can be used for tests. Copy + paste + modify from `molchanica`.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

use crate::{
    ComputationDevice, MdConfig, MdState, MolDynamics, ParamError, SimBoxInit, params::FfParamSet,
};

pub(in crate::tests) fn build_dynamics(
    mols: Vec<MolDynamics>,
    param_set: &FfParamSet,
    cfg: &MdConfig,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    #[cfg(feature = "cuda")]
    let dev = {
        let stream = {
            let ctx = CudaContext::new(0).unwrap();
            ctx.default_stream()
        };

        ComputationDevice::Gpu(stream)
    };

    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu; // todo: For now.

    // Extract explicit box side-lengths so add_copies can keep molecules inside the boundary.
    // Only meaningful for Fixed boxes; Pad boxes are sized after molecule placement so we skip them.
    let box_dims = match &cfg.sim_box {
        SimBoxInit::Fixed((lo, hi)) => Some((hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)),
        SimBoxInit::Pad(_) => None,
    };

    println!("Initializing MD state...");
    let md_state = MdState::new(&dev, &cfg, &mols, param_set)?;
    println!("MD init done.");

    Ok(md_state)
}

/// Run the dynamics in one go. Blocking.
pub(in crate::tests) fn run_dynamics_blocking(
    md_state: &mut MdState,
    dev: &ComputationDevice,
    dt: f32,
    n_steps: usize,
) {
    if n_steps == 0 {
        return;
    }

    let i_20_pc = n_steps / 5;
    let mut disp_count = 0;
    for i in 0..n_steps {
        if i.is_multiple_of(i_20_pc) {
            println!("{}% Complete", disp_count * 20);
            disp_count += 1;
        }

        md_state.step(dev, dt, None);
    }
}
