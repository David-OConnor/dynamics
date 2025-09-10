//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_ptx};

fn main() {
    #[cfg(feature = "cuda")]
    build_ptx(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/cuda.cu", "src/cuda/util.cu"],
        "dynamics",
    );
}
