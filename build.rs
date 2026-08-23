//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "cuda")]
use std::{env, path::PathBuf, process::Command};

#[cfg(feature = "cuda")]
use cuda_setup::GpuArchitecture;

fn main() {
    #[cfg(feature = "cuda")]
    build_ptx();
}

#[cfg(feature = "cuda")]
fn build_ptx() {
    let cuda_files = ["src/cuda/cuda.cu", "src/cuda/util.cu"];
    for file in cuda_files {
        println!("cargo:rerun-if-changed={file}");
    }

    // Keep generated PTX inside Cargo's build directory. Writing to the repository root made
    // debug/release builds share an ignored `dynamics.ptx`, which could leave Rust launch
    // arguments paired with a kernel compiled from an older signature.
    let output_path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dynamics.ptx");
    let output = Command::new("nvcc")
        .args([
            cuda_files[0],
            &GpuArchitecture::Rtx3.compute_val(),
            "-ptx",
            "-O3",
            "-o",
        ])
        .arg(&output_path)
        .output()
        .unwrap_or_else(|error| panic!("Unable to run nvcc; is it installed and on PATH? {error}"));

    if !output.status.success() {
        panic!(
            "CUDA PTX compilation problem:\nstatus: {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
