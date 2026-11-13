//! We use this to automatically compile CUDA C++ code when building. With the `hip` feature, we
//! compile the same source for AMD GPUs.

#[cfg(any(feature = "cuda", feature = "hip"))]
use std::{env, path::PathBuf, process::Command};

#[cfg(feature = "cuda")]
use cuda_setup::GpuArchitecture;

#[cfg(any(feature = "cuda", feature = "hip"))]
const GPU_FILES: [&str; 2] = ["src/cuda/cuda.cu", "src/cuda/util.cu"];

/// AMD GPU targets to build for, unless overridden by `DYNAMICS_HIP_ARCHS` (comma-separated).
/// CDNA (MI100, MI200, MI300), RDNA2, RDNA3, Strix Halo, and RDNA4.
#[cfg(feature = "hip")]
const HIP_ARCHS_DEFAULT: &str =
    "gfx908,gfx90a,gfx942,gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1200,gfx1201";

fn main() {
    #[cfg(feature = "cuda")]
    build_ptx();

    #[cfg(feature = "hip")]
    build_hip();
}

#[cfg(feature = "cuda")]
fn build_ptx() {
    for file in GPU_FILES {
        println!("cargo:rerun-if-changed={file}");
    }

    // Keep generated PTX inside Cargo's build directory. Writing to the repository root made
    // debug/release builds share an ignored `dynamics.ptx`, which could leave Rust launch
    // arguments paired with a kernel compiled from an older signature.
    let output_path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dynamics.ptx");
    let output = Command::new("nvcc")
        .args([
            GPU_FILES[0],
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

/// Compiles a code object bundle containing a binary for each AMD GPU target.
#[cfg(feature = "hip")]
fn build_hip() {
    for file in GPU_FILES {
        println!("cargo:rerun-if-changed={file}");
    }
    for var in ["DYNAMICS_HIP_ARCHS", "HIP_PATH", "ROCM_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let archs = env::var("DYNAMICS_HIP_ARCHS").unwrap_or_else(|_| HIP_ARCHS_DEFAULT.to_owned());
    let output_path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dynamics.hipfb");

    let hipcc = find_hipcc();
    let mut cmd = Command::new(&hipcc);
    // Our buffers are device memory, where hardware float atomics are safe.
    cmd.args(["--genco", "-O3", "-munsafe-fp-atomics", "-x", "hip"]);
    for arch in archs.split(',').map(str::trim).filter(|a| !a.is_empty()) {
        cmd.arg(format!("--offload-arch={arch}"));
    }
    cmd.arg(GPU_FILES[0]).arg("-o").arg(&output_path);

    let output = cmd.output().unwrap_or_else(|error| {
        panic!(
            "Unable to run {}; is the HIP SDK (ROCm) installed? Set HIP_PATH, or put hipcc on PATH. {error}",
            hipcc.display()
        )
    });

    if !output.status.success() {
        panic!(
            "HIP compilation problem:\nstatus: {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

/// Looks under `HIP_PATH` (set by the Windows HIP SDK) and `ROCM_PATH`, then falls back to PATH.
#[cfg(feature = "hip")]
fn find_hipcc() -> PathBuf {
    let names: &[&str] = if cfg!(windows) {
        &["hipcc.exe", "hipcc.bat"]
    } else {
        &["hipcc"]
    };

    for var in ["HIP_PATH", "ROCM_PATH"] {
        let Some(root) = env::var_os(var) else {
            continue;
        };

        for name in names {
            let path = PathBuf::from(&root).join("bin").join(name);
            if path.exists() {
                return path;
            }
        }
    }

    PathBuf::from("hipcc")
}
