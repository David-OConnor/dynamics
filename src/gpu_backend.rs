//! Dispatches GPU operations to the CUDA or HIP backend.

use std::{fmt::Display, sync::Arc};

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{
        CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, HostSlice, LaunchArgs,
        LaunchConfig, PushKernelArg, SyncOnDrop, ValidAsZeroBits, result,
    },
    nvrtc::Ptx,
};

use crate::ComputationDevice;
#[cfg(feature = "hip")]
use crate::hip::{self, HipModule, HipStream};

#[cfg(feature = "cuda")]
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/dynamics.ptx"));

/// AMD code object bundle, built by `build.rs`. Aligned, as the HIP loader parses it in place.
#[cfg(feature = "hip")]
static HIP_CODE_OBJECT: &Aligned<[u8]> =
    &Aligned(*include_bytes!(concat!(env!("OUT_DIR"), "/dynamics.hipfb")));

#[cfg(feature = "hip")]
#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

/// Element types we copy between the host and GPU.
///
/// # Safety
/// All-zero bits must be a valid value.
#[cfg(feature = "cuda")]
pub(crate) unsafe trait GpuElem:
    Copy + Send + Sync + DeviceRepr + ValidAsZeroBits
{
}
#[cfg(not(feature = "cuda"))]
pub(crate) unsafe trait GpuElem: Copy + Send + Sync {}

unsafe impl GpuElem for u8 {}
unsafe impl GpuElem for u32 {}
unsafe impl GpuElem for f32 {}
unsafe impl GpuElem for f64 {}

fn err(e: impl Display) -> String {
    e.to_string()
}

fn mixed() -> ! {
    panic!("GPU stream and buffers are from different backends")
}

impl ComputationDevice {
    /// This device's GPU stream, if it's a GPU.
    pub(crate) fn gpu_stream(&self) -> Option<GpuStream<'_>> {
        match self {
            Self::Cpu => None,
            #[cfg(feature = "cuda")]
            Self::Gpu(s) => Some(GpuStream::Cuda(s)),
            #[cfg(feature = "hip")]
            Self::Hip(s) => Some(GpuStream::Hip(s)),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum GpuStream<'a> {
    #[cfg(feature = "cuda")]
    Cuda(&'a Arc<CudaStream>),
    #[cfg(feature = "hip")]
    Hip(&'a Arc<HipStream>),
}

impl<'a> GpuStream<'a> {
    pub(crate) fn alloc_zeros<T: GpuElem>(self, len: usize) -> Result<DevBuf<T>, String> {
        Ok(match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => DevBuf::Cuda(s.alloc_zeros(len).map_err(err)?),
            #[cfg(feature = "hip")]
            Self::Hip(s) => DevBuf::Hip(s.alloc_zeros(len).map_err(err)?),
        })
    }

    /// Copies a host slice to new device memory.
    pub(crate) fn clone_htod<T: GpuElem>(self, src: &[T]) -> Result<DevBuf<T>, String> {
        Ok(match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => DevBuf::Cuda(s.clone_htod(src).map_err(err)?),
            #[cfg(feature = "hip")]
            Self::Hip(s) => DevBuf::Hip(s.clone_htod(src).map_err(err)?),
        })
    }

    pub(crate) fn memcpy_htod<T: GpuElem>(
        self,
        src: &PinnedBuf<T>,
        dst: &mut DevBuf<T>,
    ) -> Result<(), String> {
        match (self, src, dst) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(s), PinnedBuf::Cuda(src), DevBuf::Cuda(dst)) => {
                s.memcpy_htod(src, dst).map_err(err)
            }
            #[cfg(feature = "hip")]
            (Self::Hip(s), PinnedBuf::Hip(src), DevBuf::Hip(dst)) => {
                s.memcpy_htod(src, dst).map_err(err)
            }
            #[allow(unreachable_patterns)]
            _ => mixed(),
        }
    }

    pub(crate) fn memcpy_dtoh<T: GpuElem>(
        self,
        src: &DevBuf<T>,
        dst: &mut PinnedBuf<T>,
    ) -> Result<(), String> {
        match (self, src, dst) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(s), DevBuf::Cuda(src), PinnedBuf::Cuda(dst)) => {
                s.memcpy_dtoh(src, dst).map_err(err)
            }
            #[cfg(feature = "hip")]
            (Self::Hip(s), DevBuf::Hip(src), PinnedBuf::Hip(dst)) => {
                s.memcpy_dtoh(src, dst).map_err(err)
            }
            #[allow(unreachable_patterns)]
            _ => mixed(),
        }
    }

    pub(crate) fn memset_zeros<T: GpuElem>(self, dst: &mut DevBuf<T>) -> Result<(), String> {
        match (self, dst) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(s), DevBuf::Cuda(dst)) => s.memset_zeros(dst).map_err(err),
            #[cfg(feature = "hip")]
            (Self::Hip(s), DevBuf::Hip(dst)) => s.memset_zeros(dst).map_err(err),
            #[allow(unreachable_patterns)]
            _ => mixed(),
        }
    }

    pub(crate) fn synchronize(self) -> Result<(), String> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => s.synchronize().map_err(err),
            #[cfg(feature = "hip")]
            Self::Hip(s) => s.synchronize().map_err(err),
        }
    }

    /// Loads the kernel module built by `build.rs`.
    pub(crate) fn load_module(self) -> Result<GpuModule, String> {
        Ok(match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => {
                GpuModule::Cuda(s.context().load_module(Ptx::from_src(PTX)).map_err(err)?)
            }
            #[cfg(feature = "hip")]
            Self::Hip(s) => GpuModule::Hip(s.load_module(&HIP_CODE_OBJECT.0).map_err(err)?),
        })
    }

    pub(crate) fn launch_builder(self, kernel: &'a Kernel) -> Launch<'a> {
        match (self, kernel) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(s), Kernel::Cuda(f)) => Launch::Cuda(s.launch_builder(f)),
            #[cfg(feature = "hip")]
            (Self::Hip(s), Kernel::Hip(f)) => Launch::Hip(s.launch_builder(f)),
            #[allow(unreachable_patterns)]
            _ => mixed(),
        }
    }
}

/// Device memory.
pub(crate) enum DevBuf<T: GpuElem> {
    #[cfg(feature = "cuda")]
    Cuda(CudaSlice<T>),
    #[cfg(feature = "hip")]
    Hip(hip::DeviceBuf<T>),
}

impl<T: GpuElem> DevBuf<T> {
    pub(crate) fn len(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.len(),
            #[cfg(feature = "hip")]
            Self::Hip(b) => b.len(),
        }
    }
}

/// Page-locked host memory, for async copies.
pub(crate) enum PinnedBuf<T: GpuElem> {
    #[cfg(feature = "cuda")]
    Cuda(CudaPinned<T>),
    #[cfg(feature = "hip")]
    Hip(hip::PinnedBuf<T>),
}

impl<T: GpuElem> PinnedBuf<T> {
    /// Allocates zeroed memory.
    pub(crate) fn new(stream: GpuStream, len: usize) -> Self {
        match stream {
            #[cfg(feature = "cuda")]
            GpuStream::Cuda(s) => Self::Cuda(CudaPinned::new(s, len)),
            #[cfg(feature = "hip")]
            GpuStream::Hip(s) => Self::Hip(hip::PinnedBuf::new(s, len).unwrap()),
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.as_slice(),
            #[cfg(feature = "hip")]
            Self::Hip(b) => b.as_slice(),
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.as_mut_slice(),
            #[cfg(feature = "hip")]
            Self::Hip(b) => b.as_mut_slice(),
        }
    }
}

pub(crate) enum GpuModule {
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaModule>),
    #[cfg(feature = "hip")]
    Hip(Arc<HipModule>),
}

impl GpuModule {
    pub(crate) fn function(&self, name: &str) -> Result<Kernel, String> {
        Ok(match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(m) => Kernel::Cuda(m.load_function(name).map_err(err)?),
            #[cfg(feature = "hip")]
            Self::Hip(m) => Kernel::Hip(m.load_function(name).map_err(err)?),
        })
    }
}

pub(crate) enum Kernel {
    #[cfg(feature = "cuda")]
    Cuda(CudaFunction),
    #[cfg(feature = "hip")]
    Hip(hip::HipFunction),
}

/// Kernel arguments being assembled for a launch.
pub(crate) enum Launch<'a> {
    #[cfg(feature = "cuda")]
    Cuda(LaunchArgs<'a>),
    #[cfg(feature = "hip")]
    Hip(hip::LaunchArgs<'a>),
}

impl<'a> Launch<'a> {
    pub(crate) fn arg(&mut self, arg: impl KernelArg<'a>) -> &mut Self {
        arg.push(self);
        self
    }

    /// Launches one thread per element, over `n` elements.
    ///
    /// # Safety
    /// The arguments must match the kernel's signature.
    pub(crate) unsafe fn launch(&mut self, n: u32) -> Result<(), String> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(l) => unsafe { l.launch(LaunchConfig::for_num_elems(n)) }
                .map(|_| ())
                .map_err(err),
            #[cfg(feature = "hip")]
            Self::Hip(l) => unsafe { l.launch(n) }.map_err(err),
        }
    }
}

/// A value we can pass as a kernel argument: a device buffer, or a scalar.
pub(crate) trait KernelArg<'a> {
    fn push(self, launch: &mut Launch<'a>);
}

impl<'a, T: GpuElem> KernelArg<'a> for &'a DevBuf<T> {
    fn push(self, launch: &mut Launch<'a>) {
        match (launch, self) {
            #[cfg(feature = "cuda")]
            (Launch::Cuda(l), DevBuf::Cuda(b)) => {
                l.arg(b);
            }
            #[cfg(feature = "hip")]
            (Launch::Hip(l), DevBuf::Hip(b)) => {
                l.buf(b);
            }
            #[allow(unreachable_patterns)]
            _ => mixed(),
        }
    }
}

impl<'a, T: GpuElem> KernelArg<'a> for &'a mut DevBuf<T> {
    fn push(self, launch: &mut Launch<'a>) {
        match (launch, self) {
            #[cfg(feature = "cuda")]
            (Launch::Cuda(l), DevBuf::Cuda(b)) => {
                l.arg(b);
            }
            #[cfg(feature = "hip")]
            (Launch::Hip(l), DevBuf::Hip(b)) => {
                l.buf(b);
            }
            #[allow(unreachable_patterns)]
            _ => mixed(),
        }
    }
}

impl<'a, T: GpuElem> KernelArg<'a> for &'a T {
    fn push(self, launch: &mut Launch<'a>) {
        match launch {
            #[cfg(feature = "cuda")]
            Launch::Cuda(l) => {
                l.arg(self);
            }
            #[cfg(feature = "hip")]
            Launch::Hip(l) => {
                l.scalar(self);
            }
        }
    }
}

/// Page-locked host memory without the write-combined flag used by cudarc's
/// `PinnedHostSlice`. Write-combined memory is ideal for H2D staging but very
/// slow for the CPU reads required after D2H force copies.
#[cfg(feature = "cuda")]
pub(crate) struct CudaPinned<T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
unsafe impl<T: DeviceRepr> Send for CudaPinned<T> {}
#[cfg(feature = "cuda")]
unsafe impl<T: DeviceRepr> Sync for CudaPinned<T> {}

#[cfg(feature = "cuda")]
impl<T: GpuElem> CudaPinned<T> {
    /// Allocates zeroed memory.
    fn new(stream: &Arc<CudaStream>, len: usize) -> Self {
        let ptr = unsafe { result::malloc_host(len * size_of::<T>(), 0).unwrap() } as *mut T;
        assert!(!ptr.is_null());

        unsafe { std::ptr::write_bytes(ptr, 0, len) };

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

#[cfg(feature = "cuda")]
impl<T: DeviceRepr> Drop for CudaPinned<T> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = unsafe { result::free_host(self.ptr.cast()) };
    }
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr> HostSlice<T> for CudaPinned<T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) },
            SyncOnDrop::Sync(None),
        )
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) },
            SyncOnDrop::Sync(None),
        )
    }
}
