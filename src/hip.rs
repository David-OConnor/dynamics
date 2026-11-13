//! Minimal HIP runtime bindings, for running our kernels on AMD GPUs. The runtime library is loaded
//! on first use, so binaries built with the `hip` feature still start where it isn't installed.

use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::{size_of, size_of_val},
    ptr::{self, NonNull},
    sync::{Arc, OnceLock},
};

use libloading::Library;

use crate::gpu_backend::GpuElem;

type RawStream = *mut c_void;
type RawModule = *mut c_void;
type RawFunction = *mut c_void;

const HIP_SUCCESS: c_int = 0;
const MEMCPY_HOST_TO_DEVICE: c_int = 1;
const MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// Threads per block for kernel launches.
const BLOCK_SIZE: u32 = 1024;

#[cfg(windows)]
const LIB_NAMES: &[&str] = &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
#[cfg(not(windows))]
const LIB_NAMES: &[&str] = &["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"];

/// An error from the HIP runtime, or from loading it.
#[derive(Clone, Debug)]
pub struct HipError(String);

impl Display for HipError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HipError {}

/// The HIP runtime functions we use.
struct Api {
    init: unsafe extern "C" fn(c_uint) -> c_int,
    get_device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    set_device: unsafe extern "C" fn(c_int) -> c_int,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    stream_create: unsafe extern "C" fn(*mut RawStream) -> c_int,
    stream_destroy: unsafe extern "C" fn(RawStream) -> c_int,
    stream_synchronize: unsafe extern "C" fn(RawStream) -> c_int,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    free: unsafe extern "C" fn(*mut c_void) -> c_int,
    host_malloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int,
    host_free: unsafe extern "C" fn(*mut c_void) -> c_int,
    memcpy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, RawStream) -> c_int,
    memset_async: unsafe extern "C" fn(*mut c_void, c_int, usize, RawStream) -> c_int,
    module_load_data: unsafe extern "C" fn(*mut RawModule, *const c_void) -> c_int,
    module_unload: unsafe extern "C" fn(RawModule) -> c_int,
    module_get_function: unsafe extern "C" fn(*mut RawFunction, RawModule, *const c_char) -> c_int,
    /// Function, grid dims, block dims, shared mem bytes, stream, kernel params, extra.
    #[allow(clippy::type_complexity)]
    module_launch_kernel: unsafe extern "C" fn(
        RawFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        RawStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
    /// Keeps the function pointers above valid.
    _lib: Library,
}

impl Api {
    fn load() -> Result<Self, String> {
        // Copies a function pointer out of the library.
        unsafe fn sym<T: Copy>(lib: &Library, name: &str) -> Result<T, String> {
            unsafe { lib.get::<T>(name) }
                .map(|s| *s)
                .map_err(|e| format!("HIP runtime is missing `{name}`: {e}"))
        }

        let lib = LIB_NAMES
            .iter()
            .find_map(|name| unsafe { Library::new(*name) }.ok())
            .ok_or_else(|| format!("Unable to load the HIP runtime ({})", LIB_NAMES.join(", ")))?;

        unsafe {
            Ok(Self {
                init: sym(&lib, "hipInit")?,
                get_device_count: sym(&lib, "hipGetDeviceCount")?,
                set_device: sym(&lib, "hipSetDevice")?,
                device_get_name: sym(&lib, "hipDeviceGetName")?,
                get_error_string: sym(&lib, "hipGetErrorString")?,
                stream_create: sym(&lib, "hipStreamCreate")?,
                stream_destroy: sym(&lib, "hipStreamDestroy")?,
                stream_synchronize: sym(&lib, "hipStreamSynchronize")?,
                malloc: sym(&lib, "hipMalloc")?,
                free: sym(&lib, "hipFree")?,
                host_malloc: sym(&lib, "hipHostMalloc")?,
                host_free: sym(&lib, "hipHostFree")?,
                memcpy_async: sym(&lib, "hipMemcpyAsync")?,
                memset_async: sym(&lib, "hipMemsetAsync")?,
                module_load_data: sym(&lib, "hipModuleLoadData")?,
                module_unload: sym(&lib, "hipModuleUnload")?,
                module_get_function: sym(&lib, "hipModuleGetFunction")?,
                module_launch_kernel: sym(&lib, "hipModuleLaunchKernel")?,
                _lib: lib,
            })
        }
    }

    fn check(&self, code: c_int) -> Result<(), HipError> {
        if code == HIP_SUCCESS {
            return Ok(());
        }

        let msg = unsafe { (self.get_error_string)(code) };
        let msg = if msg.is_null() {
            "unknown".into()
        } else {
            unsafe { CStr::from_ptr(msg) }.to_string_lossy()
        };

        Err(HipError(format!("HIP error {code}: {msg}")))
    }
}

fn api() -> Result<&'static Api, HipError> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();

    API.get_or_init(Api::load)
        .as_ref()
        .map_err(|e| HipError(e.clone()))
}

/// A stream on an AMD GPU. Use with `ComputationDevice::Hip`.
#[derive(Debug)]
pub struct HipStream {
    raw: RawStream,
    device: c_int,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    /// Creates a stream on the device with this index. Errors if the HIP runtime or device isn't
    /// available.
    pub fn new(device: usize) -> Result<Arc<Self>, HipError> {
        let api = api()?;
        let device =
            c_int::try_from(device).map_err(|_| HipError("Invalid device index".into()))?;

        let mut raw = ptr::null_mut();
        unsafe {
            api.check((api.init)(0))?;
            api.check((api.set_device)(device))?;
            api.check((api.stream_create)(&mut raw))?;
        }

        Ok(Arc::new(Self { raw, device }))
    }

    /// The number of GPUs available to HIP.
    pub fn device_count() -> Result<usize, HipError> {
        let api = api()?;

        let mut count = 0;
        unsafe {
            api.check((api.init)(0))?;
            api.check((api.get_device_count)(&mut count))?;
        }

        Ok(count.max(0) as usize)
    }

    pub fn device_name(&self) -> Result<String, HipError> {
        let api = api()?;

        let mut name = [0 as c_char; 256];
        unsafe {
            api.check((api.device_get_name)(
                name.as_mut_ptr(),
                name.len() as c_int,
                self.device,
            ))?;
        }

        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Makes this stream's device current on the calling thread.
    fn bind(&self) -> Result<&'static Api, HipError> {
        let api = api()?;
        unsafe { api.check((api.set_device)(self.device))? };

        Ok(api)
    }

    pub(crate) fn synchronize(&self) -> Result<(), HipError> {
        let api = api()?;

        unsafe { api.check((api.stream_synchronize)(self.raw)) }
    }

    /// Allocates uninitialized device memory.
    fn alloc<T: GpuElem>(&self, len: usize) -> Result<DeviceBuf<T>, HipError> {
        let api = self.bind()?;

        let mut ptr = ptr::null_mut();
        let bytes = len * size_of::<T>();
        if bytes > 0 {
            unsafe { api.check((api.malloc)(&mut ptr, bytes))? };
        }

        Ok(DeviceBuf {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub(crate) fn alloc_zeros<T: GpuElem>(&self, len: usize) -> Result<DeviceBuf<T>, HipError> {
        let mut buf = self.alloc(len)?;
        self.memset_zeros(&mut buf)?;

        Ok(buf)
    }

    /// Copies a host slice to new device memory. Blocks until complete.
    pub(crate) fn clone_htod<T: GpuElem>(&self, src: &[T]) -> Result<DeviceBuf<T>, HipError> {
        let buf = self.alloc(src.len())?;
        unsafe {
            self.memcpy(
                buf.ptr,
                src.as_ptr().cast(),
                size_of_val(src),
                MEMCPY_HOST_TO_DEVICE,
            )?;
        }
        self.synchronize()?;

        Ok(buf)
    }

    /// Queues a copy from pinned host memory to the device.
    pub(crate) fn memcpy_htod<T: GpuElem>(
        &self,
        src: &PinnedBuf<T>,
        dst: &mut DeviceBuf<T>,
    ) -> Result<(), HipError> {
        assert!(dst.len >= src.len);

        unsafe {
            self.memcpy(
                dst.ptr,
                src.ptr.cast(),
                size_of_val(src.as_slice()),
                MEMCPY_HOST_TO_DEVICE,
            )
        }
    }

    /// Queues a copy from the device to pinned host memory.
    pub(crate) fn memcpy_dtoh<T: GpuElem>(
        &self,
        src: &DeviceBuf<T>,
        dst: &mut PinnedBuf<T>,
    ) -> Result<(), HipError> {
        assert!(dst.len >= src.len);

        unsafe {
            self.memcpy(
                dst.ptr.cast(),
                src.ptr,
                src.len * size_of::<T>(),
                MEMCPY_DEVICE_TO_HOST,
            )
        }
    }

    unsafe fn memcpy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> Result<(), HipError> {
        if bytes == 0 {
            return Ok(());
        }

        let api = self.bind()?;

        unsafe { api.check((api.memcpy_async)(dst, src, bytes, kind, self.raw)) }
    }

    /// Queues zeroing device memory.
    pub(crate) fn memset_zeros<T: GpuElem>(&self, dst: &mut DeviceBuf<T>) -> Result<(), HipError> {
        let bytes = dst.len * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }

        let api = self.bind()?;

        unsafe { api.check((api.memset_async)(dst.ptr, 0, bytes, self.raw)) }
    }

    /// Loads a code object, e.g. from `hipcc --genco`.
    pub(crate) fn load_module(&self, image: &[u8]) -> Result<Arc<HipModule>, HipError> {
        let api = self.bind()?;

        let mut raw = ptr::null_mut();
        unsafe { api.check((api.module_load_data)(&mut raw, image.as_ptr().cast()))? };

        Ok(Arc::new(HipModule { raw }))
    }

    pub(crate) fn launch_builder<'a>(&'a self, func: &'a HipFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream: self,
            func,
            args: Vec::new(),
        }
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if let Ok(api) = self.bind() {
            unsafe {
                (api.stream_synchronize)(self.raw);
                (api.stream_destroy)(self.raw);
            }
        }
    }
}

/// Device memory.
pub(crate) struct DeviceBuf<T> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuf<T> {}
unsafe impl<T: Sync> Sync for DeviceBuf<T> {}

impl<T> DeviceBuf<T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null()
            && let Ok(api) = api()
        {
            unsafe { (api.free)(self.ptr) };
        }
    }
}

/// Page-locked host memory, for async copies.
pub(crate) struct PinnedBuf<T> {
    ptr: *mut T,
    len: usize,
    /// Synchronized on drop, as queued copies may still use this memory.
    stream: Arc<HipStream>,
}

unsafe impl<T: Send> Send for PinnedBuf<T> {}
unsafe impl<T: Sync> Sync for PinnedBuf<T> {}

impl<T: GpuElem> PinnedBuf<T> {
    /// Allocates zeroed memory.
    pub(crate) fn new(stream: &Arc<HipStream>, len: usize) -> Result<Self, HipError> {
        let bytes = len * size_of::<T>();

        let ptr = if bytes == 0 {
            NonNull::dangling().as_ptr()
        } else {
            let api = stream.bind()?;

            let mut ptr = ptr::null_mut();
            unsafe { api.check((api.host_malloc)(&mut ptr, bytes, 0))? };

            let ptr = ptr.cast::<T>();
            unsafe { ptr::write_bytes(ptr, 0, len) };
            ptr
        };

        Ok(Self {
            ptr,
            len,
            stream: stream.clone(),
        })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();

        if self.len * size_of::<T>() > 0
            && let Ok(api) = api()
        {
            unsafe { (api.host_free)(self.ptr.cast()) };
        }
    }
}

pub(crate) struct HipModule {
    raw: RawModule,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    pub(crate) fn load_function(self: &Arc<Self>, name: &str) -> Result<HipFunction, HipError> {
        let api = api()?;
        let name_c = CString::new(name).map_err(|_| HipError("Invalid kernel name".into()))?;

        let mut raw = ptr::null_mut();
        unsafe {
            api.check((api.module_get_function)(
                &mut raw,
                self.raw,
                name_c.as_ptr(),
            ))?;
        }

        Ok(HipFunction {
            raw,
            _module: self.clone(),
        })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            unsafe { (api.module_unload)(self.raw) };
        }
    }
}

/// A kernel. Keeps its module loaded.
pub(crate) struct HipFunction {
    raw: RawFunction,
    _module: Arc<HipModule>,
}

unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}

/// Kernel arguments, in order. Borrows each for the launch.
pub(crate) struct LaunchArgs<'a> {
    stream: &'a HipStream,
    func: &'a HipFunction,
    /// A pointer to each argument's value.
    args: Vec<*mut c_void>,
}

impl<'a> LaunchArgs<'a> {
    pub(crate) fn buf<T>(&mut self, buf: &'a DeviceBuf<T>) -> &mut Self {
        self.args.push((&raw const buf.ptr).cast_mut().cast());
        self
    }

    pub(crate) fn scalar<T: GpuElem>(&mut self, val: &'a T) -> &mut Self {
        self.args.push(ptr::from_ref(val).cast_mut().cast());
        self
    }

    /// Launches one thread per element, over `n` elements.
    ///
    /// # Safety
    /// The arguments must match the kernel's signature.
    pub(crate) unsafe fn launch(&mut self, n: u32) -> Result<(), HipError> {
        if n == 0 {
            return Ok(());
        }

        let api = self.stream.bind()?;
        let grid = n.div_ceil(BLOCK_SIZE);

        unsafe {
            api.check((api.module_launch_kernel)(
                self.func.raw,
                grid,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                self.stream.raw,
                self.args.as_mut_ptr(),
                ptr::null_mut(),
            ))
        }
    }
}
