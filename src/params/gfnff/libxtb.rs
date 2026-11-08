//! Persistent GFN-FF calls through xTB's C API.

use std::{
    env,
    ffi::{CString, c_char, c_int, c_void},
    path::{Path, PathBuf},
    ptr,
    sync::{Mutex, MutexGuard},
};

use libloading::Library;
use lin_alg::f32::Vec3;
use na_seq::Element;

use super::{BOHR_ANGSTROM, GfnffSet, HARTREE_KCAL};
use crate::{
    ParamError,
    params::potential::{PotentialEvaluation, SystemPotential},
};

// xTB's Fortran runtime and global settings are shared by all calculators.
static API_LOCK: Mutex<()> = Mutex::new(());

type Handle = *mut c_void;
type New = unsafe extern "C" fn() -> Handle;
type Delete = unsafe extern "C" fn(*mut Handle);
type Check = unsafe extern "C" fn(Handle) -> c_int;
type GetError = unsafe extern "C" fn(Handle, *mut c_char, *const c_int);
type SetVerbosity = unsafe extern "C" fn(Handle, c_int);
type NewMolecule = unsafe extern "C" fn(
    Handle,
    *const c_int,
    *const c_int,
    *const f64,
    *const f64,
    *const c_int,
    *const f64,
    *const bool,
) -> Handle;
type UpdateMolecule = unsafe extern "C" fn(Handle, Handle, *const f64, *const f64);
type LoadGfnff = unsafe extern "C" fn(Handle, Handle, Handle, *const c_char);
type Singlepoint = unsafe extern "C" fn(Handle, Handle, Handle, Handle);
type GetValues = unsafe extern "C" fn(Handle, Handle, *mut f64);
type SetSolventV2 = unsafe extern "C" fn(
    Handle,
    Handle,
    *const c_char,
    *const c_int,
    *const f64,
    *const c_int,
    *const c_char,
);

#[derive(Debug)]
struct Api {
    _library: Library,
    _thread_libraries: Vec<Library>,
    version: c_int,
    new_environment: New,
    del_environment: Delete,
    check: Check,
    get_error: GetError,
    set_verbosity: SetVerbosity,
    new_calculator: New,
    del_calculator: Delete,
    new_results: New,
    del_results: Delete,
    new_molecule: NewMolecule,
    del_molecule: Delete,
    update_molecule: UpdateMolecule,
    load_gfnff: LoadGfnff,
    singlepoint: Singlepoint,
    get_energy: GetValues,
    get_gradient: GetValues,
    set_solvent_v2: Option<SetSolventV2>,
    set_threads: Vec<unsafe extern "C" fn(c_int)>,
}

fn lock_api() -> MutexGuard<'static, ()> {
    API_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Api {
    fn open(path: &Path) -> Result<Self, String> {
        // Symbols retain their library for the full calculator lifetime.
        unsafe {
            #[cfg(windows)]
            let library: Library = if path.is_absolute() {
                libloading::os::windows::Library::load_with_flags(path, 0x00000100 | 0x00001000)
                    .map(Into::into)
            } else {
                Library::new(path)
            }
            .map_err(|error| error.to_string())?;
            #[cfg(not(windows))]
            let library = Library::new(path).map_err(|error| error.to_string())?;

            let version = library
                .get::<unsafe extern "C" fn() -> c_int>(b"xtb_getAPIVersion\0")
                .map_err(|error| error.to_string())?();
            if !(10000..30000).contains(&version) {
                return Err(format!("unsupported libxtb C API version {version}"));
            }

            macro_rules! symbol {
                ($name:literal, $type:ty) => {
                    *library
                        .get::<$type>(concat!($name, "\0").as_bytes())
                        .map_err(|error| error.to_string())?
                };
            }

            let mut thread_libraries = Vec::new();
            let mut set_threads = Vec::new();
            for name in [c"omp_set_num_threads", c"openblas_set_num_threads"] {
                if let Ok(symbol) =
                    library.get::<unsafe extern "C" fn(c_int)>(name.to_bytes_with_nul())
                {
                    set_threads.push(*symbol);
                }
            }

            #[cfg(windows)]
            if let Some(directory) = path.parent() {
                for (filename, symbol) in [
                    ("libgomp-1.dll", c"omp_set_num_threads"),
                    ("openblas.dll", c"openblas_set_num_threads"),
                ] {
                    let dependency = directory.join(filename);
                    if !dependency.is_file() {
                        continue;
                    }

                    if let Ok(runtime) = libloading::os::windows::Library::load_with_flags(
                        &dependency,
                        0x00000100 | 0x00001000,
                    ) {
                        let runtime: Library = runtime.into();
                        if let Ok(setter) =
                            runtime.get::<unsafe extern "C" fn(c_int)>(symbol.to_bytes_with_nul())
                        {
                            set_threads.push(*setter);
                        }
                        thread_libraries.push(runtime);
                    }
                }
            }

            let api = Self {
                version,
                new_environment: symbol!("xtb_newEnvironment", New),
                del_environment: symbol!("xtb_delEnvironment", Delete),
                check: symbol!("xtb_checkEnvironment", Check),
                get_error: symbol!("xtb_getError", GetError),
                set_verbosity: symbol!("xtb_setVerbosity", SetVerbosity),
                new_calculator: symbol!("xtb_newCalculator", New),
                del_calculator: symbol!("xtb_delCalculator", Delete),
                new_results: symbol!("xtb_newResults", New),
                del_results: symbol!("xtb_delResults", Delete),
                new_molecule: symbol!("xtb_newMolecule", NewMolecule),
                del_molecule: symbol!("xtb_delMolecule", Delete),
                update_molecule: symbol!("xtb_updateMolecule", UpdateMolecule),
                load_gfnff: symbol!("xtb_loadGFNFF", LoadGfnff),
                singlepoint: symbol!("xtb_singlepoint", Singlepoint),
                get_energy: symbol!("xtb_getEnergy", GetValues),
                get_gradient: symbol!("xtb_getGradient", GetValues),
                set_solvent_v2: if version >= 20000 {
                    Some(symbol!("xtb_setSolvent", SetSolventV2))
                } else {
                    None
                },
                set_threads,
                _library: library,
                _thread_libraries: thread_libraries,
            };

            Ok(api)
        }
    }

    fn load(set: &GfnffSet) -> Result<Self, ParamError> {
        let explicit = set
            .library
            .clone()
            .or_else(|| env::var_os("XTB_LIBRARY").map(PathBuf::from));
        let mut candidates = Vec::new();
        if let Some(path) = explicit {
            if path.is_dir() {
                add_directory(&mut candidates, &path);
            } else {
                candidates.push(absolute(&path)?);
            }
        } else {
            let executable = set
                .executable
                .clone()
                .or_else(|| env::var_os("XTB_EXECUTABLE").map(PathBuf::from));
            if let Some(executable) = executable
                && let Some(directory) = absolute(&executable)?.parent()
            {
                add_directory(&mut candidates, directory);
                if let Some(root) = directory.parent() {
                    add_directory(&mut candidates, &root.join("lib"));
                    add_directory(
                        &mut candidates,
                        &root.join("libxtb").join("Library").join("bin"),
                    );
                }
            }

            if let Ok(executable) = env::current_exe()
                && let Some(directory) = executable.parent()
            {
                add_directory(&mut candidates, directory);
                add_directory(&mut candidates, &directory.join("lib"));
            }

            for directory in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
                add_directory(&mut candidates, &directory);
                if let Some(root) = directory.parent() {
                    add_directory(&mut candidates, &root.join("lib"));
                }
            }
            candidates.extend(library_names().iter().map(PathBuf::from));
        }

        let _guard = lock_api();
        let mut errors = Vec::new();
        for candidate in candidates {
            match Self::open(&candidate) {
                Ok(api) => {
                    if set.alpb_solvent.is_some() && api.set_solvent_v2.is_none() {
                        errors.push(format!(
                            "{}: ALPB requires libxtb C API 2; found {}",
                            candidate.display(),
                            api.version
                        ));
                        continue;
                    }

                    return Ok(api);
                }
                Err(error) => errors.push(format!("{}: {error}", candidate.display())),
            }
        }

        Err(ParamError::new(&format!(
            "Unable to load libxtb. Set GfnffSet::library or XTB_LIBRARY to a shared library. {}",
            errors.join("; ")
        )))
    }

    fn check(&self, environment: Handle, operation: &str) -> Result<(), ParamError> {
        unsafe {
            if (self.check)(environment) == 0 {
                return Ok(());
            }

            let mut buffer = vec![0_u8; 4096];
            let capacity = buffer.len() as c_int;
            (self.get_error)(environment, buffer.as_mut_ptr().cast(), &capacity);
            let end = buffer
                .iter()
                .position(|&byte| byte == 0)
                .unwrap_or(buffer.len());

            Err(ParamError::new(&format!(
                "libxtb {operation}: {}",
                String::from_utf8_lossy(&buffer[..end]).trim()
            )))
        }
    }
}

fn absolute(path: &Path) -> Result<PathBuf, ParamError> {
    std::path::absolute(path).map_err(Into::into)
}

fn library_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["libxtb.dll", "libxtb-6.dll", "xtb.dll"]
    } else if cfg!(target_os = "macos") {
        &["libxtb.dylib", "libxtb.6.dylib"]
    } else {
        &["libxtb.so", "libxtb.so.6"]
    }
}

fn add_directory(candidates: &mut Vec<PathBuf>, directory: &Path) {
    let Ok(directory) = absolute(directory) else {
        return;
    };

    for name in library_names() {
        let path = directory.join(name);
        if path.is_file() && !candidates.contains(&path) {
            candidates.push(path);
        }
    }
}

#[derive(Debug)]
pub(super) struct LibXtbPotential {
    api: Api,
    environment: Handle,
    molecule: Handle,
    calculator: Handle,
    results: Handle,
    elements: Vec<c_int>,
    positions: Vec<f64>,
    gradient: Vec<f64>,
    charge: f64,
    _builtin_directory: Option<tempfile::TempDir>,
    parameter_file: CString,
    solvent: Option<CString>,
    failed: bool,
}

// Handles have one owner; all Fortran calls, including deletion, hold API_LOCK.
unsafe impl Send for LibXtbPotential {}

impl LibXtbPotential {
    pub(super) fn new(
        set: &GfnffSet,
        elements: Vec<Element>,
        charge: f64,
    ) -> Result<Self, ParamError> {
        if elements.is_empty()
            || elements.len() > c_int::MAX as usize
            || elements
                .iter()
                .any(|e| !(1..=86).contains(&e.atomic_number()))
        {
            return Err(ParamError::new("GFN-FF requires atoms with elements H–Rn"));
        }

        let solvent = set
            .validated_alpb_solvent()?
            .map(CString::new)
            .transpose()
            .map_err(|_| ParamError::new("Invalid ALPB solvent"))?;
        let builtin_directory = if set.parameter_file.is_none() {
            Some(
                tempfile::Builder::new()
                    .prefix("dynamics-libxtb-")
                    .tempdir()?,
            )
        } else {
            None
        };
        let parameter_file = if let Some(path) = &set.parameter_file {
            if !path.is_file() {
                return Err(ParamError::new("GFN-FF parameter file does not exist"));
            }

            absolute(path)?
        } else {
            // A missing absolute filename selects built-ins without reading the current directory.
            builtin_directory
                .as_ref()
                .unwrap()
                .path()
                .join(".param_gfnff.xtb")
        };
        let parameter_file = CString::new(
            parameter_file
                .to_str()
                .ok_or_else(|| ParamError::new("libxtb requires a UTF-8 parameter path"))?,
        )
        .map_err(|_| ParamError::new("Invalid GFN-FF parameter path"))?;
        let count = elements.len();

        Ok(Self {
            api: Api::load(set)?,
            environment: ptr::null_mut(),
            molecule: ptr::null_mut(),
            calculator: ptr::null_mut(),
            results: ptr::null_mut(),
            elements: elements
                .iter()
                .map(|e| e.atomic_number() as c_int)
                .collect(),
            positions: vec![0.; count * 3],
            gradient: vec![0.; count * 3],
            charge,
            _builtin_directory: builtin_directory,
            parameter_file,
            solvent,
            failed: false,
        })
    }

    unsafe fn initialize(&mut self) -> Result<(), ParamError> {
        unsafe {
            self.environment = (self.api.new_environment)();
            if self.environment.is_null() {
                return Err(ParamError::new("libxtb could not allocate its environment"));
            }
            (self.api.set_verbosity)(self.environment, 0);
            self.api.check(self.environment, "set verbosity")?;

            let count = self.elements.len() as c_int;
            self.molecule = (self.api.new_molecule)(
                self.environment,
                &count,
                self.elements.as_ptr(),
                self.positions.as_ptr(),
                &self.charge,
                ptr::null(),
                ptr::null(),
                ptr::null(),
            );
            self.api.check(self.environment, "create molecule")?;
            self.calculator = (self.api.new_calculator)();
            self.results = (self.api.new_results)();
            if self.molecule.is_null() || self.calculator.is_null() || self.results.is_null() {
                return Err(ParamError::new("libxtb could not allocate its calculator"));
            }

            (self.api.load_gfnff)(
                self.environment,
                self.molecule,
                self.calculator,
                self.parameter_file.as_ptr(),
            );
            self.api.check(self.environment, "load GFN-FF")?;
            if let Some(solvent) = &self.solvent {
                let set_solvent = self
                    .api
                    .set_solvent_v2
                    .ok_or_else(|| ParamError::new("ALPB requires libxtb C API 2"))?;
                set_solvent(
                    self.environment,
                    self.calculator,
                    solvent.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    c"alpb".as_ptr(),
                );
                self.api.check(self.environment, "set ALPB solvent")?;
            }

            Ok(())
        }
    }

    unsafe fn calculate(&mut self) -> Result<f64, ParamError> {
        unsafe {
            for set_threads in &self.api.set_threads {
                set_threads(1);
            }
            if self.molecule.is_null() {
                self.initialize()?;
            } else {
                (self.api.update_molecule)(
                    self.environment,
                    self.molecule,
                    self.positions.as_ptr(),
                    ptr::null(),
                );
                self.api.check(self.environment, "update coordinates")?;
            }

            (self.api.singlepoint)(
                self.environment,
                self.molecule,
                self.calculator,
                self.results,
            );
            self.api.check(self.environment, "evaluate GFN-FF")?;
            let mut energy = f64::NAN;
            (self.api.get_energy)(self.environment, self.results, &mut energy);
            self.api.check(self.environment, "read energy")?;
            self.gradient.fill(f64::NAN);
            (self.api.get_gradient)(self.environment, self.results, self.gradient.as_mut_ptr());
            self.api.check(self.environment, "read gradient")?;
            if !energy.is_finite() || self.gradient.iter().any(|v| !v.is_finite()) {
                return Err(ParamError::new(
                    "libxtb returned non-finite energy or forces",
                ));
            }

            Ok(energy)
        }
    }
}

impl SystemPotential for LibXtbPotential {
    fn evaluate(&mut self, positions: &[Vec3]) -> Result<PotentialEvaluation, ParamError> {
        if self.failed {
            return Err(ParamError::new(
                "libxtb evaluation failed; rebuild the simulation",
            ));
        }
        if positions.len() != self.elements.len()
            || positions
                .iter()
                .flat_map(|p| p.to_arr())
                .any(|v| !v.is_finite())
        {
            return Err(ParamError::new(
                "GFN-FF requires matching, finite coordinates",
            ));
        }

        for (target, position) in self.positions.chunks_exact_mut(3).zip(positions) {
            target.copy_from_slice(&[
                position.x as f64 / BOHR_ANGSTROM,
                position.y as f64 / BOHR_ANGSTROM,
                position.z as f64 / BOHR_ANGSTROM,
            ]);
        }
        let result = {
            let _guard = lock_api();

            unsafe { self.calculate() }
        };
        let energy = match result {
            Ok(energy) => energy * HARTREE_KCAL,
            Err(error) => {
                self.failed = true;

                return Err(error);
            }
        };
        let scale = -HARTREE_KCAL / BOHR_ANGSTROM;
        let forces: Vec<_> = self
            .gradient
            .chunks_exact(3)
            .map(|g| {
                Vec3::new(
                    (g[0] * scale) as f32,
                    (g[1] * scale) as f32,
                    (g[2] * scale) as f32,
                )
            })
            .collect();
        if forces
            .iter()
            .flat_map(|f| f.to_arr())
            .any(|v| !v.is_finite())
        {
            self.failed = true;

            return Err(ParamError::new(
                "libxtb forces exceed the Dynamics numeric range",
            ));
        }

        // The C API does not expose GFN-FF charges or individual energy terms.
        Ok(PotentialEvaluation {
            energy,
            forces,
            charges: None,
            components: Vec::new(),
        })
    }
}

impl Drop for LibXtbPotential {
    fn drop(&mut self) {
        let _guard = lock_api();
        unsafe {
            if !self.results.is_null() {
                (self.api.del_results)(&mut self.results);
            }
            if !self.calculator.is_null() {
                (self.api.del_calculator)(&mut self.calculator);
            }
            if !self.molecule.is_null() {
                (self.api.del_molecule)(&mut self.molecule);
            }
            if !self.environment.is_null() {
                (self.api.del_environment)(&mut self.environment);
            }
        }
    }
}
