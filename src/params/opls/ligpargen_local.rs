//! Local LigParGen command/library entry points. No shell command strings or HTTP.
//!
//! Supports Isra3l/ligpargen (`-i`, `-cgen`) and leelasd/LigParGen (`-m`, `-l`).
//! The latter returns a ZIP containing its final, possibly canonicalized MOL and ITP.

use std::{
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

use super::ligpargen::Generated;

const TIMEOUT: Duration = Duration::from_secs(300);
const MAX_OUTPUT: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Interface {
    Israel,
    Leela,
}

#[derive(Debug)]
pub(super) struct Local {
    program: PathBuf,
    args: Vec<OsString>,
    interface: Interface,
}

impl Local {
    /// An explicit configuration must work; an incomplete detected installation
    /// must not cause the caller's molecule to be silently sent to a web service.
    pub fn discover() -> io::Result<Option<Self>> {
        if let Some(program) = std::env::var_os("DYNAMICS_LIGPARGEN") {
            let program = find_program(&program).ok_or_else(|| {
                io::Error::other("DYNAMICS_LIGPARGEN does not name an executable or Python script")
            })?;
            return Self::probe_program(program).map(Some);
        }
        let configured_python = std::env::var_os("DYNAMICS_LIGPARGEN_PYTHON");
        if configured_python.is_none() {
            for name in ["ligpargen", "LigParGen", "Converter"] {
                if let Some(program) = find_program(OsStr::new(name)) {
                    return Self::probe_program(program).map(Some);
                }
            }
        }
        // A library-only install is useful too. Avoid starting Python for every
        // unsupported ligand on machines without any indication of a BOSS setup.
        if configured_python.is_none() && std::env::var_os("BOSSdir").is_none() {
            return Ok(None);
        }
        let python = match python_program() {
            Ok(python) => python,
            Err(_) if configured_python.is_none() => return Ok(None),
            Err(error) => return Err(error),
        };
        for (module, entry) in [
            (
                "ligpargen.ligpargen",
                "from ligpargen.ligpargen import main; main()",
            ),
            (
                "LigParGen.Converter",
                "from LigParGen.Converter import main; main()",
            ),
        ] {
            // Probe the entry module to distinguish packages even on a
            // case-insensitive filesystem. Dependency failures remain diagnostic
            // when the selected installation's --help is run below.
            let probe = format!(
                "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{module}') else 1)"
            );
            let mut command = Command::new(&python);
            command.args(["-c", &probe]);
            let scratch = Scratch::new()?;
            if run(&mut command, &scratch.0, Duration::from_secs(15)).is_ok() {
                return Self::probe(python, vec!["-c".into(), entry.into()]).map(Some);
            }
        }
        if configured_python.is_some() {
            return Err(io::Error::other(
                "DYNAMICS_LIGPARGEN_PYTHON cannot import ligpargen or LigParGen",
            ));
        }
        Ok(None)
    }

    fn probe_program(program: PathBuf) -> io::Result<Self> {
        if program.extension().is_some_and(|ext| ext == "py") {
            let python = python_program()?;
            Self::probe(python, vec![program.into_os_string()])
        } else {
            Self::probe(program, Vec::new())
        }
    }

    fn probe(program: PathBuf, args: Vec<OsString>) -> io::Result<Self> {
        let scratch = Scratch::new()?;
        let mut command = Command::new(&program);
        command.args(&args).arg("--help");
        let help = run(&mut command, &scratch.0, Duration::from_secs(15))?;
        let interface = if help.contains("-cgen") {
            Interface::Israel
        } else if help.contains("--lbcc") && help.contains("--mol") {
            Interface::Leela
        } else {
            return Err(io::Error::other(format!(
                "Unrecognized LigParGen command-line interface: {}",
                program.display()
            )));
        };
        Ok(Self {
            program,
            args,
            interface,
        })
    }

    pub fn parameterize(&self, mol: &str, charge: i32) -> io::Result<Generated> {
        if !(-2..=2).contains(&charge) {
            return Err(io::Error::other(
                "Local LigParGen supports charges from -2 to +2",
            ));
        }

        // Legacy LigParGen/BOSS writes fixed intermediate names in /tmp. File
        // locking serializes our jobs across processes and releases on a crash.
        let owner = std::env::var_os("USERNAME")
            .or_else(|| std::env::var_os("USER"))
            .or_else(|| std::env::var_os("HOME"))
            .unwrap_or_default();
        let owner_hash: String = Sha256::digest(owner.as_encoded_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        let lock_name = format!("dynamics-ligpargen-{owner_hash}.lock");
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(std::env::temp_dir().join(lock_name))?;
        lock.lock()?;

        let scratch = Scratch::new()?;
        let mut value = rand::random::<u32>();
        let residue: String = (0..3)
            .map(|_| {
                let c = char::from(b'A' + (value % 26) as u8);
                value /= 26;
                c
            })
            .collect();
        let input_name = format!("{residue}.mol");
        fs::write(scratch.0.join(&input_name), mol)?;

        let mut command = Command::new(&self.program);
        command.args(&self.args);
        match self.interface {
            Interface::Israel => {
                command.args([
                    "-i",
                    &input_name,
                    "-n",
                    &residue,
                    "-p",
                    ".",
                    "-cgen",
                    if charge == 0 { "CM1A-LBCC" } else { "CM1A" },
                ]);
            }
            Interface::Leela => {
                command.args(["-m", &input_name]);
                if charge == 0 {
                    command.arg("-l");
                }
            }
        }

        command.args(["-r", &residue, "-c", &charge.to_string(), "-o", "0"]);
        run(&mut command, &scratch.0, TIMEOUT)?;
        let output = match self.interface {
            Interface::Israel => {
                // This implementation explicitly restores serialOriginal when
                // writing its GROMACS atoms AND bonded indices. Its ITP is in
                // input order, even though BOSS internally reorders the atoms.
                read_text(&scratch.0.join(format!("{residue}.gmx.itp")))
                    .map(|itp| (mol.to_owned(), itp))
            }
            Interface::Leela => {
                // Never pair this implementation's ITP with the original input:
                // CreatZmat can canonicalize it. The archived MOL is the final
                // ordering actually used to construct the BOSS Z-matrix.
                read_archive(&scratch.0.join(format!("{residue}.zip")), &residue)
            }
        };

        let (sdf, itp) = output.map_err(|e| {
            io::Error::other(format!(
                "Local LigParGen did not produce a matching MOL/ITP: {e}\n{}",
                diagnostics(&scratch.0)
            ))
        })?;

        Ok(Generated {
            sdf,
            itp,
            job_id: format!("local {} ({:?})", self.program.display(), self.interface),
        })
    }
}

fn python_program() -> io::Result<PathBuf> {
    let configured = std::env::var_os("DYNAMICS_LIGPARGEN_PYTHON");
    let candidates: Vec<OsString> = configured.map_or_else(
        || vec!["python3".into(), "python".into()],
        |value| vec![value],
    );
    candidates.iter().find_map(|p| find_program(p)).ok_or_else(|| io::Error::other(
        "No Python interpreter found; set DYNAMICS_LIGPARGEN_PYTHON to the LigParGen environment's Python"
    ))
}

fn find_program(name: &OsStr) -> Option<PathBuf> {
    let path = Path::new(name);
    let candidates = if path.is_absolute() || path.components().count() > 1 {
        vec![std::path::absolute(path).ok()?]
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .filter_map(|dir| std::path::absolute(dir.join(path)).ok())
            .collect()
    };
    for path in candidates {
        #[cfg(not(windows))]
        let paths = [path];
        #[cfg(windows)]
        let paths = {
            let mut paths = vec![path.clone()];
            if path.extension().is_none() {
                // Console entry points from pip/conda; .py also works with an
                // explicitly selected Python. No shell parsing of user arguments.
                paths.extend(["exe", "cmd", "bat", "py"].map(|ext| path.with_extension(ext)));
            }
            paths
        };
        for path in paths {
            if !path.is_file() {
                continue;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if path.extension().is_none_or(|ext| ext != "py")
                    && fs::metadata(&path).ok()?.permissions().mode() & 0o111 == 0
                {
                    continue;
                }
            }
            return Some(path);
        }
    }
    None
}

fn run(command: &mut Command, directory: &Path, timeout: Duration) -> io::Result<String> {
    command
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(File::create(directory.join("stdout.log"))?)
        .stderr(File::create(directory.join("stderr.log"))?);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn()?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let output = diagnostics(directory);
            return if status.success() {
                Ok(output)
            } else {
                Err(io::Error::other(format!(
                    "Local LigParGen exited with {status}:\n{output}"
                )))
            };
        }
        if start.elapsed() >= timeout {
            // BOSS is a descendant of the Python/CLI process. Reclaim its whole
            // process tree before removing the job directory and releasing lock.
            #[cfg(unix)]
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", child.id())])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                let _ = Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &child.id().to_string()])
                    .creation_flags(0x08000000)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "Local LigParGen timed out after {} seconds:\n{}",
                    timeout.as_secs(),
                    diagnostics(directory)
                ),
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn diagnostics(directory: &Path) -> String {
    let mut text = String::new();
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "log") {
                continue;
            }
            if let Ok(mut file) = File::open(path) {
                if let Ok(size) = file.metadata().map(|m| m.len()) {
                    let _ = file.seek(SeekFrom::Start(size.saturating_sub(8192)));
                    let mut bytes = Vec::new();
                    let _ = file.take(8192).read_to_end(&mut bytes);
                    text.push_str(&String::from_utf8_lossy(&bytes));
                    text.push('\n');
                }
            }
        }
    }
    text
}

fn read_text(path: &Path) -> io::Result<String> {
    let file = File::open(path)?;
    read_limited(file)
}

fn read_limited(reader: impl Read) -> io::Result<String> {
    let mut text = String::new();
    reader.take(MAX_OUTPUT + 1).read_to_string(&mut text)?;
    if text.len() as u64 > MAX_OUTPUT {
        return Err(io::Error::other("LigParGen output exceeds 8 MiB"));
    }
    Ok(text)
}

fn read_archive(path: &Path, residue: &str) -> io::Result<(String, String)> {
    let mut archive = zip::ZipArchive::new(File::open(path)?).map_err(io::Error::other)?;
    let mol = read_limited(
        archive
            .by_name(&format!("{residue}.mol"))
            .map_err(io::Error::other)?,
    )?;
    let itp = read_limited(
        archive
            .by_name(&format!("{residue}.itp"))
            .map_err(io::Error::other)?,
    )?;
    Ok((mol, itp))
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> io::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("dynamics-ligpargen-{:016x}", rand::random::<u64>()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
