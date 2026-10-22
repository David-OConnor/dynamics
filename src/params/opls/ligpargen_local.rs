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

use super::ligpargen::Generated;
use sha2::{Digest, Sha256};

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
        let lock_name = format!(
            "dynamics-ligpargen-{:x}.lock",
            Sha256::digest(owner.as_encoded_bytes())
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::opls::known::Ligand;

    const SILANE: &str = "Silane\nDynamics          3D\n\n  5  4  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 Si  0  0  0  0  0  0  0  0  0  0  0  0\n    0.8500    0.8500    0.8500 H   0  0  0  0  0  0  0  0  0  0  0  0\n    0.8500   -0.8500   -0.8500 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.8500    0.8500   -0.8500 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.8500   -0.8500    0.8500 H   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0  0  0  0\n  1  3  1  0  0  0  0\n  1  4  1  0  0  0  0\n  1  5  1  0  0  0  0\nM  END\n";
    const SILANE_ITP: &str = "[ defaults ]\n1 3 yes 0.5 0.5\n[ atomtypes ]\nTEST_SI 28.085 0 A 0.35 0.3\nTEST_H 1.008 0 A 0.25 0.1\n[ moleculetype ]\nSIL 3\n[ atoms ]\n1 TEST_SI 1 SIL Si 1 -0.4 28.085\n2 TEST_H 1 SIL H1 1 0.1 1.008\n3 TEST_H 1 SIL H2 1 0.1 1.008\n4 TEST_H 1 SIL H3 1 0.1 1.008\n5 TEST_H 1 SIL H4 1 0.1 1.008\n[ bonds ]\n1 2 1 0.147 200000\n1 3 1 0.147 200000\n1 4 1 0.147 200000\n1 5 1 0.147 200000\n[ angles ]\n2 1 3 1 109.47 100\n2 1 4 1 109.47 100\n2 1 5 1 109.47 100\n3 1 4 1 109.47 100\n3 1 5 1 109.47 100\n4 1 5 1 109.47 100\n";

    // Exercise a real subprocess and real output files without Python, BOSS or
    // networking. The two fake CLIs follow the upstream help/argument contracts.
    fn fake_cli(mode: &str) -> (Scratch, PathBuf, Vec<OsString>) {
        let scratch = Scratch::new().unwrap();
        let root = scratch.0.join("CLI with spaces");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("mode"), mode).unwrap();
        fs::write(
            root.join("fixture.mol"),
            include_str!("../../../param_data/opls/creatine.mol"),
        )
        .unwrap();
        fs::write(
            root.join("fixture.itp"),
            include_str!("../../../param_data/opls/creatine.itp"),
        )
        .unwrap();
        #[cfg(windows)]
        {
            let script = root.join("fake cli.ps1");
            fs::write(&script, r#"
$ErrorActionPreference = 'Stop'
$mode = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'mode') -Raw).Trim()
if ($args -contains '--help') {
    if ($mode -in @('israel', 'fail')) { Write-Output '-i -cgen' }
    elseif ($mode -eq 'leela') { Write-Output '--mol --lbcc' }
    else { Write-Output 'unrelated program' }
    exit 0
}
if ($mode -eq 'fail') { Write-Error 'BOSS executable missing'; exit 7 }
if ($mode -eq 'timeout') { Start-Sleep -Seconds 30; exit 0 }
$args | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'args.json')
$r = $args[[Array]::IndexOf($args, '-r') + 1]
if ($mode -eq 'israel') {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixture.itp') -Destination "$r.gmx.itp"
} elseif ($mode -eq 'leela') {
    New-Item -ItemType Directory -Path 'archive' | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixture.itp') -Destination "archive/$r.itp"
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixture.mol') -Destination "archive/$r.mol"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory((Join-Path $PWD 'archive'), (Join-Path $PWD "$r.zip"))
}
"#).unwrap();
            (
                scratch,
                PathBuf::from("powershell.exe"),
                vec![
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-ExecutionPolicy".into(),
                    "Bypass".into(),
                    "-File".into(),
                    script.into_os_string(),
                ],
            )
        }
        #[cfg(unix)]
        {
            let script = root.join("fake cli.sh");
            fs::write(
                &script,
                r#"#!/bin/sh
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mode=$(cat "$root/mode")
if [ "$1" = '--help' ]; then
    case "$mode" in
        israel|fail) echo '-i -cgen';;
        leela) echo '--mol --lbcc';;
        *) echo 'unrelated program';;
    esac
    exit 0
fi
if [ "$mode" = fail ]; then echo 'BOSS executable missing' >&2; exit 7; fi
if [ "$mode" = timeout ]; then sleep 30; exit 0; fi
printf '%s\n' "$@" > "$root/args.txt"
while [ "$#" -gt 0 ]; do
    if [ "$1" = '-r' ]; then shift; residue=$1; fi
    shift
done
if [ "$mode" = israel ]; then cp "$root/fixture.itp" "$residue.gmx.itp"; fi
"#,
            )
            .unwrap();
            (
                scratch,
                PathBuf::from("/bin/sh"),
                vec![script.into_os_string()],
            )
        }
    }

    fn arguments(scratch: &Scratch) -> Vec<String> {
        let root = scratch.0.join("CLI with spaces");
        #[cfg(windows)]
        {
            serde_json::from_str(&fs::read_to_string(root.join("args.json")).unwrap()).unwrap()
        }
        #[cfg(unix)]
        {
            fs::read_to_string(root.join("args.txt"))
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect()
        }
    }

    #[test]
    fn local_cli_selects_charge_model_and_maps_parameters() {
        let (scratch, program, args) = fake_cli("israel");
        let local = Local::probe(program, args).unwrap();
        assert_eq!(local.interface, Interface::Israel);
        let mol = include_str!("../../../param_data/opls/creatine.mol");
        let generated = local.parameterize(mol, 0).unwrap();
        let call = arguments(&scratch);
        assert!(call.windows(2).any(|p| p == ["-cgen", "CM1A-LBCC"]));
        assert!(call.windows(2).any(|p| p == ["-o", "0"]));
        let source = bio_files::Sdf::new(mol).unwrap();
        let mut atoms = source.atoms.clone();
        atoms.reverse();
        let template = Ligand::from_text(&generated.sdf, &generated.itp).unwrap();
        let params = template
            .build_generated(&mut atoms, &source.bonds)
            .unwrap()
            .unwrap();
        assert_eq!(params.bonds.len(), source.bonds.len());
        assert!(
            atoms
                .iter()
                .all(|a| a.force_field_type.is_some() && a.partial_charge.is_some())
        );
        for (atom, original) in atoms.iter().zip(source.atoms.iter().rev()) {
            assert_eq!(atom.serial_number, original.serial_number);
            assert_eq!(atom.posit, original.posit);
        }
        local.parameterize(mol, -1).unwrap();
        assert!(
            arguments(&scratch)
                .windows(2)
                .any(|p| p == ["-cgen", "CM1A"])
        );
    }

    #[test]
    fn legacy_cli_uses_final_mol_from_archive() {
        let (scratch, program, args) = fake_cli("leela");
        let local = Local::probe(program, args).unwrap();
        assert_eq!(local.interface, Interface::Leela);
        #[cfg(windows)]
        {
            let generated = local
                .parameterize("input before canonicalization", 0)
                .unwrap();
            assert_eq!(
                generated.sdf,
                include_str!("../../../param_data/opls/creatine.mol")
            );
            Ligand::from_text(&generated.sdf, &generated.itp).unwrap();
            assert!(arguments(&scratch).iter().any(|arg| arg == "-l"));
            local
                .parameterize("input before canonicalization", 1)
                .unwrap();
            assert!(!arguments(&scratch).iter().any(|arg| arg == "-l"));
        }
        #[cfg(unix)]
        {
            // No external zip executable is required by these tests.
            assert!(
                local
                    .parameterize("input before canonicalization", 0)
                    .is_err()
            );
            assert!(arguments(&scratch).iter().any(|arg| arg == "-l"));
            assert!(
                local
                    .parameterize("input before canonicalization", 1)
                    .is_err()
            );
            assert!(!arguments(&scratch).iter().any(|arg| arg == "-l"));
        }
    }

    #[test]
    fn local_failures_and_timeout_keep_diagnostics() {
        for (mode, timeout, expected) in [
            ("fail", Duration::from_secs(10), "BOSS executable missing"),
            ("timeout", Duration::from_millis(500), "timed out"),
        ] {
            let (scratch, program, args) = fake_cli(mode);
            let mut command = Command::new(program);
            command.args(args);
            let error = run(&mut command, &scratch.0, timeout).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
        let (_scratch, program, args) = fake_cli("unknown");
        assert!(
            Local::probe(program, args)
                .unwrap_err()
                .to_string()
                .contains("Unrecognized")
        );
    }

    #[test]
    fn archives_require_both_matching_output_files() {
        use std::io::Write;
        let scratch = Scratch::new().unwrap();
        let path = scratch.0.join("LIG.zip");
        let mut zip = zip::ZipWriter::new(File::create(&path).unwrap());
        zip.start_file("LIG.itp", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(b"topology").unwrap();
        zip.finish().unwrap();
        assert!(read_archive(&path, "LIG").is_err());
        let mut zip = zip::ZipWriter::new(File::create(&path).unwrap());
        for (name, text) in [("LIG.mol", "final atom order"), ("LIG.itp", "topology")] {
            zip.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(text.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        assert_eq!(
            read_archive(&path, "LIG").unwrap(),
            ("final atom order".into(), "topology".into())
        );
    }

    #[test]
    fn provider_uses_local_generation_offline_and_reloads_cache() {
        for mode in ["israel", "fail", "absent"] {
            let (scratch, _, _) = fake_cli(mode);
            let root = scratch.0.join("CLI with spaces");
            fs::write(root.join("fixture.itp"), SILANE_ITP).unwrap();
            #[cfg(windows)]
            let executable = {
                let path = root.join("ligpargen.cmd");
                fs::write(&path, "@\"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"%~dp0fake cli.ps1\" %*\r\n").unwrap();
                path
            };
            #[cfg(unix)]
            let executable = {
                use std::os::unix::fs::PermissionsExt;
                let path = root.join("fake cli.sh");
                fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
                path
            };
            // Environment changes belong only to this subprocess, keeping Rust's
            // parallel tests and the user's own cache/configuration untouched.
            let mut command = Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "params::opls::ligpargen_local::tests::provider_child",
                    "--nocapture",
                ])
                .env("DYNAMICS_LIGPARGEN_TEST_CHILD", mode)
                .env("DYNAMICS_LIGPARGEN", &executable)
                .env("DYNAMICS_OPLS_CACHE", scratch.0.join("cache"));
            if mode == "absent" {
                command
                    .env_remove("DYNAMICS_LIGPARGEN")
                    .env_remove("DYNAMICS_LIGPARGEN_PYTHON")
                    .env_remove("BOSSdir")
                    .env("PATH", "");
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn provider_child() {
        use crate::{
            FfMolType, MolDynamics,
            params::{FfParamSet, ParameterizationOptions},
        };
        let Ok(mode) = std::env::var("DYNAMICS_LIGPARGEN_TEST_CHILD") else {
            return;
        };
        let sdf = bio_files::Sdf::new(SILANE).unwrap();
        let mut molecule = MolDynamics {
            atoms: sdf.atoms,
            bonds: sdf.bonds,
            ff_mol_type: FfMolType::SmallOrganic,
            ..Default::default()
        };
        molecule.atoms.reverse();
        let set = FfParamSet::new_opls().unwrap();
        if mode == "absent" {
            let error = set
                .assign(&molecule, &ParameterizationOptions::local_only())
                .unwrap_err();
            assert!(
                error.descrip.contains("Remote generation is disabled"),
                "{error}"
            );
            return;
        }
        if mode == "fail" {
            // AllowRemote must still return the local diagnostic, without
            // attempting HTTP after a detected installation has failed.
            let error = set
                .assign(&molecule, &ParameterizationOptions::default())
                .unwrap_err();
            assert!(error.descrip.contains("BOSS executable missing"), "{error}");
            assert!(molecule.atoms.iter().all(|a| a.partial_charge.is_none()));
            return;
        }
        let options = ParameterizationOptions::local_only();
        let assigned = set.assign(&molecule, &options).unwrap();
        assert!(
            assigned.type_source.starts_with("local LigParGen/BOSS"),
            "{}",
            assigned.type_source
        );
        assert_eq!(assigned.parameters.explicit.bonds.len(), 4);
        assert_eq!(assigned.atom_types.last().unwrap(), "TEST_SI");
        assert_eq!(*assigned.partial_charges.values.last().unwrap(), -0.4);
        assert!(molecule.atoms.iter().all(|a| a.force_field_type.is_none()));
        // A fresh backend has no memory cache. Removing the mock executable
        // proves the persisted result bypasses discovery and local execution.
        fs::remove_file(std::env::var_os("DYNAMICS_LIGPARGEN").unwrap()).unwrap();
        let cached = FfParamSet::new_opls()
            .unwrap()
            .assign(&molecule, &options)
            .unwrap();
        assert_eq!(cached.type_source, "cached LigParGen topology");
        assert_eq!(cached.atom_types, assigned.atom_types);
        assert_eq!(
            cached.partial_charges.values,
            assigned.partial_charges.values
        );
    }
}
