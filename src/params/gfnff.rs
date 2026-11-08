//! GFN-FF through xTB, with a private working directory and retained topology.
//!
//! Requires [xTB](https://doi.org/10.1002/wcms.1493) to be installed. [xtB release binaries](https://github.com/grimme-lab/xtb/releases)
//! xTB parameterizes molecules, and computes forces at each step. We use our native integrator; not xTB's.

use std::{
    env,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use lin_alg::f32::Vec3;
use tempfile::TempDir;

use super::{
    ForceFieldBackend, ParameterizationOptions, TypedTopology,
    potential::{EnergyComponent, PotentialEvaluation, SystemPotential, SystemPotentialBackend},
};
use crate::{AtomDynamics, HydrogenConstraint, MdConfig, MolDynamics, ParamError, SimBox, Solvent};

const BOHR_ANGSTROM: f64 = 0.529_177_210_903;
const HARTREE_KCAL: f64 = 627.509_474_063_1;

/// Terms evaluated by xTB; LJ mixing and Amber PME do not apply.
pub const TERMS: &[(&str, &str)] = &[
    (
        "bond energy",
        "Dissociative, topology-dependent bond potential",
    ),
    ("angle energy", "Topology-dependent angular potential"),
    ("torsion energy", "Proper and out-of-plane torsional terms"),
    (
        "repulsion energy",
        "Bonded and nonbonded short-range repulsion",
    ),
    (
        "electrostat energy",
        "Electronegativity-equilibration (EEQ) electrostatics",
    ),
    (
        "dispersion energy",
        "Charge-scaled, topology-based D4 dispersion",
    ),
    (
        "HB energy",
        "Explicit charge-scaled hydrogen-bond correction",
    ),
    (
        "XB energy",
        "Explicit charge-scaled halogen-bond correction",
    ),
    (
        "bonded atm energy",
        "Three-body bonded repulsion correction",
    ),
    ("external energy", "External potential"),
    ("add. restraining", "Additional restraints"),
    ("Gsolv", "ALPB solvation free energy (when enabled)"),
];

#[derive(Clone, Debug)]
pub struct GfnffSet {
    /// xTB executable; otherwise XTB_EXECUTABLE, then xtb on PATH.
    pub executable: Option<PathBuf>,
    /// Optional xTB GFN-FF parameter file; otherwise xTB's built-in parameters.
    pub parameter_file: Option<PathBuf>,
    /// System charge in e. Otherwise inferred from complete input charges, or zero.
    pub total_charge: Option<f64>,
    /// Implicit ALPB solvent, using an xTB name such as `water`, `methanol`, or `dmso`.
    /// `None` evaluates in vacuum. Applies to both inspection and MD; keep
    /// `MdConfig::solvent` set to `Solvent::None` to avoid generating explicit solvent.
    pub alpb_solvent: Option<String>,
    pub timeout: Duration,
}

impl Default for GfnffSet {
    fn default() -> Self {
        Self {
            executable: None,
            parameter_file: None,
            total_charge: None,
            alpb_solvent: None,
            timeout: Duration::from_secs(300),
        }
    }
}

impl GfnffSet {
    fn validated_alpb_solvent(&self) -> Result<Option<&str>, ParamError> {
        let solvent = self.alpb_solvent.as_deref().map(str::trim);
        if solvent.is_some_and(|s| {
            s.is_empty()
                || s.starts_with('-')
                || s.chars().any(|c| c.is_whitespace() || c.is_control())
        }) {
            return Err(ParamError::new(
                "GFN-FF ALPB solvent must be a single xTB solvent name, such as water or methanol",
            ));
        }

        Ok(solvent)
    }

    fn charge(&self, charges: impl Iterator<Item = Option<f64>>) -> Result<f64, ParamError> {
        if let Some(charge) = self.total_charge {
            if charge.is_finite() && (charge - charge.round()).abs() < 1e-6 {
                return Ok(charge);
            }

            return Err(ParamError::new(
                "GFN-FF total charge must be a finite integer",
            ));
        }

        let charges: Vec<_> = charges.collect();
        if charges.iter().all(Option::is_none) {
            return Ok(0.);
        }

        if charges.iter().any(|q| q.is_none_or(|q| !q.is_finite())) {
            return Err(ParamError::new(
                "GFN-FF needs complete input charges or an explicit total charge",
            ));
        }

        let sum: f64 = charges.iter().flatten().sum();
        if (sum - sum.round()).abs() > 0.05 {
            return Err(ParamError::new(
                "GFN-FF input charges do not sum to an integer; set the total charge",
            ));
        }

        Ok(sum.round())
    }

    /// Inspect one geometry; the report includes xTB's energy components and EEQ charges.
    pub fn inspect(&self, molecule: &MolDynamics) -> Result<PotentialEvaluation, ParamError> {
        let charge = self.charge(
            molecule
                .atoms
                .iter()
                .map(|a| a.partial_charge.map(f64::from)),
        )?;
        let elements: Vec<_> = molecule.atoms.iter().map(|a| a.element).collect();

        let positions: Vec<Vec3> = match &molecule.atom_posits {
            Some(p) => p.iter().map(|p| (*p).into()).collect(),
            None => molecule.atoms.iter().map(|a| a.posit.into()).collect(),
        };
        let mut evaluator = XtbPotential::new(self, elements, charge)?;

        evaluator.evaluate(&positions)
    }
}

impl ForceFieldBackend for GfnffSet {
    fn name(&self) -> &str {
        "GFN-FF (xTB)"
    }

    fn system_potential(&self) -> Option<&dyn SystemPotentialBackend> {
        Some(self)
    }

    fn assign_atom_types<'a>(
        &'a self,
        _: &'a MolDynamics,
        _: &ParameterizationOptions,
    ) -> Result<Box<dyn TypedTopology + 'a>, ParamError> {
        Err(ParamError::new(
            "GFN-FF uses a whole-system potential, not classical parameter tables",
        ))
    }
}

impl SystemPotentialBackend for GfnffSet {
    fn validate_config(&self, cfg: &MdConfig) -> Result<(), ParamError> {
        if !matches!(cfg.solvent, Solvent::None)
            || cfg.barostat_cfg.is_some()
            || cfg.hydrogen_constraint != HydrogenConstraint::Flexible
        {
            return Err(ParamError::new(
                "GFN-FF requires no generated solvent or barostat and flexible hydrogens. Use MdConfig::for_family(Gfnn); set GfnffSet::alpb_solvent for implicit solvent.",
            ));
        }

        if cfg.overrides.bonded_disabled
            || cfg.overrides.coulomb_disabled
            || cfg.overrides.lj_disabled
        {
            return Err(ParamError::new(
                "GFN-FF cannot disable individual energy terms",
            ));
        }

        self.validated_alpb_solvent()?;

        Ok(())
    }

    fn prepare(&self, molecule: &MolDynamics) -> Result<MolDynamics, ParamError> {
        let evaluation = self.inspect(molecule)?;
        let mut result = molecule.clone();
        for (atom, charge) in result.atoms.iter_mut().zip(evaluation.charges.unwrap()) {
            atom.force_field_type = Some(format!("gfnff_{}", atom.element.to_letter()));
            atom.partial_charge = Some(charge);
        }
        result.explicit_params = None;
        result.mol_specific_params = None;

        Ok(result)
    }

    fn create(
        &self,
        atoms: &[AtomDynamics],
        _: &SimBox,
        molecules: &[MolDynamics],
    ) -> Result<Box<dyn SystemPotential>, ParamError> {
        let charge = self.charge(
            molecules
                .iter()
                .flat_map(|m| &m.atoms)
                .map(|a| a.partial_charge.map(f64::from)),
        )?;
        let elements = atoms.iter().map(|a| a.element).collect();

        Ok(Box::new(XtbPotential::new(self, elements, charge)?))
    }
}

#[derive(Debug)]
struct XtbPotential {
    directory: TempDir,
    executable: PathBuf,
    elements: Vec<na_seq::Element>,
    charge: f64,
    alpb_solvent: Option<String>,
    timeout: Duration,
    initialized: bool,
}

impl XtbPotential {
    fn new(
        set: &GfnffSet,
        elements: Vec<na_seq::Element>,
        charge: f64,
    ) -> Result<Self, ParamError> {
        if elements.is_empty()
            || elements
                .iter()
                .any(|e| !(1..=86).contains(&e.atomic_number()))
        {
            return Err(ParamError::new("GFN-FF requires atoms with elements H–Rn"));
        }

        if set.timeout.is_zero() {
            return Err(ParamError::new("xTB timeout must be positive"));
        }
        let alpb_solvent = set.validated_alpb_solvent()?.map(str::to_owned);

        let directory = tempfile::Builder::new()
            .prefix("dynamics-gfnff-")
            .tempdir()?;
        if let Some(path) = &set.parameter_file {
            fs::copy(path, directory.path().join(".param_gfnff.xtb"))?;
        }

        let executable = set
            .executable
            .clone()
            .or_else(|| env::var_os("XTB_EXECUTABLE").map(PathBuf::from))
            .unwrap_or_else(|| "xtb".into());

        let executable = if executable.components().count() > 1 {
            fs::canonicalize(&executable).map_err(|e| {
                ParamError::new(&format!("xTB executable {}: {e}", executable.display()))
            })?
        } else {
            executable
        };

        Ok(Self {
            directory,
            executable,
            elements,
            charge,
            alpb_solvent,
            timeout: set.timeout,
            initialized: false,
        })
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);

        command
            .current_dir(self.directory.path())
            .args([
                "geometry.xyz",
                "--gfnff",
                "--grad",
                "--chrg",
                &self.charge.to_string(),
            ])
            .env("XTBPATH", self.directory.path())
            .env("XTBHOME", self.directory.path())
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .stdin(Stdio::null());

        if let Some(solvent) = &self.alpb_solvent {
            command.args(["--alpb", solvent]);
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }

        command
    }

    fn run(&self) -> Result<String, ParamError> {
        let stdout = File::create(self.directory.path().join("stdout"))?;
        let stderr = File::create(self.directory.path().join("stderr"))?;
        let mut command = self.command();
        command.stdout(stdout).stderr(stderr);

        let mut child = command.spawn().map_err(|e| ParamError::new(&format!(
            "Unable to run xTB ({}): {e}. Install xTB and set its path in the force-field viewer, or on the system path.",
            self.executable.display()
        )))?;

        let start = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if start.elapsed() < self.timeout => {
                    thread::sleep(Duration::from_millis(5))
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();

                    return Err(ParamError::new("GFN-FF evaluation timed out"));
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();

                    return Err(error.into());
                }
            }
        };

        let output = fs::read_to_string(self.directory.path().join("stdout"))?;
        let errors = fs::read_to_string(self.directory.path().join("stderr"))?;
        if !status.success()
            || !errors
                .to_ascii_lowercase()
                .contains("normal termination of xtb")
        {
            let tail: String = errors
                .chars()
                .rev()
                .take(2000)
                .collect::<String>()
                .chars()
                .rev()
                .collect();

            return Err(ParamError::new(&format!("xTB failed ({status}): {tail}")));
        }

        Ok(output)
    }
}

impl SystemPotential for XtbPotential {
    fn evaluate(&mut self, positions: &[Vec3]) -> Result<PotentialEvaluation, ParamError> {
        let count = self.elements.len();
        if positions.len() != count
            || positions
                .iter()
                .flat_map(|p| p.to_arr())
                .any(|v| !v.is_finite())
        {
            return Err(ParamError::new(
                "GFN-FF requires matching, finite coordinates",
            ));
        }
        if self.initialized && !self.directory.path().join("gfnff_topo").is_file() {
            return Err(ParamError::new(
                "GFN-FF topology cache was lost; rebuild the simulation",
            ));
        }

        let mut xyz = File::create(self.directory.path().join("geometry.xyz"))?;
        writeln!(xyz, "{count}\nDynamics GFN-FF")?;
        for (element, position) in self.elements.iter().zip(positions) {
            writeln!(
                xyz,
                "{} {:.12} {:.12} {:.12}",
                element.to_letter(),
                position.x,
                position.y,
                position.z
            )?;
        }
        drop(xyz);

        // These outputs must belong to this evaluation; retain only the topology cache.
        for name in ["gradient", "gfnff_charges"] {
            let path = self.directory.path().join(name);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }

        let output = self.run()?;
        if self.initialized && !output.contains("GFN-FF topology read from file successfully") {
            return Err(ParamError::new(
                "xTB did not reuse the fixed GFN-FF topology",
            ));
        }
        let (energy, components) = parse_energy(&output, self.alpb_solvent.is_some())?;

        let gradient = fs::read_to_string(self.directory.path().join("gradient"))?;
        let lines: Vec<_> = gradient.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() != 2 * count + 3
            || lines[0].trim() != "$grad"
            || lines.last().unwrap().trim() != "$end"
        {
            return Err(ParamError::new("Invalid xTB gradient shape"));
        }

        let mut forces = Vec::with_capacity(count);
        for line in &lines[2 + count..2 + 2 * count] {
            let values: Vec<_> = line.split_whitespace().map(parse_number).collect();
            if values.len() != 3 || values.iter().any(Option::is_none) {
                return Err(ParamError::new("Invalid xTB gradient value"));
            }
            let scale = -HARTREE_KCAL / BOHR_ANGSTROM;
            forces.push(Vec3::new(
                (values[0].unwrap() * scale) as f32,
                (values[1].unwrap() * scale) as f32,
                (values[2].unwrap() * scale) as f32,
            ));
        }

        let charges = fs::read_to_string(self.directory.path().join("gfnff_charges"))?;
        let charges: Option<Vec<_>> = charges
            .split_whitespace()
            .map(|v| parse_number(v).map(|q| q as f32))
            .collect();

        let charges = charges.ok_or_else(|| ParamError::new("Invalid xTB EEQ charge"))?;
        if charges.len() != count
            || (charges.iter().map(|q| *q as f64).sum::<f64>() - self.charge).abs()
                > 1e-5 * count as f64
            || forces
                .iter()
                .flat_map(|f| f.to_arr())
                .chain(charges.iter().copied())
                .any(|v| !v.is_finite())
        {
            return Err(ParamError::new("Invalid GFN-FF forces or charges"));
        }
        self.initialized = true;

        Ok(PotentialEvaluation {
            energy,
            forces,
            charges: Some(charges),
            components,
        })
    }
}

fn parse_energy(output: &str, alpb: bool) -> Result<(f64, Vec<EnergyComponent>), ParamError> {
    let mut components = Vec::new();
    let mut energy = None;

    for line in output.lines() {
        let Some(line) = line.trim().strip_prefix("::") else {
            continue;
        };
        let Some((label_value, _)) = line.split_once(" Eh ") else {
            continue;
        };
        let mut parts: Vec<_> = label_value.split_whitespace().collect();
        let Some(value) = parts.pop().and_then(parse_number) else {
            continue;
        };
        let name = parts.join(" ");
        // xTB also prints Gborn, Gsasa, Ghb, and Gshift under Gsolv. Only the
        // aggregate belongs in the additive component list, to avoid double counting.
        let name = name.strip_prefix("-> ").unwrap_or(&name);
        if name == "total energy" {
            energy = Some(value * HARTREE_KCAL);
        } else if TERMS.iter().any(|(term, _)| *term == name) {
            components.push(EnergyComponent {
                name: name.to_owned(),
                energy: value * HARTREE_KCAL,
            });
        }
    }

    let energy = energy.ok_or_else(|| ParamError::new("Missing GFN-FF total energy"))?;
    let mut expected_terms = TERMS.iter().filter(|(name, _)| *name != "Gsolv" || alpb);
    if components.len() != TERMS.len() - usize::from(!alpb)
        || expected_terms
            .any(|(name, _)| components.iter().filter(|c| c.name == *name).count() != 1)
        || (components.iter().map(|c| c.energy).sum::<f64>() - energy).abs() > 1e-5
    {
        return Err(ParamError::new(
            "Incomplete or inconsistent GFN-FF energy components",
        ));
    }

    Ok((energy, components))
}

fn parse_number(text: &str) -> Option<f64> {
    text.replace(['D', 'd'], "E")
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use na_seq::Element;

    use super::*;
    use crate::params::ForceFieldFamily;

    const VACUUM_OUTPUT: &str = "
         :: total energy           -0.110000000000 Eh    ::
         :: gradient norm           0.010000000000 Eh/a0 ::
         :: bond energy            -0.100000000000 Eh    ::
         :: angle energy            0.002000000000 Eh    ::
         :: torsion energy          0.000000000000 Eh    ::
         :: repulsion energy        0.003000000000 Eh    ::
         :: electrostat energy     -0.010000000000 Eh    ::
         :: dispersion energy      -0.004000000000 Eh    ::
         :: HB energy              -0.001000000000 Eh    ::
         :: XB energy               0.000000000000 Eh    ::
         :: bonded atm energy       0.000000000000 Eh    ::
         :: external energy         0.000000000000 Eh    ::
         :: add. restraining        0.000000000000 Eh    ::
         :: total charge            0.000000000000 e     ::
";

    fn alpb_output() -> String {
        let mut output = VACUUM_OUTPUT.replace("-0.110000000000", "-0.120000000000");
        output.push_str(
            "
         :: total w/o Gsolv         -0.110000000000 Eh    ::
         :: total w/o Gsasa/hb      -0.113000000000 Eh    ::
         :: -> Gsolv               -0.010000000000 Eh    ::
         ::    -> Gborn            -0.003000000000 Eh    ::
         ::    -> Gsasa             0.001000000000 Eh    ::
         ::    -> Ghb              -0.008000000000 Eh    ::
         ::    -> Gshift            0.000000000000 Eh    ::
",
        );
        output
    }

    #[test]
    fn vacuum_and_alpb_energies_sum_without_double_counting() {
        let (vacuum, components) = parse_energy(VACUUM_OUTPUT, false).unwrap();
        assert!((vacuum + 0.11 * HARTREE_KCAL).abs() < 1e-8);
        assert_eq!(components.len(), 11);

        let (solvated, components) = parse_energy(&alpb_output(), true).unwrap();
        assert!((solvated + 0.12 * HARTREE_KCAL).abs() < 1e-8);
        assert_eq!(components.len(), 12);
        assert!((components.iter().map(|c| c.energy).sum::<f64>() - solvated).abs() < 1e-8);
        let solvation = components.iter().find(|c| c.name == "Gsolv").unwrap();
        assert!((solvation.energy + 0.01 * HARTREE_KCAL).abs() < 1e-8);
        assert!((solvated - vacuum - solvation.energy).abs() < 1e-8);
    }

    #[test]
    fn rejects_missing_unexpected_and_duplicate_energy_components() {
        assert!(parse_energy(VACUUM_OUTPUT, true).is_err());
        assert!(parse_energy(&alpb_output(), false).is_err());
        assert!(parse_energy(&alpb_output().replace("-> Gsolv", "-> missing"), true).is_err());
        assert!(
            parse_energy(
                &alpb_output().replace("-0.120000000000", "-0.130000000000"),
                true
            )
            .is_err()
        );
        // These zero-energy terms have the right count and sum, but one is duplicated.
        assert!(parse_energy(&alpb_output().replace("XB energy", "torsion energy"), true).is_err());
    }

    #[test]
    fn alpb_is_passed_to_each_evaluation_without_generating_solvent() {
        let set = GfnffSet {
            alpb_solvent: Some(" water ".into()),
            ..Default::default()
        };
        let cfg = MdConfig::for_family(ForceFieldFamily::Gfnn);
        set.validate_config(&cfg).unwrap();
        let mut evaluator = XtbPotential::new(&set, vec![Element::Oxygen], 0.).unwrap();
        for initialized in [false, true] {
            evaluator.initialized = initialized;
            let command = evaluator.command();
            let args: Vec<_> = command
                .get_args()
                .map(|arg| arg.to_str().unwrap())
                .collect();
            assert_eq!(
                args,
                [
                    "geometry.xyz",
                    "--gfnff",
                    "--grad",
                    "--chrg",
                    "0",
                    "--alpb",
                    "water"
                ]
            );
        }

        let mut generated = cfg;
        generated.solvent = Solvent::WaterOpc;
        assert!(set.validate_config(&generated).is_err());

        let vacuum = XtbPotential::new(&GfnffSet::default(), vec![Element::Oxygen], 0.).unwrap();
        assert!(!vacuum.command().get_args().any(|arg| arg == "--alpb"));
    }

    #[test]
    fn rejects_invalid_alpb_names_before_starting_xtb() {
        for solvent in ["", " ", "--md", "water --md", "water\0", "water\nmethanol"] {
            let set = GfnffSet {
                alpb_solvent: Some(solvent.into()),
                ..Default::default()
            };
            assert!(
                set.validate_config(&MdConfig::for_family(ForceFieldFamily::Gfnn))
                    .is_err()
            );
            assert!(XtbPotential::new(&set, vec![Element::Oxygen], 0.).is_err());
        }
    }

    #[test]
    #[ignore = "requires xTB on PATH or XTB_EXECUTABLE"]
    fn xtb_alpb_forces_match_energy_gradient_and_reuse_topology() {
        let positions = vec![
            Vec3::new(0., 0., 0.),
            Vec3::new(0.98, 0., 0.),
            Vec3::new(-0.25, 0.94, 0.),
        ];
        let elements = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];
        let set = GfnffSet {
            alpb_solvent: Some("water".into()),
            ..Default::default()
        };
        let mut evaluator = XtbPotential::new(&set, elements.clone(), 0.).unwrap();
        let result = evaluator.evaluate(&positions).unwrap();
        assert_eq!(result.forces.len(), 3);
        assert!(
            result
                .components
                .iter()
                .any(|c| c.name == "Gsolv" && c.energy.abs() > 1e-6)
        );

        let mut displaced = positions.clone();
        let delta = 0.001;
        displaced[1].x += delta;
        let plus = evaluator.evaluate(&displaced).unwrap().energy;
        displaced[1].x = positions[1].x - delta;
        let minus = evaluator.evaluate(&displaced).unwrap().energy;
        let numerical_force = -((plus - minus) / (2. * delta as f64));
        assert!((result.forces[1].x as f64 - numerical_force).abs() < 0.05);

        let vacuum = XtbPotential::new(&GfnffSet::default(), elements, 0.)
            .unwrap()
            .evaluate(&positions)
            .unwrap();
        assert!((result.energy - vacuum.energy).abs() > 1e-4);
        assert!((result.forces[1].x - vacuum.forces[1].x).abs() > 1e-4);
    }
}
