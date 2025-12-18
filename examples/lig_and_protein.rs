//! Taken from ChemForma. Shows how to bind application-specific data structures to
//! this library's API.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use bio_files::{MmCif, Mol2, Sdf, create_bonds, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, HydrogenConstraint, Integrator, MdConfig, MdState, MolDynamics,
    ParamError, SimBoxInit,
    params::{FfParamSet, prepare_peptide_mmcif},
    snapshot::{SaveType, Snapshot, SnapshotHandler},
};
use lin_alg::f64::Vec3;

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to be counted.
// Set this wide to take into account motion.
const STATIC_ATOM_DIST_THRESH: f64 = 8.;

/// Perform MD on the ligand, with nearby protein (receptor) atoms, from the docking setup as static
/// non-bonded contributors. (Vdw and coulomb)
pub fn build_dynamics(
    dev: &ComputationDevice,
    ligs: Vec<&mut Sdf>,
    peptide: &MmCif,
    param_set: &FfParamSet,
    cfg: &MdConfig,
    n_steps: u32,
    dt: f32,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    let mut mols = Vec::new();

    for lig in &ligs {
        mols.push(MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: lig.atoms.clone(),
            atom_posits: None,
            atom_init_velocities: None,
            bonds: lig.bonds.clone(),
            // Adj list is created atomatically, but can be passed from a cache.
            adjacency_list: None,
            static_: false,
            mol_specific_params: None,
            bonded_only: false,
            // Or: ..Default::default()
        });
    }

    // A demonstration of only selecting peptide atoms near a ligand.
    // We assume hetero atoms are ligands, water etc, and are not part of the protein.
    let atoms: Vec<_> = peptide
        .atoms
        .iter()
        .filter(|a| {
            let mut closest_dist = f64::MAX;
            for lig in &ligs {
                // You should probably use a special position set vs the native atom positions.
                for a in &lig.atoms {
                    let posit = a.posit;
                    let dist = (posit - a.posit).magnitude();
                    if dist < closest_dist {
                        closest_dist = dist;
                    }
                }
            }

            !a.hetero && closest_dist < STATIC_ATOM_DIST_THRESH
        })
        .map(|a| a.clone())
        .collect();

    let bonds = create_bonds(&atoms);

    mols.push(MolDynamics {
        ff_mol_type: FfMolType::Peptide,
        atoms,
        bonds,
        static_: true,
        ..Default::default()
    });

    // See also: `MolDynamics::from_sdf()`, `::from_mol2()`, and `::from_amber_geostd("CPB")`

    println!("Initializing MD state...");
    let mut md_state = MdState::new(dev, cfg, &mols, param_set)?;
    println!("Done.");

    let start = Instant::now();

    for _ in 0..n_steps {
        md_state.step(dev, dt);
    }

    let elapsed = start.elapsed();
    println!("MD complete in {:.2} s", elapsed.as_secs());

    change_snapshot(ligs, &md_state.snapshots[0]);

    Ok(md_state)
}

// todo: This is so annoying. &[T] vs [&T].
/// Set atom positions for molecules involve in dynamics to that of a snapshot.
pub fn change_snapshot(ligs: Vec<&mut Sdf>, snapshot: &Snapshot) {
    // todo: Handle peptide too!

    // todo: QC this logic.

    // Unflatten.
    let mut start_i_this_mol = 0;

    for lig in ligs {
        // todo: A/R, for your system that manages atom positions.
        let mut atom_posits = vec![Vec3::new_zero(); lig.atoms.len()];

        for (i_snap, posit) in snapshot.atom_posits.iter().enumerate() {
            if i_snap < start_i_this_mol || i_snap >= atom_posits.len() + start_i_this_mol {
                continue;
            }
            atom_posits[i_snap - start_i_this_mol] = (*posit).into();
        }

        start_i_this_mol += atom_posits.len();
    }
}

fn main() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().unwrap();

    let mut protein = MmCif::load(Path::new("1c8k.cif")).unwrap();
    // let mut mol = Mol2::load(Path::new("CPB.mol2")).unwrap();
    let mut mol = Sdf::load(Path::new("123.sdf")).unwrap();
    // Optional; the library infers FRCMOD overrides on its own.
    let _mol_specific = ForceFieldParams::load_frcmod(Path::new("CPB.frcmod")).unwrap();

    // Or, instead of loading atoms and mol-specific params separately:
    // let (mol, lig_specific) = load_prmtop("my_mol.prmtop");

    // Add Hydrogens, force field type, and partial charge to atoms in the protein; these usually aren't
    // included from RSCB PDB. You can also call `populate_hydrogens_dihedrals()`, and
    // `populate_peptide_ff_and_q() separately. Add bonds.
    let (_bonds, _dihedrals) = prepare_peptide_mmcif(
        &mut protein,
        &param_set.peptide_ff_q_map.as_ref().unwrap(),
        7.0,
    )
    .unwrap();

    // A variant of that function called `prepare_peptide` takes separate atom, residue, and chain
    // lists, for flexibility.

    let cfg = MdConfig {
        // Defaults to Langevin middle.
        integrator: Integrator::VerletVelocity,
        // If enabled, zero the drift in center of mass of the system.
        zero_com_drift: true,
        // Kelvin. Defaults to 310 K.
        temp_target: 310.,
        // Bar (Pa/100). Defaults to 1 bar.
        pressure_target: 1.,
        // Allows constraining Hydrogens to be rigid with their bonded atom, using SHAKE and RATTLE
        // algorithms. This allows for higher time steps.
        hydrogen_constraint: HydrogenConstraint::Constrained,
        // Deafults to in-memory, every step
        snapshot_handlers: vec![
            SnapshotHandler {
                save_type: SaveType::Memory,
                ratio: 1,
            },
            SnapshotHandler {
                save_type: SaveType::Dcd(PathBuf::from("output.dcd")),
                ratio: 10,
            },
        ],
        // Or sim_box: SimBoxInit::Fixed((Vec3::new(-10., -10., -10.), Vec3::new(10., 10., 10.)),
        sim_box: SimBoxInit::Pad(10.),
        ..Default::default()
    };

    let mut md = build_dynamics(&dev, vec![&mut mol], &protein, &param_set, &cfg, 100, 0.001);
}
