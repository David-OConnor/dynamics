//! A minimal example, demonstrating the most important syntax.

use std::path::Path;

use bio_files::{MmCif, Mol2};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdState, MolDynamics,
    params::{FfParamSet, prepare_peptide_mmcif},
};

fn main() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().unwrap();

    let mut protein = MmCif::load(Path::new("1c8k.cif")).unwrap();
    let mol = Mol2::load(Path::new("CPB.mol2")).unwrap();

    // Add Hydrogens, force field type, and partial charge to atoms in the protein; these usually aren't
    // included from RSCB PDB. You can also call `populate_hydrogens_dihedrals()`, and
    // `populate_peptide_ff_and_q() separately. Add bonds.
    let (_bonds, _dihedrals) = prepare_peptide_mmcif(
        &mut protein,
        &param_set.peptide_ff_q_map.as_ref().unwrap(),
        7.0,
    )
    .unwrap();

    let mols = vec![
        MolDynamics::from_mol2(&mol, None),
        MolDynamics {
            ff_mol_type: FfMolType::Peptide,
            atoms: protein.atoms.clone(),
            static_: true,
            ..Default::default()
        },
    ];

    let mut md = MdState::new(&dev, &MdConfig::default(), &mols, &param_set).unwrap();

    let n_steps = 100;
    let dt = 0.002; // picoseconds.

    for _ in 0..n_steps {
        md.step(&dev, dt);
    }

    let snap = &md.snapshots[md.snapshots.len() - 1]; // A/R.
    println!(
        "KE: {}, PE: {}, Atom posits:",
        snap.energy_kinetic, snap.energy_potential
    );
    for posit in &snap.atom_posits {
        println!("Posit: {posit}");
        // Also keeps track of velocities, and water molecule positions/velocity
    }

    // Do something with snapshot data, like displaying atom positions in your UI.
    // You can save to DCD file, and adjust the ratio they're saved at using the `MdConfig.snapshot_setup`
    // field: See the example below.
    for snap in &md.snapshots {}
}
