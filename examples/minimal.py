#! A minimal example, demonstrating the most important syntax.

from bio_files import MmCif, Mol2
from dynamics import *

def main:
    dev = ComputationDevice.Cpu
    param_set = FfParamSet.new_amber()

    protein = MmCif.load(Path("1c8k.cif"))
    mol = Mol2.load(Path("CPB.mol2"))

    # Add Hydrogens, force field type, and partial charge to atoms in the protein these usually aren't
    # included from RSCB PDB. You can also call `populate_hydrogens_dihedrals()`, and
    # populate_peptide_ff_and_q() separately. Add bonds.
    _bonds, _dihedrals = prepare_peptide_mmcif(
        protein,
        param_set.peptide_ff_q_map,
        7.0,
    )

    mols = [
        MolDynamics.from_mol2(mol),
        MolDynamics(
            ff_mol_type=FfMolType.Peptide,
            atoms=protein.atoms,
            static_=True,
        ),
    ]

    md = MdState(dev, MdConfig.default(), mols, param_set)

    n_steps = 100
    dt = 0.002  # picoseconds.

    for _ in 0..n_steps:
        md.step(dev, dt)

    snap = md.snapshots[len(md.snapshots) - 1]  # A/R.
    
    print(
        f"KE: {}, PE: {}, Atom posits:",
        snap.energy_kinetic, snap.energy_potential
    )
    
    for posit in snap.atom_posits:
        print(f"Posit: {posit}")
        # Also keeps track of velocities, and water molecule positions/velocity

    # Do something with snapshot data, like displaying atom positions in your UI.
    # You can save to DCD file, and adjust the ratio they're saved at using the `MdConfig.snapshot_setup`
    # field: See the example below.
    for snap in md.snapshots:
        pass
}
