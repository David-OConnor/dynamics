from mol_dynamics import *

def setup_dynamics(mol: Mol2, protein: MmCif, param_set: FfParamSet, lig_specific: ForceFieldParams) -> MdState:
    """
    Set up dynamics between a small molecule we treat with full dynamics, and a rigid one
    which acts on the system, but doesn't move.
    """

    # Or, consider using these terse helpers instead for small organic molecules.
    # MolDynamics.from_amber_geostd("CPB")  # Can use with a PubChem CID as well.
    # MolDynamics.from_mol2(mol, lig_specific)
    # MolDynamics.from_sdf(mol, lig_specific)

    mols = [
        MolDynamics(
            ff_mol_type=FfMolType.SmallOrganic,
            atoms=mol.atoms,
            # Pass a [Vec3] of starting atom positions. If absent,
            # will use the positions stored in atoms.
            atom_posits=None,
            atom_init_velocities=None,
            bonds=mol.bonds,
            # Pass your own from cache if you want, or it will build.
            adjacency_list=None,
            static_=False,
            # This is usually mandatory for small organic molecules. Provided, for example,
            # in Amber FRCMOD files. Overrides general params.
            mol_specific_params=lig_specific,
            bonded_only=False,
        ),
        MolDynamics(
            ff_mol_type=FfMolType.Peptide,
            atoms=protein.atoms,
            atom_posits=None,
            atom_init_velocities=None,
            bonds=[],  # Not required if static.
            adjacency_list=None,
            static_=True,
            mol_specific_params=None,
            bonded_only=False,
        ),
    ]

    return MdState(
        MdConfig(),
        mols,
        param_set,
    )


def main():
    mol = Mol2.load("CPB.mol2")
    protein = MmCif.load("1c8k.cif")

    param_set = FfParamSet.new_amber()
    lig_specific = ForceFieldParams.load_frcmod("CPB.frcmod")

    # Or, instead of loading atoms and mol-specific params separately:
    # mol, lig_specific = load_prmtop("my_mol.prmtop")

    # Add Hydrogens, force field type, and partial charge to atoms in the protein; these usually aren't
    # included from RSCB PDB. You can also call `populate_hydrogens_dihedrals()`, and
    # `populate_peptide_ff_and_q() separately. Add bonds.
    _bonds, _dihedrals = prepare_peptide_mmcif(
        protein,
        param_set.peptide_ff_q_map,
        7.0,
    )
    # A variant of that function called `prepare_peptide` takes separate atom, residue, and chain
    # lists, for flexibility.

    md = setup_dynamics(mol, protein, param_set, lig_specific)

    n_steps = 100
    dt = 0.002  # picoseconds.

    for _ in range(n_steps):
        md.step(dt)

    snap = md.snapshots[len(md.snapshots) - 1]  # A/R.
    print(f"KE: {snap.energy_kinetic}, PE: {snap.energy_potential}, Atom posits:")
    for posit in snap.atom_posits:
        print(f"Posit: {posit}")
        # Also keeps track of velocities, and water molecule positions/velocity

    # Do something with snapshot data, like displaying atom positions in your UI.
    # You can save to DCD file, and adjust the ratio they're saved at using the `MdConfig.snapshot_setup`
    # field: See the example below.
    for snap in md.snapshots:
        pass


main()