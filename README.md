# Molecular Dynamics
[![Crate](https://img.shields.io/crates/v/dynamics.svg)](https://crates.io/crates/dynamics)
[![Docs](https://docs.rs/dynamics/badge.svg)](https://docs.rs/dynamics)
[![PyPI](https://img.shields.io/pypi/v/dynamics.svg)](https://pypi.org/project/dynamics)

A Python and Rust library for molecular dynamics. Compatible with Linux, Windows, and Mac.
Uses CPU with threadpools and SIMD, or an nVidia GPU.

It uses traditional forcefield-based molecular dynamics, and is inspired by Amber.
It does not use quantum-mechanics, nor ab-initio methods.

It uses the [Bio-Files](https://github.com/david-oconnor/bio-files) dependency to load molecule 
and force-field files.

Please reference the [API documentation](https://docs.rs/dynamics) for details on the functionality
of each data structure and function. You may with to reference the [Bio Files API docs](https://docs.rs/bio-files)
as well.

**Note: The Python version is CPU-only for now**


## Goals
- Runs traditional MD algorithms accurately
- Easy to install, learn, and use
- Fast
- Easy integration into workflows, scripting, and applications


## Installation
Python: `pip install mol_dynamics`

Rust: `Ad dynamics` to `Cargo.toml`

For a GUI application that uses this library, download the [Daedalus molecule viewer](https://github.com/david-oconnor/daedalus)


## Parameters
Integrates the following [Amber parameters](https://ambermd.org/AmberModels.php):
- Small organic molecules, e.g. ligands: [General Amber Force Fields: GAFF2](https://ambermd.org/antechamber/gaff.html)
- Protein and amino acids: [FF19SB](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00591)
- Nucleic acids: Amber OL3 and RNA libraries
- Lipids: lipids21
- Carbohydrates: GLYCAM_06j
- Water: Explicit, with [OPC](https://arxiv.org/abs/1408.1679)


## The algorithm

This library uses a traditional MD workflow. We use the following components:

### Integrator
We use a Velocity-Verlet integrator for the whole system.


### Solvation
We use an explicit water solvation model: A 4-point rigid OPC model, with a point partial charge on each Hydrogen, and a M (or EP) point offset from the Oxygen.
We use the [SETTLE]() algorithm to maintain rigidity, while applying forces to each atom. Only the Oxygen atom
carries a Lennard Jones(LJ) force.


### Bonded forces
We use Amber-style spring-based forces to maintain covalent bonds. We maintain the following parameters:

- Bond length between each covalently-bonded atom pair
- Angle (sometimes called valence angle) between each 3-atom line of covalently-bonded atoms. (4atoms, 2 bonds)
- Dihedral (aka torsion) angles between each 4-atom line of covalently-bonded atoms. (4 atoms, 3 bonds).
These usually have rotational symmetry of 2 or 3 values which are stable.
- Improper Dihedral angles between 4-atoms in a hub-and-spoke configuration. These, for example, maintain stability
- where rings meet other parts of the molecule, or other rings.


### Non-bonded forces
These are Coulomb and Lennard Jones (LJ) interactions. These make up the large majority of computational effort. Coulomb
forces represent electric forces occurring from dipoles and similar effects, or ions. We use atom-centered pre-computed
partial charges for these. They occur within a molecule, between molecules, and between molecules and solvents.

We use a neighbors list (Sometimes called Verlet neighbors; not directly related to the Verlet integrator) to reduce
computational effort. We use the SPME Ewald (todo: link) approximation to reduce computation time. This algorithm
is suited for periodic boundary conditions, which we use for the solvent.

We use Amber's scaling and exclusion rules: LJ and Coulomb force is reduced between atoms separated by 1 and 2
covalent bonds, and skipped between atoms separated by 3 covalent bonds.


We have two modes of handling Hydrogen in bonded forces: The same as other atoms, and rigid, with position maintained
using SHAKE and RATTLE algorithms. The latter allows for stability under higher timesteps. (e.g. 2ps)


### Thermostat and barostate
Uses a Berendsen barostat, and a simple thermostat. These continuously update atom velocities (for molecules and
solvents) to match target pressure and temperatures.

## More info

We plan to support carbohydrates and lipids later. If you're interested in these, please add a Github Issue.

These general parameters do not need to be loaded externally; they provide the information needed to perform
MD with any amino acid sequence, and provide a baseline for dynamics of small organic molecules. You may wish to load
frcmod data over these that have overrides for specific small molecules.

This program can automatically load ligands with Amber parameters, for the
*Amber Geostd* set. This includes many common small organic molecules with force field parameters,
and partial charges included. It can infer these from the protein loaded, or be queried by identifier.

You can load these molecules with parameters directly from the GUI by typing the identifier. 
If you load an SDF molecule, the program may be able to automatically update it using Amber parameters and
partial charges.

For details on how dynamics using this parameterized approach works, see the 
[Amber Reference Manual](https://ambermd.org/doc12/Amber25.pdf). Section 3 and 15 are of particular
interest, regarding force field parameters.

Molecule-specific overrides to these general parameters can be loaded from *.frcmod* and *.dat* files.
We delegate this to the [bio files](https://github.com/david-OConnor/bio_files) library.

We load partial charges for ligands from *mol2*, *PDBQT* etc files. Protein dynamics and water can be simulated
using parameters built-in to the program (The Amber one above). Simulating ligands requires the loaded
file (e.g. *mol2*) include partial charges. we recommend including ligand-specific override
files as well, e.g. to load dihedral angles from *.frcmod* that aren't present in *Gaff2*.


Example use (Python):
```python
from mol_dynamics import *

TEMP_TGT: f32 = 310.  # K
PRESSURE_TGT: f32 = 310.  # Bar

fn setup_dynamics(mol: Mol2, protein: Mmcif, param_set: FfParamSet) -> MdState:
    """
    Set up dynamics between a small molecule we treat with full dynamics, and a rigid one 
    which acts on the system, but doesn't move.
    """

    protein.add_params_and_q(param_set.peptide)

    mols = [
        MolDynamics(
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms,
            # Pass a [Vec3] of starting atom positions. If absent,
            # will use the positions stored in atoms.
            atom_posits: None,
            bonds: mol.bonds,
            # Pass your own from cache if you want, or it will build.
            adjacency_list: None,
            static_: False,
            # This is usually mandatory for small organic molecules. Provided, for example,
            # in Amber FRCMOD files. Overrides general params.
            mol_specific_params: Some(lig_specific_params),
        ),
        MolDynamics(
            ff_mol_type: FfMolType::Peptide,
            atoms: protein.atoms,
            atom_posits: None,
            bonds: [], // Not required if static.
            adjacency_list: None,
            static_: true,
            mol_specific_params: None,
        ),
    ]

    return MdState(
        mols,
        TEMP_TGT,
        PRESSURE_TGT,
        param_set,
        # Or flexible, with a smaller time step.
        HydrogenMdType::Fixed(Vec::new()),
        1, # Take a snapshot every this many steps.
    )


def main():
    mol = Mol2.load("CPB.mol2")
    mut protein = Mmcif.load("1c8k.cif")

    # Add force field type, and partial charge to atoms in the protein; these usually aren't
    # included from RSCB PDB.
    populate_peptide_ff_and_q(mut protein.atoms, protein.residues, ff_map)
    
    param_paths = ParamGeneralPaths(
        peptide=Some("parm19.dat"),
        peptide_mod=Some("frcmod.ff19SB"),
        peptide_ff_q=Some("amino19.lib"),
        peptide_ff_q_c=Some("aminoct12.lib"),
        peptide_ff_q_n=Some("aminont12.lib"),
        small_organic=Some("gaff2.dat")),
    )
    
    let param_set = FfParamSet(param_paths)
    let mut md = setup_dynamics(mol, protein, param_set)
    
    let n_steps = 100
    let dt = 0.002  # picoseconds.
    
    for _ in range(n_steps):
        md.step(dt)
    
    snap = md.snapshots[len(md.snapshots) - 1] // A/R.
    print(f"KE: {snap.energy_kinetic}, PE: {snap.energy_potential}, Atom posits:")
    for atom in snap.atom_posits {
        print("Posit: {snap.posit}")
        # Also keeps track of velocities, and water molecule positions/velocity
    }
    
    for snap in md.snapshots:
        pass
        # Do something with snapshot data, like displaying atom positions in your UI,
        # Or saving to a file.
        
        
main()
}
```


Example use (Rust):
```rust
use dynamics::{
    ComputationDevice, MdState, MolDynamics, FfMolType, HydrogenMdType,
    params::{ForceFieldParamsIndexed, ProtFFTypeChargeMap, FfParamSet, ParamGeneralPaths, populate_peptide_ff_and_q},
    files::{Mol2, Mmcif}, // re-export of the bio-files lib.
};

const TEMP_TGT: f32 = 310.; // K
const PRESSURE_TGT: f32 = 310.; // Bar

/// Set up dynamics between a small molecule we treat with full dynamics, and a rigid one 
/// which acts on the system, but doesn't move.
fn setup_dynamics(mol: &Mol2, protein: &Mmcif, param_set: &FfParamSet) -> MdState {
    // Note: We assume you've already added hydrogens to any mmCif of other files
    // that don't include them.
    // todo: Include H addition in this lib?
    protein.add_params_and_q(&param_set.peptide);

    let mols = vec![
        MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: &mol.atoms,
            // Pass a &[Vec3] of starting atom positions. If absent,
            // will use the positions stored in atoms.
            atom_posits: None,
            bonds: &mol.bonds,
            // Pass your own from cache if you want, or it will build.
            adjacency_list: None,
            static_: false,
            // This is usually mandatory for small organic molecules. Provided, for example,
            // in Amber FRCMOD files. Overrides general params.
            mol_specific_params: Some(&lig_specific_params),
        },
        MolDynamics {
            ff_mol_type: FfMolType::Peptide,
            atoms: &protein.atoms,
            atom_posits: None,
            bonds: &[], // Not required if static.
            adjacency_list: None,
            static_: true,
            mol_specific_params: None,
        },
    ];

    MdState::new(
        &mols,
        TEMP_TGT,
        PRESSURE_TGT,
        param_set,
        // Or flexible, with a smaller time step.
        HydrogenMdType::Fixed(Vec::new()),
        1, // Take a snapshot every this many steps.
    ).unwrap()

}

fn main() {
    let mol = Mol2::load("CPB.mol2").unwrap();
    let mut protein = Mmcif::load("1c8k.cif").unwrap();

    // Add force field type, and partial charge to atoms in the protein; these usually aren't
    // included from RSCB PDB.
    populate_peptide_ff_and_q(&mut protein.atoms, &protein.residues, ff_map).unwrap();
    
    let param_paths = ParamGeneralPaths {
        peptide: Some(&Path::new("parm19.dat")),
        peptide_mod: Some(&Path::new("frcmod.ff19SB")),
        peptide_ff_q: Some(&Path::new("amino19.lib")),
        peptide_ff_q_c: Some(&Path::new("aminoct12.lib")),
        peptide_ff_q_n: Some(&Path::new("aminont12.lib")),
        small_organic: Some(&Path::new("gaff2.dat")),
        ..default()
    };
    
    let param_set = FfParamSet::new(&param_paths);
    let mut md = setup_dynamics(&mol, &protein, &param_set);
    
    let n_steps = 100;
    let dt = 0.002; // picoseconds.
    
    for _ in 0..n_steps {
        md.step(&ComputationDevice.Cpu, dt);
    }
    
    let snap = &md.snapshots[md.snapshots.len() - 1]; // A/R.
    println!("KE: {}, PE: {}, Atom posits:", snap.energy_kinetic, snap.energy_potential);
    for atom in &snap.atom_posits {
        println!("Posit: {}", snap.posit);
        // Also keeps track of velocities, and water molecule positions/velocity
    }
    
    for snap in &md.snapshots {
        // Do something with snapshot data, like displaying atom positions in your UI,
        // Or saving to a file.
    }
}
```


## Why this when OpenMM exists?
This library exists as part of a larger Rust biology infrastructure effort. It was not possible to use
[OpenMM](todo: URL) there due to language barriers. We've exposed Python bindings using the PyO3 library,
as it was convenient to do so. This currently only has a limited subset of the functionality of OpenMM.

It's unfortunate that we've embraced a model of computing replete with obstacles. In this case, the major
obstacle is the one placed between programming languages. We repeat efforts to make molecular dynamics
accessible to applications written using Rust.

In the process, we hope to jump over other obstacles as well: That of operating systems and distribution
methods. We hope that this is a bit easier to install and use than OpenMM; it can be used on any
Operating system, and any Python version >= 3.10, installable using `pip` or `cargo`, with no additional package managers. It's
intended to *just work*. OpenMM itself is easy to install with Pip, but the additional libraries
it requires to load force fields and files are higher-friction.


## Eratta
- Only a single dynamic molecule is supported
- Python is CPU-only
- CPU SIMD unsupported


## References
- [Amber forcefields](https://ambermd.org/antechamber/gaff.html)
- [Amber reference manual](https://ambermd.org/doc12/Amber25.pdf)
- SPME
- [OPC water model](https://arxiv.org/abs/1408.1679)