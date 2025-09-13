# Molecular Dynamics
[![Crate](https://img.shields.io/crates/v/dynamics.svg)](https://crates.io/crates/dynamics)
[![Docs](https://docs.rs/dynamics/badge.svg)](https://docs.rs/dynamics)
[![PyPI](https://img.shields.io/pypi/v/mol_dynamics.svg)](https://pypi.org/project/mol_dynamics)

A Python and Rust library for molecular dynamics. Compatible with Linux, Windows, and Mac.
Uses CPU with threadpools and SIMD, or an Nvidia GPU.

## Warning: Early release! Lots of missing features. If you see something, post a Github issue.

It uses traditional forcefield-based molecular dynamics, and is inspired by Amber.
It does not use quantum-mechanics, nor ab-initio methods.

It uses the [Bio-Files](https://github.com/david-oconnor/bio-files) dependency to load molecule 
and force-field files.

Please reference the [API documentation](https://docs.rs/dynamics) for details on the functionality
of each data structure and function. You may with to reference the [Bio Files API docs](https://docs.rs/bio-files)
as well.

**Note: The Python version is CPU-only for now**
**Note: We currently only support saving and loading snapshots/trajectories in a custom format.**


## Use of this library
This is intended for integration into a Rust or Python program, another Rust or Python library, or in small scripts that 
describe a workflow. When scripting, you will likely load molecule files directly, use integrated force fields or load them
from file, and save results to a reporter format like DCD. If incorporating into an application, you might do more in
memory using the [data structures](https://docs.rs/dynamics/latest/dynamics/) we provide.

Our API goal is to both provide a default terse syntax with safe defaults, and allow customization and 
flexibility that facilitates integration into bigger systems.


## Goals
- Runs traditional MD algorithms accurately
- Easy to install, learn, and use
- Fast
- Easy integration into workflows, scripting, and applications


## Installation
Python: `pip install mol_dynamics`

Rust: Add `dynamics` to `Cargo.toml`. Likely `bio_files` as well.

For a GUI application that uses this library, download the [Daedalus molecule viewer](https://github.com/david-oconnor/daedalus) This
provides an easy-to-use way to set up the simulation, and play back trajectories.


## Input topology
The simulation accepts sets of [AtomicGeneric](https://docs.rs/bio_files/latest/bio_files/struct.AtomGeneric.html) and 
[BondGeneric](https://docs.rs/bio_files/latest/bio_files/struct.BondGeneric.html). You can get these by loading molecular
file formats (mmCIF, Mol2, SDF, etc) using the [Bio Files](https://github.com/david-OConnor/bio_files) library 
([biology-files in Python](https://pypi.org/project/biology-files/)), or by creating them 
directly. See examples below and in the examples folder, and the 
docs links above; those are structs of plain data that can be built from from arbitrary input sources. For example, if
you're building an application, you might use a more complicated Atom format; you can create a function that converts 
between yours, and `AtomGeneric`.

Requirements by molecule type:
- **Proteins**/amino acids chains: No special requirements. Uses `mmCif`. (also known as PdbX). Force field type and
partial charges are inferred automatically.
- **Small organic molecules**: Must have Force Field name, and partial charge populated on all atoms. `Mol2` files from 
- Amber's GeoStd library have these. SDF files from Drugbank are generally missing these fields.  SDF files from 
PubChem usually includes partial charges, but not forcefield names. Molecule-specific
parameters (e.g. from .frcmod files) may be required; these can also be [automatically] loaded from Amber Geostd.
- **Lipids**: TBD
- **Nucleic acids**: TBD
- **Carbohydrates**: TBD
- **Lipids**: TBD


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


### Integrators, thermostats, barostats
We provide a Velocity-Verlet integrator. It can be used with a Berendsen barostat, and either a 
CSVR/Bussi, or Langevin thermostat (Middle or traditional). These continuously update atom velocities (for molecules and
solvents) to match target pressure and temperatures. The Langevin Middle thermostat is a good starting point.


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
computational effort. We use the [SPME Ewald](https://manual.gromacs.org/nightly/reference-manual/functions/long-range-electrostatics.html) approximation to reduce computation time. This algorithm
is suited for periodic boundary conditions, which we use for the solvent.

We use Amber's scaling and exclusion rules: LJ and Coulomb force is reduced between atoms separated by 1 and 2
covalent bonds, and skipped between atoms separated by 3 covalent bonds.


We have two modes of handling Hydrogen in bonded forces: The same as other atoms, and rigid, with position maintained
using SHAKE and RATTLE algorithms. The latter allows for stability under higher timesteps. (e.g. 2ps)


### Initial relaxation
We run a relaxation / energy-minimization function prior to starting each simulation. This adjusts atom
positions to reduce the amount of energy that comes from initial conditions deviating from bonded parameters.





### Floating point precision
Mixed precision: 32-bit floating points for most operations. We use 64-bit accumulators, and in thermostat 
and barostat computations.


### Saving results
Snapshots of results can be returned in memory, or saved to disk in [DCD](https://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.app.dcdfile.DCDFile.html) format.


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


def setup_dynamics(mol: Mol2, protein: MmCif, param_set: FfParamSet, lig_specific: ForceFieldParams) -> MdState:
    """
    Set up dynamics between a small molecule we treat with full dynamics, and a rigid one 
    which acts on the system, but doesn't move.
    """

    mols = [
        MolDynamics(
            ff_mol_type=FfMolType.SmallOrganic,
            atoms=mol.atoms,
            # Pass a [Vec3] of starting atom positions. If absent,
            # will use the positions stored in atoms.
            atom_posits=None,
            bonds=mol.bonds,
            # Pass your own from cache if you want, or it will build.
            adjacency_list=None,
            static_=False,
            # This is usually mandatory for small organic molecules. Provided, for example,
            # in Amber FRCMOD files. Overrides general params.
            mol_specific_params=lig_specific,
        ),
        MolDynamics(
            ff_mol_type=FfMolType.Peptide,
            atoms=protein.atoms,
            atom_posits=None,
            bonds=[],  # Not required if static.
            adjacency_list=None,
            static_=True,
            mol_specific_params=None,
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
    
    # Add Hydrogens, force field type, and partial charge to atoms in the protein; these usually aren't
    # included from RSCB PDB. You can also call `populate_hydrogens_dihedrals()`, and
    # `populate_peptide_ff_and_q() separately.
    protein.atoms = prepare_peptide(
        protein.atoms,
        protein.residues,
        protein.chains,
        param_set.peptide_ff_q_map,
        7.0,
    )
    
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
```


Example use (Rust):
```rust
use std::path::Path;

use bio_files::{MmCif, Mol2, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdState, MolDynamics,
    params::{FfParamSet, prepare_peptide},
    populate_hydrogens_dihedrals,
};

/// Set up dynamics between a small molecule we treat with full dynamics, and a rigid one
/// which acts on the system, but doesn't move.
fn setup_dynamics(
    mol: &Mol2,
    protein: &MmCif,
    param_set: &FfParamSet,
    lig_specific: &ForceFieldParams,
) -> MdState {
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
            mol_specific_params: Some(lig_specific),
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

    MdState::new(&MdConfig::default(), &mols, param_set).unwrap()
}

fn main() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().unwrap();

    let mut protein = MmCif::load(Path::new("1c8k.cif")).unwrap();
    let mol = Mol2::load(Path::new("CPB.mol2")).unwrap();
    let mol_specific = ForceFieldParams::load_frcmod(Path::new("CPB.frcmod")).unwrap();
    
    // Or, if you have a small molecule available in Amber Geostd, load it remotely:
    // let data = bio_apis::amber_geostd::load_mol_files("CPB");
    // let mol = Mol2::new(&data.mol2);
    // let mol_specific = ForceFieldParams::from_frcmod(&data.frcmod);
    
    // Add Hydrogens, force field type, and partial charge to atoms in the protein; these usually aren't
    // included from RSCB PDB. You can also call `populate_hydrogens_dihedrals()`, and
    // `populate_peptide_ff_and_q() separately.
    prepare_peptide(
        &mut protein.atoms,
        &mut protein.residues,
        &mut protein.chains,
        &param_set.peptide_ff_q_map.as_ref().unwrap(),
        7.0,
    )
        .unwrap();

    let mut md = setup_dynamics(&mol, &protein, &param_set, &mol_specific);

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
```

Example of loading your own parameter files:
```python
    param_paths = ParamGeneralPaths(
        peptide="parm19.dat",
        peptide_mod="frcmod.ff19SB",
        peptide_ff_q="amino19.lib",
        peptide_ff_q_c="aminoct12.lib",
        peptide_ff_q_n=None,
        small_organic="gaff2.dat",
    )
    
    param_set = FfParamSet(param_paths)
```

```rust
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
```

An overview of configuration parameters. You may wish to (Rust) use a baseline of the `Default` implementation,
then override specific fields you wish to change.
```rust
let cfg = MdConfig {
    // Defaults to Langevin middle.
    integrator: dynamics::Integrator::VelocityVerlet,
    // If enabled, zero the drift in center of mass of the system.
    zero_com_drift: true,
    // Kelvin. Defaults to 310 K.
    temp_target: 310.,
    // Bar (Pa/100). Defaults to 1 bar.
    pressure_target: 1.,
    // Allows constraining Hydrogens to be rigid with their bonded atom, using SHAKE and RATTLE
    // algorithms. This allows for higher time steps.
    hydrogen_constraint: dynamics::HydrogenConstraint::Fixed,
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
    sim_box: SimBoxInit::Pad(10.),
    // Or sim_box: SimBoxInit::Fixed((Vec3::new(-10., -10., -10.), Vec3::new(10., 10., 10.)),
};
```

Python config syntax:
```python
cfg = MdConfig() // Initializes with defaults.

cfg.integrator = dynamics.Integrator.VelocityVerlet
cfg.temp_target = 310.
# etc
```

## Using with GPU

We use the [Cudarc](https://github.com/coreylowman/cudarc) library for GPU (CUDA) integration.

Rust setup example with Cudarc. Pass this to the `step` function.
```rust
let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();
let module = ctx.load_module(Ptx::from_src(dynamics::PTX)).unwrap();
let dev = ComputationDevice::Gpu((stream, module));
```

**Note: Currently GPU isn't supported in the python bindings**

To use with an Nvidia GPU, enable the `cuda` feature in `Cargo.toml`. The library will generate PTX instructions
as a publicly exposed string. Set up your application to use it from `dynamics::PTX`. It requires
CUDA 13 support, which requires Nvidia driver version 580 or higher.


## On unflattening trajactory data
If you passed multiple molecules, these will be flattened during runtime, and in snapshots. You 
need to unflatten them if placing back into their original data structures. 


## Why this when OpenMM exists?
This library exists as part of a larger Rust biology infrastructure effort. It's not possible to use
[OpenMM](https://openmm.org/) there due to the language barrier. This library currently only has a limited subset of the 
functionality of OpenMM. It's unfortunate that, as a society, we've embraced a model of computing replete with obstacles. In this case, the major
one is the one placed between programming languages.

While going around this obstacle, we attempt to jump over others, to make molecular dynamics more accessible.
This includes operating systems, software distribution, and user experience. We hope that this is easier to install and use
than OpenMM; it can be used on any
Operating system, and any Python version >= 3.10, installable using `pip` or `cargo`.

This library is intended to *just work*. OpenMM itself is easy to install with Pip, but the additional libraries
it requires to load force fields are higher-friction. Additionally, it's easy to run into errors when
using it with proteins from RCSB PDB, and small molecules broadly. Getting a functional OpenMM configuration
for a given system involves work which we hope to eschew.



## Compiling from source
It requires these Amber parameter files to be present under the project's `resources` folder at compile time.
These are available in [Amber tools](https://ambermd.org/GetAmber.php). Download, unpack, then copy these files from
`dat/leap/parm` and `dat/leap/lib`:

- `amino19.lib`
- `aminoct12.lib`
- `aminont12.lib`
- `parm19.dat`
- `frcmod.ff19SB`
- `gaff2.dat`
- `ff-nucleic-OL24.lib`
- `ff-nucleic-OL24.frcmod`
- `RNA.lib`

We provide a [copy of these files](https://github.com/David-OConnor/daedalus/releases/download/0.1.3/amber_params_sept_2025.zip)
for convenience; this is a much smaller download than the entire Amber package, and prevents needing to locate the specific files.
Unpack, and place these under `resources` prior to compiling.

To build the Python library wheel, from the `python` subdirectory, run `maturin build`. You can load the library
locally for testing, once built, by running `pip install .`

## Eratta
- Python is CPU-only
- GPU operations are slower than they should, as we're passing all data between CPU and GPU each
time step.
- CPU SIMD unsupported


## References
- [Amber forcefields](https://ambermd.org/antechamber/gaff.html)
- [Amber reference manual](https://ambermd.org/doc12/Amber25.pdf)
- [Ewald Summation/SPME](https://manual.gromacs.org/nightly/reference-manual/functions/long-range-electrostatics.html)
- [OPC water model](https://arxiv.org/abs/1408.1679)