# Molecular Dynamics
[![Crate](https://img.shields.io/crates/v/dynamics.svg)](https://crates.io/crates/dynamics)
[![Docs](https://docs.rs/dynamics/badge.svg)](https://docs.rs/dynamics)
[![PyPI](https://img.shields.io/pypi/v/dynamics.svg)](https://pypi.org/project/dynamics)

A Python and Rust library for molecular dynamics. Compatible with Linux, Windows, and Mac.
Uses CPU with threadpools and SIMD, or an nVidia GPU.


## Installation
Python: `pip install dynamics`

Rust: `dynamics` Add to Cargo.toml

### Windows and Linux
[Download, unzip, and run](https://github.com/David-OConnor/daedalus/releases).


## Molecular dynamics
Integrates the following [Amber parameters](https://ambermd.org/AmberModels.php):
- Small organic molecules, e.g. ligands: [General Amber Force Fields: GAFF2](https://ambermd.org/antechamber/gaff.html)
- Protein and amino acids: [FF19SB](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00591)
- Nucleic acids: Amber OL3 and RNA libraries
- Water: [OPC](https://arxiv.org/abs/1408.1679)


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

Moleucule-specific overrides to these general parameters can be loaded from *.frcmod* and *.dat* files.
We delegate this to the [bio files](https://github.com/david-OConnor/bio_files) library.

We load partial charges for ligands from *mol2*, *PDBQT* etc files. Protein dynamics and water can be simulated
using parameters built-in to the program (The Amber one above). Simulating ligands requires the loaded
file (e.g. *mol2*) include partial charges. we recommend including ligand-specific override
files as well, e.g. to load dihedral angles from *.frcmod* that aren't present in *Gaff2*.


## References