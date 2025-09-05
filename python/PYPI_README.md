# Bio Files: Read and write common biology file formats

[![Crate](https://img.shields.io/crates/v/bio_files.svg)](https://crates.io/crates/bio_files)
[![Docs](https://docs.rs/bio_files/badge.svg)](https://docs.rs/bio_files)
[![PyPI](https://img.shields.io/pypi/v/biology-files.svg)](https://pypi.org/project/biology-files)


This Rust and Python library contains functionality to load and save data in common biology file formats. It operates
on data structures that are specific to each file format; you will need to convert to and from the structures
used by your application. The API docs, and examples below are sufficient to get started.

Note: Install the pip version with `pip install biology-files` due to a name conflict.

### Supported formats:
- mmCIF (Protein atom, residue, chain, and related data like secondary structure)
- Mol2 (Small molecules, e.g. ligands)
- SDF (Small molecules, e.g. ligands)
- PDBQT (Small molecules, e.g. ligands. Includes docking-specific fields.)
- Map (Electron density, e.g. from crystallography, Cryo EM)
- AB1 (Sequence tracing)
- DAT (Amber force field data for small molecules)
- FRCMOD (Amber force field patch data for small molecules)
- Amber .lib files, e.g. with charge data for amino acids and proteins.

### Planned:
- MTZ (Exists in Daedalus; needs to be decoupled)
- DNA (Exists in PlasCAD; needs to be decoupled)
- CIF structure formats (2fo-fc etc) (Exists in Daedalus; needs to be decoupled)


For Genbank, we recommend [gb-io](https://docs.rs/gb-io/latest/gb_io/).  We do not plan to support this format, due to this high quality library.

Each module represents a file format, and most have dedicated structs dedicated to operating on that format.

It operates using structs with public fields, which you can explore
using the [API docs](https://docs.rs/bio_files), or your IDE. These structs generally include these three methods: `new()`,
`save()` and `load()`. `new()` accepts `&str` for text files, and a `R: Read + Seek` for binary. `save()` and
`load()` accept `&Path`.
The Force Field formats use `load_dat`, `save_frcmod` instead, as they use the same structs for both formats.

## Serial numbers
Serial numbers for atoms, residues, secondary structure, and chains are generally pulled directly from atom data files
(mmCIF, Mol2 etc). These lists reference atoms, or residues, stored as `Vec<u32>`, with the `u32` being the serial number.
In your application, you may wish to adapt these generic types to custom ones that use index lookups
instead of serial numbers. We use SNs here because they're more robust, and match the input files directly;
add optimizations downstream, like converting to indices, and/or applying back-references. (e.g. the index of the residue
an atom's in, in your derived Atom struct).

Example use. (The Python version uses an equivalent API)

```rust
pub fn open_molecule(&mut self, path: &Path) -> io::Result<()> {
    let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
    let extension = binding;

    let molecule = match extension.to_str().unwrap() {
        "sdf" => Ok(Sdf::load(path)?.into()),
        "mol2" => Ok(Mol2::load(path)?.into()),
        _ => ()
    };
}

pub fn open_map(&mut self, path: &Path) -> io::Result<()> {
    let dm = DensityMap::load(path)?;

    // Call dm.density_at_point_trilinear(coord) to get density
    // Run `density_to_sig` to get sigma-normalized density, for uniform display.

    self.load_density(dm);

    Ok(())
}

/// A single endpoint to save a number of file types
pub fn save(&mut self, path: &Path) -> io::Result<()> {
    let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
    let extension = binding;

    match extension.to_str().unwrap_or_default() {
        "sdf" => match &self.ligand {
            Some(lig) => {
                lig.molecule.to_sdf().save(path)?;
            }
            None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
        },
        "mol2" => match &self.ligand {
            Some(lig) => {
                lig.molecule.to_mol2().save(path)?;
            }
            None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
        }
    }
}
```

## Amber force fields

Reference the [Amber reference manual](Amber 2025 Reference Manual, section 15](https://ambermd.org/doc12/Amber25.pdf)
for details on how we parse its files, and how to use the results. In some cases, we change the format from
the raw Amber data. For example, we store angles as radians (vice degrees), and σ vice R_min for Van der Waals
parameters. Structs and fields are documented with reference manual references.

The Amber forcefield parameter format has fields which each contain a `Vec` of a certain type of data. (Bond stretching parameters,
angle between 3 atoms, torsion/dihedral angles etc.) You may wish to parse these into a format that has faster lookups
for your application.


Note that the above examples expect that your application has a struct representing the molecule that has
`From<Mol2>`, and `to_mol2(&self)` (etc) methods. The details of these depend on the application. For example:

```rust
impl From<Sdf> for Molecule {
    fn from(m: Sdf) -> Self {
        // We've implemented `From<AtomGeneric>` and `From<ResidueGeneric>` for our application's `Atom` and
        // `Residue`
        let atoms = m.atoms.iter().map(|a| a.into()).collect();
        let residues = m.residues.iter().map(|r| r.into()).collect();

        Self::new(m.ident, atoms, m.chains.clone(), residues, None, None);
    }
}
```

A practical example of parsing a molecule from a `mmCIF` as parsed from `bio_files` into an application-specific format:
```rust
fn load() {
    let cif_data = mmcif::load("./1htm.cif");
    let mol: Molecule = cif_data.try_into().unwrap();
}

impl TryFrom<MmCif> for Molecule {
    type Error = io::Error;

    fn try_from(m: MmCif) -> Result<Self, Self::Error> {
        let mut atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let mut residues = Vec::with_capacity(m.residues.len());
        for res in &m.residues {
            residues.push(Residue::from_generic(res, &atoms)?);
        }

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        // Now that chains and residues are loaded, update atoms with their back-ref index.
        for atom in &mut atoms {
            for (i, res) in residues.iter().enumerate() {
                if res.atom_sns.contains(&atom.serial_number) {
                    atom.residue = Some(i);
                    break;
                }
            }

            for (i, chain) in chains.iter().enumerate() {
                if chain.atom_sns.contains(&atom.serial_number) {
                    atom.chain = Some(i);
                    break;
                }
            }
        }

        let mut result = Self::new(m.ident.clone(), atoms, chains, residues, None, None);

        result.experimental_method = m.experimental_method.clone();
        result.secondary_structure = m.secondary_structure.clone();

        result.bonds_hydrogen = Vec::new();
        result.adjacency_list = result.build_adjacency_list();

        Ok(result)
    }
}
```

Note: The Python version is currently missing support for some formats, and not all fields are exposed.


### References
- [Amber 2025 Reference Manual, section 15](https://ambermd.org/doc12/Amber25.pdf)