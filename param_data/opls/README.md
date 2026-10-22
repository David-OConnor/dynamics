# OPLS-AA/L parameter data

These files are copied unchanged from the GROMACS v2026.4 distribution:
https://github.com/gromacs/gromacs/tree/v2026.4/share/top/oplsaa.ff

The upstream distribution's license notice is included as COPYING (LGPL 2.1).
The force-field references are listed in forcefield.itp and atomtypes.atp.
In particular, the protein templates include the Kaminski et al. (2001)
side-chain torsion refinements (OPLS-AA/L).

The Rust implementation embeds ffnonbonded.itp, ffbonded.itp and aminoacids.rtp.
Charged terminal patches and TIP4P/ion values follow the accompanying .tdb,
tip4p.itp and ffnonbonded.itp files. No external GROMACS installation or network
access is needed for proteins or bundled ligand chemistry. Ligands outside
these tables can use the automatic LigParGen service and persistent cache
documented below.

Validation used GROMACS 2025.3 pdb2gmx with these v2026.4 tables to independently
parameterize ubiquitin (1,231 atoms). Native charges, 1,237 bonds, 2,257 angles,
216 planar impropers and 3,264 scaled pairs match. Native bonded energy agrees
with the independently generated topology. Native/exported energies and forces
were also compared to GROMACS PME calculations. Native and GROMACS electrostatic
constants and PME numerics differ slightly. Temporary validation code was removed.

## OPLS-AA ligand parameters

`foyer-oplsaa.xml` is copied unchanged from mosdef-hub/foyer at commit
`a04faae806c45ac9a2986e113cedf8954f23f828`:
https://github.com/mosdef-hub/foyer/blob/a04faae806c45ac9a2986e113cedf8954f23f828/foyer/forcefields/xml/oplsaa.xml

The upstream MIT license is included as `FOYER-LICENSE.rst`. The XML cites the
original force-field publications and contains the source attribution.
`convert_foyer.py` (Python standard library only) regenerates `foyer-ligands.tsv`;
the generated header records the XML's SHA256. Dynamics embeds the compact TSV,
with 228 SMARTS definitions, the accompanying atom types and partial charges,
and harmonic bond/angle and Ryckaert–Bellemans torsion tables. This Foyer subset
contains no improper terms. No parameters are inferred from Amber/GAFF2.

The native matcher implements this data set's Foyer SMARTS subset, including
ordered type overrides, type references and chordless rings of up to eight
atoms. All hydrogens must be present. Missing or ambiguous types, missing bonded
terms, and total charge farther than 0.005 e from an integer are rejected.
These bundled rules are not universal. When they cannot parameterize a ligand,
the general builder uses the LigParGen generation/cache path below.

Temporary validation compared 50 hydrogenated molecules against upstream
Foyer's matcher and independently assigned OpenMM bonded/nonbonded parameters.
Assigned types and charges matched. With electrostatics disabled to isolate
bonded and geometric LJ terms, the maximum energy error was 0.00025 kcal/mol
and maximum force component error was 0.006 kcal/mol/angstrom. Six further
molecules were explicitly rejected (missing atom types or non-integral charge).
Those tests exercised bundled rules only; the general builder can now generate
parameters for additional chemistry through LigParGen.
The fixtures included aliphatic, aromatic and heterocyclic compounds, alcohols,
ethers, carbonyl compounds, amides, amines, nitriles, sulfur compounds and
halogenated compounds. Temporary validation code was removed.

## Complete creatine ligand template

`creatine.itp` is an unmodified generated output from the
Jorgensen group's official LigParGen service, obtained 2026-10-21:
https://huggingface.co/spaces/lsdodda/ligpargen
https://github.com/leelasd/LigParGen_2.3

Input: `CN(CC(=O)O)C(=N)N`, net charge 0, optimization level 0,
1.14*CM1A-LBCC charge model. Job ID: `9ff3f126eaa040089eccb61f757b6a2c`.
`creatine.mol` reconstructs the matching element/bond-order graph in that ITP's
atom order from the documented input SMILES and bonded hydrogens. The previous
`creatine.sdf` was absent from the repository. The MOL coordinates are placeholders
because template recognition uses connectivity only; simulation coordinates always
come from the caller. Numeric force-field parameters in the ITP are unchanged.

This is the neutral tautomer present in the supplied PubChem 586 structure,
not a zwitterionic or protonated creatine model. The server's generated residue
label is YIB; recognition uses the entire element/bond-order graph, including
all 18 atoms and 17 bonds. No coordinates, serial numbers or atom names are
required to match. A different protonation state or tautomer does not use this
template. Recognizing this bundled template does not contact a server.

The complete template supplements Foyer's rules, which do not cover creatine's
substituted guanidine nitrogen. It supplies charges, masses, LJ values, bonds,
angles, propers/impropers, exclusions and scaled pairs. Numeric parameters are
generated OPLS-AA/CM1A-LBCC data; no BOSS executable or source is distributed.
References: Dodda et al., Nucleic Acids Res. 2017, 45, W331-W336,
doi:10.1093/nar/gkx312; Dodda et al., J. Phys. Chem. B 2017, 121, 3864-3870,
doi:10.1021/acs.jpcb.7b00272. Upstream attribution remains in the ITP header.

Temporary validation used the user's creatine coordinates, reversed atom order
and noncontiguous serial numbers. Against OpenMM's independent GROMACS topology
reader (with the same geometric LJ rule), maximum energy error was below
0.00004 kcal/mol and force component error below 0.0036 kcal/mol/angstrom, with
electrostatics disabled to isolate bonded and LJ terms. Exported GROMACS energy
agreed within 0.00005 kcal/mol. Native CPU/CUDA agreed; a native TIP4P/PME step
also passed. A changed guanidine bond order was rejected. Temporary validation
code was removed.

Generated data SHA256:
- `creatine.itp`: `fe3cb6c73375c5df89f7191b87b57fa3ca7a3c4510fe6a1b52f7681392e08316`

## Automatic ligand generation and local cache

The general ligand builder now supplements complete bundled assignments with
https://huggingface.co/spaces/lsdodda/ligpargen at runtime. It submits a MOL
structure with explicit hydrogens and formal charges recovered from ordinary
closed-shell organic valences. Ambiguous valences require an explicit topology;
protonation is never changed to make typing succeed. Submission uses a connected
heavy-atom order for BOSS's Z-matrix construction. Output graph matching maps
all parameters back without changing caller coordinates or structural names.

The generated GROMACS topology supplies every bonded/nonbonded term, including
impropers, exclusions and scaled pairs. Imports validate OPLS's geometric LJ
mixing, 0.5 LJ/Coulomb scaling and the requested net charge. Neutral requests use
CM1A-LBCC; charged requests use CM1A. BOSS itself is not bundled or installed.

Successful assignments are saved as one structure/topology cache file; a full
graph match validates reuse. The cache fingerprint ignores coordinates and atom
ordering. The public service permits five submissions in ten minutes, up to
200 atoms including hydrogens, with its own chemical coverage and charge limits.
Requests wait for quota on the calling worker; rate-limit errors are not treated
as unsupported atom types. Other server errors retain their actual diagnostic.
See the main Dynamics README for cache paths and the offline-only API.

Validation of the automatic path used live LigParGen jobs for neutral and
zwitterionic methionine, caffeine and aspirin. Each generated topology passed
an independent cache reload with reversed atom order/noncontiguous serials and
a native force calculation. Neutral methionine also completed a full-charge
TIP4P/PME step with 799 waters. Against OpenMM's independent GROMACS reader,
methionine's bonded/geometric-LJ energy differed by less than 0.000001 kcal/mol
and its largest force component by less than 0.0001 kcal/mol/angstrom, with
charges disabled to isolate imported bonded/exclusion/LJ terms. Six existing
Amber/GROMACS import regressions passed. Temporary validation code was removed.
