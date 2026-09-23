# Amber/GAFF2 inference QC

The fixtures were generated with unmodified AmberClassic source at commit
[`8e55e97ada48b96eefaec2e6a3fa849018aaeea5`](https://github.com/Amber-MD/AmberClassic/tree/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber),
compiled with GCC, using this repository's `ATOMTYPE_GFF2.DEF`, `PARMCHK.DAT`,
and `gaff2.dat`. They are reference program outputs, not expectations obtained
from the Rust implementation. `regenerate.py` reproduces both fixture files;
it needs working Amber `bondtype`, `atomtype`, `parmchk2` binaries and their data
files, but no chemistry toolkit. The original molecular graphs were generated
from the included SMILES using RDKit with explicit hydrogens and Kekule bonds.
Tests themselves require neither Amber nor RDKit nor a network connection.

## Coverage and results

- 105 molecules, 1,269 atoms: aliphatic and fused rings, five-membered
  heteroaromatics, conjugated chains, carbonyls, amines/amides, charged groups,
  sulfur/phosphorus chemistry, halogens, nucleobases, and several drug molecules.
- 104 molecules match `atomtype` exactly; guanine deliberately differs at two
  paired conjugated labels, for the reason below.
- 326 parameter records from `parmchk2`: 316 match numerically, seven improper
  records are checked against the existing GAFF2 terms instead, and three
  degenerate proper torsions are deliberately excluded. Every retained Fourier
  term, divisor, phase, and periodicity is compared.
- Additional tests cover aromatic versus Kekule input, arbitrary serial numbers,
  atom/bond ordering, ring counts, nested patterns, property AND/OR/count/prime
  semantics, directed parameter substitutions, and invalid input.

For comparison, the previous implementation disagreed with the unmodified Amber
atom-type reference on 56 molecules (245 atoms) on these same Kekule graphs.
That comparison is a QC sample, not a general error-rate estimate.

## Intentional differences from this upstream revision

1. **Conjugated label propagation:** `atomtype.c::atadjust` scans bonds and may
   seed an unvisited pair before a later ring-closing bond connects it to an
   existing component. In guanine, the resulting N7-C8 aromatic single bond joins
   `nd` to `cc`, contradicting the function's same-label rule for single bonds.
   The Rust traversal completes one connected component before seeding another:
   C6/N7 become `cd`/`nc`. Tests assert the single/multiple-bond parity constraints
   over the entire fixture set. A component's two label orientations are
   interchangeable; changing atom order may interchange paired suffixes.

2. **Improper parameter search bound:** upstream `parmchk2.c` sets
   `improperparmnum2 = impropernum` in `main`, immediately after assigning the
   other parameter-table bounds. `impropernum` counts molecular impropers;
   `improperparmnum` counts entries in the force-field table. Consequently this
   revision can miss existing table entries. Acetone is a small reproducer:
   the reference writes a 1.1 default despite the available `X-X-c-o` 10.5 term.
   Rust searches the complete table. It also permutes only the three satellites,
   holding the third (central) atom fixed, so a carbonyl oxygen need not be the
   last satellite alphabetically. Tests explicitly document the seven numerical
   differences for acetone, methyl acetate, aspirin, pyrimidine, adenine, triazine.
   Their expected 10.5 values come from `X-X-c-o` or `X-n2-ca-n2` in GAFF2.

3. **Degenerate proper torsions:** upstream's enumeration permits the first and
   fourth atom to be identical in a three-membered ring. Such a quartet does not
   define a proper dihedral. Rust requires four distinct atoms; the three omitted
   records occur in aziridine, N-methylaziridine, and epoxide.

The original executable outputs remain unchanged in the fixtures. Exceptions are
asserted explicitly in tests, so future unexpected differences fail validation.

## Implementation boundaries

This module types a supplied molecular graph; it does not implement Amber's full
valence search, hydrogen addition, formal-charge perception, or quantum chemistry.
The graph must include hydrogens and correct bond orders. Ring classification uses
Amber's AR1-AR5 counts and chordless rings of sizes 3-10 from current `ring.c`.
The older Antechamber paper and bundled DEF comments describe a limit of nine.

MOL2 aromatic bonds are resolved by a constrained single/double matching for
ordinary neutral aromatic systems. `AtomGeneric` lacks formal charges; ambiguous
charged systems (for example aromatic-only pyridinium) must use explicit Kekule
bonds. The checked API returns an error rather than guessing. `Amide` bonds count
as single; explicit `Delocalized` bonds retain Amber DL semantics. Unsupported or
malformed graph inputs are rejected. The legacy `find_ff_types` signature returns
one `du` per atom on error; use `try_find_ff_types` for diagnostics.

Missing bonded parameters remain estimates. They use directed EQUA/CORR lists,
actual position-specific columns, defaults, weights, group and conjugated-pair
penalties. A parameter's comment identifies its source and penalty. All Fourier
terms are retained. Exact/equivalent/specific/generic search stages are distinct;
missing required terms produce errors. Empirical bond/angle fitting using
`PARM_BLBA_GAFF2.DAT` is not implemented, and equality with every parmchk2 estimate
is not claimed. The 1.1 improper default is restricted to three-coordinate centers
flagged in PARMCHK, not guessed from neighboring type names.

`update_small_mol_params` still uses this crate's trained partial-charge model,
**not AM1-BCC/RESP**. Its accuracy is outside these topology/parameter tests.
ABCG2 is a charge-typing rule set, not a source of bonded FRCMOD overrides.
Updates validate the adjacency cache and stage changes before modifying atoms;
serial-number normalization is passed to the charge model internally.

## Running the checks

```sh
cargo test --lib param_inference
python src/param_inference/fixtures/regenerate.py \
  --amber-home /path/to/AmberClassic --amber-bin /path/to/AmberClassic/bin
```

The Amber executables may require `APS.DAT` and `PARM_BLBA_GAFF2.DAT` under
`dat/antechamber`; use the pinned revision for reproducible reference outputs.
The script sets both `AMBERHOME` and `AMBERCLASSICHOME` for compatibility.

## Primary references

- [Antechamber paper, sections 2.2-2.3](https://ambermd.org/antechamber/antechamber.pdf)
- [DEF evaluator and paired atom-type adjustment](https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber/atomtype.c)
- [Ring enumeration and AR classification](https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber/ring.c)
- [Bond categories](https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber/bondtype.c)
- [Missing-parameter rules and scores](https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/antechamber/parmchk2.c)
