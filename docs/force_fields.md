# Classical force-field parameterization

The simulation engine consumes resolved classical energy terms. It does not interpret
Amber, CHARMM or OPLS typing rules. Each backend implements three stages:

1. `ForceFieldBackend::assign_atom_types` selects force-field names in atom order and
   retains any residue/topology context needed by the next stages.
2. `TypedTopology::assign_partial_charges` supplies a separate charge vector and method.
3. `TypedTopology::assign_parameters` resolves masses, Lennard-Jones and bonded terms.
   This stage does not call charge inference and can run without assigned charges.

```rust
use dynamics::params::{FfParamSet, ParameterizationOptions};

fn example(molecule: &dynamics::MolDynamics) -> Result<(), dynamics::ParamError> {
let set = FfParamSet::new_amber()?;
let options = ParameterizationOptions::local_only();
let typed = set.assign_atom_types(molecule, &options)?;
let names = typed.atom_types();
let charges = typed.assign_partial_charges()?;
let parameters = typed.assign_parameters(&set.nonbonded_rules)?;

// Convenience: run all stages, validate, and return a parameterized copy.
let prepared = set.parameterize_with_options(molecule, &options)?;
Ok(())
}
```

`FfParamSet::assign` returns the three assignments together, including their sources.
`ParameterizationOptions::charge_policy` independently selects the backend default,
recomputation, or the caller's complete charge vector (`ChargePolicy::Supplied`).
The last option works with every typing backend and does not invoke its charge model.
`parameterize` applies a successful assignment to a copy of `MolDynamics`, retaining
its coordinates, atom order and serial numbers. Failure leaves the input unchanged.
Native simulation initialization and GROMACS export use this same operation. Each
molecule is resolved separately before systems are flattened, so molecule-specific
overrides cannot leak between molecules sharing type names.

Complete supplied `explicit_params` take precedence. Their charges are taken from the
accompanying atoms. Imported topologies therefore bypass automatic naming and charging.
The engine validates parameter lengths, atom indices, numerical values and LJ mixing.

## Backends and embedded data

| Backend | Protein assignments | Ligand assignments | Embedded data |
|---|---|---|---|
| `params::amber::AmberSet` | Residue libraries; ff19SB tables | GAFF2 rules, supplied or GeoStd ML charges, supplied/estimated bonded overrides | `param_data/amber/` |
| `params::charmm::CharmmSet` | Residue topologies and patches; CHARMM36m tables | Matching imported topologies; automatic CGenFF is not supplied | `param_data/charmm/` |
| `params::opls::OplsSet` | OPLS-AA/L residue templates and tables | Bundled known/Foyer assignments, LigParGen cache, optional generation | `param_data/opls/` |

All three load their supported tables with `include_str!`/`include_bytes!`; users do
not need parameter files, AmberTools, GROMACS, CHARMM, Python or BOSS installed for
bundled chemistry. The GeoStd neural network belongs specifically to Amber ligand
charges. CHARMM and OPLS use their own charge data.

Some providers supply names, charges and parameters in one response. OPLS retains that
complete response so requesting the separate stages never repeats a remote job. Its
local/remote fallback chooses a complete matching topology, rather than combining
names from one source with charges or bonded terms from another source.

`ParameterizationOptions::local_only()` prohibits remote generation, but permits
installed local LigParGen/BOSS execution. OPLS tries the cache, bundled chemistry,
and local LigParGen before considering the remote service. Local failures are
reported without silently uploading the molecule. Both local and remote generation
can block, so GUI callers should prepare on workers. For lookup on a UI thread,
`OplsSet::build_small_molecule_cached` only reads bundled/cached data. See the main
README for `DYNAMICS_LIGPARGEN` and `DYNAMICS_LIGPARGEN_PYTHON` configuration.

Water, ions, LJ mixing, 1–4 scales and NBFIX are data in `FfParamSet`, separate from
the backend's assignment algorithm. Protein templates, patches and supported molecule
classes remain backend-specific. `MdConfig::for_family` supplies simulation presets;
custom force fields should select their own cutoffs, switching and solvent choices.

## Adding a force field

For a set with already assigned names and charges, use `TableForceField`:

```rust
use dynamics::{WaterModel, params::{FfParamSet, NonBondedRules, TableForceField}};
use bio_files::md_params::ForceFieldParams;

let set = FfParamSet::custom(
    TableForceField { name: "My classical set".into(), parameters: ForceFieldParams::default() },
    NonBondedRules::default(), // Supply the set's actual mixing and scaling rules.
    WaterModel::OPC,          // Supply a compatible solvent/ion model.
);
```

This backend uses `bio_files::md_params::ForceFieldParamsIndexed::from_types`, a strict
lookup with no Amber aliases or zero-force substitutions. Its table representation
supports harmonic bonds/angles and periodic proper/improper torsions; periodic
impropers follow the representation's central-third-atom convention.

For chemical typing, custom charge methods, residue patches or additional supported
terms, implement `ForceFieldBackend` and return a `TypedTopology`. Its parameter stage
returns `params::ExplicitParams`, which also carries Urey–Bradley, harmonic improper,
CMAP, NBFIX, exclusions and special 1–4 terms. Adding such a backend requires no new
dispatch branch in MD or export. New energy functional forms still require engine support.

## Source organization and migration

* `params`: shared traits, options, conventions, strict table backend and resolved terms.
* `params::amber`: Amber tables and residue preparation; `typing` contains GAFF2 rules and
  missing-parameter estimates, and `charges` contains the GeoStd model and trainer.
* `params::charmm`: CHARMM topology/patch assignment and parameter resolution, plus CMAP support.
* `params::opls`: OPLS templates, Foyer rules, known topologies and LigParGen transport/cache.
* `bio_files`: format codecs, parameter data structures and indexed table lookup.

Amber-specific fields formerly on `FfParamSet` are now accessed through
`set.amber()`/`set.amber_mut()`; the other built-ins have equivalent typed accessors.
Generic callers should use the shared assignment interface. `parameters(molecule_type)`
exposes type-keyed tables where the backend supplies them; template backends return
their resolved terms through parameterization.

The previous `param_inference`, `partial_charge_inference`, `import::ExplicitParams`
and GROMACS `parameterize` paths remain compatibility exports/wrappers. Current callers
use the new family-specific or shared paths. Protein hydrogen/protonation preparation
helpers currently use Amber templates and live in `params::amber`; those structural helpers
are distinct from assigning the selected simulation force field.
