"""Regenerates the OpenMM reference data for the prmtop import test.

Reads `ala_dipeptide.prmtop` (alanine dipeptide, ACE-ALA-NME), builds coordinates by minimizing
from an extended chain, and writes:
- `ala_dipeptide.inpcrd`: The coordinates, centered in a 40 Å cubic box.
- `ala_dipeptide_openmm.txt`: Potential energy (kcal/mol), then per-atom forces (kcal/mol/Å). A
  final comment line breaks the energy down by force.

Non-bonded settings match the `gromacs_compare` tests: PME with α = 0.35 Å⁻¹ and a 50³ grid
(0.8 Å spacing), 10 Å cutoffs, and no LJ switching or dispersion correction.

Run from this directory: `python regenerate_openmm_reference.py`. Requires OpenMM.
"""

import numpy as np
import openmm as mm
from openmm import app, unit

BOX_A = 40.0

prmtop = app.AmberPrmtopFile("ala_dipeptide.prmtop")
n = prmtop.topology.getNumAtoms()

# Minimize in vacuum from a slightly perturbed extended chain.
rng = np.random.default_rng(1)
start = np.array([[1.5 * i, 0.3 * (i % 2), 0.0] for i in range(n)]) + rng.normal(0, 0.2, (n, 3))
vac = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
ctx = mm.Context(vac, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
ctx.setPositions(start * unit.angstrom)
mm.LocalEnergyMinimizer.minimize(ctx, 1e-3, 5000)
pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
pos = pos - pos.mean(axis=0) + BOX_A / 2
# Perturb, so that every term contributes substantial forces.
pos = pos + rng.normal(0, 0.08, pos.shape)

# Reference energy and forces with PME, at the test's settings.
box = mm.Vec3(BOX_A, 0, 0), mm.Vec3(0, BOX_A, 0), mm.Vec3(0, 0, BOX_A)
prmtop.topology.setPeriodicBoxVectors([v * 0.1 for v in box] * unit.nanometer)
system = prmtop.createSystem(
    nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=None
)
for i, force in enumerate(system.getForces()):
    force.setForceGroup(i)
    if isinstance(force, mm.NonbondedForce):
        force.setUseDispersionCorrection(False)
        force.setUseSwitchingFunction(False)
        force.setPMEParameters(3.5, 50, 50, 50)
system.setDefaultPeriodicBoxVectors(*[v * 0.1 for v in box])

ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
ctx.setPositions(pos * unit.angstrom)
state = ctx.getState(getEnergy=True, getForces=True)
energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)

with open("ala_dipeptide.inpcrd", "w") as f:
    f.write("ACE-ALA-NME, minimized with OpenMM, then perturbed\n")
    f.write(f"{n:6d}\n")
    flat = pos.reshape(-1)
    for i in range(0, len(flat), 6):
        f.write("".join(f"{v:12.7f}" for v in flat[i : i + 6]) + "\n")
    f.write("".join(f"{v:12.7f}" for v in [BOX_A, BOX_A, BOX_A, 90.0, 90.0, 90.0]) + "\n")

components = []
for i, force in enumerate(system.getForces()):
    e = ctx.getState(getEnergy=True, groups={i}).getPotentialEnergy()
    components.append(f"{type(force).__name__}={e.value_in_unit(unit.kilocalorie_per_mole):.6f}")

with open("ala_dipeptide_openmm.txt", "w") as f:
    f.write(f"{energy:.8f}\n")
    for fx, fy, fz in forces:
        f.write(f"{fx:.8f} {fy:.8f} {fz:.8f}\n")
    f.write("# " + " ".join(components) + "\n")

print(f"Wrote {n} atoms. Potential energy: {energy:.4f} kcal/mol")
