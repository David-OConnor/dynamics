"""Regenerate the CGenFF ligand fixture in `cgenff_pacp/`: paracetamol (CGenFF's model compound
PACP) as a GROMACS topology, laid out as CHARMM-GUI writes them: `forcefield.itp` with parameters
by type, and `pacp.itp` with interactions but no parameters. We check the topology against OpenMM's
CHARMM implementation, reading the original CHARMM files, before writing it.

The tests in `tests/charmm.rs` import it, compare to GROMACS, and simulate it with a protein built
from CHARMM36m.

Usage: python regenerate_cgenff_ligand.py path/to/toppar
(toppar from toppar_c36_jul24.) Needs RDKit, NetworkX, and OpenMM. Tested with OpenMM 8.3.1.
"""

import math
import os
import sys
import tempfile

import networkx as nx
import openmm as mm
from openmm import app, unit
from rdkit import Chem
from rdkit.Chem import AllChem

TOPPAR = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cgenff_pacp")
RES = "PACP"
SMILES = "CC(=O)Nc1ccc(O)cc1"
BOX_NM = 4.0

KCAL_TO_KJ = 4.184
# 2^(-1/6): Rmin to σ
RMIN_TO_SIGMA = 2 ** (-1 / 6)


def tokens(line):
    return line.split("!")[0].split()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_rtf_residue(path, name):
    atoms, bonds, impropers, masses = [], [], [], {}
    current = None
    for line in open(path, encoding="latin-1"):
        t = tokens(line)
        if not t:
            continue
        k = t[0].upper()[:4]
        if k == "MASS":
            # Parameter sections repeat MASS lines, sometimes without the element.
            if len(t) > 4 or t[2].upper() not in masses:
                masses[t[2].upper()] = (float(t[3]), t[4].upper() if len(t) > 4 else "")
        elif k in ("RESI", "PRES"):
            current = t[1].upper()
        elif current == name:
            if k == "ATOM":
                atoms.append((t[1].upper(), t[2].upper(), float(t[3])))
            elif k in ("BOND", "DOUB", "TRIP", "AROM"):
                bonds += [(t[i].upper(), t[i + 1].upper()) for i in range(1, len(t) - 1, 2)]
            elif k in ("IMPR", "IMPH"):
                impropers += [tuple(x.upper() for x in t[i : i + 4]) for i in range(1, len(t) - 3, 4)]
    return atoms, bonds, impropers, masses


def read_prm(paths):
    p = {"bonds": {}, "angles": {}, "dihedrals": {}, "impropers": {}, "nonbonded": {}, "nbfix": {}, "cmap": []}
    for path in paths:
        section, continued, cmap = None, False, None
        replaced = set()
        for line in open(path, encoding="latin-1"):
            raw = line.split("!")[0].rstrip()
            t = raw.split()
            if continued:
                continued = raw.endswith("-")
                continue
            if not t:
                continue
            k = t[0].upper()
            if not is_number(k) and k[:4] in (
                "BOND", "ANGL", "THET", "DIHE", "PHI", "IMPR", "IMPH", "CMAP", "NONB", "NBON", "NBFI",
                "HBON", "END", "ATOM",
            ) and (len(t) == 1 or k[:4] in ("NONB", "NBON", "HBON")):
                section = k[:4]
                continued = raw.endswith("-")
                continue
            if section == "BOND" and len(t) >= 4:
                p["bonds"][(t[0], t[1])] = (float(t[2]), float(t[3]))
            elif section in ("ANGL", "THET") and len(t) >= 5:
                ub = (float(t[5]), float(t[6])) if len(t) >= 7 else None
                p["angles"][(t[0], t[1], t[2])] = (float(t[3]), float(t[4]), ub)
            elif section in ("DIHE", "PHI") and len(t) >= 7:
                key = tuple(t[:4])
                # A key's terms in a later file replace the earlier file's.
                if key not in replaced:
                    p["dihedrals"][key] = []
                    replaced.add(key)
                p["dihedrals"][key] = [d for d in p["dihedrals"][key] if d[1] != int(t[5])]
                p["dihedrals"][key].append((float(t[4]), int(t[5]), float(t[6])))
            elif section in ("IMPR", "IMPH") and len(t) >= 7:
                p["impropers"][tuple(t[:4])] = (float(t[4]), float(t[6]))
            elif section == "CMAP":
                if len(t) == 9 and not is_number(t[0]):
                    cmap = {"types": t[:8], "size": int(t[8]), "values": []}
                    p["cmap"].append(cmap)
                elif cmap is not None:
                    cmap["values"] += [float(v) for v in t]
            elif section in ("NONB", "NBON") and len(t) >= 4 and not is_number(t[0]):
                special = (abs(float(t[5])), float(t[6])) if len(t) >= 7 else None
                p["nonbonded"][t[0]] = (abs(float(t[2])), float(t[3]), special)
            elif section == "NBFI" and len(t) >= 4:
                p["nbfix"][(t[0], t[1])] = (abs(float(t[2])), float(t[3]))
    return p


def lookup(table, key):
    for k in (key, key[::-1]):
        if k in table:
            return k, table[k]
    return None, None


def dihedral_key(prm, t):
    for key in (t, (t[0], "X", "X", t[3]), ("X", t[1], t[2], "X")):
        k, v = lookup(prm["dihedrals"], key)
        if k:
            return k, v
    raise KeyError(f"dihedral {t}")


def improper_key(prm, t):
    a, b, c, d = t
    for key in ((a, b, c, d), (a, "X", "X", d), ("X", b, c, d), ("X", b, c, "X"), ("X", "X", c, d)):
        k, v = lookup(prm["impropers"], key)
        if k:
            return k, v
    raise KeyError(f"improper {t}")


def sigma_nm(rmin_half):
    return 2 * rmin_half * RMIN_TO_SIGMA / 10


atoms, bond_names, impropers_named, masses = read_rtf_residue(os.path.join(TOPPAR, "top_all36_cgenff.rtf"), RES)
prm = read_prm(
    [os.path.join(TOPPAR, f) for f in ("par_all36m_prot.prm", "par_all36_cgenff.prm", "toppar_water_ions.str")]
)
for f in ("top_all36_prot.rtf", "toppar_water_ions.str"):
    masses.update(read_rtf_residue(os.path.join(TOPPAR, f), None)[3])

names = [a[0] for a in atoms]
types = [a[1] for a in atoms]
idx = {n: i for i, n in enumerate(names)}
n = len(atoms)
bonds = sorted({tuple(sorted((idx[a], idx[b]))) for a, b in bond_names})

graph = nx.Graph()
graph.add_nodes_from(range(n))
graph.add_edges_from(bonds)

angles = []
for j in range(n):
    nbrs = sorted(graph[j])
    for x in range(len(nbrs)):
        for y in range(x + 1, len(nbrs)):
            angles.append((nbrs[x], j, nbrs[y]))
dihedrals = []
for j, k in bonds:
    for i in graph[j]:
        for l in graph[k]:
            if i not in (j, k) and l not in (j, k) and i != l:
                dihedrals.append((i, j, k, l))
impropers = [tuple(idx[x] for x in imp) for imp in impropers_named]

excluded = set(bonds) | {tuple(sorted((a[0], a[2]))) for a in angles}
pairs_14 = sorted({tuple(sorted((d[0], d[3]))) for d in dihedrals} - excluded)

# Coordinates: RDKit, mapped onto CHARMM's atoms by graph isomorphism.
mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
AllChem.EmbedMolecule(mol, randomSeed=7)
AllChem.MMFFOptimizeMolecule(mol)
rd_graph = nx.Graph()
for a in mol.GetAtoms():
    rd_graph.add_node(a.GetIdx(), element=a.GetSymbol().upper())
rd_graph.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
for i in range(n):
    graph.nodes[i]["element"] = masses[types[i]][1]
matcher = nx.algorithms.isomorphism.GraphMatcher(graph, rd_graph, node_match=lambda x, y: x["element"] == y["element"])
mapping = next(matcher.isomorphisms_iter())
conf = mol.GetConformer()
center = [BOX_NM * 5] * 3  # Å
com = [sum(conf.GetAtomPosition(i)[d] for i in range(n)) / n for d in range(3)]
posits = []
for i in range(n):
    p = conf.GetAtomPosition(mapping[i])
    posits.append([p[d] - com[d] + center[d] for d in range(3)])  # Å

# ---------------- GROMACS files, laid out as CHARMM-GUI writes them.
used_types = sorted(set(types))
ff = ["; CHARMM36m and CGenFF parameters by type, as CHARMM-GUI writes them. Units: nm, kJ/mol.\n"]
ff.append("[ defaults ]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n1 2 yes 1.0 1.0\n\n")
ff.append("[ atomtypes ]\n; name at.num mass charge ptype sigma epsilon\n")
atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "NA": 11, "CL": 17}
# Like CHARMM-GUI, we also define the types in the NBFIX and CMAP entries below, which the ligand
# doesn't use.
cmap = prm["cmap"][0]
other_types = sorted({"SOD", "CLA", "OC", "NC2"} | set(cmap["types"]))
for t in used_types + other_types:
    mass, element = masses[t]
    if not element:  # The protein file's MASS lines don't list elements.
        element = min(((1.008, "H"), (12.011, "C"), (14.007, "N"), (15.999, "O")), key=lambda m: abs(m[0] - mass))[1]
    eps, rmin_half, _ = prm["nonbonded"][t]
    ff.append(f"{t} {atomic_numbers[element]} {mass:.5f} 0.000 A {sigma_nm(rmin_half):.12f} {eps * KCAL_TO_KJ:.6f}\n")

ff.append("\n[ pairtypes ]\n; i j func sigma1-4 epsilon1-4\n")
written = set()
for i, j in pairs_14:
    ti, tj = sorted((types[i], types[j]))
    si, sj = prm["nonbonded"][ti], prm["nonbonded"][tj]
    if (ti, tj) in written or (si[2] is None and sj[2] is None):
        continue
    written.add((ti, tj))
    ei, ri = si[2] or si[:2]
    ej, rj = sj[2] or sj[:2]
    sigma = (sigma_nm(ri) + sigma_nm(rj)) / 2
    ff.append(f"{ti} {tj} 1 {sigma:.12f} {math.sqrt(ei * ej) * KCAL_TO_KJ:.6f}\n")

ff.append("\n[ bondtypes ]\n; i j func b0 Kb\n")
written = set()
for i, j in bonds:
    k, (kb, b0) = lookup(prm["bonds"], (types[i], types[j]))
    if k not in written:
        written.add(k)
        ff.append(f"{k[0]} {k[1]} 1 {b0 / 10:.8f} {kb * 2 * KCAL_TO_KJ * 100:.2f}\n")

ff.append("\n[ angletypes ]\n; i j k func theta0 ktheta ub0 kub\n")
written = set()
for i, j, k_ in angles:
    k, (kt, t0, ub) = lookup(prm["angles"], (types[i], types[j], types[k_]))
    if k not in written:
        written.add(k)
        kub, s0 = ub or (0.0, 0.0)
        ff.append(
            f"{k[0]} {k[1]} {k[2]} 5 {t0:.6f} {kt * 2 * KCAL_TO_KJ:.6f} {s0 / 10:.8f} {kub * 2 * KCAL_TO_KJ * 100:.2f}\n"
        )

ff.append("\n[ dihedraltypes ]\n; i j k l func phi0 kphi mult\n")
written = set()
for d in dihedrals:
    k, terms = dihedral_key(prm, tuple(types[i] for i in d))
    if k not in written:
        written.add(k)
        for kd, mult, delta in terms:
            ff.append(f"{k[0]} {k[1]} {k[2]} {k[3]} 9 {delta:.6f} {kd * KCAL_TO_KJ:.6f} {mult}\n")

ff.append("\n[ dihedraltypes ]\n; i j k l func q0 cq\n")
written = set()
for imp in impropers:
    k, (ki, psi0) = improper_key(prm, tuple(types[i] for i in imp))
    if k not in written:
        written.add(k)
        ff.append(f"{k[0]} {k[1]} {k[2]} {k[3]} 2 {psi0:.6f} {ki * 2 * KCAL_TO_KJ:.6f}\n")

# CHARMM-GUI includes pair-specific LJ, and CMAP for proteins, whether or not a molecule uses them.
ff.append("\n[ nonbond_params ]\n; i j func sigma epsilon\n")
for (t0, t1), (eps, rmin) in prm["nbfix"].items():
    if {t0, t1} <= {"SOD", "CLA", "OC", "NC2"}:
        ff.append(f"{t0} {t1} 1 {rmin * RMIN_TO_SIGMA / 10:.12f} {eps * KCAL_TO_KJ:.6f}\n")
ff.append("\n[ cmaptypes ]\n")
values = [f"{v * KCAL_TO_KJ:.6f}" for v in cmap["values"]]
# GROMACS lists CMAP's five distinct atoms: φ's four, and ψ's last.
ff.append(" ".join(cmap["types"][:4] + cmap["types"][7:]) + f" 1 {cmap['size']} {cmap['size']}\\\n")
for r in range(0, len(values), 10):
    end = "\\\n" if r + 10 < len(values) else "\n"
    ff.append(" ".join(values[r : r + 10]) + end)

itp = [f"; {RES}: paracetamol (p-acetamide-phenol), CGenFF\n"]
itp.append(f"[ moleculetype ]\n; name nrexcl\n{RES} 3\n\n[ atoms ]\n; nr type resnr residue atom cgnr charge mass\n")
for i, (name, t, q) in enumerate(atoms):
    itp.append(f"{i + 1} {t} 1 {RES} {name} {i + 1} {q:.4f} {masses[t][0]:.5f}\n")
itp.append("\n[ bonds ]\n; ai aj funct\n")
itp += [f"{i + 1} {j + 1} 1\n" for i, j in bonds]
itp.append("\n[ pairs ]\n; ai aj funct\n")
itp += [f"{i + 1} {j + 1} 1\n" for i, j in pairs_14]
itp.append("\n[ angles ]\n; ai aj ak funct\n")
itp += [f"{i + 1} {j + 1} {k + 1} 5\n" for i, j, k in angles]
itp.append("\n[ dihedrals ]\n; ai aj ak al funct\n")
itp += [f"{a + 1} {b + 1} {c + 1} {d + 1} 9\n" for a, b, c, d in dihedrals]
itp.append("\n[ dihedrals ]\n; ai aj ak al funct\n")
itp += [f"{a + 1} {b + 1} {c + 1} {d + 1} 2\n" for a, b, c, d in impropers]

# Ions, as in a CHARMM-GUI solution system. They're close enough to each other for their
# pair-specific LJ (NBFIX) to matter, and away from the ligand's plane.
ions = [("SOD", 1.0), ("CLA", -1.0)]
ions_itp = []
for name, q in ions:
    ions_itp.append(f"[ moleculetype ]\n; name nrexcl\n{name} 1\n\n[ atoms ]\n")
    ions_itp.append(f"1 {name} 1 {name} {name} 1 {q:.3f} {masses[name][0]:.5f}\n\n")
ring = [posits[idx[a]] for a in ("C22", "C24", "C26")]
sub = lambda u, v: [u[d] - v[d] for d in range(3)]
cross = lambda u, v: [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
unit_vec = lambda u: [x / math.sqrt(sum(y * y for y in u)) for x in u]
normal = unit_vec(cross(sub(ring[1], ring[0]), sub(ring[2], ring[0])))
in_plane = unit_vec(sub(ring[1], ring[0]))
c = [sum(p[d] for p in posits) / n for d in range(3)]
ion_posits = [
    [c[d] + 5.0 * normal[d] for d in range(3)],
    [c[d] + 5.0 * normal[d] + 4.5 * in_plane[d] for d in range(3)],
]
for p_ion in ion_posits:
    assert min(math.dist(p_ion, p) for p in posits) > 3.5

top = [
    '#include "forcefield.itp"\n',
    '#include "pacp.itp"\n',
    '#include "ions.itp"\n',
    "\n[ system ]\nParacetamol, with Na+ and Cl-\n\n[ molecules ]\n",
    f"{RES} 1\n",
] + [f"{name} 1\n" for name, _ in ions]

gro = [f"{RES}, CGenFF\n", f"{n + len(ions):5d}\n"]
for i, (name, _, _) in enumerate(atoms):
    x, y, z = (v / 10 for v in posits[i])
    gro.append(f"{1:5d}{RES:<5}{name:>5}{i + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
for k, ((name, _), p_ion) in enumerate(zip(ions, ion_posits)):
    x, y, z = (v / 10 for v in p_ion)
    gro.append(f"{k + 2:5d}{name:<5}{name:>5}{n + k + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
gro.append(f"{BOX_NM:10.5f}{BOX_NM:10.5f}{BOX_NM:10.5f}\n")

os.makedirs(OUT, exist_ok=True)
for fname, lines in (
    ("forcefield.itp", ff),
    ("pacp.itp", itp),
    ("ions.itp", ions_itp),
    ("topol.top", top),
    ("conf.gro", gro),
):
    with open(os.path.join(OUT, fname), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)

# ---------------- Check: OpenMM's CHARMM implementation, from the CHARMM files, vs OpenMM reading our
# GROMACS files. Both in vacuum, without cutoffs. Positions are as written in conf.gro.
gro_file = app.GromacsGroFile(os.path.join(OUT, "conf.gro"))
positions = gro_file.getPositions()

n_all = n + len(ions)
psf = ["PSF EXT XPLOR\n\n", "         1 !NTITLE\n", f"* {RES}\n\n", f"{n_all:10d} !NATOM\n"]
for i, (name, t, q) in enumerate(atoms):
    psf.append(f"{i + 1:10d} LIG      1        {RES:<8} {name:<8} {t:<6} {q:14.6f}{masses[t][0]:14.4f}           0\n")
for k, (name, q) in enumerate(ions):
    psf.append(f"{n + k + 1:10d} ION      {k + 1:<8} {name:<8} {name:<8} {name:<6} {q:14.6f}{masses[name][0]:14.4f}           0\n")


def section(items, per_line, title):
    out = [f"\n{len(items):10d} !{title}\n"]
    flat = [str(x + 1).rjust(10) for item in items for x in item]
    width = per_line * len(items[0]) if items else 1
    for r in range(0, len(flat), width):
        out.append("".join(flat[r : r + width]) + "\n")
    return out


psf += section(bonds, 4, "NBOND: bonds")
psf += section(angles, 3, "NTHETA: angles")
psf += section(dihedrals, 2, "NPHI: dihedrals")
psf += section(impropers, 2, "NIMPHI: impropers")
psf += ["\n         0 !NDON: donors\n\n", "\n         0 !NACC: acceptors\n\n", "\n         0 !NNB\n\n"]
psf.append("".join("0".rjust(10) for _ in range(n_all)) + "\n")
psf += ["\n         1         0 !NGRP NST2\n", "         0         0         0\n", "\n"]

with tempfile.TemporaryDirectory() as tmp:
    psf_path = os.path.join(tmp, "pacp.psf")
    with open(psf_path, "w") as f:
        f.writelines(psf)
    charmm_params = app.CharmmParameterSet(
        *(
            os.path.join(TOPPAR, f)
            for f in (
                "top_all36_prot.rtf",
                "par_all36m_prot.prm",
                "top_all36_cgenff.rtf",
                "par_all36_cgenff.prm",
                "toppar_water_ions.str",
            )
        )
    )
    system_charmm = app.CharmmPsfFile(psf_path).createSystem(charmm_params, nonbondedMethod=app.NoCutoff)

# OpenMM's GROMACS reader mishandles this system when it has [ nonbond_params ] (its NBFIX code
# path), so we read it without them, and correct the energy of the one NBFIX pair, SOD-CLA, below.
with tempfile.TemporaryDirectory() as tmp:
    for fname in ("pacp.itp", "ions.itp", "topol.top"):
        with open(os.path.join(tmp, fname), "w", encoding="utf-8") as f:
            f.writelines({"pacp.itp": itp, "ions.itp": ions_itp, "topol.top": top}[fname])
    no_nbfix = "".join(ff).split("[ nonbond_params ]")[0] + "[ cmaptypes ]" + "".join(ff).split("[ cmaptypes ]")[1]
    with open(os.path.join(tmp, "forcefield.itp"), "w", encoding="utf-8") as f:
        f.write(no_nbfix)
    system_gmx = app.GromacsTopFile(os.path.join(tmp, "topol.top"), includeDir=tmp).createSystem(
        nonbondedMethod=app.NoCutoff
    )


def lj(sigma, eps, r):
    sr6 = (sigma / r) ** 6
    return 4 * eps * (sr6 * sr6 - sr6)


(s_na, e_na), (s_cl, e_cl) = ((sigma_nm(prm["nonbonded"][t][1]) * 10, prm["nonbonded"][t][0]) for t in ("SOD", "CLA"))
_, (e_fix, rmin_fix) = lookup(prm["nbfix"], ("SOD", "CLA"))
r_ions = math.dist(*(positions[i].value_in_unit(unit.angstrom) for i in (n, n + 1)))
nbfix_correction = lj(rmin_fix * RMIN_TO_SIGMA, e_fix, r_ions) - lj((s_na + s_cl) / 2, math.sqrt(e_na * e_cl), r_ions)
print(f"SOD-CLA NBFIX changes the energy by {nbfix_correction:.5f} kcal/mol, at {r_ions:.2f} Å")


def energy(system):
    for f in system.getForces():
        f.setForceGroup(0)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    return e, f


e_charmm, f_charmm = energy(system_charmm)
e_gmx, f_gmx = energy(system_gmx)
e_gmx += nbfix_correction
# The ligand's forces; the ions' differ by the NBFIX force.
max_df = abs(f_charmm[:n] - f_gmx[:n]).max()
print(f"OpenMM from CHARMM files: {e_charmm:.5f} kcal/mol; from our GROMACS files: {e_gmx:.5f}; max force diff {max_df:.2e}")
assert abs(e_charmm - e_gmx) < 1e-3 and max_df < 1e-3, "The GROMACS topology doesn't match CHARMM"
print(f"Wrote {OUT}: {len(bonds)} bonds, {len(angles)} angles, {len(dihedrals)} dihedrals, {len(impropers)} impropers, {len(pairs_14)} 1-4 pairs")
