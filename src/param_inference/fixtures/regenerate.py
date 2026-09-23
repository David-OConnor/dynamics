"""Regenerate the checked-in Amber oracle fixtures using an existing Amber build.

No RDKit is needed: the TSV contains the explicit-hydrogen molecular graphs.
Run this under Linux/WSL (or another environment where the Amber binaries run):
  python regenerate.py --amber-home /path/to/AmberClassic --amber-bin /path/to/bin
The reference revision and intentional discrepancies are documented in README.md.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amber-home", type=Path, required=True)
    parser.add_argument("--amber-bin", type=Path, required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    root = fixtures.parents[2]
    env = dict(os.environ, AMBERHOME=str(args.amber_home.resolve()),
               AMBERCLASSICHOME=str(args.amber_home.resolve()))
    data = (fixtures / "amber_atomtypes.tsv").read_text().splitlines()
    output, frcmods = [], []
    with tempfile.TemporaryDirectory(prefix="dynamics-amber-oracle-") as temp:
        temp = Path(temp)

        def run(program, *arguments):
            result = subprocess.run([str(args.amber_bin.resolve() / program), *map(str, arguments)],
                                    cwd=temp, env=env, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{program} failed:\n{result.stdout}\n{result.stderr}")

        for line in data:
            if not line or line.startswith("#"):
                output.append(line)
                continue
            name, smiles, elements, bonds, _ = line.split("\t")
            atoms = elements.split(",")
            edges = [list(map(int, edge.split(","))) for edge in bonds.split(";") if edge]
            mol2 = ["@<TRIPOS>MOLECULE", name, f"{len(atoms)} {len(edges)} 1 0 0",
                    "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
            # Coordinates are irrelevant to these topology-only reference programs.
            mol2 += [f"{i+1} {element}{i+1} {i*1.5:.3f} 0 0 {element} 1 MOL 0"
                     for i, element in enumerate(atoms)]
            mol2 += ["@<TRIPOS>BOND"]
            mol2 += [f"{i+1} {a+1} {b+1} {order}" for i, (a, b, order) in enumerate(edges)]
            (temp / "input.mol2").write_text("\n".join(mol2) + "\n", newline="\n")
            run("bondtype", "-i", "input.mol2", "-o", "bonds.ac", "-f", "mol2", "-j", "part", "-an", "no")
            run("atomtype", "-i", "bonds.ac", "-o", "typed.ac", "-d",
                root / "param_data/antechamber_defs/ATOMTYPE_GFF2.DEF", "-a", "1", "-an", "no")
            types = [line.split()[-1] for line in (temp / "typed.ac").read_text().splitlines()
                     if line.startswith("ATOM")]
            if len(types) != len(atoms):
                raise RuntimeError(f"Incorrect atom count for {name}")
            output.append("\t".join([name, smiles, elements, bonds, ",".join(types)]))
            run("parmchk2", "-i", "typed.ac", "-f", "ac", "-o", "output.frcmod", "-s", "2",
                "-p", root / "param_data/gaff2.dat", "-c", root / "param_data/antechamber_defs/PARMCHK.DAT")
            frcmods.append(f"@@{name}\n" + (temp / "output.frcmod").read_text() + "\n")
    (fixtures / "amber_atomtypes.tsv").write_text("\n".join(output) + "\n", newline="\n")
    (fixtures / "amber_frcmod.txt").write_text("".join(frcmods), newline="\n")


if __name__ == "__main__":
    main()
