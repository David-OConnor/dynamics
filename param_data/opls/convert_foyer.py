"""Generate the native OPLS tables from the adjacent, unmodified Foyer XML."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

directory = Path(__file__).resolve().parent
source = directory / 'foyer-oplsaa.xml'
root = ET.parse(source).getroot()
assert root.get('combining_rule') == 'geometric'
nonbonded = root.find('NonbondedForce')
assert nonbonded.get('coulomb14scale') == '0.5'
assert nonbonded.get('lj14scale') == '0.5'
nb = {a.get('type'): a for a in nonbonded}
lines = [f'# Generated from foyer-oplsaa.xml; SHA256 {hashlib.sha256(source.read_bytes()).hexdigest()}']
for atom in root.find('AtomTypes'):
    attrs = atom.attrib
    b = nb[attrs['name']]
    cols = ['T', attrs['name'], attrs['class'], attrs.get('element', ''), attrs['mass'],
            b.get('charge'), b.get('sigma'), b.get('epsilon'),
            attrs.get('overrides', '').replace(' ', ''), attrs.get('def', '')]
    assert all('\t' not in c and '\n' not in c for c in cols)
    lines.append('\t'.join(cols))
for section, tag, arity, values in [
    ('HarmonicBondForce', 'B', 2, ['length', 'k']),
    ('HarmonicAngleForce', 'A', 3, ['angle', 'k']),
    ('RBTorsionForce', 'R', 4, [f'c{i}' for i in range(6)]),
]:
    for term in root.find(section):
        assert term.tag in ('Bond', 'Angle', 'Proper')
        cols = [tag] + [term.get(f'class{i}', '') for i in range(1, arity + 1)]
        cols += [term.get(value) for value in values]
        lines.append('\t'.join(cols))
(directory / 'foyer-ligands.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
