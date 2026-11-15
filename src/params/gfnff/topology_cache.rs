//! On-disk cache of xTB's `gfnff_topo`. Building it scales steeply with atom count; e.g. minutes
//! for a few thousand atoms. The xTB executable reads it from its working directory; libxtb does not.

use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use lin_alg::f32::Vec3;
use na_seq::Element;
use sha2::{Digest, Sha256};

const DIR: &str = "dynamics-gfnff-topology";
/// Oldest entries beyond this are evicted; the newest is kept regardless. Files grow as N².
const MAX_BYTES: u64 = 4 << 30;
/// Covalent radius sum scale for the connectivity in the key.
const BOND_SCALE: f32 = 1.25;

fn entry(key: &str) -> PathBuf {
    env::temp_dir().join(DIR).join(format!("{key}.topo"))
}

/// Hard link where possible: files reach several GB. xTB only reads a topology it accepts.
fn link_or_copy(source: &Path, dest: &Path) -> bool {
    let _ = fs::remove_file(dest);

    fs::hard_link(source, dest).is_ok() || fs::copy(source, dest).is_ok()
}

/// Non-geometric key inputs: charge, parameters, and xTB build. xTB also rejects topologies from
/// other format versions.
pub(super) fn base(executable: &Path, charge: f64, parameters: Option<&[u8]>) -> Vec<u8> {
    let mut result = charge.to_bits().to_le_bytes().to_vec();
    if let Some(parameters) = parameters {
        result.extend(parameters);
    }

    result.extend(executable.to_string_lossy().as_bytes());
    if let Ok(meta) = fs::metadata(executable) {
        result.extend(meta.len().to_le_bytes());
        if let Ok(modified) = meta.modified()
            && let Ok(since) = modified.duration_since(UNIX_EPOCH)
        {
            result.extend(since.as_nanos().to_le_bytes());
        }
    }

    result
}

/// Key from inputs that determine the topology. Covalent connectivity stands in for geometry, so
/// rigid moves, and small displacements, reuse it; as within one xTB run.
pub(super) fn key(base: &[u8], elements: &[Element], positions: &[Vec3]) -> String {
    let radii: Vec<_> = elements
        .iter()
        .map(|e| e.covalent_radius() as f32 * BOND_SCALE)
        .collect();
    let cutoff = 2. * radii.iter().copied().fold(0., f32::max);

    // Sweep along x; only pairs within `cutoff` in x can bond.
    let mut order: Vec<_> = (0..positions.len()).collect();
    order.sort_unstable_by(|&a, &b| positions[a].x.total_cmp(&positions[b].x));

    let mut bonds = Vec::new();
    for (start, &i) in order.iter().enumerate() {
        for &j in &order[start + 1..] {
            if positions[j].x - positions[i].x > cutoff {
                break;
            }

            let r = radii[i] + radii[j];
            if (positions[j] - positions[i]).magnitude_squared() < r * r {
                bonds.push((i.min(j) as u64, i.max(j) as u64));
            }
        }
    }
    bonds.sort_unstable();

    let mut hash = Sha256::new();
    hash.update(b"gfnff-topo-v1");
    hash.update(base);
    hash.update((elements.len() as u64).to_le_bytes());
    for element in elements {
        hash.update([element.atomic_number()]);
    }
    for (i, j) in bonds {
        hash.update(i.to_le_bytes());
        hash.update(j.to_le_bytes());
    }

    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The cached topology's path, if present.
pub(super) fn lookup(key: &str) -> Option<PathBuf> {
    let path = entry(key);
    if !path.is_file() {
        return None;
    }

    // Recency for eviction. Before any copy, so the copy matches; see `store`.
    if let Ok(file) = File::options().write(true).open(&path) {
        let _ = file.set_modified(SystemTime::now());
    }

    Some(path)
}

/// Place a cached topology at `dest`. Returns whether one was found.
pub(super) fn fetch(key: &str, dest: &Path) -> bool {
    lookup(key).is_some_and(|source| link_or_copy(&source, dest))
}

/// Cache a topology xTB built. Failures only cost a rebuild later.
pub(super) fn store(key: &str, source: &Path) {
    let dest = entry(key);
    let Some(dir) = dest.parent() else {
        return;
    };
    if fs::create_dir_all(dir).is_err() {
        return;
    }

    // A link to, or copy of, the entry: xTB read rather than rebuilt it.
    if let (Ok(a), Ok(b)) = (fs::metadata(source), fs::metadata(&dest))
        && a.len() == b.len()
        && a.modified().ok() == b.modified().ok()
    {
        return;
    }

    let staged = dir.join(format!("{key}.{}.tmp", std::process::id()));
    if !link_or_copy(source, &staged) || fs::rename(&staged, &dest).is_err() {
        let _ = fs::remove_file(&staged);

        return;
    }

    evict(dir);
}

fn evict(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    let mut files: Vec<_> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "topo"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), e.path()))
        })
        .collect();
    files.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut total = 0;
    for (i, (_, len, path)) in files.iter().enumerate() {
        total += len;
        if i > 0 && total > MAX_BYTES {
            let _ = fs::remove_file(path);
        }
    }
}
