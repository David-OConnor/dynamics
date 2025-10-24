//! Misc utility functions.

use std::{
    fmt::{Display, Formatter},
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
};

use bio_files::{AtomGeneric, BondGeneric};

use crate::{COMPUTATION_TIME_RATIO, ParamError, snapshot::Snapshot};

/// Build a list of indices that relate atoms that are connected by covalent bonds.
/// For each outer atom index, the inner values are indices of the atom it's bonded to.
///
/// Note: If you store bonds with atom indices directly, you may wish to build this in a faster
/// way and cache it, vice this serial-number lookup.
pub(crate) fn build_adjacency_list(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
) -> Result<Vec<Vec<usize>>, ParamError> {
    let mut result = vec![Vec::new(); atoms.len()];

    // For each bond, record its atoms as neighbors of each other
    for bond in bonds {
        let mut atom_0 = None;
        let mut atom_1 = None;

        let mut found = false;
        for (i, atom) in atoms.iter().enumerate() {
            if atom.serial_number == bond.atom_0_sn {
                atom_0 = Some(i);
            }
            if atom.serial_number == bond.atom_1_sn {
                atom_1 = Some(i);
            }

            if let (Some(a0), Some(a1)) = (atom_0, atom_1) {
                result[a0].push(a1);
                result[a1].push(a0);

                found = true;
                break;
            }
        }

        if !found {
            return Err(ParamError::new(
                "Invalid bond to atom mapping when building adjacency list.",
            ));
        }
    }

    Ok(result)
}

pub fn save_snapshots(snapshots: &[Snapshot], path: &Path) -> io::Result<()> {
    let mut file = File::create(path)?;

    let mut result = Vec::new();
    let mut i = 0;

    // todo: Add a header if/when required.

    for snap in snapshots {
        let snap_ser = snap.to_bytes();
        result[i..i + snap_ser.len()].copy_from_slice(&snap_ser);

        i += snap_ser.len();
    }

    file.write_all(&result)?;

    Ok(())
}

pub fn load_snapshots(path: &Path) -> io::Result<Vec<Snapshot>> {
    let mut f = File::open(path)?;
    let mut out = Vec::new();

    loop {
        let mut len_buf = [0u8; 4];
        match f.read_exact(&mut len_buf) {
            Ok(()) => {
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut buf = vec![0u8; len];
                f.read_exact(&mut buf)?;
                let s = Snapshot::from_bytes(&buf)?;
                out.push(s);
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }

    Ok(out)
}

/// The output of `ComputationTime`, averaged over step count. In μs.
#[derive(Debug)]
pub struct ComputationTime {
    pub step_count: usize,
    pub bonded: u32,
    pub non_bonded_short_range: u32,
    pub ewald_long_range: u32,
    pub neighbor_all: u32,
    pub neighbor_rebuild: u32,
    pub neighbor_rebuild_ratio: f32,
    pub integration: u32,
    pub ambient: u32,
    pub snapshots: u32,
    /// Others substracted from `total`. Assumes no overlap. Uses `neighbor_all`, since `neighbor_rebuild`
    /// is part of it.
    pub other: i32,
    pub total: u32,
}

impl Display for ComputationTime {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(
            f,
            "Computation time from {} steps, averaged per step, sampled every {COMPUTATION_TIME_RATIO} steps:",
            self.step_count
        )?;

        writeln!(f, "--Bonded: {} μs", self.bonded)?;
        writeln!(
            f,
            "--Non-bonded (short-range): {} μs",
            self.non_bonded_short_range
        )?;
        writeln!(f, "--Non-bonded (long-range): {} μs", self.ewald_long_range)?;

        writeln!(f, "--Neighbor rebuild: {} μs", self.neighbor_rebuild)?;
        writeln!(
            f,
            "--Neighbor rebuild ratio: {:.2}",
            self.neighbor_rebuild_ratio
        )?;

        writeln!(f, "--Integration: {} μs", self.integration)?;
        writeln!(f, "--Baro/Thermo: {} μs", self.ambient)?;
        writeln!(f, "--Other: {} μs", self.other + self.snapshots as i32)?;
        writeln!(f, "--Total: {} μs", self.total)?;

        Ok(())
    }
}

/// We use this to monitor performance, by component. We track
/// Times are in μs. todo: Add integration time, H Shaking, Water settle, SPME rebuild etc A/R.
#[derive(Debug, Default)]
pub(crate) struct ComputationTimeSums {
    pub bonded_sum: u64,
    pub non_bonded_short_range_sum: u64,
    pub ewald_long_range_sum: u64,
    /// The ratio doesn't apply to neighbors; log each time we do this.
    /// `neighbor_all` includes code that determines when we need to rebuild.
    /// note: The neighbor rebuild makes up the large majority of the time taken of neighbor_all.
    /// todo: Consider removing one or the other.
    pub neighbor_all_sum: u64,
    /// Just the actual rebuild time; not each time.
    pub neighbor_rebuild_sum: u64,
    pub neighbor_rebuild_count: u16,
    pub integration_sum: u64,
    /// Thermostat, barostat, sim box rebuilds.
    /// todo: Split this up into these components if it's significantly large.
    pub ambient_sum: u64,
    pub snapshot_sum: u64,
    /// If the other values don't add up to nearly this, parts we haven't counted
    /// make up a significant amount of the computation time; we may need to include them.
    pub total: u64,
}

impl ComputationTimeSums {
    pub fn time_per_step(&self, num_steps: usize) -> io::Result<ComputationTime> {
        if num_steps == 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "No steps to perform the computation on",
            ));
        }

        let c = COMPUTATION_TIME_RATIO as f32 / num_steps as f32;

        let apply = |v| (v as f32 * c) as u32;

        let bonded = apply(self.bonded_sum);
        let non_bonded_short_range = apply(self.non_bonded_short_range_sum);
        let ewald_long_range = apply(self.ewald_long_range_sum);
        let neighbor_all = (self.neighbor_all_sum / num_steps as u64) as u32;
        let neighbor_rebuild = (self.neighbor_rebuild_sum / num_steps as u64) as u32;
        let integration = apply(self.integration_sum);
        let ambient = apply(self.ambient_sum);
        let snapshots = apply(self.snapshot_sum);
        let total = apply(self.total);

        let other = total as i32
            - (bonded
                + non_bonded_short_range
                + ewald_long_range
                + neighbor_all
                + integration
                + ambient
                + snapshots) as i32;

        Ok(ComputationTime {
            step_count: num_steps,
            bonded,
            non_bonded_short_range,
            ewald_long_range,
            neighbor_all,
            neighbor_rebuild,
            neighbor_rebuild_ratio: self.neighbor_rebuild_count as f32 / num_steps as f32,
            integration,
            ambient,
            snapshots,
            other,
            total,
        })
    }
}
