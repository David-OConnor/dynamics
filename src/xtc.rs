//! WIP, and wraps libs that warp an undocumented Gromacs XTC lib: libXDR

//! - XTC coordinates are conventionally in nanometers.
//! - Frame time is usually in picoseconds. :contentReference[oaicite:3]{index=3}

use std::io;
use std::path::Path;

use lin_alg::f32::Vec3;
use xdrfile::{Frame, Trajectory, XTCTrajectory};

use crate::snapshot::Snapshot;

pub const NM_PER_ANGSTROM: f32 = 0.1;
pub const ANGSTROM_PER_NM: f32 = 10.0;

fn xdr_to_io(err: xdrfile::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}

fn snapshot_num_atoms(s: &Snapshot, write_water: bool) -> usize {
    let mut n = s.atom_posits.len();
    if write_water {
        n += s.water_o_posits.len() + s.water_h0_posits.len() + s.water_h1_posits.len();
    }
    n
}

fn fill_frame_coords(frame: &mut Frame, s: &Snapshot, write_water: bool, coord_scale: f32) {
    let mut i = 0usize;

    let mut push = |v: &Vec<Vec3>| {
        for p in v {
            frame.coords[i] = [p.x * coord_scale, p.y * coord_scale, p.z * coord_scale];
            i += 1;
        }
    };

    push(&s.atom_posits);

    if write_water {
        push(&s.water_o_posits);
        push(&s.water_h0_posits);
        push(&s.water_h1_posits);
    }
}

fn default_box_vector() -> [[f32; 3]; 3] {
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
}

/// Create or append snapshots to an XTC file.
///
/// `coord_scale` is applied as: `xtc_coord = internal_coord * coord_scale`.
/// - If your internal coords are **nm**, use `coord_scale = 1.0`.
/// - If your internal coords are **Å**, use `coord_scale = NM_PER_ANGSTROM`.
pub fn append_xtc_scaled(
    snapshots: &[Snapshot],
    path: &Path,
    write_water: bool,
    coord_scale: f32,
) -> io::Result<()> {
    if snapshots.is_empty() {
        return Ok(());
    }

    let first = &snapshots[0];
    let n_atoms = snapshot_num_atoms(first, write_water);

    for s in snapshots.iter().skip(1) {
        let n2 = snapshot_num_atoms(s, write_water);
        if n2 != n_atoms {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "inconsistent atom counts",
            ));
        }
    }

    let exists_nonempty = path.exists() && path.metadata()?.len() > 0;

    if exists_nonempty {
        let mut r = XTCTrajectory::open_read(path).map_err(xdr_to_io)?;
        let nat_existing = r.get_num_atoms().map_err(xdr_to_io)?;
        if nat_existing != n_atoms {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "atom count mismatch with existing file",
            ));
        }
    }

    let mut trj = if exists_nonempty {
        XTCTrajectory::open_append(path).map_err(xdr_to_io)?
    } else {
        XTCTrajectory::open_write(path).map_err(xdr_to_io)?
    };

    let mut frame = Frame::with_len(n_atoms);
    frame.box_vector = default_box_vector();

    for (i, s) in snapshots.iter().enumerate() {
        frame.step = i;
        frame.time = s.time as f32;
        fill_frame_coords(&mut frame, s, write_water, coord_scale);
        trj.write(&frame).map_err(xdr_to_io)?;
    }

    trj.flush().map_err(xdr_to_io)?;
    Ok(())
}

/// Convenience wrapper assuming your internal coords are already in nm.
pub fn append_xtc(snapshots: &[Snapshot], path: &Path, write_water: bool) -> io::Result<()> {
    append_xtc_scaled(snapshots, path, write_water, 1.0)
}

/// Load an XTC file into Snapshots.
///
/// `coord_unscale` is applied as: `internal_coord = xtc_coord * coord_unscale`.
/// - If you want output in **nm**, use `coord_unscale = 1.0`.
/// - If you want output in **Å**, use `coord_unscale = ANGSTROM_PER_NM`.
///
/// Note: XTC doesn’t encode semantic grouping (e.g. “these atoms are water”),
/// so this loader puts all coords into `atom_posits` and leaves water empty.
pub fn load_xtc_scaled(path: &Path, coord_unscale: f32) -> io::Result<Vec<Snapshot>> {
    let mut trj = XTCTrajectory::open_read(path).map_err(xdr_to_io)?;
    let n_atoms = trj.get_num_atoms().map_err(xdr_to_io)?;

    let mut frame = Frame::with_len(n_atoms);
    let mut out = Vec::new();

    while trj.read(&mut frame).is_ok() {
        let mut atom_posits = Vec::with_capacity(n_atoms);
        for xyz in &frame.coords {
            atom_posits.push(Vec3 {
                x: xyz[0] * coord_unscale,
                y: xyz[1] * coord_unscale,
                z: xyz[2] * coord_unscale,
            });
        }

        out.push(Snapshot {
            time: frame.time as f64,
            atom_posits,
            ..Default::default()
        });
    }

    Ok(out)
}
