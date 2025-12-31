//! For teh DCD reporter format. Todo: Move to bio_files?

use std::{
    fs::{File, OpenOptions},
    io,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::Path,
    ptr::write,
};

use lin_alg::f32::Vec3;

use crate::snapshot::Snapshot;

fn rec<W: Write>(w: &mut W, payload: &[u8]) -> io::Result<()> {
    let len = payload.len() as u32;

    w.write_all(&len.to_le_bytes())?;
    w.write_all(payload)?;
    w.write_all(&len.to_le_bytes())
}

fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_record<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let len = read_u32_le(r)? as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    let len_end = read_u32_le(r)? as usize;
    if len_end != len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record length mismatch",
        ));
    }
    Ok(payload)
}

fn f32s_from_le_bytes(b: &[u8]) -> io::Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "float block not multiple of 4",
        ));
    }
    let n = b.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let j = 4 * i;
        out.push(f32::from_le_bytes(b[j..j + 4].try_into().unwrap()));
    }
    Ok(out)
}

/// Create or append snapshots to a DCD file. This is a common trajectory/reporter format
/// used by other software, including OpenMM and VMD.
pub fn append_dcd(snapshots: &[Snapshot], path: &Path, write_water: bool) -> io::Result<()> {
    if snapshots.is_empty() {
        return Ok(());
    }

    let first = &snapshots[0];

    let mut n_atoms = first.atom_posits.len();

    if write_water {
        n_atoms +=
            first.water_o_posits.len() + first.water_h0_posits.len() + first.water_h1_posits.len();
    }

    println!("N atoms: {:?} Write water: {:?}", n_atoms, write_water);

    for s in snapshots.iter().skip(1) {
        let mut n2 = s.atom_posits.len();

        if write_water {
            n2 += s.water_o_posits.len() + s.water_h0_posits.len() + s.water_h1_posits.len();
        }

        if n2 != n_atoms {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "inconsistent atom counts",
            ));
        }
    }

    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true) // todo: QC this.
        .open(path)?;

    let file_len = f.metadata()?.len();

    // write header if new/empty
    if file_len == 0 {
        let nsets = snapshots.len() as i32;
        let istart: i32 = 0;
        let nsavc: i32 = 1;
        let delta: f32 = if snapshots.len() >= 2 {
            (snapshots[1].time - snapshots[0].time) as f32
        } else {
            0.0
        };

        let mut header = Vec::with_capacity(84);
        header.extend_from_slice(b"CORD");
        let mut icntrl = [0i32; 20];
        icntrl[0] = nsets;
        icntrl[1] = istart;
        icntrl[2] = nsavc;
        icntrl[8] = 0;
        icntrl[10] = 0;
        icntrl[11] = 0;
        icntrl[19] = 1;

        for v in icntrl {
            header.extend_from_slice(&v.to_le_bytes());
        }

        header[4 + 36..4 + 40].copy_from_slice(&delta.to_le_bytes());
        rec(&mut f, &header)?;

        let title = format!("Created by Dynamics  NATOMS={}  NFRAMES={}", n_atoms, nsets);
        let mut line = [0u8; 80];
        let tb = title.as_bytes();
        let n = tb.len().min(80);
        line[..n].copy_from_slice(&tb[..n]);

        let mut title_block = Vec::with_capacity(4 + 80);
        title_block.extend_from_slice(&(1i32).to_le_bytes());
        title_block.extend_from_slice(&line);
        rec(&mut f, &title_block)?;

        let mut natom_block = Vec::with_capacity(4);
        natom_block.extend_from_slice(&(n_atoms as i32).to_le_bytes());
        rec(&mut f, &natom_block)?;
    } else {
        // verify header and NATOM; compute current NSET; then append and bump NSET
        f.seek(SeekFrom::Start(0))?;
        let l1 = read_u32_le(&mut f)?;
        let mut hdr = vec![0u8; l1 as usize];
        f.read_exact(&mut hdr)?;
        let l1e = read_u32_le(&mut f)?;
        if l1e != l1 || &hdr[0..4] != b"CORD" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a CORD/DCD file",
            ));
        }

        // Current NSET and flags
        let mut icntrl = [0i32; 20];
        for (i, item) in icntrl.iter_mut().enumerate() {
            let off = 4 + i * 4;
            *item = i32::from_le_bytes(hdr[off..off + 4].try_into().unwrap());
        }
        let cur_nset = icntrl[0];

        // Skip title
        let l2 = read_u32_le(&mut f)?;
        f.seek(SeekFrom::Current(l2 as i64))?;
        let l2e = read_u32_le(&mut f)?;
        if l2e != l2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt title block",
            ));
        }

        // Read NATOM
        let l3 = read_u32_le(&mut f)?;
        if l3 != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected NATOM record length",
            ));
        }
        let mut nb = [0u8; 4];
        f.read_exact(&mut nb)?;
        let natom_existing = i32::from_le_bytes(nb) as usize;
        let l3e = read_u32_le(&mut f)?;
        if l3e != l3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt NATOM block",
            ));
        }

        if natom_existing != n_atoms {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "atom count mismatch with existing file",
            ));
        }

        // Append frames
        f.seek(SeekFrom::End(0))?;

        let mut xs = vec![0f32; n_atoms];
        let mut ys = vec![0f32; n_atoms];
        let mut zs = vec![0f32; n_atoms];

        for s in snapshots {
            let mut i = 0usize;
            let mut push = |v: &Vec<Vec3>| {
                for p in v {
                    xs[i] = p.x;
                    ys[i] = p.y;
                    zs[i] = p.z;
                    i += 1;
                }
            };
            push(&s.atom_posits);

            if write_water {
                push(&s.water_o_posits);
                push(&s.water_h0_posits);
                push(&s.water_h1_posits);
            }

            let xb = unsafe { core::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) };
            let yb = unsafe { core::slice::from_raw_parts(ys.as_ptr() as *const u8, ys.len() * 4) };
            let zb = unsafe { core::slice::from_raw_parts(zs.as_ptr() as *const u8, zs.len() * 4) };

            rec(&mut f, xb)?;
            rec(&mut f, yb)?;
            rec(&mut f, zb)?;
        }

        // Update NSET in header (payload offset = 4-byte marker + 4 for "CORD")
        let new_nset = cur_nset
            .checked_add(snapshots.len() as i32)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "NSET overflow"))?;
        f.seek(SeekFrom::Start(8))?;
        f.write_all(&new_nset.to_le_bytes())?;
        f.flush()?;

        return Ok(());
    }

    // initial write path: write frames
    let mut xs = vec![0f32; n_atoms];
    let mut ys = vec![0f32; n_atoms];
    let mut zs = vec![0f32; n_atoms];

    for s in snapshots {
        let mut i = 0usize;
        let mut push = |v: &Vec<Vec3>| {
            for p in v {
                xs[i] = p.x;
                ys[i] = p.y;
                zs[i] = p.z;
                i += 1;
            }
        };
        push(&s.atom_posits);

        if write_water {
            push(&s.water_o_posits);
            push(&s.water_h0_posits);
            push(&s.water_h1_posits);
        }

        let xb = unsafe { core::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) };
        let yb = unsafe { core::slice::from_raw_parts(ys.as_ptr() as *const u8, ys.len() * 4) };
        let zb = unsafe { core::slice::from_raw_parts(zs.as_ptr() as *const u8, zs.len() * 4) };

        rec(&mut f, xb)?;
        rec(&mut f, yb)?;
        rec(&mut f, zb)?;
    }

    f.flush()
}

pub fn load_dcd(path: &Path) -> io::Result<Vec<Snapshot>> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    // Header
    let hdr = read_record(&mut r)?;
    if hdr.len() < 84 || &hdr[0..4] != b"CORD" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a CORD/DCD file",
        ));
    }
    let mut icntrl = [0i32; 20];
    for (i, item) in icntrl.iter_mut().enumerate() {
        let off = 4 + i * 4;
        *item = i32::from_le_bytes(hdr[off..off + 4].try_into().unwrap());
    }
    let nset_total = icntrl[0] as usize;

    // Delta is at bytes 36..40 after the "CORD"
    let delta = f32::from_le_bytes(hdr[4 + 36..4 + 40].try_into().unwrap()) as f64;

    // Title (ignored)
    let _ = read_record(&mut r)?;

    // NATOM
    let natom_block = read_record(&mut r)?;
    if natom_block.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected NATOM block size",
        ));
    }
    let n_atoms = i32::from_le_bytes(natom_block[0..4].try_into().unwrap()) as usize;

    let mut out = Vec::with_capacity(nset_total);

    for i in 0..nset_total {
        let xb = read_record(&mut r)?;
        let yb = read_record(&mut r)?;
        let zb = read_record(&mut r)?;

        if xb.len() != 4 * n_atoms || yb.len() != 4 * n_atoms || zb.len() != 4 * n_atoms {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "coordinate block size mismatch",
            ));
        }

        let xs = f32s_from_le_bytes(&xb)?;
        let ys = f32s_from_le_bytes(&yb)?;
        let zs = f32s_from_le_bytes(&zb)?;

        let mut atom_posits = Vec::with_capacity(n_atoms);
        for k in 0..n_atoms {
            atom_posits.push(Vec3 {
                x: xs[k],
                y: ys[k],
                z: zs[k],
            });
        }

        out.push(Snapshot {
            time: (i as f64) * delta,
            atom_posits,
            ..Default::default()
        });
    }

    Ok(out)
}
