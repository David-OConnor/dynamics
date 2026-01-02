//! For saving and loading to our custom snapshot format: MDT (Molecular Dynamics Trajectory)

use std::{
    fs::File,
    io,
    io::{BufReader, ErrorKind, Read, Write},
    path::Path,
};

use lin_alg::f32::Vec3;

use crate::snapshot::Snapshot;

fn write_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_f32_le(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_vec3s(out: &mut Vec<u8>, xs: &[Vec3]) {
    for p in xs {
        write_f32_le(out, p.x);
        write_f32_le(out, p.y);
        write_f32_le(out, p.z);
    }
}

fn read_vec3s(bytes: &[u8], i: &mut usize, count: usize) -> io::Result<Vec<Vec3>> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let x = read_f32_le(bytes, i)?;
        let y = read_f32_le(bytes, i)?;
        let z = read_f32_le(bytes, i)?;
        v.push(Vec3 { x, y, z });
    }
    Ok(v)
}

fn expected_len_bytes(n_atoms: usize, n_waters: usize) -> usize {
    // header: time_f32 + n_atoms_u32 + n_waters_u32
    let header = 3 * 4;

    // Vec3 arrays (each Vec3 = 12 bytes):
    // atom_posits, atom_velocities => 2 * n_atoms
    // water_o, water_h0, water_h1, water_velocities => 4 * n_waters
    let vec3s = (2 * n_atoms + 4 * n_waters) * 12;

    // scalars: energy_kinetic, energy_potential, temperature, pressure,
    // energy_potential_nonbonded, energy_potential_bonded => 6 * f32
    let scalars = 6 * 4;

    header + vec3s + scalars
}

fn read_exact<const N: usize>(bytes: &[u8], i: &mut usize) -> io::Result<[u8; N]> {
    if *i + N > bytes.len() {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            "MDT snapshot truncated",
        ));
    }
    let mut buf = [0u8; N];
    buf.copy_from_slice(&bytes[*i..*i + N]);
    *i += N;
    Ok(buf)
}

fn read_u32_le(bytes: &[u8], i: &mut usize) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4>(bytes, i)?))
}

fn read_f32_le(bytes: &[u8], i: &mut usize) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_exact::<4>(bytes, i)?))
}

impl Snapshot {
    /// E.g. for saving to file. Saves all items as 32-bit floating point.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_atoms = self.atom_posits.len();
        let n_waters = self.water_o_posits.len();

        debug_assert_eq!(self.atom_velocities.len(), n_atoms);
        debug_assert_eq!(self.water_h0_posits.len(), n_waters);
        debug_assert_eq!(self.water_h1_posits.len(), n_waters);
        debug_assert_eq!(self.water_velocities.len(), n_waters);

        let mut out = Vec::with_capacity(expected_len_bytes(n_atoms, n_waters));

        write_f32_le(&mut out, self.time as f32);
        write_u32_le(&mut out, n_atoms as u32);
        write_u32_le(&mut out, n_waters as u32);

        write_vec3s(&mut out, &self.atom_posits);
        write_vec3s(&mut out, &self.atom_velocities);

        write_vec3s(&mut out, &self.water_o_posits);
        write_vec3s(&mut out, &self.water_h0_posits);
        write_vec3s(&mut out, &self.water_h1_posits);
        write_vec3s(&mut out, &self.water_velocities);

        write_f32_le(&mut out, self.energy_kinetic);
        write_f32_le(&mut out, self.energy_potential);
        write_f32_le(&mut out, self.temperature);
        write_f32_le(&mut out, self.pressure);
        write_f32_le(&mut out, self.energy_potential_nonbonded);
        write_f32_le(&mut out, self.energy_potential_bonded);

        out
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut i = 0usize;

        let time_f32 = read_f32_le(bytes, &mut i)?;
        let n_atoms = read_u32_le(bytes, &mut i)? as usize;
        let n_waters = read_u32_le(bytes, &mut i)? as usize;

        let expected = expected_len_bytes(n_atoms, n_waters);
        if bytes.len() != expected {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "MDT snapshot length mismatch (got {}, expected {})",
                    bytes.len(),
                    expected
                ),
            ));
        }

        let atom_posits = read_vec3s(bytes, &mut i, n_atoms)?;
        let atom_velocities = read_vec3s(bytes, &mut i, n_atoms)?;

        let water_o_posits = read_vec3s(bytes, &mut i, n_waters)?;
        let water_h0_posits = read_vec3s(bytes, &mut i, n_waters)?;
        let water_h1_posits = read_vec3s(bytes, &mut i, n_waters)?;
        let water_velocities = read_vec3s(bytes, &mut i, n_waters)?;

        let energy_kinetic = read_f32_le(bytes, &mut i)?;
        let energy_potential = read_f32_le(bytes, &mut i)?;
        let temperature = read_f32_le(bytes, &mut i)?;
        let pressure = read_f32_le(bytes, &mut i)?;
        let energy_potential_nonbonded = read_f32_le(bytes, &mut i)?;
        let energy_potential_bonded = read_f32_le(bytes, &mut i)?;

        Ok(Self {
            time: time_f32 as f64,
            atom_posits,
            atom_velocities,
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            water_velocities,
            energy_kinetic,
            energy_potential,
            energy_potential_nonbonded,
            energy_potential_bonded,
            energy_potential_between_mols: Vec::new(),
            hydrogen_bonds: Vec::new(),
            temperature,
            pressure,
        })
    }
}

/// Save in our native format.
///
/// File format:
/// - magic: b"MDT1"
/// - repeated:
///   - u32 payload_len (LE)
///   - payload bytes (Snapshot::to_bytes)
pub fn save_mdt(snaps: &[Snapshot], path: &Path) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(b"MDT1")?;

    for snap in snaps {
        let payload = snap.to_bytes();
        let len_u32 = u32::try_from(payload.len())
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "snapshot too large"))?;
        file.write_all(&len_u32.to_le_bytes())?;
        file.write_all(&payload)?;
    }

    Ok(())
}

/// Load from our native format.
pub fn load_mdt(path: &Path) -> io::Result<Vec<Snapshot>> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"MDT1" {
        return Err(io::Error::new(ErrorKind::InvalidData, "bad MDT magic"));
    }

    let mut snaps = Vec::new();

    loop {
        let mut len_b = [0u8; 4];
        match r.read_exact(&mut len_b) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let len = u32::from_le_bytes(len_b) as usize;
        if len == 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "zero-length snapshot",
            ));
        }

        let mut payload = vec![0u8; len];
        r.read_exact(&mut payload)?;
        snaps.push(Snapshot::from_bytes(&payload)?);
    }

    Ok(snaps)
}
