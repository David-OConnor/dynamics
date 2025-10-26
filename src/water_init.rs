#![allow(clippy::excessive_precision)]

//! Code for initializing water molecules, including assigning quantity, initial positions, and
//! velocities. Set up to meet density, pressure, and or temperature targets. Not specific to the
//! water model used.

use std::{f32::consts::TAU, time::Instant};

use lin_alg::f32::{Mat3 as Mat3F32, Quaternion as QuaternionF32, Vec3};
use rand::{Rng, distr::Uniform, seq::SliceRandom};
use rand_distr::Distribution;

use crate::{
    ACCEL_CONVERSION_INV, AtomDynamics,
    ambient::{GAS_CONST_R, KB_A2_PS2_PER_K_PER_AMU, SimBox},
    water_opc::WaterMol,
};

// 0.997 g cm⁻³ is a good default density. We use this for initializing and maintaining
// the water density and molecule count.
const WATER_DENSITY: f32 = 0.997;

// g / mol (or AMU per molecule)
// This is similar to the Amber H and O masses we used summed, and could be explained
// by precision limits. We use it for generating atoms based on mass density.
const MASS_WATER: f32 = 18.015_28;

// Avogadro's constant. mol^-1.
const N_A: f32 = 6.022_140_76e23;

// This is ~0.0333 mol/Å³
// Multiplying this by volume in Angstrom^3 gives us AMU g cm^-3 Å^3 mol^-1
const WATER_MOLS_PER_VOL: f32 = WATER_DENSITY * N_A / (MASS_WATER * 1.0e24);

// Don't generate water molecules that are too close to other atoms.
// Vdw contact distance between water molecules and organic molecules is roughly 3.5 Å.
const MIN_NONWATER_DIST: f32 = 3.75;
// A conservative water-water (Oxygen-Oxygen) minimum distance. 2.7 - 3.2 Å is suitable.
const MIN_WATER_OO_DIST: f32 = 2.8;

const MAX_GEN_ATTEMPTS: usize = 50; // todo: Tune A/R.

// Start free-volume code ----------

#[derive(Clone)]
struct CellList {
    cell_size: f32,
    dims: [usize; 3],
    low: Vec3,
    high: Vec3,
    inv_extent: Vec3,
    buckets: Vec<Vec<usize>>,
}

impl CellList {
    fn new(cell: &SimBox, cell_size: f32, capacity_hint: usize) -> Self {
        let low = cell.bounds_low;
        let high = cell.bounds_high;
        let extent = high - low;
        let dims = [
            (extent.x / cell_size).floor().max(1.0) as usize,
            (extent.y / cell_size).floor().max(1.0) as usize,
            (extent.z / cell_size).floor().max(1.0) as usize,
        ];
        let inv_extent = Vec3::new(1.0 / extent.x, 1.0 / extent.y, 1.0 / extent.z);

        let mut buckets = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        buckets.resize_with(buckets.capacity(), Vec::new);

        let n_buckets = buckets.len();

        if capacity_hint > 0 {
            // amortize: naive pre-reserve
            for b in &mut buckets {
                b.reserve((capacity_hint as f32 / n_buckets as f32).ceil() as usize + 4);
            }
        }
        Self {
            cell_size,
            dims,
            low,
            high,
            inv_extent,
            buckets,
        }
    }

    #[inline]
    fn idx3(&self, mut ix: isize, mut iy: isize, mut iz: isize) -> usize {
        let nx = self.dims[0] as isize;
        let ny = self.dims[1] as isize;
        let nz = self.dims[2] as isize;
        // periodic wrap
        ix = (ix % nx + nx) % nx;
        iy = (iy % ny + ny) % ny;
        iz = (iz % nz + nz) % nz;
        (ix as usize) + self.dims[0] * (iy as usize) + self.dims[0] * self.dims[1] * (iz as usize)
    }

    #[inline]
    fn to_ijk(&self, p: Vec3) -> (isize, isize, isize) {
        // Map to [0,1) then scale by dims
        let r = self.inv_extent.hadamard_product(p - self.low);
        let fx = (r.x.fract() + 1.0).fract();
        let fy = (r.y.fract() + 1.0).fract();
        let fz = (r.z.fract() + 1.0).fract();
        let ix = (fx * self.dims[0] as f32).floor() as isize;
        let iy = (fy * self.dims[1] as f32).floor() as isize;
        let iz = (fz * self.dims[2] as f32).floor() as isize;
        (ix, iy, iz)
    }

    fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
    }

    fn insert(&mut self, idx: usize, p: Vec3) {
        let (ix, iy, iz) = self.to_ijk(p);
        let k = self.idx3(ix, iy, iz);
        self.buckets[k].push(idx);
    }

    /// Iterate indices in the 3×3×3 neighborhood (periodic).
    fn neighbors<'a>(&'a self, p: Vec3) -> impl Iterator<Item = usize> + 'a {
        let (ix, iy, iz) = self.to_ijk(p);
        let nx = self.dims[0] as isize;
        let ny = self.dims[1] as isize;
        let nz = self.dims[2] as isize;
        let mut out = Vec::with_capacity(64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let k = self.idx3(ix + dx, iy + dy, iz + dz);
                    out.extend(self.buckets[k].iter().copied());
                }
            }
        }
        out.into_iter()
    }
}

struct FreeMask {
    centers: Vec<Vec3>, // voxel centers that are "free"
    voxel_size: f32,
}

fn build_free_mask(cell: &SimBox, atoms: &[AtomDynamics], voxel_size: f32) -> FreeMask {
    let low = cell.bounds_low;
    let high = cell.bounds_high;
    let ext = high - low;
    let nx = (ext.x / voxel_size).floor().max(1.0) as usize;
    let ny = (ext.y / voxel_size).floor().max(1.0) as usize;
    let nz = (ext.z / voxel_size).floor().max(1.0) as usize;

    // Cell list for atoms with search cell size = MIN_WATER_NONWATER_DIST
    let mut cl_atoms = CellList::new(cell, MIN_NONWATER_DIST, atoms.len());
    cl_atoms.clear();
    for (i, a) in atoms.iter().enumerate() {
        cl_atoms.insert(i, a.posit);
    }

    let mut centers = Vec::new();
    let half = 0.5 * voxel_size;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let c = Vec3::new(
                    low.x + (ix as f32 + 0.5) * voxel_size,
                    low.y + (iy as f32 + 0.5) * voxel_size,
                    low.z + (iz as f32 + 0.5) * voxel_size,
                );

                // Quick reject using neighbors
                let mut ok = true;
                for j in cl_atoms.neighbors(c) {
                    let d = cell.min_image(atoms[j].posit - c).magnitude();
                    if d < MIN_NONWATER_DIST {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    centers.push(c);
                }
            }
        }
    }
    FreeMask {
        centers,
        voxel_size,
    }
}

// End free-volume code ---------

/// Generate water molecules to meet a temperature target, using a standard density. We deconflict
/// with (solute) atoms in the simulation, and base the number of molecules to add on the free space,
/// not the total cell volume.
pub fn make_water_mols(
    cell: &SimBox,
    temperature_tgt: f32,
    atoms: &[AtomDynamics],
) -> Vec<WaterMol> {
    println!("Initializing water molecules...");
    let start = Instant::now();

    // Build free-volume mask (excludes voxels too close to solute)
    let voxel_size = 2.8_f32.max(MIN_WATER_OO_DIST); // coarse but safe
    let mask = build_free_mask(cell, atoms, voxel_size);

    // Estimate free volume & n_mols from it
    let free_vol = (mask.centers.len() as f32) * (voxel_size.powi(3));
    let n_mols = (WATER_MOLS_PER_VOL * free_vol).round() as usize;

    println!("Solvent free volume: {:.2} / {:.2} Å³ / 1000", free_vol / 1_000., cell.volume() / 1_000.);

    // Shuffle candidates and greedily accept with fast O–O checks
    let mut rng = rand::rng();
    let mut candidates = mask.centers.clone();
    candidates.shuffle(&mut rng);

    // Build once: solute cell list for quick recheck (used after jitter)
    let mut cl_atoms = CellList::new(cell, MIN_NONWATER_DIST, atoms.len());
    cl_atoms.clear();
    for (i, a) in atoms.iter().enumerate() {
        cl_atoms.insert(i, a.posit);
    }

    let mut cl_wat = CellList::new(cell, MIN_WATER_OO_DIST, n_mols);
    cl_wat.clear();

    let uni01 = Uniform::<f32>::new(0.0, 1.0).unwrap();
    let mut result: Vec<WaterMol> = Vec::with_capacity(n_mols);

    for mut c in candidates.into_iter() {
        if result.len() >= n_mols {
            break;
        }

        // todo: This doens't seem to be working; the result still is on a grid.
        // Tiny jitter to avoid perfect grid artifacts
        let jitter = 0.3_f32.min(0.1 * MIN_WATER_OO_DIST);
        if jitter > 0.0 {
            c.x += (rng.sample(uni01) - 0.5) * 2.0 * jitter;
            c.y += (rng.sample(uni01) - 0.5) * 2.0 * jitter;
            c.z += (rng.sample(uni01) - 0.5) * 2.0 * jitter;
            c = cell.wrap(c);
        }

        // Check for conflicts with non-water atoms.
        let mut ok = true;
        for j in cl_atoms.neighbors(c) {
            let d = cell.min_image(atoms[j].posit - c).magnitude();
            if d < MIN_NONWATER_DIST {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        // Check for conflicts with other waters.

        let mut ok = true;
        for j in cl_wat.neighbors(c) {
            let d = cell.min_image(result[j].o.posit - c).magnitude();
            if d < MIN_WATER_OO_DIST {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        // Random orientation (Shoemake)
        let (u1, u2, u3) = (rng.sample(uni01), rng.sample(uni01), rng.sample(uni01));
        let sqrt1_minus_u1 = (1.0 - u1).sqrt();
        let sqrt_u1 = u1.sqrt();
        let (theta1, theta2) = (TAU * u2, TAU * u3);
        let q = QuaternionF32::new(
            sqrt1_minus_u1 * theta1.sin(),
            sqrt1_minus_u1 * theta1.cos(),
            sqrt_u1 * theta2.sin(),
            sqrt_u1 * theta2.cos(),
        )
        .to_normalized();

        let idx = result.len();
        result.push(WaterMol::new(c, Vec3::new_zero(), q));
        cl_wat.insert(idx, c);
    }

    if result.len() < n_mols {
        eprintln!(
            "Placed {} / {} waters; consider enlarging the box or loosening thresholds.",
            result.len(),
            n_mols
        );
    }

    init_velocities_rigid(&mut result, temperature_tgt, cell);

    let elapsed = start.elapsed().as_millis();
    println!("Complete in {elapsed} ms.");
    result
}

fn init_velocities_rigid(mols: &mut [WaterMol], t_target: f32, _cell: &SimBox) {
    use rand_distr::Normal;

    let mut rng = rand::rng();
    let kT = KB_A2_PS2_PER_K_PER_AMU * t_target;

    for m in mols.iter_mut() {
        // COM & relative positions
        let (r_com, m_tot) = {
            let mut r = Vec3::new_zero();
            let mut m_tot = 0.0;
            for a in [&m.o, &m.h0, &m.h1] {
                r += a.posit * a.mass;
                m_tot += a.mass;
            }
            (r / m_tot, m_tot)
        };

        let rO = m.o.posit - r_com;
        let rH0 = m.h0.posit - r_com;
        let rH1 = m.h1.posit - r_com;

        // Sample COM velocity
        let sigma_v = (kT / m_tot).sqrt();
        let n = Normal::new(0.0, sigma_v).unwrap();
        let v_com = Vec3::new(n.sample(&mut rng), n.sample(&mut rng), n.sample(&mut rng));

        // Inertia tensor about COM (world frame)
        // Build as arrays (your code)
        let inertia = |r: Vec3, mass: f32| {
            let r2 = r.dot(r);
            [
                [
                    mass * (r2 - r.x * r.x),
                    -mass * r.x * r.y,
                    -mass * r.x * r.z,
                ],
                [
                    -mass * r.y * r.x,
                    mass * (r2 - r.y * r.y),
                    -mass * r.y * r.z,
                ],
                [
                    -mass * r.z * r.x,
                    -mass * r.z * r.y,
                    mass * (r2 - r.z * r.z),
                ],
            ]
        };
        let mut I_arr = inertia(rO, m.o.mass);
        let add_I = |I: &mut [[f32; 3]; 3], J: [[f32; 3]; 3]| {
            for i in 0..3 {
                for j in 0..3 {
                    I[i][j] += J[i][j];
                }
            }
        };
        add_I(&mut I_arr, inertia(rH0, m.h0.mass));
        add_I(&mut I_arr, inertia(rH1, m.h1.mass));

        // Convert to Mat3 once, then use
        let I = Mat3F32::from_arr(I_arr);

        // Diagonalize and solve with the Mat3 methods
        let (eigvecs, eigvals) = I.eigen_vecs_vals();
        let L_principal = Vec3::new(
            Normal::new(0.0, (kT * eigvals.x.max(0.0)).sqrt())
                .unwrap()
                .sample(&mut rng),
            Normal::new(0.0, (kT * eigvals.y.max(0.0)).sqrt())
                .unwrap()
                .sample(&mut rng),
            Normal::new(0.0, (kT * eigvals.z.max(0.0)).sqrt())
                .unwrap()
                .sample(&mut rng),
        );
        let L_world = eigvecs * L_principal; // assumes Mat3 * Vec3 is implemented
        let omega = I.solve_system(L_world); // ω = I^{-1} L

        // Set atomic velocities
        m.o.vel = v_com + omega.cross(rO);
        m.h0.vel = v_com + omega.cross(rH0);
        m.h1.vel = v_com + omega.cross(rH1);
    }

    // Remove global COM drift
    remove_com_velocity(mols);

    // Optional: compute KE (translation+rotation == sum ½ m v^2 now) and rescale to T_target
    let (ke_raw, dof) = kinetic_energy_and_dof(mols); // dof = 6*N - 3
    let lambda =
        (t_target / (2.0 * (ke_raw * ACCEL_CONVERSION_INV) / (dof as f32 * GAS_CONST_R))).sqrt();
    for a in atoms_mut(mols) {
        if a.mass > 0.0 {
            a.vel *= lambda;
        }
    }
}

fn kinetic_energy_and_dof(mols: &[WaterMol]) -> (f32, usize) {
    let mut ke = 0.0;
    let mut dof = 0usize;
    for m in mols {
        for a in [&m.o, &m.h0, &m.h1] {
            ke += 0.5 * a.mass * a.vel.dot(a.vel);
            dof += 3;
        }
    }
    // remove 3 for total COM; remove constraints if you track them
    let n_constraints = 3 * mols.len();
    (ke, dof - 3 - n_constraints)
}

fn atoms_mut(mols: &mut [WaterMol]) -> impl Iterator<Item = &mut AtomDynamics> {
    mols.iter_mut()
        .flat_map(|m| [&mut m.o, &mut m.h0, &mut m.h1].into_iter())
}

/// Removes center-of-mass drift.
fn remove_com_velocity(mols: &mut [WaterMol]) {
    let mut p = Vec3::new_zero();
    let mut m_tot = 0.0;
    for a in atoms_mut(mols) {
        p += a.vel * a.mass;
        m_tot += a.mass;
    }

    let v_com = p / m_tot;
    for a in atoms_mut(mols) {
        a.vel -= v_com;
    }
}

fn too_close_to_atoms(p: Vec3, atoms: &[AtomDynamics], cell: &SimBox) -> bool {
    for a in atoms {
        let d = cell.min_image(a.posit - p).magnitude();
        if d < MIN_NONWATER_DIST {
            return true;
        }
    }
    false
}

fn too_close_to_waters(p: Vec3, waters: &[WaterMol], cell: &SimBox) -> bool {
    for w in waters {
        let d = cell.min_image(w.o.posit - p).magnitude();
        if d < MIN_WATER_OO_DIST {
            return true;
        }
    }
    false
}
