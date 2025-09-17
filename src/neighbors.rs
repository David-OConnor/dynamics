//! This module contains code for maintaining non-bonded neighbor lists.
//! This is an optimization to determine which atoms we count in Lennard Jones
//! and short-term Ewald Coulomb interactions.
//!
//! Note: GPU is probably not a good fit for rebuilding neighbor lists.

use std::time::Instant;

use lin_alg::f32::Vec3;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::gpu_interface::PerNeighborGpu;
use crate::{
    AtomDynamics, ComputationDevice, MdState, ambient::SimBox, non_bonded::LONG_RANGE_CUTOFF,
    water_opc::WaterMol,
};

#[derive(Default)]
/// Non-bonded neighbors; an important optimization for Lennard Jones and Coulomb interactions.
/// By index for fast lookups; separate fields, as these indices have different meanings for dynamic atoms,
/// static atoms, and water.
///
/// To understand how we've set up the fields, each of the three types of atoms interactions with the others,
/// but note that static atoms are sources only; they are not acted on.
///
/// Note: These historically called "Verlet lists", but we're not using that term, as we use "Verlet" to refer
/// to the integrator, which this has nothing to do with. They do have to do with their applicability to
/// non-bonded interactions, so we call them "Non-bonded neighbors".
pub struct NeighborsNb {
    // Neighbors acting on dynamic atoms:
    /// Symmetric dynamic-dynamic indices. Dynamic source and target.
    pub dy_dy: Vec<Vec<usize>>,
    /// Outer: Dynamic. Inner: water. Each is a source and target.
    pub dy_water: Vec<Vec<usize>>,
    /// Symmetric water-water indices. Dynamic source and target.
    pub water_water: Vec<Vec<usize>>,
    /// Outer: Water. Inner: dynamic. Water target, dynamic source.
    /// todo: This is a direct reverse of dy_water, but may be worth keeping in for indexing order.
    pub water_dy: Vec<Vec<usize>>,
    //
    // Reference positions used when rebuilding. Only for movable atoms.
    pub ref_pos_dyn: Vec<Vec3>,
    // /// Doesn't change.
    // pub ref_pos_static: Vec<Vec3>,
    pub ref_pos_water_o: Vec<Vec3>, // use O as proxy for the rigid water
    /// Used to determine when to rebuild neighbor lists. todo: Implement.
    pub max_displacement_sq: f32,
}

impl MdState {
    /// Call during each step; determines if we need to rebuild neighbors, and does so A/R.
    /// todo: Run on GPU?
    pub fn build_neighbors_if_needed(&mut self, dev: &ComputationDevice) {
        let start = Instant::now();

        // Current positions
        // let dyn_pos_now = positions_of(&self.atoms);
        // let water_o_pos_now = positions_of_water_o(&self.water);

        // Displacements
        let dyn_disp_sq = max_disp_dyn(&self.cell, &self.atoms, &self.neighbors_nb.ref_pos_dyn);
        let wat_disp_sq = max_disp_wat(&self.cell, &self.water, &self.neighbors_nb.ref_pos_water_o);

        let mut rebuilt_dyn = false;
        let mut rebuilt_wat = false;

        let skin_sq_div4 = self.cfg.neighbor_skin.powi(2) / 4.;

        if dyn_disp_sq > skin_sq_div4 {
            self.neighbors_nb.dy_dy = build_neighbors(
                &self.neighbors_nb.ref_pos_dyn,
                &self.neighbors_nb.ref_pos_dyn,
                &self.cell,
                true,
                self.cfg.neighbor_skin,
            );

            self.neighbors_nb.dy_water = build_neighbors(
                &self.neighbors_nb.ref_pos_dyn,
                &self.neighbors_nb.ref_pos_water_o,
                &self.cell,
                false,
                self.cfg.neighbor_skin,
            );
            self.rebuild_dy_water_inv();

            rebuilt_dyn = true;
        }

        if wat_disp_sq > skin_sq_div4 {
            self.neighbors_nb.water_water = build_neighbors(
                &self.neighbors_nb.ref_pos_water_o,
                &self.neighbors_nb.ref_pos_water_o,
                &self.cell,
                true,
                self.cfg.neighbor_skin,
            );

            if !rebuilt_dyn {
                // Don't double-run this, but it's required for both paths.
                self.neighbors_nb.dy_water = build_neighbors(
                    &self.neighbors_nb.ref_pos_dyn,
                    &self.neighbors_nb.ref_pos_water_o,
                    &self.cell,
                    false,
                    self.cfg.neighbor_skin,
                );
                self.rebuild_dy_water_inv();
            }

            rebuilt_wat = true;
        }

        // Rebuild reference position lists for next use, for use with determining when to rebuild the neighbor list.
        // (Static refs doesn't get rebuilt after init)
        if rebuilt_dyn {
            for (i, a) in self.atoms.iter().enumerate() {
                self.neighbors_nb.ref_pos_dyn[i] = a.posit;
            }
        }

        if rebuilt_wat {
            for (i, m) in self.water.iter().enumerate() {
                self.neighbors_nb.ref_pos_water_o[i] = m.o.posit;
            }
        }

        static mut PRINTED: bool = false;
        if rebuilt_dyn || rebuilt_wat {
            // let elapsed = start.elapsed();
            if !unsafe { PRINTED } {
                // println!("Neighbor build time: {:?} μs", elapsed.as_micros());
                unsafe {
                    PRINTED = true;
                }
            }

            self.setup_pairs();
            self.neighbor_rebuild_count += 1;

            #[cfg(feature = "cuda")]
            if let ComputationDevice::Gpu((stream, _)) = dev {
                self.per_neighbor_gpu = Some(PerNeighborGpu::new(
                    stream,
                    &self.nb_pairs,
                    &self.atoms,
                    &self.water,
                    &self.lj_tables,
                ));
            }
        } else {
            // println!("No rebuild needed.");
        }

        let elapsed = start.elapsed().as_micros();
        self.neighbor_rebuild_us += elapsed as u64;
    }

    /// This inverts our neighbor set between water and dynamic atoms.
    pub fn rebuild_dy_water_inv(&mut self) {
        let n_waters = self.neighbors_nb.ref_pos_water_o.len();
        self.neighbors_nb.water_dy.clear();
        self.neighbors_nb.water_dy.resize_with(n_waters, Vec::new);

        // Count degrees first
        let mut deg = vec![0usize; n_waters];
        for ws in &self.neighbors_nb.dy_water {
            for &w in ws {
                deg[w] += 1;
            }
        }
        for i in 0..n_waters {
            self.neighbors_nb.water_dy[i] = Vec::with_capacity(deg[i]);
        }
        // Fill
        for (i_dyn, ws) in self.neighbors_nb.dy_water.iter().enumerate() {
            for &iw in ws {
                self.neighbors_nb.water_dy[iw].push(i_dyn);
            }
        }
    }
}

pub fn build_neighbors(
    tgt_posits: &[Vec3],
    src_posits: &[Vec3],
    cell: &SimBox,
    symmetric: bool,
    skin: f32,
) -> Vec<Vec<usize>> {
    let rc = LONG_RANGE_CUTOFF + skin;
    let rc2 = rc * rc;

    // Replace these with your box lengths / orthorhombic extents:
    let [lx, ly, lz] = cell.extent.to_arr();
    let nx = f32::floor(lx / rc).max(1.0) as i32;
    let ny = f32::floor(ly / rc).max(1.0) as i32;
    let nz = f32::floor(lz / rc).max(1.0) as i32;
    let ncell = (nx * ny * nz) as usize;

    #[inline]
    fn wrap(i: i32, n: i32) -> i32 {
        (i % n + n) % n
    }

    #[inline]
    fn linear(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32) -> usize {
        (cx + nx * (cy + ny * cz)) as usize
    }

    // assume cell.frac(p) returns fractional coords; adapt to your API
    fn wrap01(x: f32) -> f32 {
        x - x.floor()
    } // works for negatives too

    fn cell_index_cart(
        p: Vec3,
        lx: f32,
        ly: f32,
        lz: f32,
        nx: i32,
        ny: i32,
        nz: i32,
    ) -> (i32, i32, i32) {
        let fx = wrap01(p.x / lx);
        let fy = wrap01(p.y / ly);
        let fz = wrap01(p.z / lz);

        let cx = (fx * nx as f32) as i32;
        let cy = (fy * ny as f32) as i32;
        let cz = (fz * nz as f32) as i32;

        (cx.min(nx - 1), cy.min(ny - 1), cz.min(nz - 1))
    }

    // Build linked list for src
    let mut head = vec![usize::MAX; ncell];
    let mut next = vec![usize::MAX; src_posits.len()];

    for (i, &p) in src_posits.iter().enumerate() {
        let (cx, cy, cz) = cell_index(p, lx, ly, lz, rc, nx, ny, nz);
        let c = linear(cx, cy, cz, nx, ny);
        next[i] = head[c];
        head[c] = i;
    }

    // Candidate search
    let out: Vec<Vec<usize>> = (0..tgt_posits.len())
        .into_par_iter()
        .map(|it| {
            let pt = tgt_posits[it];
            let (cx, cy, cz) = cell_index(pt, lx, ly, lz, rc, nx, ny, nz);
            let mut neigh = Vec::new();

            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let c = linear(
                            wrap(cx + dx, nx),
                            wrap(cy + dy, ny),
                            wrap(cz + dz, nz),
                            nx,
                            ny,
                        );
                        let mut j = head[c];
                        while j != usize::MAX {
                            if !symmetric || j != it {
                                let d = cell.min_image(pt - src_posits[j]);
                                if d.magnitude_squared() < rc2 {
                                    neigh.push(j);
                                }
                            }
                            j = next[j];
                        }
                    }
                }
            }
            neigh
        })
        .collect();

    if symmetric {
        // Mirror to make adjacency symmetric (same degree prealloc trick applies if needed)
        let n = out.len();
        let mut full = vec![Vec::<usize>::new(); n];
        let mut deg = vec![0usize; n];
        for i in 0..n {
            deg[i] += out[i].len();
            for &j in &out[i] {
                deg[j] += 1;
            }
        }
        for i in 0..n {
            full[i] = Vec::with_capacity(deg[i]);
        }
        for i in 0..n {
            for &j in &out[i] {
                full[i].push(j);
                full[j].push(i);
            }
        }
        full
    } else {
        out
    }
}
//
// /// [Re]build a neighbor list, used for non-bonded interactions. Run this periodically.
// pub fn build_neighbors(
//     tgt_posits: &[Vec3],
//     src_posits: &[Vec3],
//     cell: &SimBox,
//     symmetric: bool,
//     skin: f32,
// ) -> Vec<Vec<usize>> {
//     let skin_sq = (LONG_RANGE_CUTOFF + skin) * (LONG_RANGE_CUTOFF + skin);
//
//     let tgt_len = tgt_posits.len();
//     let src_len = src_posits.len();
//
//     if symmetric {
//         assert_eq!(src_len, tgt_len, "symmetric=true requires identical sets");
//         let n = tgt_len;
//
//         let half: Vec<Vec<usize>> = (0..n)
//             .into_par_iter()
//             .with_min_len(1024)
//             .map(|i| {
//                 let mut out = Vec::new();
//                 let pi = tgt_posits[i];
//                 for j in (i + 1)..n {
//                     let d = cell.min_image(pi - src_posits[j]);
//                     if d.magnitude_squared() < skin_sq {
//                         out.push(j);
//                     }
//                 }
//                 out
//             })
//             .collect();
//
//         // Compute exact degrees for each node
//         let mut deg = vec![0; n];
//         for i in 0..n {
//             deg[i] += half[i].len();
//             for &j in &half[i] {
//                 deg[j] += 1;
//             }
//         }
//
//         let mut full: Vec<Vec<usize>> = (0..n)
//             .map(|i| Vec::with_capacity(deg[i]))
//             .collect();
//
//         for i in 0..n {
//             for &j in &half[i] {
//                 full[i].push(j);
//                 full[j].push(i);
//             }
//         }
//         full
//     } else {
//         (0..tgt_len)
//             .into_par_iter()
//             .map(|i_tgt| {
//                 let mut out = Vec::new();
//                 let pt = tgt_posits[i_tgt];
//                 for i_src in 0..src_len {
//                     let d = cell.min_image(pt - src_posits[i_src]);
//                     if d.magnitude_squared() < skin_sq {
//                         out.push(i_src);
//                     }
//                 }
//                 out
//             })
//             .collect()
//     }
// }

// pub fn max_displacement_sq_since_build(
//     targets: &[Vec3],
//     neighbor_ref_posits: &[Vec3],
//     cell: &SimBox,
// ) -> f32 {
//     let mut result: f32 = 0.0;
//
//     for (i, posit) in targets.iter().enumerate() {
//         let diff_min_img = cell.min_image(*posit - neighbor_ref_posits[i]);
//         result = result.max(diff_min_img.magnitude_squared());
//     }
//     result
// }

/// Helper
fn positions_of(atoms: &[AtomDynamics]) -> Vec<Vec3> {
    atoms.iter().map(|a| a.posit).collect()
}

/// Helper
fn positions_of_water_o(waters: &[WaterMol]) -> Vec<Vec3> {
    waters.iter().map(|w| w.o.posit).collect()
}

fn max_disp_dyn(cell: &SimBox, atoms: &[AtomDynamics], ref_pos: &[Vec3]) -> f32 {
    let mut max_d2: f32 = 0.0;
    for (i, a) in atoms.iter().enumerate() {
        let d = cell.min_image(a.posit - ref_pos[i]);
        max_d2 = max_d2.max(d.magnitude_squared());
    }
    max_d2
}
fn max_disp_wat(cell: &SimBox, waters: &[WaterMol], ref_pos: &[Vec3]) -> f32 {
    let mut max_d2: f32 = 0.0;
    for (i, w) in waters.iter().enumerate() {
        let d = cell.min_image(w.o.posit - ref_pos[i]);
        max_d2 = max_d2.max(d.magnitude_squared());
    }
    max_d2
}
