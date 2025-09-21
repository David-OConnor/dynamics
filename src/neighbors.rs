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
    pub ref_pos_water_h0: Vec<Vec3>,
    pub ref_pos_water_h1: Vec<Vec3>,
    pub half_skin_sq: f32,
    /// Used to determine when to rebuild neighbor lists. todo: Implement.
    pub max_displacement_sq: f32,
}

impl MdState {
    pub(crate) fn update_max_displacement_since_rebuild(&mut self) {
        // dynamics
        for (i, a) in self.atoms.iter().enumerate() {
            let dv = self
                .cell
                .min_image(a.posit - self.neighbors_nb.ref_pos_dyn[i]);
            self.neighbors_nb.max_displacement_sq = self
                .neighbors_nb
                .max_displacement_sq
                .max(dv.magnitude_squared());
        }

        // waters: use max over O/H0/H1 to be conservative
        for (i, w) in self.water.iter().enumerate() {
            let dvo = self
                .cell
                .min_image(w.o.posit - self.neighbors_nb.ref_pos_water_o[i]);
            let dvh0 = self
                .cell
                .min_image(w.h0.posit - self.neighbors_nb.ref_pos_water_h0[i]);
            let dvh1 = self
                .cell
                .min_image(w.h1.posit - self.neighbors_nb.ref_pos_water_h1[i]);
            let d2 = dvo
                .magnitude_squared()
                .max(dvh0.magnitude_squared())
                .max(dvh1.magnitude_squared());
            self.neighbors_nb.max_displacement_sq = self.neighbors_nb.max_displacement_sq.max(d2);
        }
    }

    pub(crate) fn snapshot_ref_positions(&mut self) {
        self.neighbors_nb.ref_pos_dyn.clear();
        self.neighbors_nb
            .ref_pos_dyn
            .extend(self.atoms.iter().map(|a| self.cell.wrap(a.posit)));

        self.neighbors_nb.ref_pos_water_o.clear();
        self.neighbors_nb.ref_pos_water_h0.clear();
        self.neighbors_nb.ref_pos_water_h1.clear();
        self.neighbors_nb
            .ref_pos_water_o
            .extend(self.water.iter().map(|w| self.cell.wrap(w.o.posit)));
        self.neighbors_nb
            .ref_pos_water_h0
            .extend(self.water.iter().map(|w| self.cell.wrap(w.h0.posit)));
        self.neighbors_nb
            .ref_pos_water_h1
            .extend(self.water.iter().map(|w| self.cell.wrap(w.h1.posit)));

        self.neighbors_nb.max_displacement_sq = 0.0;
    }

    /// Call during each step; determines if we need to rebuild neighbors, and does so A/R.
    /// todo: Run on GPU?
    pub(crate) fn build_neighbors_if_needed(&mut self, dev: &ComputationDevice) {
        let start = Instant::now();

        if self.neighbors_nb.max_displacement_sq >= self.neighbors_nb.half_skin_sq {
            // current wrapped positions
            let curr_dyn = positions_of(&self.atoms, &self.cell);
            let curr_wat_o = positions_of_water_o(&self.water, &self.cell);

            self.neighbors_nb.dy_dy = build_neighbors(
                &curr_dyn,
                &curr_dyn,
                &self.cell,
                true,
                self.cfg.neighbor_skin,
            );

            self.neighbors_nb.water_water = build_neighbors(
                &curr_wat_o,
                &curr_wat_o,
                &self.cell,
                true,
                self.cfg.neighbor_skin,
            );

            self.neighbors_nb.dy_water = build_neighbors(
                &curr_dyn,
                &curr_wat_o,
                &self.cell,
                false,
                self.cfg.neighbor_skin,
            );
            self.rebuild_dy_water_inv();

            self.setup_pairs();

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

            // refresh refs + reset displacement
            self.snapshot_ref_positions();
        }

        self.neighbor_rebuild_us += start.elapsed().as_micros() as u64;
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

/// [Re]build a neighbor list, used for non-bonded interactions. Run this periodically.
pub fn build_neighbors(
    tgt_posits: &[Vec3],
    src_posits: &[Vec3],
    cell: &SimBox,
    symmetric: bool,
    skin: f32,
) -> Vec<Vec<usize>> {
    let skin_sq = (LONG_RANGE_CUTOFF + skin) * (LONG_RANGE_CUTOFF + skin);

    let tgt_len = tgt_posits.len();
    let src_len = src_posits.len();

    if symmetric {
        assert_eq!(src_len, tgt_len, "symmetric=true requires identical sets");
        let n = tgt_len;

        let half: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .with_min_len(1024)
            .map(|i| {
                let mut out = Vec::new();
                let pi = tgt_posits[i];
                for j in (i + 1)..n {
                    let d = cell.min_image(pi - src_posits[j]);
                    if d.magnitude_squared() < skin_sq {
                        out.push(j);
                    }
                }
                out
            })
            .collect();

        // Compute exact degrees for each node
        let mut deg = vec![0; n];
        for i in 0..n {
            deg[i] += half[i].len();
            for &j in &half[i] {
                deg[j] += 1;
            }
        }

        let mut full: Vec<Vec<usize>> = (0..n).map(|i| Vec::with_capacity(deg[i])).collect();

        for i in 0..n {
            for &j in &half[i] {
                full[i].push(j);
                full[j].push(i);
            }
        }
        full
    } else {
        (0..tgt_len)
            .into_par_iter()
            .map(|i_tgt| {
                let mut out = Vec::new();
                let pt = tgt_posits[i_tgt];
                for i_src in 0..src_len {
                    let d = cell.min_image(pt - src_posits[i_src]);
                    if d.magnitude_squared() < skin_sq {
                        out.push(i_src);
                    }
                }
                out
            })
            .collect()
    }
}

fn max_disp_dyn(cell: &SimBox, atoms: &[AtomDynamics], ref_pos: &[Vec3]) -> f32 {
    let mut max_d2: f32 = 0.0;
    for (i, a) in atoms.iter().enumerate() {
        let d = cell.min_image(a.posit - ref_pos[i]);
        max_d2 = max_d2.max(d.magnitude_squared());
    }
    max_d2
}

fn max_disp_wat(
    cell: &SimBox,
    waters: &[WaterMol],
    ref_o: &[Vec3],
    ref_h0: &[Vec3],
    ref_h1: &[Vec3],
) -> f32 {
    let mut max_d2: f32 = 0.0;
    for (i, w) in waters.iter().enumerate() {
        let d2 = cell
            .min_image(w.o.posit - ref_o[i])
            .magnitude_squared()
            .max(cell.min_image(w.h0.posit - ref_h0[i]).magnitude_squared())
            .max(cell.min_image(w.h1.posit - ref_h1[i]).magnitude_squared());
        max_d2 = max_d2.max(d2);
    }
    max_d2
}

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

fn positions_of(atoms: &[AtomDynamics], cell: &SimBox) -> Vec<Vec3> {
    atoms.iter().map(|a| cell.wrap(a.posit)).collect()
}

fn positions_of_water_o(waters: &[WaterMol], cell: &SimBox) -> Vec<Vec3> {
    waters.iter().map(|w| cell.wrap(w.o.posit)).collect()
}
