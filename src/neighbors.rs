//! This module contains code for maintaining non-bonded neighbor (Verlet) neighbor lists.
//! This is an optimization to determine which atoms we count in Lennard Jones
//! and short-term Ewald Coulomb interactions.
//!
//! Note: GPU is probably not a good fit for rebuilding neighbor lists.
//!
//! We generally use cell-wrapped distances for water, and direct distances for non-water.

use std::time::Instant;

use lin_alg::f32::Vec3;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::gpu_interface::PerNeighborGpu;
use crate::{
    COMPUTATION_TIME_RATIO, ComputationDevice, MdState, ambient::SimBox,
    non_bonded::LONG_RANGE_CUTOFF,
};

/// By index for fast lookups; separate fields, as these indices are applied differently for non-water atoms
/// and water.
///
/// These are historically called "Verlet lists", but we're not using that term, as we use "Verlet" to refer
/// to the integrator, which this has nothing to do with. They do have to do with their applicability to
/// non-bonded interactions, so we call them "Non-bonded neighbors".
#[derive(Default)]
pub struct NeighborsNb {
    /// Symmetric std-std indices.
    pub std_std: Vec<Vec<usize>>,
    /// Outer: standard. Inner: water.
    pub std_water: Vec<Vec<usize>>,
    /// Symmetric water-water indices.
    pub water_water: Vec<Vec<usize>>,
    //
    /// Reference positions used to determine when we rebuild.
    pub atom_posits_last_rebuild: Vec<Vec3>,
    /// We use O as proxy for the rigid water, omitting its hydrogens to save computation; they
    /// will always be near.
    pub water_o_posits_last_rebuild: Vec<Vec3>,
    /// Used to determine when to rebuild neighbor lists.
    pub max_displacement_sq: f32,
    /// These values are set up at init.
    pub half_skin_sq: f32,
    pub skin_sq_w_cutoff: f32,
}

impl NeighborsNb {
    pub fn new(skin: f32) -> Self {
        Self {
            half_skin_sq: (skin * 0.5).powi(2),
            skin_sq_w_cutoff: (LONG_RANGE_CUTOFF + skin) * (LONG_RANGE_CUTOFF + skin),
            ..Default::default()
        }
    }
}

impl MdState {
    pub(crate) fn update_max_displacement_since_rebuild(&mut self) {
        for (i, a) in self.atoms.iter().enumerate() {
            // Static atoms always have 0 displacement.
            if a.static_ {
                continue;
            }

            let dv = a.posit - self.neighbors_nb.atom_posits_last_rebuild[i];

            self.neighbors_nb.max_displacement_sq = self
                .neighbors_nb
                .max_displacement_sq
                .max(dv.magnitude_squared());
        }

        // We only track oxygen position here. The hydrogens affect the need to rebuild as well,
        // but since they're always near their oxygen, we omit them, with a slight accuracy impact.
        for (i, w) in self.water.iter().enumerate() {
            let diff_water_o = self
                .cell
                .min_image(w.o.posit - self.neighbors_nb.water_o_posits_last_rebuild[i]);

            let dist_sq = diff_water_o.magnitude_squared();

            self.neighbors_nb.max_displacement_sq =
                self.neighbors_nb.max_displacement_sq.max(dist_sq);
        }
    }

    fn save_rebuild_posits(&mut self) {
        self.neighbors_nb.atom_posits_last_rebuild.clear();
        self.neighbors_nb
            .atom_posits_last_rebuild
            .extend(self.atoms.iter().map(|a| a.posit));

        self.neighbors_nb.water_o_posits_last_rebuild.clear();

        self.neighbors_nb
            .water_o_posits_last_rebuild
            .extend(self.water.iter().map(|w| self.cell.wrap(w.o.posit)));

        self.neighbors_nb.max_displacement_sq = 0.0;
    }

    /// This rebuilds all neighbor lists.
    pub(crate) fn build_all_neighbors(&mut self, dev: &ComputationDevice) {
        let atom_posits: Vec<_> = self.atoms.iter().map(|a| a.posit).collect();
        let water_posits: Vec<_> = self
            .water
            .iter()
            .map(|m| self.cell.wrap(m.o.posit))
            .collect();

        // Compute a static mask. We use this to prevent building static-static neighbors; this is an
        // optimization, as their displacement will always be 0.
        let is_static: Vec<_> = self.atoms.iter().map(|a| a.static_).collect();

        self.neighbors_nb.std_std = build_neighbors(
            &atom_posits,
            &atom_posits,
            Some(&is_static),
            &self.cell,
            true,
            self.neighbors_nb.skin_sq_w_cutoff,
        );

        self.neighbors_nb.std_water = build_neighbors(
            &atom_posits,
            &water_posits,
            None,
            &self.cell,
            false,
            self.neighbors_nb.skin_sq_w_cutoff,
        );

        self.neighbors_nb.water_water = build_neighbors(
            &water_posits,
            &water_posits,
            None,
            &self.cell,
            true,
            self.neighbors_nb.skin_sq_w_cutoff,
        );

        self.setup_pairs();

        #[cfg(feature = "cuda")]
        if let ComputationDevice::Gpu((modules)) = dev {
            self.per_neighbor_gpu = Some(PerNeighborGpu::new(
                &modules.stream,
                &self.nb_pairs,
                &self.atoms,
                &self.water,
                &self.lj_tables,
            ));
        }

        // Refresh refs and reset displacement
        self.save_rebuild_posits();
    }

    /// Call during each step; determines if we need to rebuild neighbors, and if so, do it.
    pub(crate) fn build_neighbors_if_needed(&mut self, dev: &ComputationDevice) {
        if self.neighbors_nb.max_displacement_sq >= self.neighbors_nb.half_skin_sq {
            let start = Instant::now();

            self.build_all_neighbors(dev);
            self.computation_time.neighbor_rebuild_count += 1;

            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.neighbor_rebuild_sum += elapsed;
        }
    }
}

/// [Re]build a neighbor list, used for non-bonded interactions. Run this periodically.
/// The static mask both prevents computing distance for re-build here, and prevents
/// running unnecessary non-bonded calculations downstream.
///
/// Result Outer index: target atoms. Inner: Source atoms that are within our cutoff distance.
/// These get converted to pairs, and passed to the GPU or CPU.
///
/// Dynamic nodes will include static neighbors, and static nodes will have empty lists.
pub fn build_neighbors(
    posits_outer: &[Vec3],
    posits_inner: &[Vec3],
    // This helps us skip static-static rebuilds. Symmetric only. Indices must match source and tgt posits.
    is_static: Option<&[bool]>,
    cell: &SimBox,
    symmetric: bool,
    skin_sq_w_cutoff: f32,
) -> Vec<Vec<usize>> {
    let outer_len = posits_outer.len();
    let inner_len = posits_inner.len();

    if is_static.is_some() && !symmetric {
        panic!("Invalid neighbor build config; can't pass static indices if non-symmetric.")
    }

    if symmetric {
        assert_eq!(
            inner_len, outer_len,
            "symmetric=true requires identical sets"
        );
        let n = outer_len;

        let half: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i_outer| {
                let mut out = Vec::new();
                let posit_tgt = posits_outer[i_outer];
                for i_inner in (i_outer + 1)..n {
                    // Skip this computation for static-static. Note that we have downstream
                    // ways to prevent the actual NB computations between static-static.
                    let mut st_st = false;

                    if let Some(st) = is_static {
                        if st[i_outer] && st[i_inner] {
                            st_st = true;
                        }
                    }

                    if !st_st {
                        let d = cell.min_image(posit_tgt - posits_inner[i_inner]);
                        if d.magnitude_squared() < skin_sq_w_cutoff {
                            out.push(i_inner);
                        }
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
        (0..outer_len)
            .into_par_iter()
            .map(|i_outer| {
                let mut out = Vec::new();
                let pos_outer = posits_outer[i_outer];

                for i_inner in 0..inner_len {
                    // Skip this computation for static-static. Note that we have downstream
                    // ways to prevent the actual NB computations between static-static.
                    let mut st_st = false;

                    if let Some(st) = is_static {
                        if st[i_outer] && st[i_inner] {
                            st_st = true;
                        }
                    }

                    if !st_st {
                        let pos_inner = posits_inner[i_inner];
                        // todo: Only take the min image for water?
                        let d = cell.min_image(pos_outer - pos_inner);
                        if d.magnitude_squared() < skin_sq_w_cutoff {
                            out.push(i_inner);
                        }
                    }
                }
                out
            })
            .collect()
    }
}
