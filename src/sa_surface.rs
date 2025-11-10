//! This is a copy+paste + modify from dadedalus. We use it to compute the area
//! taken up by molecules when adding water, for the purpose of choosing how many water molecules
//! to add in a given volume


use lin_alg::f32::Vec3;
use mcubes::{MarchingCubes, MeshSide};

use crate::AtomDynamics;

const SOLVENT_RAD: f32 = 1.4; // water probe

// Lower means more precise. A higher value runs faster, but slightly underestimates volume.
const PRECISION: f32 = 0.5;

/// Find the volume taken up by atoms, e.g. the volume that we don't need to fill with solvent.
/// We subtract this from the simulation box volume when determining how many water
/// molecules to add.
pub fn vol_take_up_by_atoms(atoms: &[AtomDynamics]) -> f32 {
    if atoms.is_empty() {
        return 0.;
    }

    // Bounding box and grid
    let mut bb_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut bb_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut r_max: f32 = 0.0;
    
    for a in atoms {
        let r = a.element.vdw_radius() + SOLVENT_RAD;
        r_max = r_max.max(r);

        bb_min = Vec3::new(
            bb_min.x.min(a.posit.x ),
            bb_min.y.min(a.posit.y ),
            bb_min.z.min(a.posit.z ),
        );

        bb_max = Vec3::new(
            bb_max.x.max(a.posit.x ),
            bb_max.y.max(a.posit.y ),
            bb_max.z.max(a.posit.z ),
        );
    }
    bb_min -= Vec3::splat(r_max + PRECISION);
    bb_max += Vec3::splat(r_max + PRECISION);

    let dim_v = (bb_max - bb_min) / PRECISION;
    let grid_dim = (
        dim_v.x.ceil() as usize + 1,
        dim_v.y.ceil() as usize + 1,
        dim_v.z.ceil() as usize + 1,
    );

    let n_voxels = grid_dim.0 * grid_dim.1 * grid_dim.2;

    // This can be any that is guaranteed to be well outside the SAS surface.
    // It prevents holes from appearing in the mesh due to not having a value outside to compare to.
    let far_val = (r_max + PRECISION).powi(2) + 1.0;
    let mut field = vec![far_val; n_voxels];

    // Helper to flatten (x, y, z)
    let idx = |x: usize, y: usize, z: usize| -> usize { (z * grid_dim.1 + y) * grid_dim.0 + x };

    // Fill signed-squared-distance field
    for a in atoms {
        let center: Vec3 = a.posit.into();
        let rad = a.element.vdw_radius() + SOLVENT_RAD;
        let rad2 = rad * rad;

        let lo = ((center - Vec3::splat(rad)) - bb_min) / PRECISION;
        let hi = ((center + Vec3::splat(rad)) - bb_min) / PRECISION;

        let (xi0, yi0, zi0) = (
            lo.x.floor().max(0.0) as usize,
            lo.y.floor().max(0.0) as usize,
            lo.z.floor().max(0.0) as usize,
        );
        let (xi1, yi1, zi1) = (
            hi.x.ceil().min((grid_dim.0 - 1) as f32) as usize,
            hi.y.ceil().min((grid_dim.1 - 1) as f32) as usize,
            hi.z.ceil().min((grid_dim.2 - 1) as f32) as usize,
        );

        for z in zi0..=zi1 {
            for y in yi0..=yi1 {
                for x in xi0..=xi1 {
                    let p = bb_min + Vec3::new(x as f32, y as f32, z as f32) * PRECISION;
                    let d2 = (p - center).magnitude_squared();
                    let v = d2 - rad2;
                    let f = &mut field[idx(x, y, z)];
                    if v < *f {
                        *f = v;
                    }
                }
            }
        }
    }

    // Convert to a mesh using Marchine Cubes.
    let sampling_interval = (
        grid_dim.0 as f32 - 1.0,
        grid_dim.1 as f32 - 1.0,
        grid_dim.2 as f32 - 1.0,
    );

    //  scale = precision because size / sampling_interval = precision
    let size = (
        sampling_interval.0 * PRECISION,
        sampling_interval.1 * PRECISION,
        sampling_interval.2 * PRECISION,
    );

    // todo: The holes in our mesh seem related to the iso level chosen.
    let mc = MarchingCubes::new(grid_dim, size, sampling_interval, bb_min, field, 0.)
        .expect("marching cubes init");

    let mesh = mc.generate(MeshSide::InsideOnly);

    mesh.volume()
}
