#![allow(clippy::excessive_precision)]

//! Code for initializing solvent molecules, including assigning quantity, initial positions, and
//! velocities. Set up to meet density, pressure, and or temperature targets. Not specific to the
//! solvent model used. Populates simulation boxes with solvents, with a PBC assumption.
//!
//! This involves creating, saving and loading templates, and generating water molecules given a template,
//! sim box, and solute.

use std::{fs, io, path::Path, time::Instant};

use bincode::{Decode, Encode};
use bio_files::{gromacs, gromacs::gro::Gro};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::{Quaternion as QuaternionF64, Vec3 as Vec3F64},
};
use rand::Rng;

use crate::{
    AtomDynamics, ComputationDevice, MdState, MolDynamics, Solvent,
    barostat::SimBox,
    partial_charge_inference::{files::load_from_bytes_bincode, save},
    sa_surface,
    solvent::WaterMolOpc,
};
// 0.997 g cm⁻³ is a good default density for biological pressures. We use this for initializing
// and maintaining the solvent density and molecule count.
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

// Don't generate solvent molecules that are too close to other atoms.
// Vdw contact distance between solvent molecules and organic molecules is roughly 3.5 Å.
// todo: Hmm. We could get lower, but there's some risk of an H atom being too close,
// todo and we're currently only measuring solvent o dist to atoms.
const MIN_NONWATER_DIST: f32 = 1.7;
const MIN_NONWATER_DIST_SQ: f32 = MIN_NONWATER_DIST * MIN_NONWATER_DIST;

// Direct O-O overlap check — prevents truly coincident molecules.
const MIN_WATER_O_O_DIST: f32 = 1.7;
pub(in crate::solvent) const MIN_WATER_O_O_DIST_SQ: f32 = MIN_WATER_O_O_DIST * MIN_WATER_O_O_DIST;

// PBC-boundary exclusion distance.
// When a smaller box is filled from a larger template (e.g. 30 Å from a 60 Å template),
// molecules near opposite faces become PBC neighbours even though they were ~30 Å apart in
// the template and were never equilibrated at that short distance.  These pairs can slip
// through the 1.7 Å check with PBC distances of 2.0–2.8 Å, where LJ energy is 6–74 kcal/mol
// per pair — enough to push pressure into the tens-of-thousands-of-bar range.
// 2.8 Å is just below the first RDF peak (~2.82 Å).  We only apply this stricter threshold
// when PBC wrapping actually shortens the distance (i.e. `min_image_dist < direct_dist`),
// so interior template molecules at their natural 2.5–2.8 Å first-shell distances are
// accepted while un-equilibrated cross-boundary pairs are rejected.
const PBC_MIN_WATER_O_O_DIST: f32 = 2.8;
const PBC_MIN_WATER_O_O_DIST_SQ: f32 = PBC_MIN_WATER_O_O_DIST * PBC_MIN_WATER_O_O_DIST;

// Higher is more accurate, but slower. After hydrogen bond networks are settled, higher doensn't
// improve things. Note that we initialize from a pre-equilibrated template, so we shouldn't
// need many effects. This mainly deals with template tiling effects, and solvent-solute conflicts.
const NUM_EQUILIBRATION_STEPS_WATER: usize = 200;
// todo: Experimenting.
const NUM_EQUILIBRATION_STEPS_OTHER_SOLVENT: usize = 200;

// Like in our normal setup with constraint H, 0.002ps may be the safe upper bound.
// We seem to get better settling results with a low dt.
const DT_EQUILIBRATION: f32 = 0.0005;

// We generate and use this externally, for example, when passing it to GROMACS in Molchanica.
pub const WATER_TEMPLATE_60A: &[u8] =
    include_bytes!("../../param_data/water_60A.water_init_template");

// Included with GROMACS. 4-point water model. 30Å per side?
pub const WATER_TEMPLATE_TIP4: &str = include_str!("../../param_data/tip4p.gro");
// We generated this using a shrinking box.
pub const OCTANOL_WATER_TEMPLATE: &str =
    include_str!("../../param_data/octanol_water_saturated.gro");

#[derive(Clone, Debug, PartialEq, Default, Decode, Encode)]
pub enum SolventTemplateType {
    Water60A,
    /// Also usable for any other 4-pt water model. Note that currently we discard the M
    /// site, adding it manually; consider using the TIP3 template instead, as it's slightly smaller.
    #[default]
    Tip4Gromacs,
    /// Octanol saturated with water at 300C and 1 bar.
    /// 46Å per side box. 356 octanol mols, 135 water mols.
    OctanolWithWater,
    Custom(WaterInitTemplate),
}

impl SolventTemplateType {
    pub fn get_template(&self) -> io::Result<WaterInitTemplate> {
        match self {
            Self::Water60A => load_from_bytes_bincode(WATER_TEMPLATE_60A),
            Self::Tip4Gromacs => WaterInitTemplate::from_gro(WATER_TEMPLATE_TIP4),
            // Currently this method is only for WaterInitTemplate; it's not general-purpose.
            Self::OctanolWithWater => Ok(Default::default()),
            Self::Custom(t) => Ok(t.clone()),
        }
    }
}

/// For 3 and 4 point water models.
///
/// We store pre-equilibrated solvent molecules in a template, and use it to initialize solvent for a simulation.
/// This keeps the equilibration steps relatively low. Note that edge effects from tiling will require
/// equilibration, as well as adjusting a template for the runtime temperature target.
///
/// Struct-of-array layout. (Corresponding indices)
/// Public so it can be created by the application after a run.
///
/// 108 bytes/mol. Size on disk/mem: for a 60Å side len: ~780kb. (Hmm: We're getting a bit less)
/// 80Å/side: 1.20Mb.
///
/// M/EP positions are not included: They can be inferred after.
#[derive(Clone, Debug, PartialEq, Default, Encode, Decode)]
pub struct WaterInitTemplate {
    // velocity is o velocity, instead of 3 separate velocities
    o_posits: Vec<Vec3>,
    h0_posits: Vec<Vec3>,
    h1_posits: Vec<Vec3>,
    o_velocities: Vec<Vec3>,
    h0_velocities: Vec<Vec3>,
    h1_velocities: Vec<Vec3>,
    /// One corner; the opposite. This must correspond to the positions.
    cell: SimBox,
}

impl WaterInitTemplate {
    /// Load a previously-saved template from a file path. Currently, this saves using bincode.
    /// todo: Make it save as a .gro as well.
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        load_from_bytes_bincode(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        load_from_bytes_bincode(bytes)
    }

    pub fn from_gro(gro_text: &str) -> io::Result<Self> {
        const NM_TO_ANGSTROM: f32 = 10.;

        let gro = Gro::new(gro_text)?;

        let mut o_posits = Vec::new();
        let mut h0_posits = Vec::new();
        let mut h1_posits = Vec::new();

        let mut o_velocities = Vec::new();
        let mut h0_velocities = Vec::new();
        let mut h1_velocities = Vec::new();

        for atom in gro.atoms {
            match atom.atom_type.as_ref() {
                "OW" => {
                    let p: Vec3 = atom.posit.into();
                    o_posits.push(p * NM_TO_ANGSTROM);
                    let Some(vel) = &atom.velocity else {
                        return Err(io::Error::other("Missing velocity on tip4 water template"));
                    };
                    let v: Vec3 = (*vel).into();
                    o_velocities.push(v * NM_TO_ANGSTROM.into());
                }
                "HW1" => {
                    let p: Vec3 = atom.posit.into();
                    h0_posits.push(p * NM_TO_ANGSTROM);
                    let Some(vel) = &atom.velocity else {
                        return Err(io::Error::other("Missing velocity on tip4 water template"));
                    };
                    let v: Vec3 = (*vel).into();
                    h0_velocities.push(v * NM_TO_ANGSTROM);
                }
                "HW2" => {
                    let p: Vec3 = atom.posit.into();
                    h1_posits.push(p * NM_TO_ANGSTROM);
                    let Some(vel) = &atom.velocity else {
                        return Err(io::Error::other("Missing velocity on tip4 water template"));
                    };
                    let v: Vec3 = (*vel).into();
                    h1_velocities.push(v * NM_TO_ANGSTROM);
                }
                _ => (),
            }
        }

        Ok(Self {
            o_posits,
            h0_posits,
            h1_posits,
            o_velocities,
            h0_velocities,
            h1_velocities,
            cell: SimBox::new(
                (-gro.box_vec * NM_TO_ANGSTROM as f64 / 2.).into(),
                (gro.box_vec * NM_TO_ANGSTROM as f64 / 2.).into(),
            ), // todo: Is this right??
        })
    }

    /// Construct from the current state, and save to file.
    /// Call this explicitly. (todo: Determine a formal or informal approach)
    pub fn create_and_save(water: &[WaterMolOpc], cell: SimBox, path: &Path) -> io::Result<()> {
        let n = water.len();

        let mut o_posits = Vec::with_capacity(n);
        let mut h0_posits = Vec::with_capacity(n);
        let mut h1_posits = Vec::with_capacity(n);

        let mut o_velocities = Vec::with_capacity(n);
        let mut h0_velocities = Vec::with_capacity(n);
        let mut h1_velocities = Vec::with_capacity(n);

        // let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));

        // Sort solvent by position so that it iterates out from the center. This makes initialization
        // easier for cases where this template is larger than the target sim box.
        let water = {
            let ctr = cell.center();

            let mut w = water.to_vec();
            w.sort_by(|a, b| {
                let da = (a.o.posit - ctr).magnitude_squared();
                let db = (b.o.posit - ctr).magnitude_squared();
                da.total_cmp(&db)
            });

            w
        };

        for mol in water {
            o_posits.push(mol.o.posit);
            h0_posits.push(mol.h0.posit);
            h1_posits.push(mol.h1.posit);

            o_velocities.push(mol.o.vel);
            h0_velocities.push(mol.h0.vel);
            h1_velocities.push(mol.h1.vel);

            // min = min.min(mol.o.posit);
            // max = max.max(mol.o.posit);
        }

        let result = Self {
            o_posits,
            h0_posits,
            h1_posits,
            o_velocities,
            h0_velocities,
            h1_velocities,
            cell,
        };

        save(path, &result)
    }

    // todo: Identical structs; we could consolidate.
    /// Note: `gmx solvate`` handles tiling, centering, and solute deconfliction; we can
    /// send it the raw template.
    pub fn to_gromacs(&self) -> gromacs::solvate::WaterInitTemplate {
        gromacs::solvate::WaterInitTemplate {
            o_posits: self.o_posits.clone(),
            h0_posits: self.h0_posits.clone(),
            h1_posits: self.h1_posits.clone(),
            o_velocities: self.o_velocities.clone(),
            h0_velocities: self.h0_velocities.clone(),
            h1_velocities: self.h1_velocities.clone(),
            bounds: (self.cell.bounds_low, self.cell.bounds_high),
        }
    }
}

/// Determine the number of solvent molecules to add, based on box size and solute.
pub(in crate::solvent) fn n_water_mols(cell: &SimBox, solute_atoms: &[AtomDynamics]) -> usize {
    let cell_volume = cell.volume();
    let mol_volume = sa_surface::vol_take_up_by_atoms(solute_atoms);
    let free_vol = cell_volume - mol_volume;

    let dims = format!(
        "{}:.2 x {:.2} x {:.2}",
        (cell.bounds_high.x - cell.bounds_low.x).abs(),
        (cell.bounds_high.y - cell.bounds_low.y).abs(),
        (cell.bounds_high.z - cell.bounds_low.z).abs()
    );

    println!(
        "Solvent-free vol: {:.2} Cell vol: {:.2} (Å³ / 1,000). Dims: {dims} Å",
        free_vol / 1_000.,
        cell_volume / 1_000.
    );

    // Estimate free volume & n_mols from it
    (WATER_MOLS_PER_VOL * free_vol).round() as usize
}

/// Create solvent molecules from a template, tiling it as many times as needed to fill the cell.
/// Works for any cell size: smaller than, equal to, or larger than the template. Deconflcits with
/// solute molecules, and adds the proper amount based on the free volume (Volume of the cell not
/// taken up by solute).
///
/// The template is always centered on the cell. For cells smaller than the template only tile
/// (0,0,0) contributes; for larger cells neighboring tiles fill in the rest.
/// Water-solvent conflict detection uses min-image distances so molecules are never placed too
/// close to a PBC image of an already-placed molecule.
///
/// `template_override`: if provided, use this template instead of the built-in 60 Å water one.
pub fn water_mols_from_template(
    cell: &SimBox,
    solute: &[AtomDynamics],
    specify_num_water: Option<usize>,
    template_type: &SolventTemplateType,
    // When true, skip the PBC-boundary proximity check (the 2.8 Å same-tile cross-boundary
    // filter) entirely.  Only the hard-overlap 1.7 Å direct/PBC-distance check remains.
    // Use when generating a template at the correct equilibrium density: same-tile boundary
    // molecules that the filter would reject are acceptable starting points; the MD
    // equilibration run will push them to their natural first-shell distances.
    skip_pbc_filter: bool,
) -> Vec<WaterMolOpc> {
    println!("Initializing solvent molecules...");
    let start = Instant::now();

    let template = match template_type.get_template() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("\nError initializing water; can't read the template: {e}");
            return Vec::new();
        }
    };

    let n_mols = specify_num_water.unwrap_or_else(|| n_water_mols(cell, solute));
    let mut result = Vec::with_capacity(n_mols);

    if n_mols == 0 {
        println!("Complete in {} ms.", start.elapsed().as_millis());
        return result;
    }

    let solute_posits: Vec<_> = solute.iter().map(|a| a.posit).collect();

    let template_size = template.cell.extent;
    let template_ctr = template.cell.center();

    let cell_ctr = (cell.bounds_low + cell.bounds_high) / 2.;

    // Align tile (0,0,0) center to the cell center.
    let base_offset = cell_ctr - template_ctr;

    // Number of half-tiles needed to cover the cell in each direction (+1 for safety).
    let cell_size = cell.bounds_high - cell.bounds_low;
    let half_x = (cell_size.x / (2.0 * template_size.x)).ceil() as i32 + 1;
    let half_y = (cell_size.y / (2.0 * template_size.y)).ceil() as i32 + 1;
    let half_z = (cell_size.z / (2.0 * template_size.z)).ceil() as i32 + 1;

    let mut loops_used = 0;

    // Parallel vecs: tile index of each molecule placed in `result`.
    // Used to restrict the PBC soft filter to same-tile pairs (see comment below).
    let mut placed_tiles: Vec<(i32, i32, i32)> = Vec::with_capacity(n_mols);

    'tiles: for ix in -half_x..=half_x {
        for iy in -half_y..=half_y {
            for iz in -half_z..=half_z {
                let tile_offset = base_offset
                    + Vec3::new(
                        ix as f32 * template_size.x,
                        iy as f32 * template_size.y,
                        iz as f32 * template_size.z,
                    );

                'mol: for i in 0..template.o_posits.len() {
                    let o_posit = template.o_posits[i] + tile_offset;
                    let h0_posit = template.h0_posits[i] + tile_offset;
                    let h1_posit = template.h1_posits[i] + tile_offset;

                    loops_used += 1;

                    if !cell.contains(o_posit) {
                        continue;
                    }

                    // Conflict with solute atoms.
                    for &atom_p in &solute_posits {
                        if (atom_p - o_posit).magnitude_squared() < MIN_NONWATER_DIST_SQ {
                            continue 'mol;
                        }
                    }

                    // Conflict with already-placed solvent.
                    //
                    // Hard overlap (1.7 Å): always reject, regardless of PBC or tile.
                    //
                    // PBC soft filter (2.8 Å): only applies to SAME-TILE pairs.
                    //   - Large-template (e.g. Water60A, 60 Å) in a small cell: only one tile
                    //     contributes, so same-tile molecules from opposite ends of the template
                    //     can land at unequilibrated PBC distances of 2–3 Å.  The filter rejects
                    //     them. ✓
                    //   - Small-template (e.g. tip4p, 18.68 Å) in a large cell: adjacent tiles
                    //     tile the cell; cross-tile molecules near the cell boundary ARE legitimate
                    //     PBC neighbours (equilibrated as such in the template).  Applying the
                    //     filter to cross-tile pairs incorrectly rejects ~10 % of molecules.
                    //     Restricting it to same-tile pairs prevents this over-rejection. ✓
                    for (j, w) in result.iter().enumerate() {
                        let diff = w.o.posit - o_posit;
                        let direct_sq = diff.magnitude_squared();
                        if direct_sq < MIN_WATER_O_O_DIST_SQ {
                            continue 'mol;
                        }
                        let min_image_sq = cell.min_image(diff).magnitude_squared();
                        // Always reject PBC hard overlaps (PBC distance < 1.7 Å) even when
                        // skip_pbc_filter is true, to prevent catastrophic initial forces.
                        if min_image_sq < MIN_WATER_O_O_DIST_SQ {
                            continue 'mol;
                        }
                        if !skip_pbc_filter && placed_tiles[j] == (ix, iy, iz) {
                            if min_image_sq < PBC_MIN_WATER_O_O_DIST_SQ && min_image_sq < direct_sq
                            {
                                continue 'mol;
                            }
                        }
                    }

                    let mut mol = WaterMolOpc::new(
                        Vec3::new_zero(),
                        Vec3::new_zero(),
                        Quaternion::new_identity(),
                    );

                    // todo: I'm not sure how we're handling the M/EP point. I guess it's placed
                    // todo: automatically during integration.

                    mol.o.posit = o_posit;
                    mol.h0.posit = h0_posit;
                    mol.h1.posit = h1_posit;

                    mol.o.vel = template.o_velocities[i];
                    mol.h0.vel = template.h0_velocities[i];
                    mol.h1.vel = template.h1_velocities[i];

                    result.push(mol);
                    placed_tiles.push((ix, iy, iz));

                    if result.len() == n_mols {
                        break 'tiles;
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!(
        "Added {} / {n_mols} solvent mols in {elapsed} ms. Used {loops_used} loops",
        result.len()
    );

    result
}

/// Pack copies of each custom solvent molecule into the simulation box on a regular grid,
/// deconflicting with already-placed atoms (e.g. the solute). Returns one `MolDynamics` per
/// copy placed, each with `atom_posits` set to its chosen world-space positions.
///
/// Strategy (mirrors `make_water_mols_grid`):
/// - Divide the box into an n_x × n_y × n_z lattice with one slot per molecule.
/// - At each grid cell, try `MAX_ROT_ATTEMPTS` random orientations and keep the one whose
///   true minimum atom–atom distance to all already-placed atoms is largest.
/// - If the best orientation achieves ≥ `SOFT_MIN` (1.4 Å) we place immediately.
/// - If it achieves ≥ `HARD_MIN` (0.5 Å, the `check_for_overlaps_oob` cutoff) we place
///   with a warning; the energy minimiser resolves the soft overlap.
/// - If every orientation produces a < 0.5 Å overlap the grid cell is skipped; the
///   barostat will compensate for the slightly lower count.
///
/// **Critical**: the minimum-distance scan inside each rotation attempt does NOT break
/// early at the soft threshold.  The previous greedy implementation broke out of the scan
/// loop as soon as any atom pair was found at < 1.4 Å, which meant a hidden pair at
/// 0.289 Å could go undetected and corrupt the "best" candidate.  Here we scan all
/// nearby atom pairs to find the *true* minimum, breaking only on a catastrophically
/// small distance (< 0.01 Å) where further scanning is pointless.
///
/// Only supports `SimBoxInit::Fixed` boxes — the caller is responsible for ensuring this.
pub(crate) fn pack_custom_solvent(
    cell: SimBox,
    existing_posits: &[Vec3F64], // declared positions of already-placed atoms (e.g. solute)
    mols_solvent: &[(MolDynamics, usize)],
    // template: Option<Vec<AtomDynamics>>,
) -> Vec<MolDynamics> {
    // Minimum inter-atom distance that `check_for_overlaps_oob` enforces. Grid cells whose
    // best rotation falls below this are skipped rather than placed.
    const HARD_MIN_SQ: f64 = 0.5 * 0.5; // Å²
    // "Good enough" threshold: once the best rotation achieves this we stop trying more.
    const SOFT_MIN_SQ: f64 = 1.4 * 1.4; // Å²
    // Threshold below which further scanning of a rotation is pointless.
    const CATASTROPHIC_SQ: f64 = 0.01 * 0.01; // Å²
    // Keep every atom at least this far from each box face.
    const WALL_MARGIN: f64 = 0.6; // Å  (slightly > check_for_overlaps_oob's 0.5 Å limit)
    const MAX_ROT_ATTEMPTS: usize = 200;

    let mut rng = rand::rng();

    let box_ctr: Vec3F64 = cell.center().into();

    // Grows as copies are committed; starts with the solute atom positions.
    let mut placed_posits: Vec<Vec3F64> = existing_posits.to_vec();

    let mut result: Vec<MolDynamics> = Vec::new();

    for (mol, count) in mols_solvent {
        let count = *count;
        if count == 0 {
            continue;
        }

        // Template positions in world space; prefer atom_posits override.
        let template_world: Vec<Vec3F64> = if let Some(ap) = &mol.atom_posits {
            ap.clone()
        } else {
            mol.atoms.iter().map(|a| a.posit).collect()
        };

        let n_atoms = template_world.len();
        if n_atoms == 0 {
            continue;
        }

        // Centroid and centroid-relative locals.
        let centroid = template_world
            .iter()
            .fold(Vec3F64::new(0., 0., 0.), |s, &p| s + p)
            * (1.0 / n_atoms as f64);
        let local: Vec<Vec3F64> = template_world.iter().map(|&p| p - centroid).collect();
        let bounding_r: f64 = local.iter().map(|p| p.magnitude()).fold(0.0_f64, f64::max);

        // Spatial early-reject: only compare placed atoms within this squared distance of
        // the candidate centroid. The factor of 2 accounts for both molecules' extents.
        let search_sq = (bounding_r * 2.0 + 2.0).powi(2);

        // Box dimensions and per-atom wall half-extents.
        let lx = (cell.bounds_high.x - cell.bounds_low.x) as f64;
        let ly = (cell.bounds_high.y - cell.bounds_low.y) as f64;
        let lz = (cell.bounds_high.z - cell.bounds_low.z) as f64;
        let (hx, hy, hz) = (
            lx * 0.5 - WALL_MARGIN,
            ly * 0.5 - WALL_MARGIN,
            lz * 0.5 - WALL_MARGIN,
        );

        // Regular grid layout — same strategy as `make_water_mols_grid`.
        let base = (count as f64).cbrt().round().max(1.0) as usize;
        let n_x = base;
        let n_y = base;
        let n_z = count.div_ceil(n_x * n_y);
        let (sx, sy, sz) = (lx / n_x as f64, ly / n_y as f64, lz / n_z as f64);

        let total_cells = n_x * n_y * n_z;
        let mut num_placed = 0;
        let mut num_skipped = 0;

        'grid: for idx in 0..total_cells {
            if num_placed >= count {
                break;
            }

            let a = idx % n_x;
            let b = (idx / n_x) % n_y;
            let c = idx / (n_x * n_y);

            let world_ctr = Vec3F64::new(
                cell.bounds_low.x as f64 + (a as f64 + 0.5) * sx,
                cell.bounds_low.y as f64 + (b as f64 + 0.5) * sy,
                cell.bounds_low.z as f64 + (c as f64 + 0.5) * sz,
            );

            let mut best_min_sq = f64::NEG_INFINITY;
            let mut best_posits: Vec<Vec3F64> = Vec::new();

            'rot: for _ in 0..MAX_ROT_ATTEMPTS {
                let (w, x, y, z): (f64, f64, f64, f64) =
                    (rng.random(), rng.random(), rng.random(), rng.random());
                let rot = QuaternionF64::new(w, x, y, z).to_normalized();

                let new_posits: Vec<Vec3F64> = local
                    .iter()
                    .map(|&l| rot.rotate_vec(l) + world_ctr)
                    .collect();

                // Wall check in box-centred coordinates.
                if !new_posits.iter().all(|p| {
                    let dp = *p - box_ctr;
                    dp.x.abs() <= hx && dp.y.abs() <= hy && dp.z.abs() <= hz
                }) {
                    continue 'rot;
                }

                // Find the TRUE minimum atom–atom distance to all placed atoms.
                // We do NOT break early at the soft threshold — that was the source of the
                // bug where a hidden 0.289 Å pair went undetected. We only break on a
                // catastrophically small distance where no further scanning can help.
                let mut min_sq = f64::MAX;
                'pairs: for &np in &new_posits {
                    for &pp in &placed_posits {
                        // Spatial early-reject: skip atoms far from the candidate centroid.
                        let diff_ctr = pp - world_ctr;
                        let diff_ctr_f32: Vec3 = diff_ctr.into();
                        let pbc_ctr_f64: Vec3F64 = cell.min_image(diff_ctr_f32).into();
                        let dctr_sq = diff_ctr
                            .magnitude_squared()
                            .min(pbc_ctr_f64.magnitude_squared());
                        if dctr_sq > search_sq {
                            continue;
                        }

                        // PBC-aware inter-atom distance.
                        let diff = np - pp;
                        let diff_f32: Vec3 = diff.into();
                        let pbc_diff_f64: Vec3F64 = cell.min_image(diff_f32).into();
                        let dsq = diff
                            .magnitude_squared()
                            .min(pbc_diff_f64.magnitude_squared());

                        if dsq < min_sq {
                            min_sq = dsq;
                            if min_sq < CATASTROPHIC_SQ {
                                // Catastrophic overlap — no point scanning further.
                                break 'pairs;
                            }
                        }
                    }
                }

                if min_sq > best_min_sq {
                    best_min_sq = min_sq;
                    best_posits = new_posits;
                }
                if best_min_sq >= SOFT_MIN_SQ {
                    break 'rot; // Clean placement found; no need for more attempts.
                }
            }

            // Skip this cell if no rotation achieved a safe inter-atom distance.
            if best_posits.is_empty() || best_min_sq < HARD_MIN_SQ {
                num_skipped += 1;
                continue 'grid;
            }

            if best_min_sq < SOFT_MIN_SQ {
                eprintln!(
                    "pack_custom_solvent: mol {num_placed}: best min atom dist {:.2} Å — \
                     placing with soft overlap (energy minimiser will resolve).",
                    best_min_sq.sqrt()
                );
            }

            placed_posits.extend_from_slice(&best_posits);

            let mut mol_copy = mol.clone();
            mol_copy.atom_posits = Some(best_posits);
            result.push(mol_copy);
            num_placed += 1;
        }

        println!(
            "pack_custom_solvent: placed {num_placed} / {count} molecules \
             ({num_skipped} grid cells skipped — no clean orientation found).",
        );
    }

    result
}

impl MdState {
    fn mark_solute_static_for_init_relaxation(&mut self) -> Vec<bool> {
        let mut static_state = Vec::with_capacity(self.atoms.len());

        for (i, atom) in self.atoms.iter_mut().enumerate() {
            static_state.push(atom.static_);
            if i < self.solute_atom_count {
                atom.static_ = true;
            }
        }

        static_state
    }

    fn restore_static_state(&mut self, static_state: &[bool]) {
        for (atom, &was_static) in self.atoms.iter_mut().zip(static_state.iter()) {
            atom.static_ = was_static;
        }
    }

    /// Use this to help initialize solvent molecules to realistic geometry of hydrogen bond networks,
    /// prior to the first proper simulation step. Runs MD on solvent only.
    /// Make sure to only run this after state is properly initialized, e.g. towards the end
    /// of init; not immediately after populating waters.
    ///
    /// This will result in an immediate energy bump as solvent positions settle from their grid
    /// into position. As they settle, the thermostat will bring the velocities down to set
    /// the target temp. This sim should run long enough to the solvent is stable by the time
    /// the main sim starts.
    pub fn md_on_solute_only(&mut self, dev: &ComputationDevice) {
        println!("Initializing solvent structure prior to production MD...");
        let start = Instant::now();

        // This disables things like snapshot saving, and certain prints.
        self.solvent_only_sim_at_init = true;
        let thermo_dof_prev = self.thermo_dof;

        // Freeze the solute atoms while leaving explicit solvent atoms in `self.atoms`
        // free to relax alongside rigid OPC water.
        let static_state = self.mark_solute_static_for_init_relaxation();
        self.thermo_dof = self.dof_for_thermo();

        let steps = match self.cfg.solvent {
            Solvent::None => 0,
            Solvent::WaterOpc | Solvent::WaterOpcSpecifyMolCount(_) => {
                NUM_EQUILIBRATION_STEPS_WATER
            }
            Solvent::OctanolWithWater | Solvent::Custom(_) => NUM_EQUILIBRATION_STEPS_OTHER_SOLVENT,
        };

        for _ in 0..steps {
            self.step(dev, DT_EQUILIBRATION, None);
        }

        self.restore_static_state(&static_state);
        self.solvent_only_sim_at_init = false;
        self.thermo_dof = thermo_dof_prev;
        self.step_count = 0; // Reset.

        let elapsed = start.elapsed().as_millis();
        println!("Solvent initialization MD complete in {elapsed} ms");
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtomDynamics, MdState};

    #[test]
    fn init_relaxation_freezes_only_solute_atoms() {
        let mut state = MdState {
            atoms: vec![
                AtomDynamics {
                    static_: false,
                    ..Default::default()
                },
                AtomDynamics {
                    static_: false,
                    ..Default::default()
                },
                AtomDynamics {
                    static_: false,
                    ..Default::default()
                },
            ],
            solute_atom_count: 2,
            ..Default::default()
        };

        let static_state = state.mark_solute_static_for_init_relaxation();

        assert!(state.atoms[0].static_);
        assert!(state.atoms[1].static_);
        assert!(!state.atoms[2].static_);

        state.restore_static_state(&static_state);

        assert!(!state.atoms[0].static_);
        assert!(!state.atoms[1].static_);
        assert!(!state.atoms[2].static_);
    }

    #[test]
    fn thermo_dof_includes_dynamic_solvent_atoms_during_init_relaxation() {
        let mut state = MdState {
            atoms: vec![
                AtomDynamics {
                    static_: true,
                    ..Default::default()
                },
                AtomDynamics {
                    static_: false,
                    ..Default::default()
                },
            ],
            solvent_only_sim_at_init: true,
            ..Default::default()
        };

        assert_eq!(state.dof_for_thermo(), 3);
    }
}
