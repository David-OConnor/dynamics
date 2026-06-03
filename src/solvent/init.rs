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
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    AtomDynamics, ComputationDevice, MdState, Solvent,
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
const NUM_EQUILIBRATION_STEPS_OTHER_SOLVENT: usize = 600;

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

/// Contains variants of templates we have built into this library. These are
/// included in the binary of applications which use this.
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

    pub fn from_parts(
        o_posits: Vec<Vec3>,
        h0_posits: Vec<Vec3>,
        h1_posits: Vec<Vec3>,
        o_velocities: Vec<Vec3>,
        h0_velocities: Vec<Vec3>,
        h1_velocities: Vec<Vec3>,
        cell: SimBox,
    ) -> io::Result<Self> {
        let n = o_posits.len();
        let lengths = [
            h0_posits.len(),
            h1_posits.len(),
            o_velocities.len(),
            h0_velocities.len(),
            h1_velocities.len(),
        ];

        if lengths.iter().any(|len| *len != n) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WaterInitTemplate component lengths must match.",
            ));
        }

        Ok(Self {
            o_posits,
            h0_posits,
            h1_posits,
            o_velocities,
            h0_velocities,
            h1_velocities,
            cell,
        })
    }

    pub fn len(&self) -> usize {
        self.o_posits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.o_posits.is_empty()
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

    pub fn from_water_mols(water: &[WaterMolOpc], cell: SimBox) -> io::Result<Self> {
        let n = water.len();

        let mut o_posits = Vec::with_capacity(n);
        let mut h0_posits = Vec::with_capacity(n);
        let mut h1_posits = Vec::with_capacity(n);

        let mut o_velocities = Vec::with_capacity(n);
        let mut h0_velocities = Vec::with_capacity(n);
        let mut h1_velocities = Vec::with_capacity(n);

        for mol in water {
            o_posits.push(mol.o.posit);
            h0_posits.push(mol.h0.posit);
            h1_posits.push(mol.h1.posit);

            o_velocities.push(mol.o.vel);
            h0_velocities.push(mol.h0.vel);
            h1_velocities.push(mol.h1.vel);
        }

        Self::from_parts(
            o_posits,
            h0_posits,
            h1_posits,
            o_velocities,
            h0_velocities,
            h1_velocities,
            cell,
        )
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
    // The solute is used for deconfliction and default molecule-count estimation.
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
    match water_mols_from_template_in_region(
        cell,
        cell,
        solute,
        specify_num_water,
        template_type,
        skip_pbc_filter,
    ) {
        Ok(water) => water,
        Err(e) => {
            eprintln!("\nError initializing water: {e}");
            Vec::new()
        }
    }
}

fn validate_positive_cell(label: &str, cell: &SimBox) -> io::Result<()> {
    if cell.extent.x.is_finite()
        && cell.extent.y.is_finite()
        && cell.extent.z.is_finite()
        && cell.extent.x > 0.0
        && cell.extent.y > 0.0
        && cell.extent.z > 0.0
    {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{label} must have positive finite dimensions; got low={:?}, high={:?}.",
                cell.bounds_low, cell.bounds_high
            ),
        ))
    }
}

/// Create water molecules from a template inside a rectangular sub-region of a
/// larger simulation cell.
///
/// `region` and `cell` are in the same coordinate frame. Molecules are accepted
/// only when their O/H sites are inside `region`, while solute and water-water
/// conflict checks use `cell` for minimum-image distances. This is useful for
/// slabs and other partial-cell solvent layouts where bulk solvation would fill
/// the whole simulation box.
pub fn water_mols_from_template_in_region(
    cell: &SimBox,
    region: &SimBox,
    solute: &[AtomDynamics],
    specify_num_water: Option<usize>,
    template_type: &SolventTemplateType,
    skip_pbc_filter: bool,
) -> io::Result<Vec<WaterMolOpc>> {
    water_mols_from_template_in_region_avoiding(
        cell,
        region,
        solute,
        &[],
        specify_num_water,
        template_type,
        skip_pbc_filter,
    )
}

#[derive(Clone, Copy)]
struct WaterTemplateCandidate {
    tile: (i32, i32, i32),
    o_posit: Vec3,
    h0_posit: Vec3,
    h1_posit: Vec3,
    o_velocity: Vec3,
    h0_velocity: Vec3,
    h1_velocity: Vec3,
}

fn candidate_fits_region(
    candidate: &WaterTemplateCandidate,
    cell: &SimBox,
    region: &SimBox,
) -> bool {
    [candidate.o_posit, candidate.h0_posit, candidate.h1_posit]
        .into_iter()
        .all(|posit| region.contains(posit) && cell.contains(posit))
}

fn spatially_interleave_candidates(
    candidates: Vec<WaterTemplateCandidate>,
    region: &SimBox,
    desired_count: usize,
) -> Vec<WaterTemplateCandidate> {
    let bins_per_axis = (desired_count as f32).cbrt().ceil() as usize;
    let mut bins = vec![Vec::new(); bins_per_axis.pow(3)];
    let bin_coord = |value: f32, low: f32, extent: f32| {
        (((value - low) / extent * bins_per_axis as f32).floor() as usize).min(bins_per_axis - 1)
    };

    for candidate in candidates {
        let x = bin_coord(candidate.o_posit.x, region.bounds_low.x, region.extent.x);
        let y = bin_coord(candidate.o_posit.y, region.bounds_low.y, region.extent.y);
        let z = bin_coord(candidate.o_posit.z, region.bounds_low.z, region.extent.z);
        bins[x + bins_per_axis * (y + bins_per_axis * z)].push(candidate);
    }

    let max_bin_len = bins.iter().map(Vec::len).max().unwrap_or(0);
    let mut bin_order: Vec<_> = (0..bins.len()).collect();
    bin_order.sort_unstable_by_key(|index| (*index as u32).wrapping_mul(0x9E37_79B9));
    let mut result = Vec::new();

    for round in 0..max_bin_len {
        for &bin_i in &bin_order {
            let bin = &bins[bin_i];
            if let Some(candidate) = bin.get(round) {
                result.push(*candidate);
            }
        }
    }

    result
}

pub(crate) fn water_mols_from_template_in_region_avoiding(
    cell: &SimBox,
    region: &SimBox,
    solute: &[AtomDynamics],
    prior_water: &[WaterMolOpc],
    specify_num_water: Option<usize>,
    template_type: &SolventTemplateType,
    skip_pbc_filter: bool,
) -> io::Result<Vec<WaterMolOpc>> {
    validate_positive_cell("Simulation cell", cell)?;
    validate_positive_cell("Water placement region", region)?;

    if !cell.contains(region.bounds_low) || !cell.contains(region.bounds_high) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Water placement region must be fully inside the simulation cell.",
        ));
    }

    println!("Initializing solvent molecules...");
    let start = Instant::now();

    let template = template_type.get_template()?;
    validate_positive_cell("Water template cell", &template.cell)?;

    let solute_for_count: Vec<_> = if cell == region {
        Vec::new()
    } else {
        solute
            .iter()
            .filter(|atom| region.contains(atom.posit))
            .cloned()
            .collect()
    };
    let n_mols = specify_num_water.unwrap_or_else(|| {
        if cell == region {
            n_water_mols(region, solute)
        } else {
            n_water_mols(region, &solute_for_count)
        }
    });
    let mut result = Vec::with_capacity(n_mols);

    if n_mols == 0 {
        println!("Complete in {} ms.", start.elapsed().as_millis());
        return Ok(result);
    }

    let solute_posits: Vec<_> = solute.iter().map(|a| a.posit).collect();

    let template_size = template.cell.extent;
    let template_ctr = template.cell.center();

    let region_ctr = region.center();

    // Align tile (0,0,0) center to the placement region center.
    let base_offset = region_ctr - template_ctr;

    // Number of half-tiles needed to cover the cell in each direction (+1 for safety).
    let region_size = region.extent;
    let half_x = (region_size.x / (2.0 * template_size.x)).ceil() as i32 + 1;
    let half_y = (region_size.y / (2.0 * template_size.y)).ceil() as i32 + 1;
    let half_z = (region_size.z / (2.0 * template_size.z)).ceil() as i32 + 1;

    let mut loops_used = 0;

    // Parallel vecs: tile index of each molecule placed in `result`.
    // Used to restrict the PBC soft filter to same-tile pairs (see comment below).
    let mut placed_tiles: Vec<(i32, i32, i32)> = Vec::with_capacity(n_mols);

    let make_candidate =
        |tile: (i32, i32, i32), tile_offset: Vec3, i: usize| WaterTemplateCandidate {
            tile,
            o_posit: template.o_posits[i] + tile_offset,
            h0_posit: template.h0_posits[i] + tile_offset,
            h1_posit: template.h1_posits[i] + tile_offset,
            o_velocity: template.o_velocities[i],
            h0_velocity: template.h0_velocities[i],
            h1_velocity: template.h1_velocities[i],
        };

    let mut place_candidate = |candidate: WaterTemplateCandidate| {
        if !candidate_fits_region(&candidate, cell, region) {
            return false;
        }

        // Conflict with solute atoms.
        for &atom_p in &solute_posits {
            if cell
                .min_image(atom_p - candidate.o_posit)
                .magnitude_squared()
                < MIN_NONWATER_DIST_SQ
            {
                return false;
            }
        }

        // Regions are populated independently, so their template phases may not line
        // up. Keep new molecules outside the first-shell distance of waters accepted
        // for earlier regions to avoid artificial high-energy contacts at boundaries.
        for w in prior_water {
            let diff = cell.min_image(w.o.posit - candidate.o_posit);
            if diff.magnitude_squared() < PBC_MIN_WATER_O_O_DIST_SQ {
                return false;
            }
        }

        // Conflict with already-placed solvent.
        //
        // Hard overlap (1.7 Å): always reject, regardless of PBC or tile.
        //
        // PBC soft filter (2.8 Å): only applies to SAME-TILE pairs.
        //   - Large-template (e.g. Water60A, 60 Å) in a small cell: only one tile
        //     contributes, so same-tile molecules from opposite ends of the template
        //     can land at unequilibrated PBC distances of 2-3 Å. The filter rejects
        //     them.
        //   - Small-template (e.g. tip4p, 18.68 Å) in a large cell: adjacent tiles
        //     tile the cell; cross-tile molecules near the cell boundary are legitimate
        //     PBC neighbours equilibrated in the template. Applying the filter to
        //     cross-tile pairs incorrectly rejects about 10 percent of molecules.
        for (j, w) in result.iter().enumerate() {
            let diff = w.o.posit - candidate.o_posit;
            let direct_sq = diff.magnitude_squared();
            if direct_sq < MIN_WATER_O_O_DIST_SQ {
                return false;
            }
            let min_image_sq = cell.min_image(diff).magnitude_squared();
            // Always reject PBC hard overlaps (PBC distance < 1.7 Å) even when
            // skip_pbc_filter is true, to prevent catastrophic initial forces.
            if min_image_sq < MIN_WATER_O_O_DIST_SQ {
                return false;
            }
            if !skip_pbc_filter && placed_tiles[j] == candidate.tile {
                if min_image_sq < PBC_MIN_WATER_O_O_DIST_SQ && min_image_sq < direct_sq {
                    return false;
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

        mol.o.posit = candidate.o_posit;
        mol.h0.posit = candidate.h0_posit;
        mol.h1.posit = candidate.h1_posit;

        mol.o.vel = candidate.o_velocity;
        mol.h0.vel = candidate.h0_velocity;
        mol.h1.vel = candidate.h1_velocity;
        mol.update_virtual_site();

        result.push(mol);
        placed_tiles.push(candidate.tile);
        result.len() == n_mols
    };

    if specify_num_water.is_some() {
        let mut candidates = Vec::new();

        for ix in -half_x..=half_x {
            for iy in -half_y..=half_y {
                for iz in -half_z..=half_z {
                    let tile = (ix, iy, iz);
                    let tile_offset = base_offset
                        + Vec3::new(
                            ix as f32 * template_size.x,
                            iy as f32 * template_size.y,
                            iz as f32 * template_size.z,
                        );

                    for i in 0..template.o_posits.len() {
                        loops_used += 1;
                        let candidate = make_candidate(tile, tile_offset, i);
                        if candidate_fits_region(&candidate, cell, region) {
                            candidates.push(candidate);
                        }
                    }
                }
            }
        }

        for candidate in spatially_interleave_candidates(candidates, region, n_mols) {
            if place_candidate(candidate) {
                break;
            }
        }
    } else {
        'tiles: for ix in -half_x..=half_x {
            for iy in -half_y..=half_y {
                for iz in -half_z..=half_z {
                    let tile = (ix, iy, iz);
                    let tile_offset = base_offset
                        + Vec3::new(
                            ix as f32 * template_size.x,
                            iy as f32 * template_size.y,
                            iz as f32 * template_size.z,
                        );

                    for i in 0..template.o_posits.len() {
                        loops_used += 1;
                        if place_candidate(make_candidate(tile, tile_offset, i)) {
                            break 'tiles;
                        }
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

    Ok(result)
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
            Solvent::WaterOpc
            | Solvent::WaterOpcSpecifyMolCount(_)
            | Solvent::WaterOpcCustomRegions(_) => NUM_EQUILIBRATION_STEPS_WATER,
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
