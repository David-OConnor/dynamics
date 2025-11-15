use std::path::{Path, PathBuf};

use bio_files::{Mol2, md_params::ForceFieldParams};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, loss};
use rand::seq::SliceRandom;

use crate::param_inference::{
    MolGNN, AtomVocab,
    files::{GEOSTD_PATH, MODEL_PATH, VOCAB_PATH, find_paths},
    frcmod::{DIHEDRAL_FEATS, MAX_DIHEDRAL_TERMS},
    save,
};

const GAFF2: &str = include_str!("../../param_data/gaff2.dat");

// Higher = perhaps better training, but slower to train.
// todo: Try setting max of 50-100 epochs, and stop early A/R if val loss
// todo hasn't improved.
const N_EPOCHS: u8 = 50;
// Stop training if we have this many epochs without improvement.
const EARLY_STOPPING_PATIENCE: u8 = 7;

// Bigger hidden dim: more capacity to learn patterns, but slower and easier to overfit.
// Smaller hidden dim: faster, less capacity.
const HIDDEN_DIM: usize = 128; // todo: Try 256 as well, and compare.

// Higher learning rate: Faster, but can overshoot. Lower: Safer but slower.
// 1e-3 is a good default.
const LEARNING_RATE: f64 = 1e-3; // todo: What should this be? What is it?

pub struct Batch {
    pub elem_ids: Tensor,
    pub coords: Tensor,
    pub edge_index: Tensor,
    pub type_ids: Tensor,
    pub has_type: Tensor,
    pub charges: Tensor,
    // todo: Valence angle and bond stretching params too, although those are rare.
    //
    pub dihedral_index: Tensor,  // [n_dih, 4] i64, atom indices
    pub dihedral_params: Tensor, // [n_dih, MAX_DIHEDRAL_TERMS, DIH_FEATS]
    pub dihedral_mask: Tensor,   // [n_dih, MAX_DIHEDRAL_TERMS]
    //
    pub improper_index: Tensor,
    pub improper_params: Tensor,
    pub improper_mask: Tensor,
    //
    pub num_atoms: usize,
}

struct GeoStdMol2Dataset {
    mol2_paths: Vec<PathBuf>,
    frcmod_paths: Vec<PathBuf>,
    vocabs: AtomVocab,
    gaff2: ForceFieldParams,
}

impl GeoStdMol2Dataset {
    pub fn new(
        mol2_paths: &[PathBuf],
        frcmod_paths: &[PathBuf],
        vocabs: AtomVocab,
        gaff2: ForceFieldParams,
    ) -> Self {
        Self {
            mol2_paths: mol2_paths.to_vec(),
            frcmod_paths: frcmod_paths.to_vec(),
            vocabs,
            gaff2,
        }
    }

    pub fn len(&self) -> usize {
        self.mol2_paths.len()
    }

    pub fn get(&self, i_mol: usize, device: &Device) -> candle_core::Result<Batch> {
        let mol = Mol2::load(&self.mol2_paths[i_mol])?;
        let frcmod = ForceFieldParams::load_frcmod(&self.frcmod_paths[i_mol])?;

        let atoms = &mol.atoms;
        let bonds = &mol.bonds;
        let n = atoms.len();

        let mut elem_ids = Vec::with_capacity(n);
        let mut has_type = Vec::with_capacity(n);
        let mut type_ids = Vec::with_capacity(n);
        let mut charges = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n * 3);

        let oov_elem_id = self.vocabs.el.len();

        for atom in atoms.iter() {
            let el_id = self
                .vocabs
                .el
                .get(&atom.element.to_letter())
                .cloned()
                .unwrap_or(oov_elem_id);
            elem_ids.push(el_id as i64);

            coords.push(atom.posit.x as f32);
            coords.push(atom.posit.y as f32);
            coords.push(atom.posit.z as f32);

            if let Some(ff) = &atom.force_field_type {
                if let Some(tid) = self.vocabs.atom_type.get(ff) {
                    has_type.push(1.0f32);
                    type_ids.push(*tid as i64);
                } else {
                    has_type.push(0.0f32);
                    type_ids.push(-1);
                }
            } else {
                has_type.push(0.0f32);
                type_ids.push(-1);
            }

            if let Some(pc) = atom.partial_charge {
                charges.push(pc);
            } else {
                charges.push(0.0);
            }
        }

        let mut edge_index_vec: Vec<i64> = Vec::new();
        for bond in bonds.iter() {
            let i = (bond.atom_0_sn - 1) as i64;
            let j = (bond.atom_1_sn - 1) as i64;
            edge_index_vec.push(i);
            edge_index_vec.push(j);
            edge_index_vec.push(j);
            edge_index_vec.push(i);
        }

        let elem_ids = Tensor::from_slice(&elem_ids, (n,), device)?;
        let coords = Tensor::from_slice(&coords, (n, 3), device)?;
        let type_ids = Tensor::from_slice(&type_ids, (n,), device)?;
        let has_type = Tensor::from_slice(&has_type, (n,), device)?;
        let charges = Tensor::from_slice(&charges, (n,), device)?;

        let edge_index = if edge_index_vec.is_empty() {
            Tensor::zeros((0, 2), DType::I64, device)?
        } else {
            let m = edge_index_vec.len() / 2;
            Tensor::from_slice(&edge_index_vec, (m, 2), device)?
        };

        // FRCMOD code here -------

        // todo: Use your own adacency list builder fn instead.
        // build adjacency
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in bonds {
            let i = (b.atom_0_sn - 1) as usize;
            let j = (b.atom_1_sn - 1) as usize;
            adj[i].push(j);
            adj[j].push(i);
        }

        // collect dihedrals (i - j - k - l)
        let mut dih_indices: Vec<i64> = Vec::new(); // flat, we’ll shape later
        let mut dih_params: Vec<f32> = Vec::new(); // flat: n_dih * MAX_DIHEDRAL_TERMS * DIH_FEATS
        let mut dih_mask: Vec<f32> = Vec::new(); // flat: n_dih * MAX_DIHEDRAL_TERMS

        for b in bonds {
            let j = (b.atom_0_sn - 1) as usize;
            let k = (b.atom_1_sn - 1) as usize;

            for &i in adj[j].iter() {
                if i == k {
                    continue;
                }
                for &l in adj[k].iter() {
                    if l == j {
                        continue;
                    }

                    // this is one dihedral: i - j - k - l
                    dih_indices.push(i as i64);
                    dih_indices.push(j as i64);
                    dih_indices.push(k as i64);
                    dih_indices.push(l as i64);

                    // we have atom FF types in training data, so we can look up the terms
                    let ti = atoms[i].force_field_type.as_ref();
                    let tj = atoms[j].force_field_type.as_ref();
                    let tk = atoms[k].force_field_type.as_ref();
                    let tl = atoms[l].force_field_type.as_ref();

                    // default: no terms
                    let mut this_terms: Vec<f32> = vec![0.0; MAX_DIHEDRAL_TERMS * DIHEDRAL_FEATS];
                    let mut this_mask: Vec<f32> = vec![0.0; MAX_DIHEDRAL_TERMS];

                    if let (Some(ti), Some(tj), Some(tk), Some(tl)) = (ti, tj, tk, tl) {
                        let key = (ti.clone(), tj.clone(), tk.clone(), tl.clone());
                        let key_rev = (tl.clone(), tk.clone(), tj.clone(), ti.clone());

                        let src_terms = if let Some(terms) = frcmod.dihedral.get(&key) {
                            Some(terms)
                        } else if let Some(terms) = frcmod.dihedral.get(&key_rev) {
                            Some(terms)
                        } else if let Some(terms) = self.gaff2.dihedral.get(&key) {
                            Some(terms)
                        } else if let Some(terms) = self.gaff2.dihedral.get(&key_rev) {
                            Some(terms)
                        } else {
                            None
                        };

                        if let Some(terms) = src_terms {
                            for (t_idx, term) in terms.iter().take(MAX_DIHEDRAL_TERMS).enumerate() {
                                this_terms[t_idx * DIHEDRAL_FEATS + 0] = term.barrier_height;
                                this_terms[t_idx * DIHEDRAL_FEATS + 1] = term.phase;
                                this_terms[t_idx * DIHEDRAL_FEATS + 2] = term.periodicity as f32;
                                this_mask[t_idx] = 1.0;
                            }
                        }
                    }

                    dih_params.extend_from_slice(&this_terms);
                    dih_mask.extend_from_slice(&this_mask);
                }
            }
        }

        let n_dih = dih_indices.len() / 4;
        let dihedral_index = if n_dih == 0 {
            Tensor::zeros((0, 4), DType::I64, device)?
        } else {
            Tensor::from_slice(&dih_indices, (n_dih, 4), device)?
        };
        let dihedral_params = if n_dih == 0 {
            Tensor::zeros((0, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS), DType::F32, device)?
        } else {
            Tensor::from_slice(
                &dih_params,
                (n_dih, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS),
                device,
            )?
        };
        let dihedral_mask = if n_dih == 0 {
            Tensor::zeros((0, MAX_DIHEDRAL_TERMS), DType::F32, device)?
        } else {
            Tensor::from_slice(&dih_mask, (n_dih, MAX_DIHEDRAL_TERMS), device)?
        };

        let improper_index = Tensor::zeros((0, 4), DType::I64, device)?;
        let improper_params =
            Tensor::zeros((0, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS), DType::F32, device)?;
        let improper_mask = Tensor::zeros((0, MAX_DIHEDRAL_TERMS), DType::F32, device)?;

        Ok(Batch {
            elem_ids,
            coords,
            edge_index,
            type_ids,
            has_type,
            charges,
            dihedral_index,
            dihedral_params,
            dihedral_mask,
            improper_index,
            improper_params,
            improper_mask,
            num_atoms: n,
        })
    }
}

/// This is the entry point for our application to train parameters.
pub(crate) fn run_training() -> candle_core::Result<()> {
    // #[cfg(feature = "cuda")]
    // let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    // #[cfg(not(feature = "cuda"))]

    let device = Device::Cpu;
    let mut rng = rand::rng();

    println!("Training on GeoStd data with device: {device:?}");

    let (paths_mol2, paths_frcmod) = find_paths(Path::new(GEOSTD_PATH))?;

    // Load Gaff2 parameters, which FRCMOD data provides overrides for, for dihedrals and
    // improper dihedrals (And rarely valence angles and bond stretching params).
    let gaff2_params = ForceFieldParams::from_dat(GAFF2)?;

    let vocabs = AtomVocab::new(&paths_mol2)?;
    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();

    save(Path::new(VOCAB_PATH), &vocabs)?;
    println!("Vocabs built and saved to {VOCAB_PATH}");

    let dataset = GeoStdMol2Dataset::new(&paths_mol2, &paths_frcmod, vocabs, gaff2_params);

    let mut varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);

    let model = MolGNN::new(vb, n_elems, n_atom_types, HIDDEN_DIM)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    // ---- train/val split ----
    let mut all_idxs: Vec<usize> = (0..dataset.len()).collect();
    all_idxs.shuffle(&mut rng);
    let split = (dataset.len() as f32 * 0.8) as usize;
    let train_idxs = all_idxs[..split].to_vec();
    let val_idxs = all_idxs[split..].to_vec();

    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improve: u8 = 0;

    for epoch in 0..N_EPOCHS {
        println!("Running epoch {epoch} / {N_EPOCHS}...");

        let mut train_order = train_idxs.clone();
        train_order.shuffle(&mut rng);

        let mut running_loss = 0.;

        for i in train_order.iter() {
            let batch = dataset.get(*i, &device)?;

            let (type_logits, charges_pred, dih_pred, imp_pred) = model.forward(
                &batch.elem_ids,
                &batch.coords,
                &batch.edge_index,
                &batch.dihedral_index,
                &batch.improper_index,
            )?;

            let diff = (charges_pred - &batch.charges)?;
            let charge_loss = diff.sqr()?.mean_all()?;

            let type_ids_host = batch.type_ids.to_vec1::<i64>()?;
            let mut valid_idx: Vec<i64> = Vec::new();
            let mut valid_targets: Vec<i64> = Vec::new();
            for (idx, tid) in type_ids_host.iter().enumerate() {
                if *tid >= 0 {
                    valid_idx.push(idx as i64);
                    valid_targets.push(*tid);
                }
            }

            let type_loss = if !valid_idx.is_empty() {
                let idx_tensor = Tensor::from_slice(&valid_idx, (valid_idx.len(),), &device)?;
                let logits_sel = type_logits.index_select(&idx_tensor, 0)?;
                let targets_sel =
                    Tensor::from_slice(&valid_targets, (valid_targets.len(),), &device)?;
                loss::cross_entropy(&logits_sel, &targets_sel)?
            } else {
                Tensor::zeros((), DType::F32, &device)?
            };

            let mut dihedral_loss = Tensor::zeros((), DType::F32, &device)?;
            if batch.dihedral_index.dims()[0] > 0 {
                let diff = (dih_pred - &batch.dihedral_params)?;
                let diff2 = diff.sqr()?;
                let mask = batch
                    .dihedral_mask
                    .unsqueeze(2)?
                    .broadcast_as(diff2.dims())?;
                let masked = (diff2 * &mask)?;

                let sum = masked.sum_all()?;

                let mask_host = batch.dihedral_mask.to_vec2::<f32>()?;
                let mut count = 0f32;
                for row in mask_host {
                    for v in row {
                        count += v;
                    }
                }
                if count > 0.0 {
                    let denom = Tensor::new(count, &device)?;
                    dihedral_loss = (&sum / &denom)?;
                }
            }

            let mut improper_loss = Tensor::zeros((), DType::F32, &device)?;
            if batch.improper_index.dims()[0] > 0 {
                let diff = (imp_pred - &batch.improper_params)?;
                let diff2 = diff.sqr()?;
                let mask = batch
                    .improper_mask
                    .unsqueeze(2)?
                    .broadcast_as(diff2.dims())?;
                let masked = (diff2 * &mask)?;
                let sum = masked.sum_all()?;

                let mask_host = batch.improper_mask.to_vec2::<f32>()?;
                let mut count = 0f32;
                for row in mask_host {
                    for v in row {
                        count += v;
                    }
                }
                if count > 0.0 {
                    let denom = Tensor::new(count, &device)?;
                    improper_loss = (&sum / &denom)?;
                }
            }

            let loss = (&charge_loss + &type_loss + &dihedral_loss + &improper_loss)?;

            opt.backward_step(&loss)?;

            running_loss += f32::from(loss.to_scalar::<f32>()?);
        }

        let train_avg = running_loss / train_idxs.len() as f32;

        // ---- validation loss ----
        let mut val_loss_sum = 0f32;
        for i in val_idxs.iter() {
            let batch = dataset.get(*i, &device)?;

            let (type_logits, charges_pred, dih_pred, imp_pred) = model.forward(
                &batch.elem_ids,
                &batch.coords,
                &batch.edge_index,
                &batch.dihedral_index,
                &batch.improper_index,
            )?;

            let diff = (charges_pred - &batch.charges)?;
            let charge_loss = diff.sqr()?.mean_all()?;

            let type_ids_host = batch.type_ids.to_vec1::<i64>()?;
            let mut valid_idx: Vec<i64> = Vec::new();
            let mut valid_targets: Vec<i64> = Vec::new();
            for (idx, tid) in type_ids_host.iter().enumerate() {
                if *tid >= 0 {
                    valid_idx.push(idx as i64);
                    valid_targets.push(*tid);
                }
            }

            let type_loss = if !valid_idx.is_empty() {
                let idx_tensor = Tensor::from_slice(&valid_idx, (valid_idx.len(),), &device)?;
                let logits_sel = type_logits.index_select(&idx_tensor, 0)?;
                let targets_sel =
                    Tensor::from_slice(&valid_targets, (valid_targets.len(),), &device)?;
                loss::cross_entropy(&logits_sel, &targets_sel)?
            } else {
                Tensor::zeros((), DType::F32, &device)?
            };

            let loss = (&charge_loss + &type_loss)?;
            val_loss_sum += f32::from(loss.to_scalar::<f32>()?);
        }

        let val_avg = val_loss_sum / val_idxs.len() as f32;

        println!("Epoch {epoch} done. Train avg loss: {train_avg}, Val avg loss: {val_avg}");

        if val_avg < best_val_loss {
            best_val_loss = val_avg;
            epochs_without_improve = 0;
            varmap.save(MODEL_PATH)?; // keep best
            println!("New best val loss. Saved model to {MODEL_PATH}");
        } else {
            epochs_without_improve += 1;
            if epochs_without_improve >= EARLY_STOPPING_PATIENCE {
                println!(
                    "Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                );
                break;
            }
        }
    }

    // save all learned parameters
    varmap.save(MODEL_PATH)?;

    println!("Saved model to {MODEL_PATH}");

    Ok(())
}
