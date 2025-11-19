use std::path::{Path, PathBuf};

use bio_files::{Mol2, md_params::ForceFieldParams};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder};
use rand::seq::SliceRandom;

use crate::partial_charge_inference::{
    AtomVocab, MolGNN,
    files::{GEOSTD_PATH, MODEL_PATH, VOCAB_PATH, find_paths},
    save,
};

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
    pub num_atoms: usize,
}

struct GeoStdMol2Dataset {
    mol2_paths: Vec<PathBuf>,
    vocabs: AtomVocab,
}

impl GeoStdMol2Dataset {
    pub fn new(mol2_paths: &[PathBuf], vocabs: AtomVocab) -> Self {
        Self {
            mol2_paths: mol2_paths.to_vec(),
            vocabs,
        }
    }

    pub fn len(&self) -> usize {
        self.mol2_paths.len()
    }

    pub fn get(&self, i_mol: usize, device: &Device) -> candle_core::Result<Batch> {
        let mol = Mol2::load(&self.mol2_paths[i_mol])?;

        let atoms = &mol.atoms;
        let bonds = &mol.bonds;
        let n = atoms.len();

        let mut elem_ids = Vec::with_capacity(n);
        let mut has_type = Vec::with_capacity(n);
        let mut type_ids = Vec::with_capacity(n);
        let mut charges = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n * 3);

        let oov_elem_id = self.vocabs.el.len();
        let oov_type_id = self.vocabs.atom_type.len();

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

            let t_id = atom
                .force_field_type
                .as_ref()
                .and_then(|ff| self.vocabs.atom_type.get(ff))
                .cloned()
                .unwrap_or(oov_type_id);
            type_ids.push(t_id as i64);
            has_type.push(1.0f32);

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

        Ok(Batch {
            elem_ids,
            coords,
            edge_index,
            type_ids,
            has_type,
            charges,
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

    let paths_mol2 = find_paths(Path::new(GEOSTD_PATH))?;

    let vocabs = AtomVocab::new(&paths_mol2)?;
    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();

    save(Path::new(VOCAB_PATH), &vocabs)?;
    println!("Vocabs built and saved to {VOCAB_PATH}");

    let dataset = GeoStdMol2Dataset::new(&paths_mol2, vocabs);

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

            let charges_pred = model.forward(
                &batch.elem_ids,
                &batch.type_ids,
                &batch.coords,
                &batch.edge_index,
            )?;

            let diff = (charges_pred - &batch.charges)?;
            let charge_loss = diff.sqr()?.mean_all()?;

            let loss = &charge_loss;

            opt.backward_step(&loss)?;

            running_loss += f32::from(loss.to_scalar::<f32>()?);
        }

        let train_avg = running_loss / train_idxs.len() as f32;

        // ---- validation loss ----
        let mut val_loss_sum = 0f32;
        for i in val_idxs.iter() {
            let batch = dataset.get(*i, &device)?;

            let charges_pred = model.forward(
                &batch.elem_ids,
                &batch.type_ids,
                &batch.coords,
                &batch.edge_index,
            )?;

            let diff = (charges_pred - &batch.charges)?;
            let charge_loss = diff.sqr()?.mean_all()?;

            let loss = charge_loss;
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
