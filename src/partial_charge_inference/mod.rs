//! For inferring partial charge of small organic molecules using Amber's
//! GeoStd library as training data. Uses a neural net. Force field types must be assigned
//! prior to running this, e.g. from the `param_inference` module.

pub(crate) mod files;

pub(crate) mod experimenting;
pub(crate) mod train;
// Pub so the training program can access it.

use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, BondGeneric, BondType, mol2::Mol2};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn as nn;
use candle_nn::{Embedding, Linear, VarBuilder, ops::sigmoid};

use crate::partial_charge_inference::files::load_from_bytes;

// Load the model and vocab from bytes. Note: This has implications for the portability
// of this library.
// todo: Test and address A/R.
// Model: ~1.5Mb. Vocab: ~440 bytes.
const PARAM_INFERENCE_MODEL: &[u8] =
    include_bytes!("../../param_data/ml_models/geostd_model.safetensors");
const PARAM_INFERENCE_VOCAB: &[u8] =
    include_bytes!("../../param_data/ml_models/geostd_model.vocab");

const N_BOND_TYPES: usize = 5; // single, double, triple, aromatic, other

/// We save this to file during training, and load it during inference.
#[derive(Debug, Encode, Decode)]
pub(crate) struct AtomVocab {
    pub el: HashMap<String, usize>,
    pub atom_type: HashMap<String, usize>,
    pub charge_mean: f32,
    pub charge_std: f32,
}

impl AtomVocab {
    pub fn new(mol2_paths: &[PathBuf]) -> candle_core::Result<Self> {
        let mut elems: BTreeSet<String> = BTreeSet::new();
        let mut ff_types: BTreeSet<String> = BTreeSet::new();

        for path in mol2_paths {
            let mol = Mol2::load(path)?;

            for atom in mol.atoms.iter() {
                elems.insert(atom.element.to_letter());

                // Ideally we won't encounter this with the Geostd data set.
                let Some(ff) = &atom.force_field_type else {
                    eprintln!("Error: Missing FF type on Geostd atom: {atom}");
                    continue;
                };

                ff_types.insert(ff.clone());
            }
        }

        let mut el_map = HashMap::new();
        for (i, el) in elems.into_iter().enumerate() {
            el_map.insert(el, i);
        }

        let mut atom_type_map = HashMap::new();
        for (i, t) in ff_types.into_iter().enumerate() {
            atom_type_map.insert(t, i);
        }

        Ok(Self {
            el: el_map,
            atom_type: atom_type_map,
            charge_mean: 0.,
            charge_std: 0.,
        })
    }
}

// -------- GRU cell (hidden_dim -> hidden_dim) --------

struct GruCell {
    w_ih: Linear, // in -> 3*h
    w_hh: Linear, // h -> 3*h
    hidden_dim: usize,
}

impl GruCell {
    fn new(vb: VarBuilder, hidden_dim: usize) -> candle_core::Result<Self> {
        let w_ih = nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("w_ih"))?;
        let w_hh = nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("w_hh"))?;
        Ok(Self {
            w_ih,
            w_hh,
            hidden_dim,
        })
    }

    fn forward(&self, x: &Tensor, h: &Tensor) -> candle_core::Result<Tensor> {
        // x, h: [N, H]
        let ih = self.w_ih.forward(x)?; // [N, 3H]
        let hh = self.w_hh.forward(h)?; // [N, 3H]

        let hsize = self.hidden_dim;
        let i_r = ih.narrow(1, 0, hsize)?;
        let i_z = ih.narrow(1, hsize, hsize)?;
        let i_n = ih.narrow(1, 2 * hsize, hsize)?;

        let h_r = hh.narrow(1, 0, hsize)?;
        let h_z = hh.narrow(1, hsize, hsize)?;
        let h_n = hh.narrow(1, 2 * hsize, hsize)?;

        let r = sigmoid(&(i_r + h_r)?)?;
        let z = sigmoid(&(i_z + h_z)?)?;
        let n = (i_n + (r * h_n)?)?.tanh()?;

        let one = Tensor::ones_like(&z)?;
        let one_minus_z = one.sub(&z)?;

        (&one_minus_z * n)? + (&z * h)?
    }
}

// -------- Message passing layer --------

struct MessagePassingLayer {
    msg: Linear,
    gru: GruCell,
}

impl MessagePassingLayer {
    fn new(vb: VarBuilder, hidden_dim: usize) -> candle_core::Result<Self> {
        let msg = nn::linear(hidden_dim * 2, hidden_dim, vb.pp("msg"))?;
        let gru = GruCell::new(vb.pp("gru"), hidden_dim)?;
        Ok(Self { msg, gru })
    }

    fn forward(
        &self,
        h: &Tensor,
        edge_index: &Tensor,
        edge_types: &Tensor,
        bond_emb: &Embedding,
    ) -> candle_core::Result<Tensor> {
        if edge_index.dims()[0] == 0 {
            return Ok(h.clone());
        }

        let src = edge_index.i((.., 0))?.contiguous()?;
        let dst = edge_index.i((.., 1))?.contiguous()?;

        let h_src = h.index_select(&src, 0)?;
        let h_dst = h.index_select(&dst, 0)?;

        let b = bond_emb.forward(edge_types)?;
        let h_src_b = (h_src + b)?;

        let m_in = Tensor::cat(&[h_src_b, h_dst], 1)?;
        let m = self.msg.forward(&m_in)?.relu()?;

        let mut agg = Tensor::zeros_like(h)?.contiguous()?;
        let m = m.contiguous()?;

        agg = agg.index_add(&dst, &m, 0)?;

        let h_new = self.gru.forward(&agg, h)?;
        Ok(h_new)
    }
}

// -------- Model --------

pub(in crate::partial_charge_inference) struct MolGNN {
    elem_emb: Embedding,
    type_emb: Embedding,
    coord_lin: Linear,
    mp1: MessagePassingLayer,
    mp2: MessagePassingLayer,
    mp3: MessagePassingLayer,
    charge_head: Linear,
    bond_emb: Embedding,
}

impl MolGNN {
    pub fn new(
        vb: VarBuilder,
        n_elems: usize,
        n_atom_types: usize,
        hidden_dim: usize,
    ) -> candle_core::Result<Self> {
        let elem_emb = nn::embedding(n_elems + 1, hidden_dim, vb.pp("elem_emb"))?;
        let type_emb = nn::embedding(n_atom_types + 1, hidden_dim, vb.pp("type_emb"))?;
        let bond_emb = nn::embedding(N_BOND_TYPES, hidden_dim, vb.pp("bond_emb"))?;

        let coord_lin = nn::linear(3, hidden_dim, vb.pp("coord_lin"))?;
        let mp1 = MessagePassingLayer::new(vb.pp("mp1"), hidden_dim)?;
        let mp2 = MessagePassingLayer::new(vb.pp("mp2"), hidden_dim)?;
        let mp3 = MessagePassingLayer::new(vb.pp("mp3"), hidden_dim)?;
        let charge_head = nn::linear(hidden_dim, 1, vb.pp("charge_head"))?;

        Ok(Self {
            elem_emb,
            type_emb,
            coord_lin,
            mp1,
            mp2,
            mp3,
            charge_head,
            bond_emb,
        })
    }

    pub fn forward(
        &self,
        elem_ids: &Tensor,
        type_ids: &Tensor,
        coords: &Tensor,
        edge_index: &Tensor,
        edge_types: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let coord_mean = coords.mean(0)?; // [3]
        let coords_centered = coords.broadcast_sub(&coord_mean)?; // [N,3]

        let h_coord = self.coord_lin.forward(&coords_centered)?;

        let h_elem = self.elem_emb.forward(elem_ids)?;
        let h_type = self.type_emb.forward(type_ids)?;

        let mut h = (h_elem + h_type)?;
        h = (h + h_coord)?;

        h = self
            .mp1
            .forward(&h, edge_index, edge_types, &self.bond_emb)?;
        h = self
            .mp2
            .forward(&h, edge_index, edge_types, &self.bond_emb)?;
        h = self
            .mp3
            .forward(&h, edge_index, edge_types, &self.bond_emb)?;

        let charges = self.charge_head.forward(&h)?.squeeze(1)?;
        Ok(charges)
    }
}

// -------- Inference --------

/// Find partial charges for each atom.
fn run_inference(
    model: &MolGNN,
    vocabs: &AtomVocab,
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    device: &Device,
) -> candle_core::Result<Vec<f32>> {
    let mut elem_ids = Vec::with_capacity(atoms.len());
    let mut type_ids = Vec::with_capacity(atoms.len());
    let mut coords = Vec::with_capacity(atoms.len() * 3);

    let oov_elem_id = vocabs.el.len();
    let oov_type_id = vocabs.atom_type.len();

    for atom in atoms.iter() {
        let el = &atom.element;
        elem_ids.push(
            vocabs
                .el
                .get(&el.to_letter())
                .cloned()
                .unwrap_or(oov_elem_id) as i64,
        );

        let t_id = atom
            .force_field_type
            .as_ref()
            .and_then(|ff| vocabs.atom_type.get(ff))
            .cloned()
            .unwrap_or(oov_type_id);
        type_ids.push(t_id as i64);

        coords.push(atom.posit.x as f32);
        coords.push(atom.posit.y as f32);
        coords.push(atom.posit.z as f32);
    }

    let mut edge_index_vec: Vec<i64> = Vec::new();
    let mut edge_types_vec: Vec<i64> = Vec::new();

    for bond in bonds.iter() {
        let i = (bond.atom_0_sn - 1) as i64;
        let j = (bond.atom_1_sn - 1) as i64;

        let bt_id = match bond.bond_type {
            BondType::Single => 0,
            BondType::Double => 1,
            BondType::Triple => 2,
            BondType::Aromatic => 3,
            _ => 4,
        };

        edge_index_vec.push(i);
        edge_index_vec.push(j);
        edge_types_vec.push(bt_id);

        edge_index_vec.push(j);
        edge_index_vec.push(i);
        edge_types_vec.push(bt_id);
    }

    let elem_ids = Tensor::from_slice(&elem_ids, (atoms.len(),), device)?;
    let type_ids = Tensor::from_slice(&type_ids, (atoms.len(),), device)?;
    let coords = Tensor::from_slice(&coords, (atoms.len(), 3), device)?;

    let edge_index = if edge_index_vec.is_empty() {
        Tensor::zeros((0, 2), DType::I64, device)?
    } else {
        Tensor::from_slice(&edge_index_vec, (edge_index_vec.len() / 2, 2), device)?
    };

    let edge_types = if edge_types_vec.is_empty() {
        Tensor::zeros((0,), DType::I64, device)?
    } else {
        Tensor::from_slice(&edge_types_vec, (edge_types_vec.len(),), device)?
    };
    let charges_t = model.forward(&elem_ids, &type_ids, &coords, &edge_index, &edge_types)?;

    let mean = vocabs.charge_mean;
    let std = vocabs.charge_std;

    let mean_t = Tensor::from_slice(&[mean], (1,), device)?;
    let std_t = Tensor::from_slice(&[std], (1,), device)?;

    // charges = charges_t * std + mean
    let charges_t = charges_t.broadcast_mul(&std_t)?.broadcast_add(&mean_t)?;

    let charges: Vec<f32> = charges_t.to_vec1()?;
    Ok(charges)
}

/// Infer partial charge for each atom.
pub fn infer_charge(atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> candle_core::Result<Vec<f32>> {
    let start = Instant::now();

    // Note: CUDA doesn't seem faster here.
    let dev_candle = Device::Cpu;

    let model_bytes = PARAM_INFERENCE_MODEL;
    let vocab_bytes = PARAM_INFERENCE_VOCAB;

    let vocabs: AtomVocab = load_from_bytes(&vocab_bytes)?;

    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();
    let hidden_dim = 128;

    let vb = VarBuilder::from_slice_safetensors(model_bytes, DType::F32, &dev_candle)?;
    let model = MolGNN::new(vb, n_elems, n_atom_types, hidden_dim)?;

    let charges = run_inference(&model, &vocabs, atoms, bonds, &dev_candle)?;

    let elapsed = start.elapsed().as_millis();
    println!("Inference complete in {elapsed} ms");

    // We know that most molecules in the GeoStd set don't specify  bond or
    // valence angles (although some do) We can therefor assume that every 3
    // sets of linear atoms should be represented by gaff2.dat. Conduct a review
    // of these, identify missing ones, and adjust FF types IOC to match valid gaff2.dat types.
    // todo: Update Dihedrals based on this (?)

    Ok(charges)
}

// C+P from graphics.
/// Save to file, using Bincode. We currently use this for preference files.
pub(crate) fn save<T: Encode>(path: &Path, data: &T) -> io::Result<()> {
    let config = bincode::config::standard();

    let encoded: Vec<u8> = bincode::encode_to_vec(data, config).unwrap();

    let mut file = File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}
