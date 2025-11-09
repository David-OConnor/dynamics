//! For inferring force field type and partial charge of small organic molecules using Amber's
//! GeoStd library as training data. Uses a neural net.
//!
//! todo: Use this to handle frcmod data as well.

pub(crate) mod files;
pub(crate) mod frcmod;
// Pub so the training program can access it.

use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric,
    md_params::{DihedralParams, ForceFieldParams},
    mol2::Mol2,
};
use candle_core::{CudaDevice, DType, Device, IndexOp, Module, Tensor};
use candle_nn as nn;
use candle_nn::{Embedding, Linear, VarBuilder, ops::sigmoid};

use crate::param_inference::{
    files::{MODEL_PATH, VOCAB_PATH},
    frcmod::{DIHEDRAL_FEATS, MAX_DIHEDRAL_TERMS},
};

/// We save this to file during training, and load it during inference.
#[derive(Debug, Encode, Decode)]
pub(crate) struct Vocabs {
    pub el: HashMap<String, usize>,
    pub atom_type: HashMap<String, usize>,
}

impl Vocabs {
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
    // hidden_dim: usize,
}

impl MessagePassingLayer {
    fn new(vb: VarBuilder, hidden_dim: usize) -> candle_core::Result<Self> {
        let msg = nn::linear(hidden_dim * 2, hidden_dim, vb.pp("msg"))?;
        let gru = GruCell::new(vb.pp("gru"), hidden_dim)?;
        Ok(Self {
            msg,
            gru,
            // hidden_dim,
        })
    }

    fn forward(&self, h: &Tensor, edge_index: &Tensor) -> candle_core::Result<Tensor> {
        if edge_index.dims()[0] == 0 {
            return Ok(h.clone());
        }

        let src = edge_index.i((.., 0))?.contiguous()?;
        let dst = edge_index.i((.., 1))?.contiguous()?;

        let h_src = h.index_select(&src, 0)?;
        let h_dst = h.index_select(&dst, 0)?;

        let m_in = Tensor::cat(&[h_src, h_dst], 1)?;
        let m = self.msg.forward(&m_in)?.relu()?;

        let mut agg = Tensor::zeros_like(h)?.contiguous()?;
        let m = m.contiguous()?;

        agg = agg.index_add(&dst, &m, 0)?;

        let h_new = self.gru.forward(&agg, h)?;
        Ok(h_new)
    }
}

// -------- Model --------

pub(crate) struct MolGNN {
    elem_emb: Embedding,
    coord_lin: Linear,
    mp1: MessagePassingLayer,
    mp2: MessagePassingLayer,
    mp3: MessagePassingLayer,
    type_head: Linear,
    charge_head: Linear,
    dihedral_head: Linear, // new
    improper_head: Linear,
}

impl MolGNN {
    pub fn new(
        vb: VarBuilder,
        n_elems: usize,
        n_atom_types: usize,
        hidden_dim: usize,
    ) -> candle_core::Result<Self> {
        let elem_emb = nn::embedding(n_elems + 1, hidden_dim, vb.pp("elem_emb"))?;

        let coord_lin = nn::linear(3, hidden_dim, vb.pp("coord_lin"))?;
        let mp1 = MessagePassingLayer::new(vb.pp("mp1"), hidden_dim)?;
        let mp2 = MessagePassingLayer::new(vb.pp("mp2"), hidden_dim)?;
        let mp3 = MessagePassingLayer::new(vb.pp("mp3"), hidden_dim)?;
        let type_head = nn::linear(hidden_dim, n_atom_types, vb.pp("type_head"))?;
        let charge_head = nn::linear(hidden_dim, 1, vb.pp("charge_head"))?;

        let dihedral_head = nn::linear(
            hidden_dim * 4,
            MAX_DIHEDRAL_TERMS * DIHEDRAL_FEATS,
            vb.pp("dih_head"),
        )?;
        let improper_head = nn::linear(
            hidden_dim * 4,
            MAX_DIHEDRAL_TERMS * DIHEDRAL_FEATS,
            vb.pp("improper_head"),
        )?;

        Ok(Self {
            elem_emb,
            coord_lin,
            mp1,
            mp2,
            mp3,
            type_head,
            charge_head,
            dihedral_head,
            improper_head,
        })
    }

    pub fn forward(
        &self,
        elem_ids: &Tensor,
        coords: &Tensor,
        edge_index: &Tensor,
        dihedral_index: &Tensor,
        improper_index: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let h_emb = self.elem_emb.forward(elem_ids)?;
        let h_coord = self.coord_lin.forward(coords)?;
        let mut h = (h_emb + h_coord)?;

        h = self.mp1.forward(&h, edge_index)?;
        h = self.mp2.forward(&h, edge_index)?;
        h = self.mp3.forward(&h, edge_index)?;

        let type_logits = self.type_head.forward(&h)?;
        let charges = self.charge_head.forward(&h)?.squeeze(1)?;
        // dihedrals
        let dih_pred = if dihedral_index.dims()[0] == 0 {
            Tensor::zeros(
                (0, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS),
                DType::F32,
                h.device(),
            )?
        } else {
            let i_idx = dihedral_index.i((.., 0))?.contiguous()?;
            let j_idx = dihedral_index.i((.., 1))?.contiguous()?;
            let k_idx = dihedral_index.i((.., 2))?.contiguous()?;
            let l_idx = dihedral_index.i((.., 3))?.contiguous()?;

            let hi = h.index_select(&i_idx, 0)?;
            let hj = h.index_select(&j_idx, 0)?;
            let hk = h.index_select(&k_idx, 0)?;
            let hl = h.index_select(&l_idx, 0)?;

            let dih_in = Tensor::cat(&[hi, hj, hk, hl], 1)?;
            let dih_flat = self.dihedral_head.forward(&dih_in)?;
            let n_dih = dihedral_index.dims()[0];
            dih_flat.reshape((n_dih, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS))?
        };

        // impropers (same pattern)
        let improper_pred = if improper_index.dims()[0] == 0 {
            Tensor::zeros(
                (0, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS),
                DType::F32,
                h.device(),
            )?
        } else {
            let i_idx = improper_index.i((.., 0))?.contiguous()?;
            let j_idx = improper_index.i((.., 1))?.contiguous()?;
            let k_idx = improper_index.i((.., 2))?.contiguous()?;
            let l_idx = improper_index.i((.., 3))?.contiguous()?;

            let hi = h.index_select(&i_idx, 0)?;
            let hj = h.index_select(&j_idx, 0)?;
            let hk = h.index_select(&k_idx, 0)?;
            let hl = h.index_select(&l_idx, 0)?;

            let improper_in = Tensor::cat(&[hi, hj, hk, hl], 1)?;
            let improper_flat = self.improper_head.forward(&improper_in)?;
            let n_imp = improper_index.dims()[0];
            improper_flat.reshape((n_imp, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS))?
        };

        Ok((type_logits, charges, dih_pred, improper_pred))
    }
}

// -------- Inference --------

fn build_dih_tensors(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let n = atoms.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in bonds {
        let i = (b.atom_0_sn - 1) as usize;
        let j = (b.atom_1_sn - 1) as usize;
        adj[i].push(j);
        adj[j].push(i);
    }

    let mut dih_indices: Vec<i64> = Vec::new();
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
                dih_indices.push(i as i64);
                dih_indices.push(j as i64);
                dih_indices.push(k as i64);
                dih_indices.push(l as i64);
            }
        }
    }

    let n_dih = dih_indices.len() / 4;
    let dihedral_index = if n_dih == 0 {
        Tensor::zeros((0, 4), DType::I64, device)?
    } else {
        Tensor::from_slice(&dih_indices, (n_dih, 4), device)?
    };

    let improper_index = Tensor::zeros((0, 4), DType::I64, device)?;

    Ok((dihedral_index, improper_index))
}

/// Find bond force field type, partial charge, and parameter overrides. (Paramater overrides
/// are generally Dihedral and improper only, but we've observed bond and angle as well.
fn run_inference(
    model: &MolGNN,
    vocabs: &Vocabs,
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    device: &Device,
) -> candle_core::Result<(
    Vec<String>,
    Vec<f32>,
    Tensor, // dihedral_pred: [n_dih, MAX_DIHEDRAL_TERMS, DIHEDRAL_FEATS]
    Tensor, // dihedral_index: [n_dih, 4]
)> {
    let mut elem_ids = Vec::with_capacity(atoms.len());
    let mut coords = Vec::with_capacity(atoms.len() * 3);

    let oov_elem_id = vocabs.el.len();

    for atom in atoms.iter() {
        let el = &atom.element;
        elem_ids.push(
            vocabs
                .el
                .get(&el.to_letter())
                .cloned()
                .unwrap_or(oov_elem_id) as i64,
        );

        coords.push(atom.posit.x as f32);
        coords.push(atom.posit.y as f32);
        coords.push(atom.posit.z as f32);
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

    let elem_ids = Tensor::from_slice(&elem_ids, (atoms.len(),), device)?;
    let coords = Tensor::from_slice(&coords, (atoms.len(), 3), device)?;
    let edge_index = if edge_index_vec.is_empty() {
        Tensor::zeros((0, 2), DType::I64, device)?
    } else {
        Tensor::from_slice(&edge_index_vec, (edge_index_vec.len() / 2, 2), device)?
    };

    // rebuild dihedrals for this molecule
    let (dihedral_index, improper_index) = build_dih_tensors(atoms, bonds, device)?;

    let (type_logits, charges_t, dihedral_pred, _improper_pred) = model.forward(
        &elem_ids,
        &coords,
        &edge_index,
        &dihedral_index,
        &improper_index,
    )?;

    let type_ids = type_logits.argmax(1)?.to_dtype(DType::I64)?;
    let type_ids: Vec<i64> = type_ids.to_vec1()?;
    let charges: Vec<f32> = charges_t.to_vec1()?;

    let inv_type_vocab: HashMap<usize, String> = vocabs
        .atom_type
        .iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    let mut ff_types = Vec::with_capacity(atoms.len());
    for tid in type_ids {
        ff_types.push(inv_type_vocab[&(tid as usize)].clone());
    }

    Ok((ff_types, charges, dihedral_pred, dihedral_index))
}

/// Infer force field type, partial charge, and dihedral atoms for a molecule for
/// which we don't have them.
///
/// We infer FF type and partial charge for each atom. We infer missing FRCMODM params (Generally
/// dihedrals and impropers; occasionally others) based partially on which ones (defined by sets of
/// 4 FF types) are present in this mol, but aren't in GAFF2. Note: This is trickier for Impropers,
/// as it's authorized to ommit these in some cases. Proper dihedrals, on the other hand, are
/// not permitted to be missing.
pub fn infer_params(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    // todo: Consider bond stretching and valence angles eventually as well.
    dihedrals_missing: Vec<(String, String, String, String)>,
    improper_missing: Vec<(String, String, String, String)>,
    gaff2: &ForceFieldParams,
) -> candle_core::Result<(Vec<String>, Vec<f32>, ForceFieldParams)> {
    // Note: CUDA doesn't seem faster here.
    let dev_candle = Device::Cpu;

    let start = Instant::now();

    let vocabs: Vocabs = load(&Path::new(VOCAB_PATH))?;
    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();
    let hidden_dim = 128;

    let mut varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &dev_candle);
    let model = MolGNN::new(vb, n_elems, n_atom_types, hidden_dim)?;
    varmap.load(MODEL_PATH)?;

    let (ff_types, charges, dihedrals_pred, dihedral_index) =
        run_inference(&model, &vocabs, atoms, bonds, &dev_candle)?;

    let mut params = ForceFieldParams::default();

    let dihedrals = dihedrals_pred.to_vec3::<f32>()?;
    let dih_idx = dihedral_index.to_vec2::<i64>()?;

    // cache to reuse for missing list
    let mut produced: std::collections::HashMap<
        (String, String, String, String),
        Vec<bio_files::md_params::DihedralParams>,
    > = std::collections::HashMap::new();

    // 1) all 4-atom lines we found in the mol
    for (d_i, idxs) in dih_idx.iter().enumerate() {
        let i = idxs[0] as usize;
        let j = idxs[1] as usize;
        let k = idxs[2] as usize;
        let l = idxs[3] as usize;

        let ti = ff_types[i].clone();
        let tj = ff_types[j].clone();
        let tk = ff_types[k].clone();
        let tl = ff_types[l].clone();

        let key = (ti.clone(), tj.clone(), tk.clone(), tl.clone());

        if let Some(terms) =
            gaff2.get_dihedral(&(ti.clone(), tj.clone(), tk.clone(), tl.clone()), false)
        {
            params.dihedral.insert(key.clone(), terms.clone());
            produced.insert(key, terms.clone());
            continue;
        }

        let mut terms_vec = Vec::new();
        for t in 0..MAX_DIHEDRAL_TERMS {
            let barrier = dihedrals[d_i][t][0];
            let phase = dihedrals[d_i][t][1];
            let periodicity = dihedrals[d_i][t][2];

            if barrier.abs() < 1e-5 {
                continue;
            }

            terms_vec.push(bio_files::md_params::DihedralParams {
                atom_types: (ti.clone(), tj.clone(), tk.clone(), tl.clone()),
                divider: 1,
                barrier_height: barrier,
                phase,
                periodicity: periodicity.round() as u8,
                comment: None,
            });
        }

        if terms_vec.is_empty() {
            let barrier0 = dihedrals[d_i][0][0];
            let phase0 = dihedrals[d_i][0][1];
            let per0 = dihedrals[d_i][0][2];

            terms_vec.push(DihedralParams {
                atom_types: (ti.clone(), tj.clone(), tk.clone(), tl.clone()),
                divider: 1,
                barrier_height: if barrier0.abs() < 1e-5 { 0.1 } else { barrier0 },
                phase: if phase0.abs() < 1e-5 { 0.0 } else { phase0 },
                periodicity: if per0.abs() < 0.5 {
                    1
                } else {
                    per0.round() as u8
                },
                comment: None,
            });
        }

        // 2) caller-supplied missing dihedrals (already checked both orders upstream)
        for (a, b, c, d) in &dihedrals_missing {
            let key = (a.clone(), b.clone(), c.clone(), d.clone());
            if params.dihedral.contains_key(&key) {
                continue;
            }

            if let Some(terms) =
                gaff2.get_dihedral(&(a.clone(), b.clone(), c.clone(), d.clone()), false)
            {
                params.dihedral.insert(key, terms.clone());
                continue;
            }

            if let Some(terms) = produced.get(&(a.clone(), b.clone(), c.clone(), d.clone())) {
                params
                    .dihedral
                    .insert((a.clone(), b.clone(), c.clone(), d.clone()), terms.clone());
                continue;
            }

            params.dihedral.insert(
                (a.clone(), b.clone(), c.clone(), d.clone()),
                vec![DihedralParams {
                    atom_types: (a.clone(), b.clone(), c.clone(), d.clone()),
                    divider: 1,
                    barrier_height: 0.1,
                    phase: 0.0,
                    periodicity: 1,
                    comment: None,
                }],
            );
        }

        params.dihedral.insert(key.clone(), terms_vec.clone());
        produced.insert(key, terms_vec);
    }

    let elapsed = start.elapsed().as_millis();
    println!("Inference complete in {elapsed} ms");

    Ok((ff_types, charges, params))
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

// C+P from graphics.
/// Load from file, using Bincode. We currently use this for preference files.
pub(crate) fn load<T: Decode<()>>(path: &Path) -> io::Result<T> {
    let config = bincode::config::standard();

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let (decoded, _len) = match bincode::decode_from_slice(&buffer, config) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error loading from file. Did the format change?");
            return Err(io::Error::new(ErrorKind::Other, "error loading"));
        }
    };
    Ok(decoded)
}
