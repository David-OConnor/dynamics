//! Handles molecule-specific parameter inference from FRCMOD files. This is generally
//! dihedral and improper angles, but sometimes bond and angle parameters. Each GeoStd molecule
//! has both Mol2, and FRCMOD.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use bio_files::{AtomGeneric, BondGeneric, md_params::ForceFieldParams};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::param_inference::{
    MolGNN, Vocabs,
    files::{MODEL_PATH, VOCAB_PATH},
    load, run_inference,
};

// Load every frcmod you have (parallel to your mol2s) and fold them in.
pub(crate) fn build_param_db(paths_frcmod: &[PathBuf]) -> candle_core::Result<ForceFieldParams> {
    let mut db = ForceFieldParams::default();

    for p in paths_frcmod {
        let f = ForceFieldParams::load_frcmod(p)?;

        for (k, v) in f.bond {
            db.bond.insert(k, v);
        }
        for (k, v) in f.angle {
            db.angle.insert(k, v);
        }
        for (k, v) in f.dihedral {
            db.dihedral.insert(k, v);
        }
        for (k, v) in f.improper {
            db.improper.insert(k, v);
        }
        for (k, v) in f.mass {
            db.mass.insert(k, v);
        }
        for (k, v) in f.lennard_jones {
            db.lennard_jones.insert(k, v);
        }
    }

    Ok(db)
}

pub fn build_params_for_mol(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    ff_types: &[String],
    db: &ForceFieldParams,
) -> ForceFieldParams {
    let mut out = ForceFieldParams {
        bond: HashMap::new(),
        angle: HashMap::new(),
        dihedral: HashMap::new(),
        improper: HashMap::new(),
        mass: HashMap::new(),
        lennard_jones: HashMap::new(),
    };

    // todo: Eval these; they're similar to our ParamsIndexed::new() code.

    // bonds → adjacency
    let n = atoms.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in bonds {
        let i = (b.atom_0_sn - 1) as usize;
        let j = (b.atom_1_sn - 1) as usize;
        adj[i].push(j);
        adj[j].push(i);
    }

    // bonds
    for b in bonds {
        let i = (b.atom_0_sn - 1) as usize;
        let j = (b.atom_1_sn - 1) as usize;
        let ti = &ff_types[i];
        let tj = &ff_types[j];

        let ff_names = (ti.clone(), tj.clone());
        if let Some(p) = db.get_bond(&ff_names) {
            out.bond.insert(ff_names, p.clone());
        }
    }

    // angles
    for b in bonds {
        let i = (b.atom_0_sn - 1) as usize;
        let j = (b.atom_1_sn - 1) as usize;

        for &k in adj[i].iter() {
            if k == j {
                continue;
            }
            let tk = &ff_types[k];
            let ti = &ff_types[i];
            let tj = &ff_types[j];

            let ff_names = (tk.clone(), ti.clone(), tj.clone());
            if let Some(p) = db.get_valence_angle(&ff_names) {
                out.angle.insert(ff_names, p.clone());
            }
        }
    }

    // dihedrals
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
                let ti = &ff_types[i];
                let tj = &ff_types[j];
                let tk = &ff_types[k];
                let tl = &ff_types[l];

                let ff_names = (ti.clone(), tj.clone(), tk.clone(), tl.clone());

                if let Some(v) = db.get_dihedral(&ff_names, false) {
                    out.dihedral.insert(ff_names, v.clone());
                }
            }
        }
    }

    // todo: Handle improper!

    // todo: SKipping these for now. Check if they're ever included in FRCMOD, and apply A/R.as
    // // masses, LJ: these are per-type; we can fill them if present in DB
    // for t in ff_types.iter() {
    //     if let Some(m) = db.mass.get(t) {
    //         out.mass.insert(t.clone(), m.clone());
    //     }
    //     if let Some(lj) = db.lennard_jones.get(t) {
    //         out.lennard_jones.insert(t.clone(), lj.clone());
    //     }
    // }

    out
}

pub fn infer_params(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    db: &ForceFieldParams,
) -> candle_core::Result<(Vec<String>, Vec<f32>, ForceFieldParams)> {
    let dev_candle = Device::Cpu;

    let vocabs: Vocabs = load(&Path::new(VOCAB_PATH)).unwrap();
    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();
    let hidden_dim = 128;

    let mut varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &dev_candle);
    let model = MolGNN::new(vb, n_elems, n_atom_types, hidden_dim).unwrap();
    varmap.load(MODEL_PATH).unwrap();

    let (ff_types, charges, _) = run_inference(&model, &vocabs, atoms, bonds, &dev_candle)?;

    let params = build_params_for_mol(atoms, bonds, &ff_types, db);

    Ok((ff_types, charges, params))
}
