//! This is the entry point for a standalone application to train a neural net model
//! to provide Amber-style force field paramers for small organic molecules not present in
//! Amber's GAFF2.dat. It generates per-atom FF name and partial charge, and per-molecule
//! FRCMOD-style overrides for bonded params. (Generally Dihedral and improper angles)
//!
//! Releavnt code is in teh `partial_charge_inference` folder`. We seem to need this entry point at the top level
//! to work properly.
//!
//! Run `cargo b --release --bin train --features "train-bin"`

mod partial_charge_inference;

fn main() {
    if let Err(e) = partial_charge_inference::train::run_training() {
        eprintln!("Error training: {e}");
    }
}
