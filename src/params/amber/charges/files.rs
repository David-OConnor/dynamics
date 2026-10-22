//! For managing inference-related files

use std::{
    fs, io,
    path::{Path, PathBuf},
};

#[cfg(feature = "train-bin")]
pub const MODEL_PATH: &str = "geostd_model.safetensors";
#[cfg(feature = "train-bin")]
pub const VOCAB_PATH: &str = "geostd_model.vocab";

pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/Amber/amber_geostd";
// pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/amber_geostd_test"; // todo temp

/// Find Mol2  paths. Assumes there are per-letter subfolders one-layer deep.
/// todo: FRCmod as well
pub fn find_mol2_paths(geostd_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut result = Vec::new();

    for entry in fs::read_dir(geostd_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            for subentry in fs::read_dir(&path)? {
                let subentry = subentry?;
                let subpath = subentry.path();

                if subpath
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    == Some("mol2".to_string())
                {
                    result.push(subpath);
                }
            }
        }
    }

    Ok(result)
}
