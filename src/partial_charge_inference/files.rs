//! For managing inference-related files

use std::{
    fs,
    fs::File,
    io,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
};

use bincode::Decode;

pub const MODEL_PATH: &str = "geostd_model.safetensors";
pub const VOCAB_PATH: &str = "geostd_model.vocab";

// pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/amber_geostd";
pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/amber_geostd_test"; // todo temp

/// Find Mol2 and FRCMOD paths. Assumes there are per-letter subfolders one-layer deep.
/// todo: FRCmod as well
pub fn find_paths(geostd_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut mol2_paths = Vec::new();

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
                    mol2_paths.push(subpath);
                }
            }
        }
    }

    Ok(mol2_paths)
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

/// Load from file, using Bincode. We currently use this for preference files.
pub(crate) fn load_from_bytes<T: Decode<()>>(buffer: &[u8]) -> io::Result<T> {
    let config = bincode::config::standard();

    let (decoded, _len) = match bincode::decode_from_slice(&buffer, config) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error loading from file. Did the format change?");
            return Err(io::Error::new(ErrorKind::Other, "error loading"));
        }
    };
    Ok(decoded)
}
