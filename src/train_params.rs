//! Train the Amber GeoStd partial-charge model.

fn main() {
    if let Err(e) = dynamics::params::amber::charges::train::run_training() {
        eprintln!("Error training partial charges: {e}");
        std::process::exit(1);
    }
}
