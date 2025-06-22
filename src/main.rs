use anyhow::{Result, bail};
use pico_args::Arguments;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = Arguments::from_env();

    if args.contains(["-h", "--help"]) {
        println!(
            "nokozero
-h, --help        print this message
-d, --game-dir    path to directory containing game files"
        );
        return Ok(());
    }

    let game_dir_path: PathBuf = args.value_from_str(["-d", "--game-dir"])?;

    if !game_dir_path.is_dir() {
        bail!("`-d`/`--game-dir`: path does not point to a directory");
    }

    Ok(())
}
