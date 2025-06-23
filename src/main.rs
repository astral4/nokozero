use anyhow::{Context, Error, Result, bail};
use pico_args::Arguments;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    let game_exe_path = {
        // Use VSync patch if present
        let path = game_dir_path.join("vpatch.exe");
        if path.is_file() {
            let dll_path = game_dir_path.join("vpatch_th15.dll");
            if !dll_path.is_file() {
                bail!(
                    "vpatch DLL not found; no file exists at {}",
                    dll_path.display()
                );
            }
            path
        } else {
            let path = game_dir_path.join("th15.exe");
            if !path.is_file() {
                bail!(
                    "game executable not found; no file exists at {}",
                    path.display()
                );
            }
            path
        }
    };

    println!("Using game executable at {}", game_exe_path.display());

    // Run game using Wine
    start_game(&game_exe_path).context("failed to start game")?;

    Ok(())
}

fn start_game(exe_path: &Path) -> Result<()> {
    let status = Command::new("wine")
        .arg(exe_path)
        .env("LC_ALL", "ja_JP.UTF-8") // Run with locale set to Japanese
        .env("WINEDEBUG", "-all") // Disable Wine's debug logging
        .env("WINEESYNC", "1") // Enable esync optimization
        .env("STAGING_SHARED_MEMORY", "1") // Use shared memory to optimize wineserver calls
        .status()
        .map_err(|e| {
            let not_found = e.kind() == ErrorKind::NotFound;
            let err = Error::new(e);
            if not_found {
                err.context("wine command not found")
            } else {
                err
            }
        })?;

    if !status.success() {
        match status.code() {
            Some(code) => bail!("Wine failed with status code {code}"),
            None => bail!("Wine was terminated by a signal"),
        }
    }

    Ok(())
}
