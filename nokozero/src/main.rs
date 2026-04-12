use anyhow::{bail, Context, Error, Result};
use pico_args::Arguments;
use std::fs::{canonicalize, write};
use std::io::{ErrorKind, Result as IoResult};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::sleep;
use std::time::Duration;
use tap::Pipe;

const GAME_LAUNCHER: &[u8] = include_bytes!(env!("LAUNCHER_PATH"));
const GAME_HOOK: &[u8] = include_bytes!(env!("HOOK_PATH"));

fn main() -> Result<()> {
    let mut args = Arguments::from_env();

    if args.contains(["-h", "--help"]) {
        println!(
            "nokozero\n\
             -h, --help        print this message\n\
             -d, --game-dir    path to directory containing game files"
        );
        return Ok(());
    }

    let game_dir_path: PathBuf = args.value_from_str(["-d", "--game-dir"])?;
    if !game_dir_path.is_dir() {
        bail!("`-d`/`--game-dir`: path does not point to a directory");
    }

    let game_path = {
        let path = game_dir_path.join("th15.exe");
        if !path.is_file() {
            bail!(
                "game executable not found; no file exists at {}",
                path.display()
            );
        }
        canonicalize(path).context("failed to process game executable path")?
    };

    println!("Using game executable at {}", game_path.display());

    let launcher_path = game_dir_path.join("nokozero_launcher.exe");
    let hook_path = (|| -> Result<_> {
        let hook_path = game_dir_path.join("nokozero_hook.dll");
        write(&launcher_path, GAME_LAUNCHER)?;
        write(&hook_path, GAME_HOOK)?;
        Ok(canonicalize(hook_path)?)
    })()
    .context("failed to load game hooking utilities")?;

    // Run game using Wine
    start_game(&launcher_path, &game_path, &hook_path).context("failed to start game")?;

    if !game_process_exists() {
        bail!("game process not found after launch");
    }
    println!("Found game process");

    // Set up graceful shutdown
    let is_running = Arc::new(AtomicBool::new(true));
    let r = is_running.clone();

    ctrlc::set_handler(move || {
        println!("\nReceived Ctrl+C, shutting down...");
        r.store(false, Ordering::Relaxed);
    })
    .context("failed to set Ctrl+C handler")?;

    // Wait for game to exit
    while is_running.load(Ordering::Relaxed) {
        sleep(Duration::from_millis(100));

        if !game_process_exists() {
            println!("Game process terminated");
            break;
        }
    }

    println!("Shutdown complete");

    Ok(())
}

/// Adds context to the result of running a command if the result is `Err(_)`.
/// A specific message is included if the command was not found.
fn add_command_context<T>(res: IoResult<T>, command_name: &str) -> Result<T> {
    res.map_err(|e| {
        let not_found = e.kind() == ErrorKind::NotFound;
        let err = Error::new(e);
        if not_found {
            err.context(format!("{command_name} command not found"))
        } else {
            err
        }
    })
    .with_context(|| format!("failed to run {command_name} command"))
}

/// Runs the game executable at the provided path using Wine.
/// This function returns an error if running the `wine` command was unsuccessful.
fn start_game(launcher_path: &Path, game_path: &Path, hook_path: &Path) -> Result<()> {
    // When running a Windows program through Wine, the Linux filesystem is mapped to the Z: drive.
    // For example, the Linux root directory "/" becomes "Z:\".
    // `game_path` and `hook_path` refer to files on our Linux filesystem,
    // but we need to convert them for the launcher executable running under Wine.
    let status = Command::new("wine")
        .arg(launcher_path)
        .env("GAME_PATH", unix_to_windows_path(game_path))
        .env("HOOK_PATH", unix_to_windows_path(hook_path))
        .env("LC_ALL", "ja_JP.UTF-8") // Run with locale set to Japanese
        .env("WINEDEBUG", "-all") // Disable Wine's debug logging
        .env("WINEESYNC", "1") // Enable esync optimization
        .env("STAGING_SHARED_MEMORY", "1") // Use shared memory to optimize wineserver calls
        .status()
        .pipe(|res| add_command_context(res, "wine"))?;

    if !status.success() {
        match status.code() {
            Some(code) => bail!("Wine failed with status code {code}"),
            None => bail!("Wine was terminated by a signal"),
        }
    }

    Ok(())
}

/// Converts a Unix-style path to a Windows-style path under the Z: drive volume.
fn unix_to_windows_path(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    format!(
        "Z:\\{}",
        path_str
            .strip_prefix('/')
            .unwrap_or(&path_str)
            .replace('/', "\\")
    )
}

/// Returns `true` if a game process is currently running.
fn game_process_exists() -> bool {
    Command::new("pgrep")
        .arg("th15.exe")
        .output()
        .is_ok_and(|output| output.status.success())
}
