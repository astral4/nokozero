use anyhow::{Context, Error, Result, bail};
use pico_args::Arguments;
use std::fs::write;
use std::io::{ErrorKind, Result as IoResult};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::sleep;
use std::time::Duration;
use tap::Pipe;

const HOOK_DLL: &[u8] = include_bytes!(env!("HOOK_PATH"));

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

    let game_dir: PathBuf = args.value_from_str(["-d", "--game-dir"])?;
    if !game_dir.is_dir() {
        bail!("`-d`/`--game-dir`: path does not point to a directory");
    }

    if !game_dir.join("th15.exe").is_file() {
        bail!(
            "game executable not found; no file exists at {}",
            game_dir.join("th15.exe").display()
        );
    }

    // Deploy the hook library as a `dinput8.dll` proxy in the game directory.
    // Windows DLL search order makes it load before the real system DLL, so we can do hooking.
    write(game_dir.join("dinput8.dll"), HOOK_DLL).context("failed to deploy hook library")?;

    // Use vpatch if available, otherwise run the game directly
    let exe = if game_dir.join("vpatch.exe").is_file() {
        println!("Using vpatch");
        game_dir.join("vpatch.exe")
    } else {
        game_dir.join("th15.exe")
    };

    start_game(&exe, &game_dir).context("failed to start game")?;

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
fn start_game(exe: &Path, game_dir: &Path) -> Result<()> {
    let status = Command::new("wine")
        .arg(exe)
        .current_dir(game_dir)
        .env("WINEDLLOVERRIDES", "dinput8=n,b") // Load hook library, then fall back to the built-in real DLL
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

/// Returns `true` if a game process is currently running.
fn game_process_exists() -> bool {
    Command::new("pgrep")
        .arg("th15.exe")
        .output()
        .is_ok_and(|output| output.status.success())
}
