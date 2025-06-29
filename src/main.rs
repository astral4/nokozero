use anyhow::{Context, Error, Result, bail};
use nix::unistd::Pid;
use nokozero::reader::StateReader;
use pico_args::Arguments;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tap::Pipe;

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

    let game_pid = get_game_pid().context("failed to query game process ID")?;
    println!("Found game process with PID {game_pid}");

    // Set up graceful shutdown
    let is_running = Arc::new(AtomicBool::new(true));
    let r = is_running.clone();

    ctrlc::set_handler(move || {
        println!("\nReceived Ctrl+C, shutting down...");
        r.store(false, Ordering::Relaxed);
    })
    .context("failed to set Ctrl+C handler")?;

    // Create game state reader
    let mut reader = StateReader::new(game_pid).context("failed to create game state reader")?;

    // Main game loop - read state every 6 frames
    let frame_duration = Duration::from_millis(100);
    let mut last_read = Instant::now();

    println!("Starting game state monitoring");

    while is_running.load(Ordering::Relaxed) {
        let now = Instant::now();

        if now.duration_since(last_read) >= frame_duration {
            match reader.get_state() {
                Ok(Some(state)) => {
                    println!(
                        "position: ({}, {}), hitbox radius: {}, is focused: {}, items: {:?}",
                        state.player.pos_x,
                        state.player.pos_y,
                        state.player.hitbox_radius,
                        state.player.is_focused,
                        state.items,
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("Failed to read game state: {e:?}");
                    if get_game_pid().is_err() {
                        println!("Game process terminated");
                        break;
                    }
                }
            }

            last_read = now;
        }

        sleep(Duration::from_millis(10));
    }

    println!("Shutdown complete");

    Ok(())
}

/// Adds context to the result of running a command if the result is `Err(_)`.
/// A specific message is included if the command was not found.
fn add_command_context<T>(res: std::io::Result<T>, command_name: &str) -> Result<T> {
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
fn start_game(exe_path: &Path) -> Result<()> {
    let status = Command::new("wine")
        .arg(exe_path)
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

/// Finds the ID of the game process.
/// Returns an error if:
/// - running the `pgrep` command was unsuccessful
/// - parsing `pgrep` output was unsuccessful
/// - no processes matching the game name were found
/// - multiple processes matching the game name were found
fn get_game_pid() -> Result<Pid> {
    let output = Command::new("pgrep")
        .arg("th15.exe")
        .output()
        .pipe(|res| add_command_context(res, "pgrep"))?;

    if !output.status.success() {
        match output.status.code() {
            Some(1) => bail!("no game process found"),
            Some(code) => bail!("pgrep failed with status code {code}"),
            None => bail!("pgrep was terminated by a signal"),
        }
    }

    let pids: Vec<i32> = String::from_utf8(output.stdout)
        .context("pgrep output was not valid UTF-8")?
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()
        .context("failed to parse pgrep output as i32 values")?;

    match pids.len() {
        0 => bail!("no game process found"),
        1 => Ok(Pid::from_raw(pids[0])),
        _ => bail!("multiple game processes found"),
    }
}
