use anyhow::{Context, Error, Result, bail};
use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use pico_args::Arguments;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use std::thread::sleep;
use std::time::Duration;

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

    // Terminate game process when receiving termination signal
    ctrlc::set_handler(move || {
        println!("\nReceived termination signal, attempting graceful shutdown...");

        let _ = kill(game_pid, Signal::SIGTERM);

        // Give process some time to shut down gracefully
        sleep(Duration::from_secs(2));

        match kill(game_pid, Signal::SIGKILL) {
            // `kill()` returns ESRCH if process was already terminated
            // Reference: https://pubs.opengroup.org/onlinepubs/9799919799/functions/kill.html
            Ok(()) | Err(nix::Error::ESRCH) => println!("Terminated game process"),
            Err(e) => {
                let error = Error::new(e)
                    .context(format!("failed to terminate game process (pid {game_pid})"));
                eprintln!("{error}");
            }
        }

        exit(128);
    })
    .context("failed to set up termination signal handler")?;

    Ok(())
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
        .map_err(|e| {
            let not_found = e.kind() == ErrorKind::NotFound;
            let err = Error::new(e);
            if not_found {
                err.context("pgrep command not found")
            } else {
                err
            }
        })?;

    if !output.status.success() {
        match output.status.code() {
            Some(1) => bail!("no game process found"),
            Some(code) => bail!("pgrep failed with status code {code}"),
            None => bail!("pgrep was terminated by a signal"),
        }
    }

    let pids: Vec<i32> = String::from_utf8(output.stdout)
        .context("pgrep output is not valid UTF-8")?
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
