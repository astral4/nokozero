use anyhow::{Context, Error, Result, bail};
use pico_args::Arguments;
use std::env::home_dir;
use std::fs::{create_dir_all, write};
use std::io::{ErrorKind, Result as IoResult};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio, exit};
use std::thread::sleep;
use std::time::Duration;

const HOOK_DLL: &[u8] = include_bytes!(env!("HOOK_PATH"));

fn main() -> Result<()> {
    let mut args = Arguments::from_env();

    if args.contains(["-h", "--help"]) {
        println!(
            "nokozero\n\
             -h, --help          print this message\n\
             -d, --game-dir      path to directory containing game files\n\
             -n, --instances     number of game instances to run, each with a separate custom Wine prefix\n\
                                 (default: 1, with default Wine prefix)"
        );
        return Ok(());
    }

    let game_dir: PathBuf = args.value_from_str(["-d", "--game-dir"])?;
    let num_instances: Option<usize> = args.opt_value_from_str(["-n", "--instances"])?;
    let num_instances: Option<NonZeroUsize> = num_instances
        .map(|n| NonZeroUsize::new(n).context("`-n`/`--instances`: must be at least 1"))
        .transpose()?;

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

    // When `-n` is specified, each instance gets its own Wine prefix for process isolation
    let prefix_dir = if num_instances.is_some() {
        let dir = home_dir()
            .context("could not determine home directory")?
            .join(".local/share/nokozero/prefixes");
        create_dir_all(&dir).context("failed to create prefix directory")?;
        Some(dir)
    } else {
        None
    };

    // Suppress the default SIGINT behavior so we can wait for children to exit.
    // Children receive SIGINT from the process group and shut down on their own.
    ctrlc::set_handler({
        let mut times = 0u32;
        move || {
            times += 1;
            if times == 1 {
                eprintln!("\nShutting down, waiting for instances to exit...");
                eprintln!("Press Ctrl+C again to force exit.");
            } else {
                exit(1);
            }
        }
    })
    .context("failed to set Ctrl+C handler")?;

    let count = num_instances.map_or(1, NonZeroUsize::get);
    let mut children: Vec<(usize, ChildGuard)> = Vec::with_capacity(count);
    for i in 0..count {
        let prefix = prefix_dir.as_ref().map(|dir| dir.join(i.to_string()));
        let child = spawn_game(&exe, &game_dir, prefix.as_deref())
            .with_context(|| format!("failed to start instance {i}"))?;
        children.push((i, ChildGuard(child)));
    }

    // Wait for all instances to exit
    while !children.is_empty() {
        children.retain_mut(|(i, child)| match child.try_wait() {
            Ok(Some(status)) => {
                println!("Instance {i} exited ({status})");
                false
            }
            Ok(None) => true,
            Err(e) => {
                eprintln!("Instance {i}: {e}");
                false
            }
        });
        if !children.is_empty() {
            sleep(Duration::from_millis(100));
        }
    }

    println!("Shutdown complete");

    Ok(())
}

/// Spawns a game instance under Wine and returns the child process handle.
fn spawn_game(exe: &Path, game_dir: &Path, wine_prefix: Option<&Path>) -> Result<Child> {
    let mut cmd = Command::new("wine");
    cmd.arg(exe)
        .current_dir(game_dir)
        .env("WINEDLLOVERRIDES", "dinput8=n,b") // Load hook library, then fall back to the built-in real DLL
        .env("LC_ALL", "ja_JP.UTF-8") // Run with locale set to Japanese
        .env("WINEDEBUG", "-all") // Disable Wine's debug logging
        .env("WINEESYNC", "1") // Enable esync optimization
        .env("STAGING_SHARED_MEMORY", "1"); // Use shared memory to optimize wineserver calls

    if let Some(prefix) = wine_prefix {
        cmd.env("WINEPREFIX", prefix);
    }

    cmd.spawn().map_err(|e| {
        let not_found = e.kind() == ErrorKind::NotFound;
        let err = Error::new(e);
        if not_found {
            err.context("wine command not found")
        } else {
            err
        }
    })
}

/// Owns a spawned [`Child`] process, terminating and reaping it on drop.
///
/// This prevents leaking Wine instances when one of several spawns fails partway through.
#[derive(Debug)]
struct ChildGuard(Child);

impl ChildGuard {
    fn try_wait(&mut self) -> IoResult<Option<ExitStatus>> {
        self.0.try_wait()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        // Both calls are no-ops once the child has been reaped via `try_wait()`.
        // `Child` tracks the reaped state to avoid PID recycling.
        self.0.kill().ok();
        self.0.wait().ok();
    }
}
