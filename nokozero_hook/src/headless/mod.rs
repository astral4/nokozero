//! Logic for running the game in headless mode.

mod backing;
mod d3d9;
mod out;
mod window;

use crate::env::headless;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::Foundation::HMODULE;

static HEADLESS: AtomicBool = AtomicBool::new(false);

/// This should be called during `DLL_PROCESS_ATTACH`, before [`install`].
pub(crate) fn init_from_env() {
    HEADLESS.store(headless(), Ordering::Relaxed);
}

#[must_use]
pub(crate) fn is_enabled() -> bool {
    HEADLESS.load(Ordering::Relaxed)
}

/// # Safety
///
/// `game` must be a loaded module handle. This function must be called during `DLL_PROCESS_ATTACH`, before the game's entry point runs.
pub(crate) unsafe fn install(game: HMODULE) {
    unsafe {
        d3d9::install(game);
        window::install(game);
    }
}
