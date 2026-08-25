//! Typed access to the game's process memory at absolute addresses.

use crate::addrs::{
    GAME_THREAD_PTR_VA, GAMEMODE_INGAME, GAMEMODE_VA, LOADER_DONE_VA, LOADER_RUNNING_VA,
};
use std::ptr::with_exposed_provenance;

/// # Safety
///
/// `addr` must point to a readable, mapped, initialized `T`.
pub(crate) unsafe fn read<T: Copy>(addr: usize) -> T {
    unsafe { with_exposed_provenance::<T>(addr).read_unaligned() }
}

/// Returns whether a stage is fully live (i.e. an in-game scene is active and the stage loader has finished).
///
/// # Safety
///
/// The game image must be loaded.
pub(crate) unsafe fn stage_stable() -> bool {
    unsafe {
        read::<u32>(GAMEMODE_VA) == GAMEMODE_INGAME
            && read::<u32>(LOADER_RUNNING_VA) == 0
            && read::<u32>(LOADER_DONE_VA) == 1
    }
}

/// Returns whether [`stage_stable`] holds and the game thread is live.
///
/// # Safety
///
/// The game image must be loaded.
pub(crate) unsafe fn game_live() -> bool {
    unsafe { stage_stable() && read::<u32>(GAME_THREAD_PTR_VA) != 0 }
}
