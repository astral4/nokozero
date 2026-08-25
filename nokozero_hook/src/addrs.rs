//! Absolute addresses and values of th15.exe v1.00b `.data` globals.

/// The active top-level scene.
pub(crate) const GAMEMODE_VA: usize = 0x004e_7ec8;

/// Scene IDs held by [`GAMEMODE_VA`].
pub(crate) const GAMEMODE_INGAME: u32 = 7;

/// The inverse of [`LOADER_RUNNING_VA`]. Set when a load has completed.
pub(crate) const LOADER_DONE_VA: usize = 0x004e_817c;
/// Set when `0x44d7f0` spawns the stage loader thread. Cleared in `GameThread::thread_start`'s epilogue.
pub(crate) const LOADER_RUNNING_VA: usize = 0x004e_8180;

pub(crate) const BULLET_MANAGER_PTR_VA: usize = 0x004e_9a6c;
pub(crate) const ENEMY_MANAGER_PTR_VA: usize = 0x004e_9a80;
pub(crate) const GAME_THREAD_PTR_VA: usize = 0x004e_9a94;
pub(crate) const ITEM_MANAGER_PTR_VA: usize = 0x004e_9a9c;
pub(crate) const LASER_MANAGER_PTR_VA: usize = 0x004e_9ba0;
pub(crate) const PLAYER_PTR_VA: usize = 0x004e_9bb8;
