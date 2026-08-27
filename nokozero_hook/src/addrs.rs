//! Absolute addresses and values of th15.exe v1.00b `.data` globals.

/// The stage that actually loaded. The retry reload (`0x44f6c0`) copies [`STAGE_SELECT_VA`] here.
pub(crate) const STAGE_CURRENT_VA: usize = 0x004e_73f0;
/// The menu-selected stage, 1–7. 7 = Extra.
pub(crate) const STAGE_SELECT_VA: usize = 0x004e_73f4;
pub(crate) const CURRENT_CHAPTER_VA: usize = 0x004e_73f8;
/// The number of frames since the stage started.
pub(crate) const GAME_TICK_VA: usize = 0x004e_73fc;
/// The selected character, 0-3 (Reimu/Marisa/Sanae/Reisen).
pub(crate) const CHARACTER_VA: usize = 0x004e_7404;
/// The current score divided by 10.
pub(crate) const SCORE_DIV10_VA: usize = 0x004e_740c;
/// The current difficulty, 0–4. 4 = Extra.
pub(crate) const DIFFICULTY_VA: usize = 0x004e_7410;
pub(crate) const GRAZE_VA: usize = 0x004e_741c;
/// The current PIV. Stored as `value * 100`.
pub(crate) const VALUE_VA: usize = 0x004e_7434;
/// The current power. Stored as `power * 100`.
pub(crate) const POWER_VA: usize = 0x004e_7440;
/// Negative after a game over.
pub(crate) const LIVES_VA: usize = 0x004e_7450;
pub(crate) const LIFE_FRAGMENTS_VA: usize = 0x004e_7454;
pub(crate) const BOMBS_VA: usize = 0x004e_745c;
pub(crate) const BOMB_FRAGMENTS_VA: usize = 0x004e_7460;
pub(crate) const ENEMIES_SPAWNED_IN_CHAPTER_VA: usize = 0x004e_7484;

pub(crate) const DEVICE_PTR_VA: u32 = 0x004e_77d8;
/// The active top-level scene.
pub(crate) const GAMEMODE_VA: usize = 0x004e_7ec8;
pub(crate) const GAMEMODE_TO_SWITCH_TO_VA: usize = 0x004e_7ecc;

/// Scene IDs held by [`GAMEMODE_VA`] and [`GAMEMODE_TO_SWITCH_TO_VA`].
pub(crate) const GAMEMODE_MENU: u32 = 4;
pub(crate) const GAMEMODE_INGAME: u32 = 7;
pub(crate) const GAMEMODE_RETRY: u32 = 10;

/// The inverse of [`LOADER_RUNNING_VA`]. Set when a load has completed.
pub(crate) const LOADER_DONE_VA: usize = 0x004e_817c;
/// Set (with [`LOADER_DONE_VA`] cleared) inside the spawn helper `0x44d7f0` before the loader thread is created.
/// Cleared (with [`LOADER_DONE_VA`] set) in scene-loader epilogues.
pub(crate) const LOADER_RUNNING_VA: usize = 0x004e_8180;

pub(crate) const BULLET_MANAGER_PTR_VA: usize = 0x004e_9a6c;
pub(crate) const ENEMY_MANAGER_PTR_VA: usize = 0x004e_9a80;
pub(crate) const GUI_PTR_VA: usize = 0x004e_9a8c;
pub(crate) const GAME_THREAD_PTR_VA: usize = 0x004e_9a94;
pub(crate) const ITEM_MANAGER_PTR_VA: usize = 0x004e_9a9c;
pub(crate) const LASER_MANAGER_PTR_VA: usize = 0x004e_9ba0;
pub(crate) const PLAYER_PTR_VA: usize = 0x004e_9bb8;
pub(crate) const MAIN_MENU_PTR_VA: usize = 0x004e_9be0;
