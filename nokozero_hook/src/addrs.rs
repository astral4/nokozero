//! Absolute addresses and values of th15.exe v1.00b `.data` globals.

/// The game's globals block. The fields below are its members.
pub(crate) const GLOBALS_VA: usize = 0x004e_73f0;
pub(crate) const GLOBALS_INNER_LEN: usize = 0x1cc;
/// The stage that actually loaded. The retry reload (`0x44f6c0`) copies [`STAGE_SELECT_VA`] here.
pub(crate) const STAGE_CURRENT_VA: usize = GLOBALS_VA;
/// The menu-selected stage, 1–7. 7 = Extra.
pub(crate) const STAGE_SELECT_VA: usize = GLOBALS_VA + 0x4;
pub(crate) const CURRENT_CHAPTER_VA: usize = GLOBALS_VA + 0x8;
/// The number of frames since the stage started.
pub(crate) const GAME_TICK_VA: usize = GLOBALS_VA + 0xc;
/// The number of frames since the current chapter started.
pub(crate) const TIME_IN_CHAPTER_VA: usize = GLOBALS_VA + 0x10;
/// The selected character, 0-3 (Reimu/Marisa/Sanae/Reisen).
pub(crate) const CHARACTER_VA: usize = GLOBALS_VA + 0x14;
/// The current score divided by 10.
pub(crate) const SCORE_DIV10_VA: usize = GLOBALS_VA + 0x1c;
/// The current difficulty, 0–4. 4 = Extra.
pub(crate) const DIFFICULTY_VA: usize = GLOBALS_VA + 0x20;
pub(crate) const GRAZE_VA: usize = GLOBALS_VA + 0x2c;
/// The current spell card's ID.
pub(crate) const SPELL_ID_VA: usize = GLOBALS_VA + 0x34;
/// The number of deaths so far in the run.
pub(crate) const MISS_COUNT_VA: usize = GLOBALS_VA + 0x3c;
/// The current PIV. Stored as `value * 100`.
pub(crate) const VALUE_VA: usize = GLOBALS_VA + 0x44;
/// The current power. Stored as `power * 100`.
pub(crate) const POWER_VA: usize = GLOBALS_VA + 0x50;
/// Negative after a game over.
pub(crate) const LIVES_VA: usize = GLOBALS_VA + 0x60;
pub(crate) const LIFE_FRAGMENTS_VA: usize = GLOBALS_VA + 0x64;
pub(crate) const BOMBS_VA: usize = GLOBALS_VA + 0x6c;
pub(crate) const BOMB_FRAGMENTS_VA: usize = GLOBALS_VA + 0x70;
pub(crate) const ENEMIES_SPAWNED_IN_CHAPTER_VA: usize = GLOBALS_VA + 0x94;
/// The run's mode flags, Legacy/Pointdevice/Practice.
pub(crate) const MODEFLAGS_VA: usize = GLOBALS_VA + 0x3a4;

/// The replay-unsafe RNG for visual effects only.
pub(crate) const RNG_UNSAFE_VA: usize = 0x004e_9a40;
/// The replay-safe RNG for gameplay.
pub(crate) const RNG_VA: usize = 0x004e_9a48;
/// The replay-safe RNG's call counter.
pub(crate) const RNG_COUNT_VA: usize = 0x004e_9a4c;

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
/// The live spell card object, or null.
pub(crate) const SPELLCARD_PTR_VA: usize = 0x004e_9a70;
pub(crate) const MAIN_MENU_PTR_VA: usize = 0x004e_9be0;
