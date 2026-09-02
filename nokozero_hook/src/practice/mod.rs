//! Practice navigation, stage-load interception, resource injection, ECL section patching, and death suppression.

mod catalog;
mod chapter;
mod data;
mod ecl;
mod hit;
mod load;
#[forbid(unsafe_code)]
mod machine;

pub(crate) use crate::practice::data::MAX_CHARACTER;
pub(crate) use crate::practice::hit::take_forced_step;
pub(crate) use crate::practice::load::Generation;
pub(crate) use crate::practice::machine::Outcome;

use crate::addrs::{
    BOMB_FRAGMENTS_VA, BOMBS_VA, CHARACTER_VA, DIFFICULTY_VA, GAME_THREAD_PTR_VA, GAMEMODE_INGAME,
    GAMEMODE_RETRY, GAMEMODE_TO_SWITCH_TO_VA, GLOBALS_INNER_LEN, GLOBALS_VA, GRAZE_VA, GUI_PTR_VA,
    LIFE_FRAGMENTS_VA, LIVES_VA, LOADER_RUNNING_VA, PLAYER_PTR_VA, POWER_VA, RNG_COUNT_VA,
    RNG_UNSAFE_VA, RNG_VA, SCORE_DIV10_VA, STAGE_CURRENT_VA, STAGE_SELECT_VA, VALUE_VA,
};
use crate::log::fatal;
use crate::mem::{game_live, read, read_ptr, stage_stable, write};
use crate::patch::{NearBranchSite, Site, op_abs32};
use crate::practice::catalog::apply_section;
use crate::practice::data::{EXTRA_DIFFICULTY, rank_matches_stage, section_mapped, section_stage};
use crate::practice::ecl::{Ecl, EclUndo};
use crate::practice::hit::count;
use crate::practice::load::{HANDOFF, LoadStatus, PerLoad, load_generation};
use crate::practice::machine::{Lifecycle, ReloadPlan, TickDecision};
use crate::thread::{MainCell, MainThread, MainToken};
use crate::{Action, READ_INTERVAL};
use std::arch::naked_asm;
use std::ffi::c_void;
use std::mem::transmute;
use std::ptr::{copy_nonoverlapping, with_exposed_provenance, with_exposed_provenance_mut};

#[derive(Clone, Copy)]
pub(crate) struct PracticeParams {
    /// Whether to warp to `section` and `phase` at the next stage load.
    active: bool,
    /// The warp target ID; see the section ID constants. Read only when `active`.
    section: u32,
    /// The sub-phase within the section. Read only when `active`.
    phase: u32,
    /// The difficulty (0-4) applied at the reset. 4 = Extra.
    difficulty: u32,
    /// The character (0–3) required by the reset.
    character: u32,
    score: i64,
    graze: i32,
    value: i32,
    power: i32,
    lives: i32,
    life_fragments: i32,
    bombs: i32,
    bomb_fragments: i32,
    rng_seed: u16,
    /// Game frames per RL step for this episode (1-[`MAX_STEP_INTERVAL`]).
    step_interval: u32,
    /// Episode behavior flags; see the `FLAG_*` constants.
    flags: u32,
    /// The action held from the reset's acceptance until the driver's first ACT.
    initial_action: u32,
    /// The player's position on the stage's first frame.
    player_x: f32,
    player_y: f32,
}

impl PracticeParams {
    pub(crate) fn initial_action(&self) -> u32 {
        self.initial_action
    }

    pub(crate) fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

/// The slowest supported control rate (one decision per second).
pub(crate) const MAX_STEP_INTERVAL: u32 = 60;

/// Leave dialogue to the controller instead of skipping it. The controller's action, `SKIP` included, reaches the game unchanged.
/// This is needed to replay human inputs that follow their own dialogue timing.
pub(crate) const FLAG_RAW_DIALOGUE: u32 = 1 << 0;
/// Let hits run the game's own death sequence instead of suppressing it. Hits are still counted.
pub(crate) const FLAG_REAL_DEATHS: u32 = 1 << 1;
pub(crate) const FLAG_RECORD: u32 = 1 << 2;
const KNOWN_FLAGS: u32 = FLAG_RAW_DIALOGUE | FLAG_REAL_DEATHS | FLAG_RECORD;

pub(crate) const RECORD_LEN: usize = 0x238 + 0xa4;
const RECORD_GLOBALS: usize = 0x14;
const RECORD_PLAYER_POS: usize = 0xc;
const INFO_GAME_THREAD: usize = 0x238 + 0x18;
const GAME_THREAD_INNER: usize = 0x24;
const GAME_THREAD_INNER_LEN: usize = 0x6c;

/// A stage record plus its run's info block.
pub(crate) struct StageRecord(pub(crate) [u8; RECORD_LEN]);

/// The record of the reset in flight, taken when the reset is consumed.
/// The value is `Some` iff the pending reset's params carry [`FLAG_RECORD`].
static PENDING_RECORD: MainCell<Option<Box<StageRecord>>> = MainCell::new(None);

/// Stores the record carried by a just-accepted RESET.
pub(crate) fn stash_record(thread: MainThread, record: Option<Box<StageRecord>>) {
    PENDING_RECORD.set(thread, record);
}

#[repr(C)]
struct WireParams {
    section: u32,
    active: u32,
    score: i64,
    graze: i32,
    value: i32,
    power: i32,
    lives: i32,
    life_fragments: i32,
    bombs: i32,
    bomb_fragments: i32,
    phase: u32,
    difficulty: i32,
    character: i32,
    rng_seed: u32,
    step_interval: u32,
    flags: u32,
    initial_action: u32,
    player_x: f32,
    player_y: f32,
}

pub(crate) const PARAMS_LEN: usize = size_of::<WireParams>();

const _: () = assert!(PARAMS_LEN == 80, "wire params block is fixed at 80 bytes");

impl PracticeParams {
    /// Validates and decodes a RESET command's params block. Returns `None` if there is a protocol violation.
    pub(crate) fn parse(payload: &[u8]) -> Option<Self> {
        if payload.len() != size_of::<WireParams>() {
            return None;
        }
        // SAFETY: The length check guarantees `size_of::<WireParams>()` readable bytes.
        // `WireParams` has only integer fields, so every bit pattern is valid. `read_unaligned` tolerates the buffer's alignment.
        let wire = unsafe { payload.as_ptr().cast::<WireParams>().read_unaligned() };

        let (Ok(difficulty), Ok(character)) = (
            u32::try_from(wire.difficulty),
            u32::try_from(wire.character),
        ) else {
            return None;
        };
        if difficulty > EXTRA_DIFFICULTY || character > MAX_CHARACTER {
            return None;
        }
        let rng_seed = u16::try_from(wire.rng_seed).ok()?;
        if !(1..=MAX_STEP_INTERVAL).contains(&wire.step_interval) {
            return None;
        }
        if wire.flags & !KNOWN_FLAGS != 0 {
            return None;
        }
        Action::from_wire(wire.initial_action)?;
        if !(wire.player_x.is_finite() && wire.player_y.is_finite()) {
            return None;
        }

        let active = wire.active != 0;
        if active {
            let stage = section_stage(wire.section)?;
            if !rank_matches_stage(stage, difficulty) {
                return None;
            }
            if !section_mapped(wire.section, wire.phase) {
                return None;
            }
        }

        let score = wire.score.clamp(0, 9_999_999_990);
        let graze = wire.graze.clamp(0, 999_999);
        let value = wire.value.clamp(0, 999_990);
        let power = wire.power.clamp(0, 400);
        let lives = wire.lives.clamp(0, 8);
        // 5 fragments per life in Extra; 3 fragments per life elsewhere
        let life_fragments = wire
            .life_fragments
            .clamp(0, if difficulty == EXTRA_DIFFICULTY { 5 } else { 3 });
        let bombs = wire.bombs.clamp(0, 8);
        let bomb_fragments = wire.bomb_fragments.clamp(0, 4);

        Some(Self {
            active,
            section: wire.section,
            phase: wire.phase,
            difficulty,
            character,
            score,
            graze,
            value,
            power,
            lives,
            life_fragments,
            bombs,
            bomb_fragments,
            rng_seed,
            step_interval: wire.step_interval,
            flags: wire.flags,
            initial_action: wire.initial_action,
            player_x: wire.player_x,
            player_y: wire.player_y,
        })
    }
}

struct LiveWarp {
    /// The stage affected by the ECL patches.
    stage: u32,
    undo: EclUndo,
}

/// The value is `None` while the value is checked out (i.e. inside [`with_lifecycle`]).
static LIFECYCLE: MainCell<Option<Lifecycle>> = MainCell::new(Some(Lifecycle::INIT));

/// Checks the lifecycle out of its cell for the duration of `f`, aborting if it is already checked out.
fn with_lifecycle<R>(thread: MainThread, f: impl FnOnce(&mut Lifecycle) -> R) -> R {
    let Some(mut lc) = LIFECYCLE.replace(thread, None) else {
        fatal!("lifecycle re-entered");
    };
    let result = f(&mut lc);
    LIFECYCLE.set(thread, Some(lc));
    result
}

/// Observes one supervisor frame, consuming any unrequested publish.
pub(crate) fn observe_loads(thread: MainThread) {
    let status = HANDOFF.status();
    with_lifecycle(thread, |lc| lc.observe(status));
}

/// Accepts a decoded RESET command. Returns `false` if a reset is already pending.
#[must_use]
pub(crate) fn accept_reset(thread: MainThread, seq: u32, params: PracticeParams) -> bool {
    with_lifecycle(thread, |lc| lc.accept(seq, params))
}

/// Starts a requested reset's reload. The request remains pending unless the game is in stable gameplay,
/// no other scene switch is queued, and the previous stage loader has fully published.
/// This should only be called from the input hook after [`observe_loads`] within the same tick.
pub(crate) fn apply_pending_reset(token: MainToken) {
    let thread = token.thread();
    let stable =
        unsafe { stage_stable() && read::<u32>(GAMEMODE_TO_SWITCH_TO_VA) == GAMEMODE_INGAME };
    let Some(plan) = try_start_reload(thread, stable) else {
        return;
    };

    if plan.arms_load {
        HANDOFF.arm();
    }

    if let Some(stage) = plan.stage_select {
        unsafe { write(token, STAGE_SELECT_VA, stage) };
    }
    unsafe {
        write(token, DIFFICULTY_VA, plan.difficulty);
        write(token, GAMEMODE_TO_SWITCH_TO_VA, GAMEMODE_RETRY);
    }
}

/// Attempts `Requested` -> `Reloading`, returning the writes for the caller to perform.
/// Returns `None` if the state should stay `Requested` or the reset was refused.
fn try_start_reload(thread: MainThread, stable_ingame: bool) -> Option<ReloadPlan> {
    let status = HANDOFF.status();
    let loader_running = unsafe { read::<u32>(LOADER_RUNNING_VA) } == 1;
    let current_stage = unsafe { read::<u32>(STAGE_CURRENT_VA) };
    let current_character = unsafe { read::<u32>(CHARACTER_VA) };
    with_lifecycle(thread, |lc| {
        lc.try_start_reload(
            stable_ingame,
            status,
            loader_running,
            current_stage,
            current_character,
        )
    })
}

pub(crate) struct WireMeta {
    pub(crate) load_generation: Generation,
    pub(crate) reset_seq: u32,
    pub(crate) reset_outcome: Outcome,
    pub(crate) applied_section: u32,
    pub(crate) hits: u32,
}

impl WireMeta {
    #[must_use]
    pub(crate) fn read(thread: MainThread) -> Self {
        let (load_generation, reset_seq, reset_outcome, applied_section) =
            with_lifecycle(thread, |lc| lc.wire());
        Self {
            load_generation,
            reset_seq,
            reset_outcome,
            applied_section,
            hits: count(thread, load_generation),
        }
    }
}

/// A detour handler's decision, returned to its naked trampoline in `eax`.
/// The trampolines run `test eax, eax`, so the return type must define all 32 bits of the register —
/// hence a `#[repr(u32)]` enum rather than `bool` (which would only define `al`).
#[repr(u32)]
enum Verdict {
    /// Fall through into the displaced original path.
    Run = 0,
    /// Divert from the original path (suppress the death, hold the tick, skip the scoring).
    Divert = 1,
}

/// The entry of `GameThread::on_tick`.
const GT_TICK: Site<6> = Site::new(
    0x0043_cc50,
    [0x55, 0x8B, 0xEC, 0x83, 0xE4, 0xF8],
    "game-tick guard",
);
static GT_TICK_CONTINUE_VA: u32 = GT_TICK.after();

// Runs on every `GameThread::on_tick`.
#[unsafe(naked)]
unsafe extern "C" fn gt_tick_trampoline() -> ! {
    naked_asm!(
        "push ecx",
        "push ebp",
        "mov ebp, esp",
        "and esp, -16",
        "call {handler}",
        "mov esp, ebp",
        "pop ebp",
        "test eax, eax",
        "pop ecx",
        "jnz 2f",
        "push ebp",
        "mov ebp, esp",
        "and esp, -8",
        "jmp dword ptr [{cont}]",
        "2:",
        "mov eax, 1",
        "ret",
        handler = sym on_gt_tick,
        cont = sym GT_TICK_CONTINUE_VA,
    )
}

/// Returns [`Verdict::Divert`] while the guard is holding this tick back.
extern "C" fn on_gt_tick() -> Verdict {
    let thread = MainThread::current();
    let status = HANDOFF.status();
    let run = with_lifecycle(thread, |lc| guard_tick(thread, lc, status));
    if run { Verdict::Run } else { Verdict::Divert }
}

fn guard_tick(thread: MainThread, lc: &mut Lifecycle, status: LoadStatus) -> bool {
    let (params, generation) = match lc.tick_decision(status) {
        TickDecision::Run => return true,
        TickDecision::Hold => return false,
        TickDecision::Consume { params, generation } => (params, generation),
    };

    if unsafe { read::<u32>(GUI_PTR_VA) } == 0 {
        lc.defer_consume();
        return false;
    }

    // SAFETY: The loader has published, so nothing else touches ECL or resource memory for the rest of this tick.
    let token = unsafe { MainToken::new(thread) };
    let stage = unsafe { read::<u32>(STAGE_CURRENT_VA) };

    unsafe { revert_previous_warp(token, stage, lc.last_warp_mut()) };

    let (outcome, section) = if params.active {
        unsafe { apply_warp(token, stage, generation, &params, lc.last_warp_mut()) }
    } else {
        (Outcome::Vanilla, 0)
    };
    let record = PENDING_RECORD.replace(thread, None);
    // This frame's input hook placed the player before the player's own update (see `place_before_consume`),
    // so that update has already moved it from the start position. Holding this one tick makes the next frame the stage's first.
    let landed = matches!(outcome, Outcome::Applied | Outcome::Vanilla);
    if landed {
        write_resources(token, &params);
        if let Some(record) = &record {
            unsafe { apply_record(token, record) };
        }
        write_rng(token, params.rng_seed);
        STEP_INTERVAL.set(thread, generation, params.step_interval);
        FLAGS.set(thread, generation, params.flags);
    }
    lc.finish_reload(generation, outcome, section);
    !landed
}

/// Reverts the previous warp's patches if `current_stage` matches the recorded stage patched.
///
/// # Safety
///
/// This must only be called while the enemy manager pointer is valid to dereference
/// (e.g. in a loaded stage or once the loader has published).
unsafe fn revert_previous_warp(
    token: MainToken,
    current_stage: u32,
    last_warp: &mut Option<LiveWarp>,
) {
    let Some(live) = last_warp.take() else {
        return;
    };
    if live.stage != current_stage {
        return;
    }
    if !unsafe { live.undo.revert(token) } {
        *last_warp = Some(live);
    }
}

/// Patches the loaded ECL to the target section, returning the reset's [`Outcome`] and the now-live section.
/// The section is `0` if the outcome is not [`Outcome::Applied`].
///
/// This function should be called within [`on_gt_tick`] on the first post-load tick before the tick body decodes any ECL.
/// This ensures that the game's ECL VM doesn't run the unpatched script.
///
/// # Safety
///
/// The loader must have published an armed load; see `LoadHandoff::enter` in `load.rs`.
unsafe fn apply_warp(
    token: MainToken,
    stage: u32,
    generation: Generation,
    params: &PracticeParams,
    last_warp: &mut Option<LiveWarp>,
) -> (Outcome, u32) {
    if section_stage(params.section) != Some(stage) {
        // The reload happened, but the requested section is not on the loaded stage.
        return (Outcome::FailedStageMismatch, 0);
    }
    // SAFETY: The ECL files are loaded by the time the loader publishes, and they stay loaded until the next teardown.
    let Some(mut ecl) = (unsafe { Ecl::new(token) }) else {
        return (Outcome::FailedNoEcl, 0);
    };

    let intent = unsafe { apply_section(&mut ecl, params.section, params.phase) };
    intent.schedule(token.thread(), generation);
    *last_warp = Some(LiveWarp {
        stage,
        undo: ecl.take_undo(),
    });
    (Outcome::Applied, params.section)
}

/// Writes the reset's starting resources.
fn write_resources(token: MainToken, p: &PracticeParams) {
    // `PracticeParams::parse` clamped these, so `score / 10` and `value * 100` fit in 32-bit words.
    #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    unsafe {
        write(token, SCORE_DIV10_VA, (p.score / 10) as u32);
        write(token, GRAZE_VA, p.graze as u32);
        write(token, VALUE_VA, (p.value * 100) as u32);
        write(token, POWER_VA, p.power as u32);
        write(token, LIVES_VA, p.lives as u32);
        write(token, LIFE_FRAGMENTS_VA, p.life_fragments as u32);
        write(token, BOMBS_VA, p.bombs as u32);
        write(token, BOMB_FRAGMENTS_VA, p.bomb_fragments as u32);
    }
}

/// Applies a stage record the way the game's replay playback does at a stage start.
///
/// # Safety
///
/// This must run on the reset's consuming tick, after the load published and before the tick body.
unsafe fn apply_record(_token: MainToken, record: &StageRecord) {
    let bytes = &record.0;
    unsafe {
        copy_nonoverlapping(
            bytes[RECORD_GLOBALS..RECORD_GLOBALS + GLOBALS_INNER_LEN].as_ptr(),
            with_exposed_provenance_mut::<u8>(GLOBALS_VA),
            GLOBALS_INNER_LEN,
        );
        if let Some(game_thread) = read_ptr(GAME_THREAD_PTR_VA) {
            copy_nonoverlapping(
                bytes[INFO_GAME_THREAD..INFO_GAME_THREAD + GAME_THREAD_INNER_LEN].as_ptr(),
                with_exposed_provenance_mut::<u8>(game_thread + GAME_THREAD_INNER),
                GAME_THREAD_INNER_LEN,
            );
        }
    }
}

/// Returns the player's start position for a reset, in 1/128 units.
fn start_position(params: &PracticeParams, record: Option<&StageRecord>) -> [i32; 2] {
    if let Some(record) = record {
        let bytes = &record.0;
        return [
            i32::from_le_bytes(
                bytes[RECORD_PLAYER_POS..RECORD_PLAYER_POS + 4]
                    .try_into()
                    .unwrap(),
            ),
            i32::from_le_bytes(
                bytes[RECORD_PLAYER_POS + 4..RECORD_PLAYER_POS + 8]
                    .try_into()
                    .unwrap(),
            ),
        ];
    }
    #[expect(clippy::cast_possible_truncation)]
    [
        (params.player_x * 128.0).round() as i32,
        (params.player_y * 128.0).round() as i32,
    ]
}

/// Places the player on the input hook of the frame whose tick consumes the reset.
/// This should be called exactly once per input-hook frame while connected, after `observe_loads`.
pub(crate) fn place_before_consume(token: MainToken) {
    let thread = token.thread();
    let status = HANDOFF.status();
    let Some(params) = with_lifecycle(thread, |lc| lc.pending_consume(status)) else {
        return;
    };
    let record = PENDING_RECORD.replace(thread, None);
    let position = start_position(&params, record.as_deref());
    PENDING_RECORD.set(thread, record);
    if place_player(token, position) {
        with_lifecycle(thread, Lifecycle::mark_placed);
    }
}

const PLAYER_SET_POSITION_VA: usize = 0x0045_b510;

/// Places the player at a fixed-point position (1/128 units) through the game's own routine.
/// Returns `false` if there is no player, in which case this function is a no-op.
fn place_player(_token: MainToken, fixed: [i32; 2]) -> bool {
    if unsafe { read_ptr(PLAYER_PTR_VA) }.is_none() {
        return false;
    }
    let set_position = unsafe {
        transmute::<*const (), unsafe extern "stdcall" fn(*const [i32; 2])>(
            with_exposed_provenance(PLAYER_SET_POSITION_VA),
        )
    };
    unsafe { set_position(&raw const fixed) };
    true
}

/// Seeds both RNGs and zeroes the replay-safe call counter.
/// This mirrors the game's own stage start and runs on the reset's first tick, before any ECL.
fn write_rng(token: MainToken, seed: u16) {
    unsafe {
        write(token, RNG_VA, seed);
        write(token, RNG_UNSAFE_VA, seed);
        write(token, RNG_COUNT_VA, 0u32);
    }
}

/// The number of live stage frames the input hook has seen for the current load.
static STAGE_FRAMES: PerLoad<u32> = PerLoad::new(0);

/// Game frames per RL step for the current load.
static STEP_INTERVAL: PerLoad<u32> = PerLoad::new(READ_INTERVAL);

/// The current load's episode flags (`FLAG_*`).
static FLAGS: PerLoad<u32> = PerLoad::new(0);

/// Returns whether the current load has `flag` set.
pub(crate) fn episode_flag(thread: MainThread, flag: u32) -> bool {
    FLAGS.get(thread, load_generation()) & flag != 0
}

/// Counts this frame as a live stage frame of the current load and returns whether an RL step is due on it.
/// Returns `None` outside a live stage or until the tick guard has consumed the latest load.
/// This should be called exactly once per input-hook frame.
pub(crate) fn stage_step_due(thread: MainThread) -> Option<bool> {
    if !unsafe { game_live() } {
        return None;
    }
    let generation = load_generation();
    if with_lifecycle(thread, |lc| lc.wire().0) != generation {
        return None;
    }
    let interval = STEP_INTERVAL.get(thread, generation);
    STAGE_FRAMES.update(thread, generation, |count| {
        let index = *count;
        *count = count.wrapping_add(1);
        Some(index.is_multiple_of(interval))
    })
}

const STAGE_LOADER_THREAD_ENTRY_VA: u32 = 0x0043_c690;

const STAGE_LOADER_THREAD_SPAWN_VA: u32 = 0x0043_cbde;

/// Marks the load in flight, runs `GameThread::thread_start`, then publishes its outcome.
unsafe extern "C" fn stage_loader_thread(arg: *mut c_void) -> u32 {
    HANDOFF.enter();
    let original = unsafe {
        transmute::<*const (), unsafe extern "C" fn(*mut c_void) -> u32>(with_exposed_provenance(
            STAGE_LOADER_THREAD_ENTRY_VA as usize,
        ))
    };
    let result = unsafe { original(arg) };
    HANDOFF.publish();
    result
}

/// Call once during `DLL_PROCESS_ATTACH`.
///
/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
pub(crate) unsafe fn install() {
    unsafe {
        const {
            NearBranchSite::new(0x0043_c02a, 0x85, 0x0043_c504, "loader quit-bail skip").skip()
        }
        .apply();
        #[expect(clippy::cast_possible_truncation)]
        Site::new(
            STAGE_LOADER_THREAD_SPAWN_VA,
            op_abs32(0x68, STAGE_LOADER_THREAD_ENTRY_VA),
            "loader thread entry redirect",
        )
        .patch(op_abs32(
            0x68,
            (stage_loader_thread as *mut ()).expose_provenance() as u32,
        ))
        .apply();
        GT_TICK.jmp(gt_tick_trampoline as *mut ());
        chapter::install();
        hit::install();
    }
}

#[cfg(test)]
mod test_support {
    use super::{PARAMS_LEN, PracticeParams, WireParams};
    use std::mem::transmute;

    /// Returns a valid `PARAMS_LEN`-byte params block with the given categorical selectors.
    pub(super) fn wire_bytes(section: u32, active: u32, phase: u32) -> [u8; PARAMS_LEN] {
        let wire = WireParams {
            section,
            active,
            score: 0,
            graze: 0,
            value: 0,
            power: 400,
            lives: 2,
            life_fragments: 0,
            bombs: 3,
            bomb_fragments: 0,
            phase,
            difficulty: 3,
            character: 0,
            rng_seed: 0xBEEF,
            step_interval: 3,
            flags: 0,
            initial_action: 0,
            player_x: 0.0,
            player_y: 400.0,
        };
        // SAFETY: `WireParams` is `#[repr(C)]` with only integer fields and no padding, so every byte is initialized.
        unsafe { transmute(wire) }
    }

    /// Returns parsed params for the categorical selectors.
    pub(super) fn params(section: u32, active: u32, phase: u32) -> PracticeParams {
        PracticeParams::parse(&wire_bytes(section, active, phase)).expect("valid params")
    }

    /// Returns parsed params at `difficulty` rather than the default of Lunatic (`3`).
    pub(super) fn params_at(
        section: u32,
        active: u32,
        phase: u32,
        difficulty: i32,
    ) -> PracticeParams {
        let mut bytes = wire_bytes(section, active, phase);
        bytes[48..52].copy_from_slice(&difficulty.to_le_bytes());
        PracticeParams::parse(&bytes).expect("valid params")
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{params, wire_bytes};
    use super::{KNOWN_FLAGS, MAX_STEP_INTERVAL, PARAMS_LEN, PracticeParams};

    #[test]
    fn parse_accept_valid_params() {
        let params = params(1202, 1, 0);
        assert!(params.active);
        assert_eq!(params.section, 1202);
        assert_eq!(params.difficulty, 3);
        assert_eq!(params.character, 0);
    }

    #[test]
    fn parse_length_exact() {
        let bytes = wire_bytes(1202, 1, 0);
        assert!(PracticeParams::parse(&bytes[..PARAMS_LEN - 1]).is_none());
        let mut long = bytes.to_vec();
        long.push(0);
        assert!(PracticeParams::parse(&long).is_none());
    }

    #[test]
    fn parse_reject_out_of_range_selectors() {
        // Difficulty 5 does not exist.
        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[48..52].copy_from_slice(&5i32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
        // Stage 9 does not exist.
        assert!(PracticeParams::parse(&wire_bytes(9101, 1, 0)).is_none());
    }

    #[test]
    fn parse_extra_stage_and_difficulty() {
        let mut bytes = wire_bytes(7201, 1, 0);

        assert!(PracticeParams::parse(&bytes).is_none());

        bytes[48..52].copy_from_slice(&4i32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_some());

        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[48..52].copy_from_slice(&4i32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());

        let mut bytes = wire_bytes(0, 0, 0);
        bytes[48..52].copy_from_slice(&4i32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_some());
    }

    #[test]
    fn parse_ignore_warp_fields_when_inactive() {
        let params = params(9101, 0, 0);
        assert!(!params.active);
        assert_eq!(params.difficulty, 3);
    }

    #[test]
    fn parse_reject_wide_rng_seed() {
        let mut bytes = wire_bytes(1202, 1, 0);
        assert_eq!(PracticeParams::parse(&bytes).unwrap().rng_seed, 0xBEEF);
        bytes[56..60].copy_from_slice(&0x1_0000u32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
        bytes[56..60].copy_from_slice(&0xFFFFu32.to_le_bytes());
        assert_eq!(PracticeParams::parse(&bytes).unwrap().rng_seed, 0xFFFF);
    }

    #[test]
    fn parse_reject_non_finite_player_position() {
        let mut bytes = wire_bytes(1202, 1, 0);
        assert!(PracticeParams::parse(&bytes).is_some());
        bytes[72..76].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
    }

    #[test]
    fn parse_reject_initial_action_outside_mask() {
        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[68..72].copy_from_slice(&0x9u32.to_le_bytes());
        assert_eq!(PracticeParams::parse(&bytes).unwrap().initial_action(), 0x9);
        bytes[68..72].copy_from_slice(&0x100u32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
    }

    #[test]
    fn parse_reject_unknown_flags() {
        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[64..68].copy_from_slice(&KNOWN_FLAGS.to_le_bytes());
        assert_eq!(PracticeParams::parse(&bytes).unwrap().flags, KNOWN_FLAGS);
        bytes[64..68].copy_from_slice(&(KNOWN_FLAGS | (1 << 31)).to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
    }

    #[test]
    fn parse_reject_step_interval_outside_bounds() {
        let mut bytes = wire_bytes(1202, 1, 0);
        assert_eq!(PracticeParams::parse(&bytes).unwrap().step_interval, 3);
        bytes[60..64].copy_from_slice(&0u32.to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
        bytes[60..64].copy_from_slice(&(MAX_STEP_INTERVAL + 1).to_le_bytes());
        assert!(PracticeParams::parse(&bytes).is_none());
        bytes[60..64].copy_from_slice(&MAX_STEP_INTERVAL.to_le_bytes());
        assert_eq!(
            PracticeParams::parse(&bytes).unwrap().step_interval,
            MAX_STEP_INTERVAL
        );
    }

    #[test]
    fn parse_clamp_resources() {
        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[28..32].copy_from_slice(&99i32.to_le_bytes()); // lives
        let params = PracticeParams::parse(&bytes).unwrap();
        assert_eq!(params.lives, 8);
    }
}
