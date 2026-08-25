//! Practice navigation, stage-load interception, resource injection, and ECL section patching.

mod catalog;
mod chapter;
mod data;
mod ecl;
mod load;
#[forbid(unsafe_code)]
mod machine;

pub(crate) use crate::practice::load::{Generation, load_generation};
pub(crate) use crate::practice::machine::Outcome;

use crate::practice::catalog::apply_section;
use crate::practice::data::section_stage;
use crate::practice::ecl::{Ecl, EclUndo};
use crate::practice::load::HANDOFF;
use crate::practice::machine::{Lifecycle, ReloadPlan};

use crate::addrs::{
    BOMB_FRAGMENTS_VA, BOMBS_VA, CHARACTER_VA, DIFFICULTY_VA, GAMEMODE_INGAME, GAMEMODE_RETRY,
    GAMEMODE_TO_SWITCH_TO_VA, GRAZE_VA, GUI_PTR_VA, LIFE_FRAGMENTS_VA, LIVES_VA, LOADER_RUNNING_VA,
    POWER_VA, SCORE_DIV10_VA, STAGE_CURRENT_VA, STAGE_SELECT_VA, VALUE_VA,
};
use crate::mem::{read, stage_stable, write};
use crate::patch::{NearBranchSite, Site, op_abs32};
use crate::thread::{MainCell, MainThread, MainToken};
use std::arch::naked_asm;
use std::ffi::c_void;
use std::mem::transmute;
use std::process::abort;
use std::ptr::with_exposed_provenance;

#[derive(Clone, Copy)]
pub(crate) struct PracticeParams {
    /// Whether to warp/inject at the next stage load.
    active: bool,
    /// The warp target ID; see the section ID constants.
    section: u32,
    /// The sub-phase within the section.
    phase: u32,
    /// The difficulty (0-4) applied at the next reset. Negative = leave as-is.
    difficulty: i32,
    /// The character (0–3) applied at the next reset. Negative = leave as-is.
    character: i32,
    score: i64,
    graze: i32,
    value: i32,
    power: i32,
    lives: i32,
    life_fragments: i32,
    bombs: i32,
    bomb_fragments: i32,
}

/// The RESET params block. Its resource run is in the same order as the observation meta
/// block's (`features::put_meta`), so a value read off an observation replays into the
/// field of the same name.
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
}

pub(crate) const PARAMS_LEN: usize = size_of::<WireParams>();

const _: () = assert!(PARAMS_LEN == 56, "wire params block is fixed at 56 bytes");

/// Validates and decodes a RESET command's params block. Returns `None` if there is a protocol violation.
pub(crate) fn parse_params(payload: &[u8]) -> Option<PracticeParams> {
    if payload.len() != size_of::<WireParams>() {
        return None;
    }
    // SAFETY: The length check guarantees `size_of::<WireParams>()` readable bytes.
    // `WireParams` has only integer fields, so every bit pattern is valid. `read_unaligned` tolerates the buffer's alignment.
    let wire = unsafe { payload.as_ptr().cast::<WireParams>().read_unaligned() };

    if wire.difficulty > 4 || wire.character > 3 {
        return None;
    }
    if wire.active != 0 && section_stage(wire.section).is_none() {
        return None;
    }
    if wire.active != 0 && (section_stage(wire.section) == Some(7)) != (wire.difficulty == 4) {
        return None;
    }

    let score = wire.score.clamp(0, 9_999_999_990);
    let graze = wire.graze.clamp(0, 999_999);
    let value = wire.value.clamp(0, 999_990);
    let power = wire.power.clamp(0, 400);
    let lives = wire.lives.clamp(0, 8);
    // 5 fragments per life in Extra; 3 fragments per life elsewhere
    let life_fragments = wire
        .life_fragments
        .clamp(0, if wire.difficulty == 4 { 5 } else { 3 });
    let bombs = wire.bombs.clamp(0, 8);
    let bomb_fragments = wire.bomb_fragments.clamp(0, 4);

    Some(PracticeParams {
        active: wire.active != 0,
        section: wire.section,
        phase: wire.phase,
        difficulty: wire.difficulty,
        character: wire.character,
        score,
        graze,
        value,
        power,
        lives,
        life_fragments,
        bombs,
        bomb_fragments,
    })
}

struct LiveWarp {
    /// The stage affected by the ECL patches.
    stage: u32,
    undo: EclUndo,
}

/// The value is `None` while the value is checked out (i.e. inside one of the transition
/// wrappers below, or between [`ConsumeGrant::begin`] and [`ConsumeGrant::finish`]).
static LIFECYCLE: MainCell<Option<Lifecycle>> = MainCell::new(Some(Lifecycle::INIT));

/// Takes the lifecycle out of its cell, aborting if it is already checked out.
fn take_lifecycle(thread: MainThread) -> Lifecycle {
    if let Some(lc) = LIFECYCLE.replace(thread, None) {
        lc
    } else {
        eprintln!("nokozero_hook: practice: lifecycle re-entered");
        abort();
    }
}

/// Returns the checked-out lifecycle to its cell.
fn put_lifecycle(thread: MainThread, lc: Lifecycle) {
    LIFECYCLE.set(thread, Some(lc));
}

/// Accepts a decoded RESET command. Returns `false` if a reset is already in flight.
#[must_use]
pub(crate) fn accept_reset(thread: MainThread, seq: u32, params: PracticeParams) -> bool {
    let mut lc = take_lifecycle(thread);
    let accepted = lc.accept(seq, params);
    put_lifecycle(thread, lc);
    accepted
}

/// Observes one supervisor frame for the load counter and "loader running" flag.
fn observe(thread: MainThread, generation: Generation, loader_running: bool) {
    let mut lc = take_lifecycle(thread);
    lc.observe(generation, loader_running);
    put_lifecycle(thread, lc);
}

/// Attempts `Requested` -> `Reloading`, returning the writes for the caller to perform.
/// Returns `None` if the state should stay `Requested`.
fn try_start_reload(thread: MainThread, stable_ingame: bool) -> Option<ReloadPlan> {
    let mut lc = take_lifecycle(thread);
    let plan = lc.try_start_reload(stable_ingame);
    put_lifecycle(thread, lc);
    plan
}

/// Returns whether the tick guard should be holding ticks back.
fn reset_reloading(thread: MainThread) -> bool {
    let lc = take_lifecycle(thread);
    let reloading = lc.reset_reloading();
    put_lifecycle(thread, lc);
    reloading
}

/// Returns the wire values `(load_generation, reset_seq, reset_outcome, applied_section)`.
fn wire(thread: MainThread) -> (Generation, u32, Outcome, u32) {
    let lc = take_lifecycle(thread);
    let tuple = lc.wire();
    put_lifecycle(thread, lc);
    tuple
}

/// Guard for exclusive checkout of the lifecycle across ECL patch application/reversal and resource writes.
#[must_use]
struct ConsumeGrant {
    lc: Lifecycle,
}

impl ConsumeGrant {
    /// Checks the lifecycle out for the guard's consuming tick, taking the value out of [`LIFECYCLE`].
    /// Aborts if the reset state isn't `Reloading`.
    fn begin(thread: MainThread) -> Self {
        let lc = take_lifecycle(thread);
        if !lc.reset_reloading() {
            eprintln!("nokozero_hook: practice: consume began without a reload in progress");
            abort();
        }
        Self { lc }
    }

    /// Returns the in-progress reset's warp params. `None` means vanilla behavior.
    fn params(&self) -> Option<PracticeParams> {
        self.lc.reloading_params()
    }

    /// Returns the live warp's undo slot for [`revert_previous_warp`] and [`apply_managed_warp`] to operate on in place.
    fn last_warp(&mut self) -> &mut Option<LiveWarp> {
        self.lc.last_warp_mut()
    }

    /// Puts the lifecycle value back into [`LIFECYCLE`] with the indicated `generation`, `outcome`, and `section` applied.
    fn finish(self, thread: MainThread, generation: Generation, outcome: Outcome, section: u32) {
        let mut lc = self.lc;
        lc.finish_managed(generation, outcome, section);
        put_lifecycle(thread, lc);
    }
}

pub(crate) struct WireMeta {
    pub(crate) load_generation: Generation,
    pub(crate) reset_seq: u32,
    pub(crate) reset_outcome: Outcome,
    pub(crate) applied_section: u32,
}

impl WireMeta {
    #[must_use]
    pub(crate) fn read(thread: MainThread) -> Self {
        let (load_generation, reset_seq, reset_outcome, applied_section) = wire(thread);
        Self {
            load_generation,
            reset_seq,
            reset_outcome,
            applied_section,
        }
    }
}

/// The entry of `GameThread::on_tick`.
const GT_TICK: Site<6> = Site::new(
    0x0043_cc50,
    [0x55, 0x8b, 0xec, 0x83, 0xe4, 0xf8],
    "game-tick guard",
);
static GT_TICK_CONTINUE_VA: u32 = GT_TICK.after();

const RUN_TICK: u32 = 0;
const SKIP_TICK: u32 = 1;

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

/// Returns whether the guard is holding this tick back. The guard only holds while a reset's reload is in progress.
extern "C" fn on_gt_tick() -> u32 {
    let thread = MainThread::current();

    if !reset_reloading(thread) {
        return RUN_TICK;
    }

    let gui = unsafe { read::<u32>(GUI_PTR_VA) };
    if gui == 0 {
        return SKIP_TICK;
    }

    let Some((generation, managed)) = HANDOFF.consume() else {
        return SKIP_TICK;
    };

    let mut grant = ConsumeGrant::begin(thread);

    // SAFETY: The stage loader thread no longer touches ECL/resource memory from this point onward in the tick.
    let token = unsafe { MainToken::new(thread) };
    let stage = unsafe { read::<u32>(STAGE_CURRENT_VA) };

    unsafe { revert_previous_warp(token, stage, grant.last_warp()) };

    let (reset_outcome, section) = if managed {
        let params = grant.params();
        unsafe { apply_managed_warp(token, stage, generation, params, grant.last_warp()) }
    } else {
        (Outcome::Vanilla, 0)
    };

    grant.finish(thread, generation, reset_outcome, section);

    RUN_TICK
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

/// Writes the starting resources and patches the loaded ECL to the target section, returning the reset's [`Outcome`]
/// and the now-live section. The section is `0` if the outcome is not [`Outcome::Applied`].
///
/// This function should be called within [`on_gt_tick`] on the first post-load tick before the tick body decodes any ECL.
/// This ensures that the game's ECL VM doesn't run the unpatched script.
///
/// # Safety
///
/// The loader must have published a managed outcome; see `LOAD_APPLY` in `load.rs`.
unsafe fn apply_managed_warp(
    token: MainToken,
    stage: u32,
    generation: Generation,
    params: Option<PracticeParams>,
    last_warp: &mut Option<LiveWarp>,
) -> (Outcome, u32) {
    let Some(params) = params else {
        return (Outcome::Vanilla, 0);
    };
    if section_stage(params.section) != Some(stage) {
        // The reload happened, but no managed section is live.
        return (Outcome::FailedStageMismatch, 0);
    }
    // SAFETY: The ECL files are loaded by the time the loader publishes, and they stay loaded until the next teardown.
    let Some(mut ecl) = (unsafe { Ecl::new(token) }) else {
        return (Outcome::FailedNoEcl, 0);
    };

    let intent = unsafe { apply_section(&mut ecl, params.section, params.phase) };
    let result = if let Some(intent) = intent {
        intent.arm(token.thread(), generation);
        write_resources(token, &params);
        (Outcome::Applied, params.section)
    } else {
        (Outcome::FailedUnmapped, 0)
    };
    // `revert_previous_warp` ran earlier this tick and cleared the prior record, so this record can be stored for the next load to undo.
    *last_warp = Some(LiveWarp {
        stage,
        undo: ecl.take_undo(),
    });
    result
}

/// Writes the reset's starting resources.
fn write_resources(token: MainToken, p: &PracticeParams) {
    // `parse_params` clamped these, so `score / 10` and `value * 100` fit in 32-bit words.
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

pub(crate) fn observe_loads(thread: MainThread) {
    let generation = load_generation();
    let loader_running = unsafe { read::<u32>(LOADER_RUNNING_VA) } == 1;
    observe(thread, generation, loader_running);
}

/// Starts a requested reset's reload. The request remains pending unless the game is in stable gameplay,
/// no other scene switch is queued, and the previous stage loader thread has fully exited.
/// This should only be called from the input hook after [`observe_loads`] within the same tick.
pub(crate) fn apply_pending_reset(token: MainToken) {
    let thread = token.thread();
    let stable =
        unsafe { stage_stable() && read::<u32>(GAMEMODE_TO_SWITCH_TO_VA) == GAMEMODE_INGAME };
    let Some(plan) = try_start_reload(thread, stable) else {
        return;
    };

    HANDOFF.arm(plan.managed);

    if let Some(stage) = plan.stage_select {
        unsafe { write(token, STAGE_SELECT_VA, stage) };
    }
    if let Some(difficulty) = plan.difficulty {
        unsafe { write(token, DIFFICULTY_VA, difficulty) };
    }
    if let Some(character) = plan.character {
        unsafe { write(token, CHARACTER_VA, character) };
    }
    unsafe { write(token, GAMEMODE_TO_SWITCH_TO_VA, GAMEMODE_RETRY) };
}

const STAGE_LOADER_THREAD_ENTRY_VA: u32 = 0x0043_c690;

const STAGE_LOADER_THREAD_SPAWN_VA: u32 = 0x0043_cbde;

/// Runs `GameThread::thread_start`, then publishes its outcome.
unsafe extern "C" fn stage_loader_thread(arg: *mut c_void) -> u32 {
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
    }
}

#[cfg(test)]
mod test_support {
    use super::{PARAMS_LEN, PracticeParams, WireParams, parse_params};
    use std::mem::transmute;

    /// Returns a valid 56-byte params block with the given categorical selectors.
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
        };
        // SAFETY: `WireParams` is `#[repr(C)]` with only integer fields and no padding, so every byte is initialized.
        unsafe { transmute(wire) }
    }

    /// Returns parsed params for the categorical selectors.
    pub(super) fn params(section: u32, active: u32, phase: u32) -> PracticeParams {
        parse_params(&wire_bytes(section, active, phase)).expect("valid params")
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{params, wire_bytes};
    use super::{PARAMS_LEN, parse_params};

    #[test]
    fn parse_accept_valid_params() {
        let params = params(1202, 1, 0);
        assert!(params.active);
        assert_eq!(params.section, 1202);
        assert_eq!(params.difficulty, 3);
    }

    #[test]
    fn parse_length_exact() {
        let bytes = wire_bytes(1202, 1, 0);
        assert!(parse_params(&bytes[..PARAMS_LEN - 1]).is_none());
        let mut long = bytes.to_vec();
        long.push(0);
        assert!(parse_params(&long).is_none());
    }

    #[test]
    fn parse_reject_out_of_range_selectors() {
        // Difficulty 5 does not exist.
        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[48..52].copy_from_slice(&5i32.to_le_bytes());
        assert!(parse_params(&bytes).is_none());
        // Stage 9 does not exist.
        assert!(parse_params(&wire_bytes(9101, 1, 0)).is_none());
    }

    #[test]
    fn parse_extra_stage_and_difficulty() {
        let mut bytes = wire_bytes(7201, 1, 0);

        assert!(parse_params(&bytes).is_none());

        bytes[48..52].copy_from_slice(&4i32.to_le_bytes());
        assert!(parse_params(&bytes).is_some());

        bytes[48..52].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(parse_params(&bytes).is_none());

        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[48..52].copy_from_slice(&4i32.to_le_bytes());
        assert!(parse_params(&bytes).is_none());

        let mut bytes = wire_bytes(0, 0, 0);
        bytes[48..52].copy_from_slice(&4i32.to_le_bytes());
        assert!(parse_params(&bytes).is_some());
    }

    #[test]
    fn parse_ignore_warp_fields_when_inactive() {
        let params = params(9101, 0, 0);
        assert!(!params.active);
    }

    #[test]
    fn parse_clamp_resources() {
        let mut bytes = wire_bytes(1202, 1, 0);
        bytes[28..32].copy_from_slice(&99i32.to_le_bytes()); // lives
        let params = parse_params(&bytes).unwrap();
        assert_eq!(params.lives, 8);
    }
}
