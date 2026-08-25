//! Logic and hooking for the player death sequence.

use crate::ipc::is_connected;
use crate::patch::Site;
use crate::practice::{Generation, load_generation};
use crate::thread::{MainThread, PerLoad};
use std::arch::naked_asm;
use std::mem::take;

const PLAYER_DIE: Site<6> = Site::new(
    0x0045_6540,
    [0x55, 0x8b, 0xec, 0x83, 0xec, 0x14], // `push ebp; mov ebp, esp; sub esp, 0x14`
    "player-die detour",
);
static PLAYER_DIE_CONTINUE_VA: u32 = PLAYER_DIE.after();

/// The number of suppressed `Player::die` calls since the most recent stage load.
static HITS: PerLoad<u32> = PerLoad::new(0);

/// Set when [`HITS`] goes from 0 to 1. Cleared when the input hook consumes it.
static FORCED_STEP_OWED: PerLoad<bool> = PerLoad::new(false);

/// Returns the number of hits for the stage associated with `generation`.
pub(crate) fn count(thread: MainThread, generation: Generation) -> u32 {
    HITS.get(thread, generation)
}

/// Consumes the episode's pending first-hit forced step, returning whether one was owed.
#[must_use]
pub(crate) fn take_forced_step(thread: MainThread) -> bool {
    FORCED_STEP_OWED
        .update(thread, load_generation(), |owed| take(owed).then_some(()))
        .is_some()
}

/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
pub(crate) unsafe fn install() {
    unsafe {
        PLAYER_DIE.jmp(player_die_trampoline as *mut ());
    }
}

#[unsafe(naked)]
unsafe extern "C" fn player_die_trampoline() -> ! {
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
        "sub esp, 0x14",
        "jmp dword ptr [{cont}]",
        "2:",
        "ret",
        handler = sym on_player_die,
        cont = sym PLAYER_DIE_CONTINUE_VA,
    )
}

/// Returning `1` suppresses the player death sequence. Returning `0` lets it run.
extern "C" fn on_player_die() -> u32 {
    if !is_connected() {
        return 0;
    }
    let thread = MainThread::current();
    let generation = load_generation();
    let first_hit = HITS.update(thread, generation, |hits| {
        let first = *hits == 0;
        *hits = hits.wrapping_add(1);
        Some(first)
    });
    if first_hit == Some(true) {
        FORCED_STEP_OWED.set(thread, generation, true);
    }
    1
}
