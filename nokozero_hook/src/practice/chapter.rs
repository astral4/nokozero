//! Patches for chapter semantics and state.

use super::catalog::ChapterIntent;
use super::load::{Generation, load_generation};
use crate::addrs::{CURRENT_CHAPTER_VA, ENEMIES_SPAWNED_IN_CHAPTER_VA};
use crate::mem::write;
use crate::patch::Site;
use crate::thread::{MainThread, MainToken, PerLoad};
use std::arch::naked_asm;

/// The committed warp's [`ChapterIntent`].
#[derive(Clone, Copy)]
struct ScheduledIntent {
    /// How many more chapter-end events should skip completion scoring.
    skip_remaining: i32,
    /// The value of `ECLSetChapter(n)` to be set as the current chapter.
    set_chapter: Option<i32>,
    /// Whether the extra stage chapter bonus write is needed. See [`on_st7_chapter_bonus`].
    st7_bonus: bool,
}

impl ScheduledIntent {
    const NONE: Self = Self {
        skip_remaining: 0,
        set_chapter: None,
        st7_bonus: false,
    };
}

static SCHEDULED_INTENT: PerLoad<ScheduledIntent> = PerLoad::new(ScheduledIntent::NONE);

impl ChapterIntent {
    /// Schedules the intent recorded by the committed warp for this load's chapter hooks.
    pub(super) fn schedule(self, thread: MainThread, generation: Generation) {
        SCHEDULED_INTENT.set(
            thread,
            generation,
            ScheduledIntent {
                skip_remaining: self.skip,
                set_chapter: self.set,
                st7_bonus: self.st7_bonus,
            },
        );
    }
}

const CHAPTER_SCORE: Site<6> = Site::new(
    0x0043_d0ad,
    [0x8b, 0x87, 0xb0, 0x00, 0x00, 0x00], // `mov eax, dword ptr [edi + 0xb0]`
    "chapter-score detour",
);

static CHAPTER_SCORE_CONTINUE_VA: u32 = CHAPTER_SCORE.after();

static CHAPTER_SCORE_SKIP_VA: u32 = 0x0043_d0d5;

#[unsafe(naked)]
unsafe extern "C" fn chapter_score_trampoline() -> ! {
    naked_asm!(
        "push ebp",
        "mov ebp, esp",
        "and esp, -16",
        "call {handler}",
        "mov esp, ebp",
        "pop ebp",
        "test eax, eax",
        "jnz 2f",
        "mov eax, dword ptr [edi + 0xb0]",
        "jmp dword ptr [{cont}]",
        "2:",
        "jmp dword ptr [{skip}]",
        handler = sym on_chapter_score,
        cont = sym CHAPTER_SCORE_CONTINUE_VA,
        skip = sym CHAPTER_SCORE_SKIP_VA,
    )
}

/// Returns `1` to suppress the current chapter-end event's completion scoring. Returns `0` to let it run.
extern "C" fn on_chapter_score() -> u32 {
    let thread = MainThread::current();
    let suppressed = SCHEDULED_INTENT.update(thread, load_generation(), |intent| {
        if intent.skip_remaining == 0 {
            None
        } else {
            intent.skip_remaining -= 1;
            Some(())
        }
    });
    u32::from(suppressed.is_some())
}

const CHAPTER_SET: Site<6> = Site::new(
    0x0043_dd58,
    [0x8d, 0x93, 0x9c, 0x00, 0x00, 0x00],
    "chapter-set detour",
);

static CHAPTER_SET_CONTINUE_VA: u32 = CHAPTER_SET.after();

#[unsafe(naked)]
unsafe extern "C" fn chapter_set_trampoline() -> ! {
    naked_asm!(
        "push ebp",
        "mov ebp, esp",
        "and esp, -16",
        "call {handler}",
        "mov esp, ebp",
        "pop ebp",
        "lea edx, [ebx + 0x9c]",
        "jmp dword ptr [{cont}]",
        handler = sym on_chapter_set,
        cont = sym CHAPTER_SET_CONTINUE_VA,
    )
}

extern "C" fn on_chapter_set() {
    let thread = MainThread::current();
    if let Some(value) = SCHEDULED_INTENT.update(thread, load_generation(), |intent| {
        intent.set_chapter.take()
    }) {
        // SAFETY: This runs inside the game's own Pointdevice snapshot routine on the update thread,
        // so nothing is concurrently touching `CURRENT_CHAPTER`.
        unsafe {
            let token = MainToken::new(thread);
            write(token, CURRENT_CHAPTER_VA, value);
        }
    }
}

const ST7_CHAPTER_BONUS: Site<5> = Site::new(
    0x0043_dece,
    [0xc2, 0x04, 0x00, 0xcc, 0xcc],
    "st7 chapter-bonus detour",
);

#[unsafe(naked)]
unsafe extern "C" fn st7_chapter_bonus_trampoline() -> ! {
    naked_asm!(
        "push eax",
        "push ebp",
        "mov ebp, esp",
        "and esp, -16",
        "call {handler}",
        "mov esp, ebp",
        "pop ebp",
        "pop eax",
        "ret 4",
        handler = sym on_st7_chapter_bonus,
    )
}

extern "C" fn on_st7_chapter_bonus() {
    let thread = MainThread::current();
    let owed = SCHEDULED_INTENT.update(thread, load_generation(), |intent| {
        if intent.st7_bonus {
            intent.st7_bonus = false;
            Some(())
        } else {
            None
        }
    });
    if owed.is_none() {
        return;
    }
    // SAFETY: This runs inside the game's own chapter-scoring path on the update thread, so nothing is concurrently touching the flag.
    unsafe {
        let token = MainToken::new(thread);
        write(token, ENEMIES_SPAWNED_IN_CHAPTER_VA, 1i32);
    }
}

/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
pub(super) unsafe fn install() {
    unsafe {
        CHAPTER_SCORE.jmp(chapter_score_trampoline as *mut ());
        CHAPTER_SET.jmp(chapter_set_trampoline as *mut ());
        ST7_CHAPTER_BONUS.jmp(st7_chapter_bonus_trampoline as *mut ());
    }
}
