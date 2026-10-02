//! Patches for chapter semantics and state.

use super::Verdict;
use super::load::{Generation, PerLoad, load_generation};
use crate::addrs::{CURRENT_CHAPTER_VA, ENEMIES_SPAWNED_IN_CHAPTER_VA};
use crate::mem::{read, write};
use crate::patch::Site;
use crate::thread::{MainThread, MainToken};
use std::arch::naked_asm;
use std::mem::take;

/// Chapter effects requested by the dispatched section and scheduled for the load's chapter hooks.
#[derive(Clone, Copy)]
pub(super) struct ChapterIntent {
    /// How many more chapter-end events should skip completion scoring.
    pub(super) skip_remaining: i32,
    /// The value of `ECLSetChapter(n)` to be set as the current chapter.
    pub(super) set_chapter: Option<i32>,
    /// Whether the extra stage chapter bonus write is needed. See [`on_st7_chapter_bonus`].
    pub(super) st7_bonus: bool,
}

impl ChapterIntent {
    /// The empty intent with no chapter effects.
    pub(super) const NONE: Self = Self {
        skip_remaining: 0,
        set_chapter: None,
        st7_bonus: false,
    };

    /// Schedules the intent recorded by the committed warp for this load's chapter hooks.
    pub(super) fn schedule(self, thread: MainThread, generation: Generation) {
        SCHEDULED_INTENT.set(thread, generation, self);
    }
}

/// The committed warp's [`ChapterIntent`].
static SCHEDULED_INTENT: PerLoad<ChapterIntent> = PerLoad::new(ChapterIntent::NONE);

/// Stage scripts set the chapter value/word on the third live frame of a load.
const COUNT_FROM_FRAME: u32 = 3;

/// Chapter word cap. (There are 81 words in total in the game.)
const COUNTED_WORDS: usize = 256;

/// Chapter load counter.
#[derive(Clone, Copy)]
struct Entries {
    /// The word on the last frame counted.
    current: Option<u32>,
    /// The number of times a load has entered each chapter word.
    counts: [u8; COUNTED_WORDS],
}

impl Entries {
    const NONE: Self = Self {
        current: None,
        counts: [0; COUNTED_WORDS],
    };

    /// Counts a frame showing `word`. Returns whether the word differed from the previous frame's.
    fn show(&mut self, word: u32) -> bool {
        if self.current == Some(word) {
            return false;
        }
        self.current = Some(word);
        if let Some(count) = self.counts.get_mut(word as usize) {
            *count = count.saturating_add(1);
        }
        true
    }

    /// The number of times the current word has been entered so far.
    /// Returns 0 before any frame is counted and for words past [`COUNTED_WORDS`].
    fn count(&self) -> u32 {
        self.current
            .and_then(|word| self.counts.get(word as usize))
            .map_or(0, |&count| u32::from(count))
    }
}

/// The current load's [`Entries`].
static ENTRIES: PerLoad<Entries> = PerLoad::new(Entries::NONE);

/// Counts the chapter word shown by the game on live stage frame `index` of the load of `generation`.
pub(super) fn count_chapter(thread: MainThread, generation: Generation, index: u32) {
    if index < COUNT_FROM_FRAME {
        return;
    }
    // SAFETY: The chapter word is a fixed global, readable whenever a stage is live.
    let word = unsafe { read::<u32>(CURRENT_CHAPTER_VA) };
    ENTRIES.update(thread, generation, |entries| {
        entries.show(word).then_some(())
    });
}

/// How many times the load of `generation` has entered its current chapter word.
pub(super) fn entry_count(thread: MainThread, generation: Generation) -> u32 {
    ENTRIES.get(thread, generation).count()
}

const CHAPTER_SCORE: Site<6> = Site::new(
    0x0043_d0ad,
    [0x8B, 0x87, 0xB0, 0x00, 0x00, 0x00], // `mov eax, dword ptr [edi + 0xb0]`
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

/// Returns [`Verdict::Divert`] to suppress the current chapter-end event's completion scoring.
extern "C" fn on_chapter_score() -> Verdict {
    let thread = MainThread::current();
    let suppressed = SCHEDULED_INTENT.update(thread, load_generation(), |intent| {
        (intent.skip_remaining != 0).then(|| intent.skip_remaining -= 1)
    });
    if suppressed.is_some() {
        Verdict::Divert
    } else {
        Verdict::Run
    }
}

const CHAPTER_SET: Site<6> = Site::new(
    0x0043_dd58,
    [0x8D, 0x93, 0x9C, 0x00, 0x00, 0x00],
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
    [0xC2, 0x04, 0x00, 0xCC, 0xCC],
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
    // `update` only stores the modified value back when the closure returns `Some(_)`,
    // so the unconditional `take` cannot clear a flag it does not consume.
    let owed = SCHEDULED_INTENT.update(thread, load_generation(), |intent| {
        take(&mut intent.st7_bonus).then_some(())
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

#[cfg(test)]
mod tests {
    use super::{COUNTED_WORDS, Entries};

    #[test]
    fn a_word_entered_again_counts_its_second_occurrence() {
        // Stage 1 at 1.00 power: the fourth chapter times out into a second chapter 2.
        let mut entries = Entries::NONE;
        assert_eq!(entries.count(), 0);
        let mut seen = Vec::new();
        for word in [0, 0, 1, 2, 2, 4, 4, 2, 2, 41] {
            entries.show(word);
            seen.push(entries.count());
        }
        assert_eq!(seen, [1, 1, 1, 1, 1, 1, 1, 2, 2, 1]);
    }

    #[test]
    fn only_a_change_of_word_is_an_entry() {
        let mut entries = Entries::NONE;
        assert!(entries.show(63));
        assert!(!entries.show(63));
        assert!(entries.show(64));
        assert!(entries.show(63));
        assert_eq!(entries.count(), 2);
    }

    #[test]
    fn a_word_past_the_table_has_no_count() {
        let mut entries = Entries::NONE;
        #[expect(clippy::cast_possible_truncation)]
        let past = COUNTED_WORDS as u32;
        assert!(entries.show(past));
        assert_eq!(entries.count(), 0);
        entries.show(5);
        assert_eq!(entries.count(), 1);
    }
}
