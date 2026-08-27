//! Title-menu navigation into practice-mode stages.

use crate::addrs::MAIN_MENU_PTR_VA;
use crate::features::Scene;
use crate::log::fatal;
use crate::mem::{read, read_ptr};
use crate::patch::BranchSite;
use crate::thread::{MainCell, MainThread};
use crate::{InputFlags, TAP_INTERVAL};
use std::sync::OnceLock;

const MENU_ID_OFFSET: usize = 0x18;
// 2 = interactive (input is honored)
const MENU_SUB_STATE_OFFSET: usize = 0x20;
const MENU_CURSOR_OFFSET: usize = 0x24;

// Entered while the menu object decides what to show during boot and on the way back from a finished run.
const MENU_BOOTING: u32 = 0;
const MENU_TITLE: u32 = 1;
// Entered on quit and while the game starts.
const MENU_TERMINAL: u32 = 2;
const MENU_DIFFICULTY: u32 = 6;
const MENU_CHARACTER: u32 = 7;
const MENU_PRACTICE_STAGE: u32 = 9;

/// The title menu cursor index (0 = Game Start, 1 = Extra Start, 2 = Practice, etc.).
const TITLE_PRACTICE_START: u32 = 2;

/// The selected character.
static CHARACTER: OnceLock<u32> = OnceLock::new();

/// This should be called during `DLL_PROCESS_ATTACH`, before [`install`].
pub(crate) fn init(character: u32) {
    let _ = CHARACTER.set(character);
}

/// Call once during `DLL_PROCESS_ATTACH`.
///
/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
pub(crate) unsafe fn install() {
    unsafe {
        // Practice stage selection normally rejects stages locked in the score file.
        const { BranchSite::new(0x0046_78a2, 0x75, 0x0046_78b9, "practice stage unlock").force() }
            .apply();
        // Character selection normally hangs the game on a score file without any records.
        const {
            BranchSite::new(0x0046_6c3a, 0x75, 0x0046_6c7b, "character-select pool skip").force()
        }
        .apply();
    }
}

#[derive(Clone, Copy)]
enum Screen {
    Booting,
    Title,
    Terminal,
    Difficulty,
    Character,
    PracticeStage,
}

#[derive(Clone, Copy)]
struct MenuView {
    screen: Option<Screen>,
    /// The base pointer for the menu navigation fields.
    base: usize,
}

impl MenuView {
    /// Classifies the menu object for this frame. Returns `None` if no menu object exists.
    #[must_use]
    fn classify(_thread: MainThread) -> Option<Self> {
        // SAFETY: The menu object is created and destroyed on this thread,
        // so a non-null pointer read here stays valid for the rest of the tick.
        let base = unsafe { read_ptr(MAIN_MENU_PTR_VA) }?;
        let screen = match unsafe { read::<u32>(base + MENU_ID_OFFSET) } {
            MENU_BOOTING => Some(Screen::Booting),
            MENU_TITLE => Some(Screen::Title),
            MENU_TERMINAL => Some(Screen::Terminal),
            MENU_DIFFICULTY => Some(Screen::Difficulty),
            MENU_CHARACTER => Some(Screen::Character),
            MENU_PRACTICE_STAGE => Some(Screen::PracticeStage),
            _ => None,
        };
        Some(Self { screen, base })
    }

    /// Returns whether [`navigate`] is actively driving the screen classified by this menu view.
    #[must_use]
    fn managed(self) -> bool {
        self.screen.is_some()
    }

    /// Returns the input for this tick's screen.
    fn input(self) -> InputFlags {
        // SAFETY: `self.base` was non-null when classified this tick; see `MenuView::classify`.
        if unsafe { read::<u32>(self.base + MENU_SUB_STATE_OFFSET) } != 2 {
            return InputFlags::empty();
        }

        match self.screen {
            Some(Screen::Booting | Screen::Terminal) => InputFlags::empty(),
            Some(Screen::Title) => {
                // SAFETY: `self.base` was non-null when classified this tick; see `MenuView::classify`.
                if unsafe { read::<u32>(self.base + MENU_CURSOR_OFFSET) } == TITLE_PRACTICE_START {
                    InputFlags::SHOOT
                } else {
                    InputFlags::DOWN
                }
            }
            Some(Screen::Character) => {
                let Some(&character) = CHARACTER.get() else {
                    fatal!("the character-select screen was reached before the character was set");
                };
                // SAFETY: `self.base` was non-null when classified this tick; see `MenuView::classify`.
                if unsafe { read::<u32>(self.base + MENU_CURSOR_OFFSET) } == character {
                    InputFlags::SHOOT
                } else {
                    InputFlags::RIGHT
                }
            }
            Some(Screen::Difficulty | Screen::PracticeStage) => InputFlags::SHOOT,
            None => InputFlags::BOMB,
        }
    }
}

/// Classifies the menu object and decides this frame's scene class and injected input.
/// This function should be called exactly once per input-hook frame while the game is in the menu gamemode.
#[must_use]
pub(crate) fn navigate(thread: MainThread) -> (Scene, InputFlags) {
    static TICK: MainCell<u32> = MainCell::new(0);

    let tick = TICK.get(thread);
    TICK.set(thread, tick.wrapping_add(1));

    let Some(view) = MenuView::classify(thread) else {
        return (Scene::Other, InputFlags::empty());
    };

    let scene = if view.managed() {
        Scene::Menu
    } else {
        Scene::Other
    };

    let input = if tick.is_multiple_of(TAP_INTERVAL) {
        view.input()
    } else {
        InputFlags::empty()
    };

    (scene, input)
}
