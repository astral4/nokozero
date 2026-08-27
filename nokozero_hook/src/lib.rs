#[cfg(not(target_arch = "x86"))]
compile_error!("nokozero_hook targets i686-pc-windows-gnu");

// See `build.rs`.
#[cfg(needs_unwind_resume_stub)]
std::arch::global_asm!(".globl __Unwind_Resume", "__Unwind_Resume:", "ud2");

mod addrs;
mod dialog;
mod dinput8;
mod env;
mod features;
mod headless;
mod iat;
mod ipc;
mod mem;
mod menu;
mod patch;
mod practice;
mod reader;
mod thread;

use crate::addrs::{GAMEMODE_INGAME, GAMEMODE_MENU, GAMEMODE_VA, GUI_PTR_VA};
use crate::features::{Meta, Scene, build as build_features};
use crate::headless::init_from_env;
use crate::ipc::{Command, ObsFrame, is_connected, step};
use crate::mem::{game_live, read, read_ptr};
use crate::menu::navigate;
use crate::patch::{CallSite, NearBranchSite};
use crate::practice::{
    WireMeta, accept_reset, apply_pending_reset, observe_loads, take_forced_step,
};
use crate::reader::{GameState, Resources};
use crate::thread::{MainCell, MainThread, MainToken};
use bitflags::bitflags;
use std::ffi::c_void;
use std::process::abort;
use std::ptr::null;
use windows_sys::Win32::Foundation::{HINSTANCE, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{DisableThreadLibraryCalls, GetModuleHandleA};
use windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows_sys::core::BOOL;

/// The number of frames between RL steps.
const READ_INTERVAL: u32 = 3;

/// Rising-edge cadence for injected inputs (e.g. during menu navigation and dialogue).
const TAP_INTERVAL: u32 = 3;

struct StepBufs {
    state: GameState,
    frame_buf: Vec<u8>,
}

impl StepBufs {
    fn new() -> Self {
        Self {
            state: GameState::new(),
            frame_buf: Vec::new(),
        }
    }
}

/// A controller-supplied action. The action space is a subset of [`InputFlags`].
#[derive(Clone, Copy)]
struct Action(u32);

impl Action {
    // SHOOT | BOMB | FOCUS | UP | DOWN | LEFT | RIGHT
    const MASK: u32 = 0b1111_1011;

    fn from_wire(bits: u32) -> Option<Self> {
        (bits & !Self::MASK == 0).then_some(Self(bits))
    }

    const fn neutral() -> Self {
        Self(0)
    }
}

bitflags! {
    #[repr(transparent)]
    struct InputFlags: u32 {
        const SHOOT = 0x1; // Z
        const BOMB = 0x2; // X
        const FOCUS = 0x8; // Shift
        const UP = 0x10;
        const DOWN = 0x20;
        const LEFT = 0x40;
        const RIGHT = 0x80;
        const SKIP = 0x200; // Ctrl, C

        const _ = !0;
    }
}

impl From<Action> for InputFlags {
    fn from(action: Action) -> Self {
        // `Action::from_wire` already proved every set bit is in `MASK`.
        Self::from_bits_retain(action.0)
    }
}

static FRAME_COUNT: MainCell<u32> = MainCell::new(0);

static STEP_BUFS: MainCell<Option<StepBufs>> = MainCell::new(None);

/// Set when an in-game frame delivers an action differing from [`LAST_ACTION`].
static INPUT_OVERRIDDEN: MainCell<bool> = MainCell::new(false);

/// The last controller action. Repeated on the frames between exchanges.
static LAST_ACTION: MainCell<Action> = MainCell::new(Action::neutral());

/// Returns whether a boss dialogue is live in this frame.
fn dialogue_active() -> bool {
    const GUI_MSG_VM_OFFSET: usize = 0x1b8;

    if !unsafe { game_live() } {
        return false;
    }
    let Some(gui) = (unsafe { read_ptr(GUI_PTR_VA) }) else {
        return false;
    };
    // SAFETY: The GUI object is live at this point.
    unsafe { read::<u32>(gui + GUI_MSG_VM_OFFSET) != 0 }
}

extern "system" fn get_joypad_input_hook(_base: InputFlags) -> InputFlags {
    let thread = MainThread::claim();
    // SAFETY: This hook is called from the game's update loop, so its thread is the update thread.
    let token = unsafe { MainToken::new(thread) };

    let gamemode = unsafe { read::<u32>(GAMEMODE_VA) };
    let connected = is_connected();
    let (scene, menu_input) = match gamemode {
        GAMEMODE_MENU => navigate(thread),
        GAMEMODE_INGAME => (Scene::InGame, InputFlags::empty()),
        _ => (Scene::Other, InputFlags::empty()),
    };

    observe_loads(thread);

    if connected {
        let frame = FRAME_COUNT.get(thread);
        FRAME_COUNT.set(thread, frame.wrapping_add(1));

        let forced = take_forced_step(thread);
        if frame.is_multiple_of(READ_INTERVAL) || forced {
            let resources = Resources::read();
            let mut bufs = STEP_BUFS
                .replace(thread, None)
                .unwrap_or_else(StepBufs::new);

            let StepBufs { state, frame_buf } = &mut bufs;
            let state = state.read();
            let wire = WireMeta::read(thread);
            let mut obs = ObsFrame::begin(frame_buf);
            build_features(
                obs.payload(),
                state,
                &Meta {
                    step: frame,
                    scene,
                    wire,
                    overrode_input: INPUT_OVERRIDDEN.replace(thread, false),
                },
                &resources,
            );

            match step(obs) {
                Some(Command::Act(action)) => LAST_ACTION.set(thread, action),
                Some(Command::Reset { seq, params }) => {
                    if !accept_reset(thread, seq, params) {
                        eprintln!(
                            "nokozero_hook: ipc: RESET rejected; another reset is still pending"
                        );
                        abort();
                    }
                    LAST_ACTION.set(thread, Action::neutral());
                }
                None => {}
            }

            STEP_BUFS.set(thread, Some(bufs));
        }

        apply_pending_reset(token);
    }

    match (connected, gamemode) {
        (true, GAMEMODE_INGAME) => {
            let mut input: InputFlags = LAST_ACTION.get(thread).into();
            if dialogue_active() {
                INPUT_OVERRIDDEN.set(thread, true);
                input.remove(InputFlags::SHOOT);
                if FRAME_COUNT.get(thread).is_multiple_of(TAP_INTERVAL) {
                    input.insert(InputFlags::SHOOT);
                }
                input.insert(InputFlags::SKIP);
            }
            input
        }
        (true, GAMEMODE_MENU) => menu_input,
        _ => InputFlags::empty(),
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(h_module: HINSTANCE, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == DLL_PROCESS_ATTACH {
        unsafe { DisableThreadLibraryCalls(h_module as HMODULE) };
        init_from_env();
        unsafe { install() };
    }
    1
}

/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
unsafe fn install() {
    unsafe {
        // Lets multiple game instances run in parallel.
        const {
            NearBranchSite::new(0x0047_13ec, 0x85, 0x0047_15a9, "instance mutex disable").force()
        }
        .apply();

        CallSite::new(0x0040_22fa, 0x0040_1b20, "GetJoypadInput call detour")
            .retarget(get_joypad_input_hook as *mut ());

        practice::install();
        menu::install();

        let game = GetModuleHandleA(null());

        dialog::install(game);

        if headless::is_enabled() {
            headless::install(game);
        }
    }

    ipc::init();
}
