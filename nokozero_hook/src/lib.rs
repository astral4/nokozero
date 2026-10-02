#[cfg(not(target_arch = "x86"))]
compile_error!("nokozero_hook targets i686-pc-windows-gnu");

// See `build.rs`.
#[cfg(needs_unwind_resume_stub)]
std::arch::global_asm!(".globl __Unwind_Resume", "__Unwind_Resume:", "ud2");

mod addrs;
mod anm;
mod dialog;
mod dinput8;
mod env;
mod features;
mod headless;
mod iat;
mod ipc;
mod log;
mod mem;
mod menu;
mod patch;
mod practice;
mod reader;
mod score;
mod thread;

use crate::addrs::{GAMEMODE_INGAME, GAMEMODE_MENU, GAMEMODE_VA, GUI_PTR_VA};
use crate::env::Config;
use crate::features::{InputRecord, Meta, Scene, build as build_features};
use crate::ipc::{Command, CommandBuf, ObsFrame, is_connected, step};
use crate::log::fatal;
use crate::mem::{game_live, read, read_ptr};
use crate::menu::navigate;
use crate::patch::{CallSite, NearBranchSite};
use crate::practice::{
    StageFrame, WireMeta, accept_reset, apply_pending_reset, observe_loads, place_before_consume,
    reset_pending, stage_frame, take_forced_step,
};
use crate::reader::{GameState, Resources};
use crate::thread::{MainCell, MainThread, MainToken};
use bitflags::bitflags;
use std::ffi::c_void;
use std::ptr::null;
use windows_sys::Win32::Foundation::{HINSTANCE, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{DisableThreadLibraryCalls, GetModuleHandleA};
use windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows_sys::core::BOOL;

/// The number of frames between RL steps outside a live stage, and the default inside one
/// (a reset sets each episode's own; see `PracticeParams::step_interval`).
pub(crate) const READ_INTERVAL: u32 = 3;

/// Rising-edge cadence for injected inputs (e.g. during menu navigation and dialogue).
const TAP_INTERVAL: u32 = 3;

struct StepBufs {
    state: GameState,
    frame_buf: Vec<u8>,
    cmd_buf: CommandBuf,
}

impl StepBufs {
    fn new() -> Self {
        Self {
            state: GameState::new(),
            frame_buf: Vec::new(),
            cmd_buf: CommandBuf::new(),
        }
    }
}

/// A controller-supplied action. The action space is a subset of [`InputFlags`].
#[derive(Clone, Copy)]
struct Action(u32);

impl Action {
    // SHOOT | BOMB | FOCUS | UP | DOWN | LEFT | RIGHT | SKIP
    const MASK: u32 = 0b0010_1111_1011;

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

/// What the controller's last command plays on the live stage frames until the next exchange.
struct Controller {
    /// The last controller action. Repeated on the frames between exchanges.
    last: Action,
    /// Whether the last command was a raw tape. If `true`, then its frames are played as sent.
    raw: bool,
    /// The rest of the last TAPE command, with actions from oldest to newest.
    tape: Vec<Action>,
}

impl Controller {
    const NEUTRAL: Self = Self {
        last: Action::neutral(),
        raw: false,
        tape: Vec::new(),
    };
}

static CONTROLLER: MainCell<Controller> = MainCell::new(Controller::NEUTRAL);

/// The inputs that the game read on each live stage frame since the previous exchange.
static INPUT_LOG: MainCell<Vec<InputRecord>> = MainCell::new(Vec::new());

/// The largest number of frames between two exchanges.
const MAX_INPUT_LOG: usize = ipc::MAX_TAPE_FRAMES + 2 * practice::MAX_STEP_INTERVAL as usize;

/// Starts a load's input state on its first live frame.
fn begin_load(thread: MainThread) {
    CONTROLLER.set(thread, Controller::NEUTRAL);
    INPUT_LOG.with(thread, Vec::clear);
}

/// Applies the tape's next action if one is pending. Returns whether a tape frame was consumed.
fn play_tape_frame(thread: MainThread) -> bool {
    CONTROLLER.with(thread, |controller| {
        let Some(action) = controller.tape.pop() else {
            return false;
        };
        controller.last = action;
        true
    })
}

/// Returns the input for live stage frame `live`.
fn ingame_input(thread: MainThread, live: StageFrame) -> InputFlags {
    let (last, raw) = CONTROLLER.with(thread, |controller| (controller.last, controller.raw));
    let mut input: InputFlags = last.into();
    if dialogue_active() && !raw {
        input.remove(InputFlags::SHOOT);
        if live.index.is_multiple_of(TAP_INTERVAL) {
            input.insert(InputFlags::SHOOT);
        }
        input.insert(InputFlags::SKIP);
    }
    input
}

/// Appends the input that the game reads on live stage frame `index` to the log carried by the next observation.
fn log_input(thread: MainThread, index: u32, input: &InputFlags) {
    INPUT_LOG.with(thread, |log| {
        if log.len() >= MAX_INPUT_LOG {
            fatal!("{} live stage frames without an exchange", log.len());
        }
        log.push([index, input.bits()]);
    });
}

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
    let mut live = None;
    if connected {
        place_before_consume(token);
        let frame = FRAME_COUNT.get(thread);
        FRAME_COUNT.set(thread, frame.wrapping_add(1));

        // In a live stage, the cadence follows the load's own frame index and step interval (see `stage_frame`), so a step always covers
        // the same frames of the stage however many hook frames the menus and loads before it took. A tape consumes live stage frames
        // instead of stepping on them, and leaves an owed forced step to its first frame after rather than taking it.
        live = stage_frame(thread);
        if live.is_some_and(|live| live.index == 0) {
            begin_load(thread);
        }
        let taping = live.is_some() && play_tape_frame(thread);
        let due = live.map_or_else(|| frame.is_multiple_of(READ_INTERVAL), |live| live.due);
        let forced = !taping && take_forced_step(thread);
        if !taping && (due || forced) {
            let resources = Resources::read();
            let mut bufs = STEP_BUFS.take(thread).unwrap_or_else(StepBufs::new);

            let StepBufs {
                state,
                frame_buf,
                cmd_buf,
            } = &mut bufs;
            let state = state.read();
            let wire = WireMeta::read(thread);
            let mut obs = ObsFrame::begin(frame_buf);
            INPUT_LOG.with(thread, |inputs| {
                build_features(
                    obs.payload(),
                    state,
                    &Meta {
                        step: frame,
                        scene,
                        wire,
                        inputs,
                    },
                    &resources,
                );
                inputs.clear();
            });

            match step(obs, cmd_buf) {
                Some(Command::Tape {
                    first,
                    mut rest,
                    raw,
                }) => {
                    if !rest.is_empty() && reset_pending(thread) {
                        fatal!("TAPE while a reset is pending");
                    }
                    // `first` is this frame's action. The rest are stored from oldest to newest so each following frame's is a `Vec::pop`.
                    rest.reverse();
                    CONTROLLER.set(
                        thread,
                        Controller {
                            last: first,
                            raw,
                            tape: rest,
                        },
                    );
                }
                Some(Command::Reset {
                    seq,
                    params,
                    record,
                }) => {
                    if !accept_reset(thread, seq, params, record) {
                        fatal!("RESET rejected; another reset is still pending");
                    }
                    CONTROLLER.set(thread, Controller::NEUTRAL);
                }
                None => {}
            }

            STEP_BUFS.set(thread, Some(bufs));
        }

        apply_pending_reset(token);
    }

    match (connected, gamemode) {
        (true, GAMEMODE_INGAME) => live.map_or_else(InputFlags::empty, |live| {
            let input = ingame_input(thread, live);
            log_input(thread, live.index, &input);
            input
        }),
        (true, GAMEMODE_MENU) => menu_input,
        _ => InputFlags::empty(),
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(h_module: HINSTANCE, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == DLL_PROCESS_ATTACH {
        unsafe { DisableThreadLibraryCalls(h_module as HMODULE) };

        let config = Config::from_env();
        menu::init(config.character);
        unsafe { install(config.headless) };
        ipc::init(config.connect_addr);
    }
    1
}

/// # Safety
///
/// The game image must be loaded at its fixed base. This function must be called during `DLL_PROCESS_ATTACH`,
/// before the game's entry point runs.
unsafe fn install(headless: bool) {
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
        anm::install();
        score::install();

        let game = GetModuleHandleA(null());

        dialog::install(game);

        if headless {
            headless::install(game);
        }
    }
}
