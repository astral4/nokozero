#[cfg(not(target_arch = "x86"))]
compile_error!("nokozero_hook targets i686-pc-windows-gnu");

// See `build.rs`.
#[cfg(needs_unwind_resume_stub)]
std::arch::global_asm!(".globl __Unwind_Resume", "__Unwind_Resume:", "ud2");

mod addrs;
mod dinput8;
mod mem;
mod patch;
mod practice;
mod reader;
mod thread;

use crate::patch::{CallSite, NearBranchSite};
use crate::practice::{apply_pending_reset, observe_loads};
use crate::reader::GameState;
use crate::thread::{MainCell, MainThread, MainToken};
use bitflags::bitflags;
use std::ffi::c_void;
use windows_sys::Win32::Foundation::{HINSTANCE, HMODULE};
use windows_sys::Win32::System::LibraryLoader::DisableThreadLibraryCalls;
use windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows_sys::core::BOOL;

/// The number of frames between each game state read.
const READ_INTERVAL: u32 = 6;

static FRAME_COUNT: MainCell<u32> = MainCell::new(0);

static GAME_STATE: MainCell<Option<GameState>> = MainCell::new(None);

bitflags! {
    #[repr(transparent)]
    struct InputFlags: u32 {
        const SHOOT = 0x1;
        const BOMB = 0x2;
        const FOCUS = 0x8;
        const UP = 0x10;
        const DOWN = 0x20;
        const LEFT = 0x40;
        const RIGHT = 0x80;
        const PAUSE = 0x100;
        const SKIP = 0x200;
        const ITEM = 0x400;
        const CHANGE = 0x800;
        const RETRY = 0x20000;
        const SCREENSHOT = 0x40000;
        const ENTER = 0x80000;

        // The source may set any bits
        const _ = !0;
    }
}

extern "system" fn get_joypad_input_hook(base: InputFlags) -> InputFlags {
    let thread = MainThread::claim();
    // SAFETY: This hook is called from the game's update loop, so its thread is the update thread.
    let token = unsafe { MainToken::new(thread) };

    observe_loads(thread);

    let frame = FRAME_COUNT.get(thread);
    FRAME_COUNT.set(thread, frame.wrapping_add(1));

    if frame.is_multiple_of(READ_INTERVAL) {
        let mut state = GAME_STATE
            .replace(thread, None)
            .unwrap_or_else(GameState::new);
        if let Some(_state) = state.read() {
            // TODO: send the observation
        }
        GAME_STATE.set(thread, Some(state));
    }

    apply_pending_reset(token);

    // TODO: send inputs
    base
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(h_module: HINSTANCE, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == DLL_PROCESS_ATTACH {
        unsafe { DisableThreadLibraryCalls(h_module as HMODULE) };

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
    }
}
