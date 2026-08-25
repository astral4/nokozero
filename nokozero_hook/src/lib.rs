#[cfg(not(target_arch = "x86"))]
compile_error!("nokozero_hook targets i686-pc-windows-gnu");

// See `build.rs`.
#[cfg(needs_unwind_resume_stub)]
std::arch::global_asm!(".globl __Unwind_Resume", "__Unwind_Resume:", "ud2");

mod dinput8;
mod patch;
mod thread;
pub mod reader;

use crate::patch::{CallSite, NearBranchSite};
use crate::thread::{MainCell, MainThread};
use bitflags::bitflags;
use reader::StateReader;
use std::ffi::c_void;
use std::ptr::{NonNull, null};
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{HINSTANCE, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{DisableThreadLibraryCalls, GetModuleHandleA};
use windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows_sys::core::BOOL;

/// The number of frames between each game state read.
const READ_INTERVAL: u32 = 6;

static READER: OnceLock<StateReader> = OnceLock::new();

static FRAME_COUNT: MainCell<u32> = MainCell::new(0);

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

    // SAFETY: `READER` is set in `DllMain` before `install` retargets the call to this hook,
    // so it is initialized by the time this code is reached.
    let reader = unsafe { READER.get().unwrap_unchecked() };

    if reader.is_game_active() {
        let frame = FRAME_COUNT.get(thread);
        FRAME_COUNT.set(thread, frame.wrapping_add(1));

        if frame % READ_INTERVAL == 0 {
            if let Some(_state) = reader.get_state() {
                // TODO: send game state
            }
        }

        // TODO: send inputs
        InputFlags::SHOOT | base
    } else {
        base
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(h_module: HINSTANCE, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == DLL_PROCESS_ATTACH {
        unsafe { DisableThreadLibraryCalls(h_module as HMODULE) };

        let module = unsafe { GetModuleHandleA(null()) };
        let base = NonNull::new(module.cast()).expect("game module handle should be valid");

        // SAFETY: `base` points to the start of the game's PE image,
        // which is loaded for the lifetime of the process.
        READER.set(unsafe { StateReader::new(base) }).unwrap();

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
    }
}
