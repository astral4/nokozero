#[cfg(not(target_arch = "x86"))]
compile_error!("nokozero_hook targets i686-pc-windows-gnu");

// See `build.rs`.
#[cfg(needs_unwind_resume_stub)]
std::arch::global_asm!(".globl __Unwind_Resume", "__Unwind_Resume:", "ud2");

mod dinput8;
pub mod reader;

use bitflags::bitflags;
use reader::StateReader;
use std::ffi::c_void;
use std::mem::transmute;
use std::ptr::{NonNull, copy_nonoverlapping, null};
use std::sync::{
    OnceLock,
    atomic::{AtomicU32, Ordering},
};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

type GetJoypadInputFn = extern "system" fn(InputFlags) -> InputFlags;

const GET_JOYPAD_INPUT_ADDR: usize = 0x1b20;
const GET_JOYPAD_INPUT_HOOK_ADDR: usize = 0x22fa;

/// The number of frames between each game state read.
const READ_INTERVAL: u32 = 6;

static GET_JOYPAD_INPUT_ORIGINAL: OnceLock<GetJoypadInputFn> = OnceLock::new();
static READER: OnceLock<StateReader> = OnceLock::new();

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
    // SAFETY: `READER` is set in `DllMain` before `patch_call()` installs this hook,
    // so it is initialized by the time this code is reached.
    let reader = unsafe { READER.get().unwrap_unchecked() };

    if reader.is_game_active() {
        static FRAME_COUNT: AtomicU32 = AtomicU32::new(0);
        let frame = FRAME_COUNT.fetch_add(1, Ordering::Relaxed);

        if frame % READ_INTERVAL == 0 {
            if let Some(_state) = reader.get_state() {
                // TODO: send game state
            }
        }

        // TODO: send inputs
        InputFlags::SHOOT | base
    } else if let Some(original) = GET_JOYPAD_INPUT_ORIGINAL.get() {
        original(base)
    } else {
        base
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(_h_module: *mut c_void, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == DLL_PROCESS_ATTACH {
        // `DLL_PROCESS_ATTACH` runs during process initialization, before the game's entry point.
        // The functions we hook are only reachable from the game's main loop,
        // so no thread can be executing them yet. This guarantees that the reader,
        // original function pointer, and code patch are all in place before any hooked code runs.
        let module = unsafe { GetModuleHandleA(null()) };
        let base = NonNull::new(module.cast()).expect("game module handle should be valid");

        // SAFETY: `base` points to the start of the game's PE image,
        // which is loaded for the lifetime of the process.
        READER.set(unsafe { StateReader::new(base) }).unwrap();

        // SAFETY: `GET_JOYPAD_INPUT_ADDR` is the offset of the original function.
        // Its signature matches the definition of `GetJoypadInputFn`.
        let original_fn: GetJoypadInputFn =
            unsafe { transmute(base.byte_add(GET_JOYPAD_INPUT_ADDR).as_ptr()) };
        GET_JOYPAD_INPUT_ORIGINAL.set(original_fn).unwrap();

        // We use the address of `get_joypad_input_hook` to calculate
        // the relative offset for an x86 CALL instruction.
        // Since we won't reconstruct a pointer from this address, we don't expose provenance
        // via `get_joypad_input_hook as usize` or, equivalently,
        // `(get_joypad_input_hook as *const ()).expose_provenance()`.
        let hook_addr = (get_joypad_input_hook as *const ()).addr();
        let hook_target = unsafe { base.byte_add(GET_JOYPAD_INPUT_HOOK_ADDR).as_ptr() };
        unsafe { patch_call(hook_target, hook_addr) };
    }

    1
}

/// # Safety
/// `target` must point to a 5-byte CALL instruction that no other thread is executing.
unsafe fn patch_call(target: *mut u8, func: usize) {
    let mut patch = [0u8; 5];
    patch[0] = 0xE8; // CALL opcode

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let offset = func.wrapping_sub(target.addr()).wrapping_sub(5) as i32;
    patch[1..5].copy_from_slice(&offset.to_le_bytes());

    unsafe { patch_bytes(target, &patch) };
}

/// # Safety
/// `dst` must be valid for writes of `src.len()` bytes,
/// and no other thread may be executing the code at `dst`.
unsafe fn patch_bytes(dst: *mut u8, src: &[u8]) {
    let mut old_protect = 0;
    let mut temp = 0;

    unsafe {
        assert_ne!(
            VirtualProtect(
                dst.cast(),
                src.len(),
                PAGE_EXECUTE_READWRITE,
                &raw mut old_protect,
            ),
            0,
        );

        copy_nonoverlapping(src.as_ptr(), dst, src.len());

        assert_ne!(
            VirtualProtect(dst.cast(), src.len(), old_protect, &raw mut temp),
            0
        );
        assert_ne!(
            FlushInstructionCache(GetCurrentProcess(), dst.cast(), src.len()),
            0,
        );
    }
}
