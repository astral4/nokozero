use bitflags::bitflags;
use std::ffi::c_void;
use std::mem::transmute;
use std::ptr;
use std::sync::OnceLock;
use windows::Win32::System::Memory::{
    PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS, VirtualProtect,
};
use windows::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows::core::BOOL;

type GetJoypadInputFn = extern "stdcall" fn(InputFlags) -> InputFlags;

const GAME_THREAD_PTR: usize = 0x4e9a94;
const GET_JOYPAD_INPUT_ADDR: usize = 0x401b20;
const GET_JOYPAD_INPUT_HOOK_ADDR: usize = 0x4022fa;

static GET_JOYPAD_INPUT_ORIGINAL: OnceLock<GetJoypadInputFn> = OnceLock::new();

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
        const ITEM = 0x400;
        const CHANGE = 0x800;
        const RETRY = 0x20000;
        const SCREENSHOT = 0x40000;
        const ENTER = 0x80000;

        // The source may set any bits
        const _ = !0;
    }
}

extern "stdcall" fn get_joypad_input_hook(base: InputFlags) -> InputFlags {
    // Note on provenance: The entire premise of DLL injection
    // involves operations that are outside Rust's formal memory model.
    // There's no previously exposed provenance for `GAME_THREAD_PTR`;
    // it's a hardcoded address obtained from reverse engineering.
    // To make a pointer from `GAME_THREAD_PTR`, we could cast to `*const usize`
    // or, equivalently, use `std::ptr::with_exposed_provenance()`.
    // However, according to the strict provenance model, both approaches
    // are technically undefined behavior because we're creating a pointer
    // from an integer that never had valid provenance to begin with.
    let game_thread = unsafe { ptr::read_volatile(GAME_THREAD_PTR as *const usize) };

    if game_thread != 0 {
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
        unsafe {
            // We save the original function in case it is needed.
            // There is a race condition when the hooked function is called
            // before `OnceLock::set()` completes. We assume this won't happen;
            // i.e. no other threads are executing game code during `DLL_PROCESS_ATTACH`.
            let original_fn: GetJoypadInputFn = unsafe { transmute(GET_JOYPAD_INPUT_ADDR) };
            GET_JOYPAD_INPUT_ORIGINAL.set(original_fn).unwrap();

            // We use the address to calculate the relative offset for an x86 CALL instruction.
            // Since we won't reconstruct a pointer from this address, we don't expose provenance
            // via `get_joypad_input_hook as usize` or, equivalently,
            // `(get_joypad_input_hook as *const ()).expose_provenance()`.
            let hook_addr = (get_joypad_input_hook as *const ()).addr();
            unsafe { patch_call(GET_JOYPAD_INPUT_HOOK_ADDR, hook_addr) };
        }
    }

    BOOL(1) // Return `TRUE`
}

unsafe fn patch_call(target: usize, func: usize) {
    let mut patch = [0u8; 5];
    patch[0] = 0xE8; // CALL opcode

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let offset = func.wrapping_sub(target).wrapping_sub(5) as i32;
    patch[1..5].copy_from_slice(&offset.to_le_bytes());

    // See previous note on provenance. There is no previously exposed
    // provenance for `GET_JOYPAD_INPUT_HOOK_ADDR`.
    unsafe { patch_bytes(target as *mut u8, &patch) };
}

unsafe fn patch_bytes(dst: *mut u8, src: &[u8]) {
    let mut old_protect = PAGE_PROTECTION_FLAGS(0);
    let mut temp = PAGE_PROTECTION_FLAGS(0);

    unsafe {
        VirtualProtect(
            dst.cast(),
            src.len(),
            PAGE_EXECUTE_READWRITE,
            &raw mut old_protect,
        )
        .unwrap();

        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());

        VirtualProtect(dst.cast(), src.len(), old_protect, &raw mut temp).unwrap();
    }
}
