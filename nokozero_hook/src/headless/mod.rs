//! Logic for running the game in headless mode.

mod backing;
mod d3d9;
mod dsound;
mod out;
mod window;

use crate::addrs::{DEVICE_PTR_VA, GAMEMODE_INGAME, GAMEMODE_VA};
use crate::env::headless;
use crate::patch::{BranchSite, Site, op_abs32};
use std::arch::naked_asm;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::Foundation::HMODULE;

static HEADLESS: AtomicBool = AtomicBool::new(false);

/// This should be called during `DLL_PROCESS_ATTACH`, before [`install`].
pub(crate) fn init_from_env() {
    HEADLESS.store(headless(), Ordering::Relaxed);
}

#[must_use]
pub(crate) fn is_enabled() -> bool {
    HEADLESS.load(Ordering::Relaxed)
}

/// # Safety
///
/// `game` must be a loaded module handle. This function must be called during `DLL_PROCESS_ATTACH`, before the game's entry point runs.
pub(crate) unsafe fn install(game: HMODULE) {
    unsafe {
        d3d9::install(game);
        dsound::install(game);
        window::install(game);
        install_frame_limiter_skips();
        install_draw_chain_skip();
    }
}

unsafe fn install_frame_limiter_skips() {
    unsafe {
        const {
            BranchSite::new(
                0x0047_276d,
                0x7e,
                0x0047_2772,
                "frame limiter skip (vsync-paced tick)",
            )
            .force()
        }
        .apply();
        const {
            BranchSite::new(
                0x0047_2b3f,
                0x7e,
                0x0047_2b48,
                "frame limiter skip (present-interval tick)",
            )
            .force()
        }
        .apply();
        const {
            BranchSite::new(
                0x0047_27de,
                0x72,
                0x0047_27e8,
                "frame limiter skip (software-timed tick)",
            )
            .redirect(0x0047_282a)
        }
        .apply();
    }
}

static DRAW_NORMAL_LANDING_VA: usize = 0x0047_290d;
static DRAW_NORMAL_CONTINUE_VA: usize = 0x0047_28bb;
static DRAW_FF_LANDING_VA: usize = 0x0047_24fd;
static DRAW_FF_CONTINUE_VA: usize = 0x0047_24ab;

/// Jumps to `$land` (the frame-skip reset + Present) if in a stage, eliding the draw pass.
/// Otherwise, runs the displaced `mov eax, [device]` and continues into the draw block at `$cont`.
macro_rules! draw_skip_trampoline {
    ($name:ident, $land:ident, $cont:ident) => {
        #[unsafe(naked)]
        unsafe extern "C" fn $name() -> ! {
            naked_asm!(
                // `cmp` clobbers EFLAGS mid-function, but this is fine because flags are dead at both patch sites.
                "cmp dword ptr [{gamemode}], {ingame}",
                "jne 2f",
                "jmp dword ptr [{land}]",
                "2:",
                "mov eax, dword ptr [{device}]",
                "jmp dword ptr [{cont}]",
                gamemode = const GAMEMODE_VA,
                ingame = const GAMEMODE_INGAME,
                device = const DEVICE_PTR_VA,
                land = sym $land,
                cont = sym $cont,
            )
        }
    };
}

draw_skip_trampoline!(
    draw_skip_normal_trampoline,
    DRAW_NORMAL_LANDING_VA,
    DRAW_NORMAL_CONTINUE_VA
);

draw_skip_trampoline!(
    draw_skip_ff_trampoline,
    DRAW_FF_LANDING_VA,
    DRAW_FF_CONTINUE_VA
);

unsafe fn install_draw_chain_skip() {
    // `mov eax, [device]`
    const DRAW_SITE_BYTES: [u8; 5] = op_abs32(0xa1, DEVICE_PTR_VA);

    unsafe {
        Site::new(
            0x0047_28b6,
            DRAW_SITE_BYTES,
            "game draw-chain skip (normal frame)",
        )
        .jmp(draw_skip_normal_trampoline as *mut ());
        Site::new(
            0x0047_24a6,
            DRAW_SITE_BYTES,
            "game draw-chain skip (fast-forward frame)",
        )
        .jmp(draw_skip_ff_trampoline as *mut ());
    }
}
