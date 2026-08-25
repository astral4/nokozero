//! Logic for automatically hiding windows during headless execution.

use crate::iat::{ImportRef, hook_import};
use crate::patch::{NearBranchSite, Site, op_abs32};
use std::ffi::c_void;
use std::mem::transmute;
use std::process::abort;
use std::ptr::{null_mut, with_exposed_provenance, with_exposed_provenance_mut};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use windows_sys::Win32::Foundation::{HMODULE, HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallWindowProcA, CreateWindowExA, GWLP_WNDPROC, HMENU, SWP_SHOWWINDOW, SetWindowLongA,
    WINDOWPOS, WM_WINDOWPOSCHANGING, WNDPROC, WS_MAXIMIZEBOX, WS_MINIMIZEBOX, WS_SYSMENU,
    WS_VISIBLE,
};

const CREATE_STYLE: u32 = WS_VISIBLE | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;

static MAIN_HWND: AtomicPtr<c_void> = AtomicPtr::new(null_mut());
static PREV_WNDPROC: AtomicUsize = AtomicUsize::new(0);

/// # Safety
///
/// `game` must be a loaded module handle. This function must be called during `DLL_PROCESS_ATTACH`, before the game's entry point runs.
pub(super) unsafe fn install(game: HMODULE) {
    unsafe {
        Site::new(
            0x0047_30bc,
            op_abs32(0x68, CREATE_STYLE),
            "create game window hidden",
        )
        .patch(op_abs32(0x68, CREATE_STYLE & !WS_VISIBLE))
        .apply();

        const {
            NearBranchSite::new(0x0047_1b90, 0x84, 0x0047_1c9a, "window mode change skip").force()
        }
        .apply();

        hook_import(
            game,
            ImportRef::Name("CreateWindowExA"),
            hook_create_window_ex_a as *mut (),
        );
    }
}

unsafe extern "system" fn hook_create_window_ex_a(
    dw_ex_style: u32,
    lp_class_name: *const u8,
    lp_window_name: *const u8,
    dw_style: u32,
    x: i32,
    y: i32,
    n_width: i32,
    n_height: i32,
    h_wnd_parent: HWND,
    h_menu: HMENU,
    h_instance: HMODULE,
    lp_param: *mut c_void,
) -> HWND {
    let is_main = h_wnd_parent.is_null() && unsafe { is_base_class(lp_class_name) };
    let hwnd = unsafe {
        CreateWindowExA(
            dw_ex_style,
            lp_class_name,
            lp_window_name,
            dw_style,
            x,
            y,
            n_width,
            n_height,
            h_wnd_parent,
            h_menu,
            h_instance,
            lp_param,
        )
    };
    if is_main && !hwnd.is_null() {
        MAIN_HWND.store(hwnd, Ordering::Release);
        unsafe { install_visibility_override(hwnd) };
    }
    hwnd
}

/// Returns whether `lp_class_name` is the game's main window class.
unsafe fn is_base_class(lp_class_name: *const u8) -> bool {
    if lp_class_name.addr() <= 0xFFFF {
        return false;
    }
    let target = b"BASE\0";
    for (i, &b) in target.iter().enumerate() {
        // SAFETY: The loop stops at the first mismatch, including the terminating NUL.
        if unsafe { *lp_class_name.add(i) } != b {
            return false;
        }
    }
    true
}

unsafe fn install_visibility_override(hwnd: HWND) {
    #[expect(clippy::cast_possible_truncation)]
    let filter = ((filter_wndproc as *const ()).expose_provenance() as u32).cast_signed();
    let prev = unsafe { SetWindowLongA(hwnd, GWLP_WNDPROC, filter) };
    // A real window always has a non-null wndproc, so 0 means failure.
    if prev == 0 {
        eprintln!("nokozero_hook: failed to override window visiblity");
        abort();
    }
    PREV_WNDPROC.store(prev.cast_unsigned() as usize, Ordering::Release);
}

/// Overrides any attempt to map the game window.
unsafe extern "system" fn filter_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_WINDOWPOSCHANGING && hwnd == MAIN_HWND.load(Ordering::Acquire) {
        let pos: *mut WINDOWPOS = with_exposed_provenance_mut(lparam.cast_unsigned());
        if !pos.is_null() {
            // SAFETY: For this message, `lparam` is a `WINDOWPOS` kept valid by the sender for the duration of the call.
            // Modifying in place is the intended way to alter the pending change.
            unsafe { (*pos).flags &= !SWP_SHOWWINDOW };
        }
    }
    let prev = PREV_WNDPROC.load(Ordering::Acquire);
    // SAFETY: `WNDPROC` is `Option<fn>`, but the store happens right after the `SetWindowLongA` swap on the window's own thread,
    // so `None` is unobservable.
    let prev = unsafe { transmute::<*const (), WNDPROC>(with_exposed_provenance(prev)) };
    unsafe { CallWindowProcA(prev, hwnd, msg, wparam, lparam) }
}
