//! Logic for suppressing modal windows, including the startup resolution dialog, log message boxes, and CRT runtime error boxes.

use crate::iat::{ImportRef, hook_import};
use crate::patch::{BranchSite, NearBranchSite, Site};
use std::ptr::{read_volatile, with_exposed_provenance_mut, write_volatile};
use windows_sys::Win32::Foundation::{HMODULE, HWND, LPARAM, WPARAM};
use windows_sys::Win32::UI::Controls::{BST_UNCHECKED, CheckDlgButton, CheckRadioButton};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    BN_CLICKED, CreateDialogParamA, DLGPROC, PostMessageA, WM_COMMAND,
};

const DIALOG_TEMPLATE_ID: usize = 0xCB;
const DIALOG_PROC_VA: usize = 0x0047_3de0;

const FULLSCREEN_CHECKBOX_ID: i32 = 0xCB;
const RES_RADIO_FIRST_ID: i32 = 0xCD; // 640x480
const RES_RADIO_LAST_ID: i32 = 0xCF; // 1280x960
const OK_BUTTON_ID: u32 = 0xD0;

const EXIT_FLAG_VA: usize = 0x004e_6d1c;
const EXIT_FLAG_BIT: u32 = 0x0008_0000;

/// # Safety
///
/// `game` must be a loaded module handle. This function must be called during `DLL_PROCESS_ATTACH`, before the game's entry point runs.
pub(crate) unsafe fn install(game: HMODULE) {
    unsafe {
        hook_import(
            game,
            ImportRef::Name("CreateDialogParamA"),
            hook_create_dialog_param_a as *mut (),
        );
        Site::new(0x0047_1620, [0x05], "force dialog hidden")
            .patch([0x00])
            .apply();

        const { BranchSite::new(0x0047_1dee, 0x74, 0x0047_1e04, "log message box skip").force() }
            .apply();

        const {
            NearBranchSite::new(0x0049_3f28, 0x84, 0x0049_403d, "CRT errors to stderr").force()
        }
        .apply();
    }
}

unsafe extern "system" fn hook_create_dialog_param_a(
    hinst: HMODULE,
    template: *const u8,
    parent: HWND,
    proc: DLGPROC,
    init_param: LPARAM,
) -> HWND {
    let hwnd = unsafe { CreateDialogParamA(hinst, template, parent, proc, init_param) };

    let template_id = template.addr();
    let proc_va = proc.map_or(0, |f| (f as *const ()).addr());

    if hwnd.is_null() || template_id != DIALOG_TEMPLATE_ID || proc_va != DIALOG_PROC_VA {
        return hwnd;
    }

    unsafe {
        CheckRadioButton(
            hwnd,
            RES_RADIO_FIRST_ID,
            RES_RADIO_LAST_ID,
            RES_RADIO_FIRST_ID,
        );
        CheckDlgButton(hwnd, FULLSCREEN_CHECKBOX_ID, BST_UNCHECKED);
        let wparam = ((BN_CLICKED << 16) | OK_BUTTON_ID) as WPARAM;
        PostMessageA(hwnd, WM_COMMAND, wparam, 0);
    };

    let flag: *mut u32 = with_exposed_provenance_mut(EXIT_FLAG_VA);
    let prev = unsafe { read_volatile(flag) };
    unsafe { write_volatile(flag, prev | EXIT_FLAG_BIT) };

    hwnd
}
