//! `dinput8.dll` proxy that loads the real System32 `DirectInput8Create` and forwards to it.

use crate::log::fatal;
use std::ffi::c_void;
use std::mem::transmute;
use std::sync::LazyLock;
use windows_sys::Win32::Foundation::{HINSTANCE, MAX_PATH};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;
use windows_sys::core::{GUID, HRESULT};

type DirectInput8CreateFn = unsafe extern "system" fn(
    HINSTANCE,
    u32,
    *const GUID,
    *mut *mut c_void,
    *mut c_void,
) -> HRESULT;

static REAL: LazyLock<DirectInput8CreateFn> = LazyLock::new(load_real);

fn load_real() -> DirectInput8CreateFn {
    const SUFFIX: [u16; 13] = {
        let s = b"\\dinput8.dll";
        let mut out = [0u16; 13];
        let mut i = 0;
        while i < s.len() {
            assert!(s[i] < 0x80, "suffix must be ASCII to widen byte-for-byte");
            out[i] = s[i] as u16;
            i += 1;
        }
        out
    };

    // We load using the full path so the bare name doesn't resolve back to us.
    let mut buf = [0u16; MAX_PATH as usize];
    let len = unsafe { GetSystemDirectoryW(buf.as_mut_ptr(), MAX_PATH) } as usize;
    let end = len + SUFFIX.len();
    if len == 0 || end > buf.len() {
        fatal!("GetSystemDirectoryW failed");
    }
    buf[len..end].copy_from_slice(&SUFFIX);

    let dll = unsafe { LoadLibraryW(buf.as_ptr()) };
    if dll.is_null() {
        fatal!("failed to load the real dinput8.dll");
    }
    if let Some(create) = unsafe { GetProcAddress(dll, c"DirectInput8Create".as_ptr().cast()) } {
        // SAFETY: The real export's signature matches `DirectInput8CreateFn`.
        unsafe { transmute::<unsafe extern "system" fn() -> isize, DirectInput8CreateFn>(create) }
    } else {
        fatal!("DirectInput8Create is not exported by the real dinput8.dll");
    }
}

#[unsafe(no_mangle)]
unsafe extern "system" fn DirectInput8Create(
    hinst: HINSTANCE,
    dw_version: u32,
    riidltf: *const GUID,
    ppv_out: *mut *mut c_void,
    punk_outer: *mut c_void,
) -> HRESULT {
    unsafe { (*REAL)(hinst, dw_version, riidltf, ppv_out, punk_outer) }
}
