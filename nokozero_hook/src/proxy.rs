// When this library is deployed as `dinput8.dll` in the game directory,
// Windows DLL search order makes it load before the real system DLL, so we can do hooking.
// We forward all `dinput8.dll` exports to the system DLL.

use std::ffi::c_void;
use std::mem::transmute;
use std::sync::LazyLock;
use windows::Win32::Foundation::{HMODULE, MAX_PATH};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::Win32::System::SystemInformation::GetSystemDirectoryA;
use windows::core::{PCSTR, s};

struct ModuleHandle(HMODULE);

// SAFETY: The module handle is valid for the lifetime of the process
// and is only used for read-only operations (`GetProcAddress`).
unsafe impl Send for ModuleHandle {}
unsafe impl Sync for ModuleHandle {}

static MODULE: LazyLock<ModuleHandle> = LazyLock::new(|| {
    const SUFFIX: &[u8] = b"\\dinput8.dll\0";
    let mut path = [0u8; MAX_PATH as usize];
    let len = unsafe { GetSystemDirectoryA(Some(&mut path)) } as usize;
    assert!(len > 0, "failed to get system directory");
    path[len..][..const { SUFFIX.len() }].copy_from_slice(SUFFIX);
    let module = unsafe { LoadLibraryA(PCSTR::from_raw(path.as_ptr())) }
        .expect("failed to load real dinput8.dll");
    ModuleHandle(module)
});

/// Resolves an export from the real `dinput8.dll` by name.
fn get_proc(name: PCSTR) -> unsafe extern "system" fn() -> isize {
    unsafe { GetProcAddress(MODULE.0, name) }.expect("failed to resolve dinput8.dll export")
}

// SAFETY: The real `dinput8.dll` exports this function with the signature defined by `F`.
// This applies to all forwarding functions below.
#[unsafe(no_mangle)]
extern "system" fn DirectInput8Create(
    hinst: *mut c_void,
    version: u32,
    riid: *const c_void,
    out: *mut *mut c_void,
    outer: *mut c_void,
) -> i32 {
    type F = unsafe extern "system" fn(
        *mut c_void,
        u32,
        *const c_void,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32;

    unsafe {
        let f: F = transmute(get_proc(s!("DirectInput8Create")));
        f(hinst, version, riid, out, outer)
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllCanUnloadNow() -> i32 {
    type F = unsafe extern "system" fn() -> i32;
    unsafe {
        let f: F = transmute(get_proc(s!("DllCanUnloadNow")));
        f()
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllGetClassObject(
    rclsid: *const c_void,
    riid: *const c_void,
    ppv: *mut *mut c_void,
) -> i32 {
    type F = unsafe extern "system" fn(*const c_void, *const c_void, *mut *mut c_void) -> i32;
    unsafe {
        let f: F = transmute(get_proc(s!("DllGetClassObject")));
        f(rclsid, riid, ppv)
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllRegisterServer() -> i32 {
    type F = unsafe extern "system" fn() -> i32;
    unsafe {
        let f: F = transmute(get_proc(s!("DllRegisterServer")));
        f()
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllUnregisterServer() -> i32 {
    type F = unsafe extern "system" fn() -> i32;
    unsafe {
        let f: F = transmute(get_proc(s!("DllUnregisterServer")));
        f()
    }
}
