use anyhow::{Context, Result, bail};
use scopeguard::guard;
use std::ffi::{CStr, CString, c_void};
use std::mem::transmute;
use windows::Win32::Foundation::{CloseHandle, GetLastError, HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::Diagnostics::Debug::WriteProcessMemory;
use windows::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAllocEx, VirtualFreeEx,
};
use windows::Win32::System::Threading::{
    CREATE_SUSPENDED, CreateProcessA, CreateRemoteThread, GetExitCodeThread, INFINITE,
    PROCESS_INFORMATION, ResumeThread, STARTUPINFOA, WaitForSingleObject,
};
use windows::core::{PCSTR, s};

type ThreadStartRoutine = unsafe extern "system" fn(*mut c_void) -> u32;

fn main() -> Result<()> {
    unsafe {
        // TODO: rework path reading
        let game_path = CString::new("C:\\users\\user\\th15\\th15.exe")?;
        let dll_path = CString::new("C:\\users\\user\\th15\\nokozero_hook.dll")?;

        #[allow(clippy::cast_possible_truncation)]
        let startup_info = STARTUPINFOA {
            cb: size_of::<STARTUPINFOA>() as u32,
            ..Default::default()
        };
        let mut process_info = PROCESS_INFORMATION::default();

        // Create the game process in suspended state
        CreateProcessA(
            PCSTR::from_raw(game_path.as_ptr().cast()),
            None,
            None,
            None,
            false,
            CREATE_SUSPENDED,
            None,
            None,
            &raw const startup_info,
            &raw mut process_info,
        )
        .context("failed to create game process")?;

        // If DLL injection fails, we clean up all existing handles
        // before returning the error value.
        let result = inject_dll(process_info.hProcess, &dll_path).context("failed to inject DLL");

        // Resume the main thread only after DLL injection is complete
        if ResumeThread(process_info.hThread) == u32::MAX {
            GetLastError()
                .to_hresult()
                .ok()
                .context("failed to resume game thread")?;
            bail!("failed to resume game thread");
        }

        // Clean up handles
        CloseHandle(process_info.hThread)?;
        CloseHandle(process_info.hProcess)?;

        result?;

        Ok(())
    }
}

unsafe fn inject_dll(process: HANDLE, dll_path: &CStr) -> Result<()> {
    unsafe {
        // `CStr::count_bytes()` does not include the nul terminator,
        // so we add 1 to the returned value.
        let dll_path_len = dll_path.count_bytes() + 1;

        // Allocate memory for the DLL path in the game process
        let remote_buffer = VirtualAllocEx(
            process,
            None,
            dll_path_len,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );

        if remote_buffer.is_null() {
            bail!("failed to allocate memory in game process");
        }

        // Ensure the allocated memory is freed if an error is encountered
        let cleanup = guard((), |()| {
            drop(VirtualFreeEx(process, remote_buffer, 0, MEM_RELEASE));
        });

        // Write the DLL path to the allocated memory
        WriteProcessMemory(
            process,
            remote_buffer,
            dll_path.as_ptr().cast(),
            dll_path_len,
            None,
        )
        .context("failed to write DLL path to game process memory")?;

        // We need the raw address of `LoadLibraryA` within `kernel32.dll`.
        // However, when we import `LoadLibraryA` from the `windows` crate,
        // we aren't getting the raw address. Instead, we're getting
        // an indirect reference through our injector's import address table (IAT).
        // This IAT entry only exists in our injector process, not in the game process.
        // `GetProcAddress` returns the raw address of `LoadLibraryA` inside `kernel32.dll`,
        // which is valid in both processes since `kernel32.dll` is loaded at the same location in each.
        let kernel_handle =
            GetModuleHandleA(s!("kernel32.dll")).context("failed to get handle to kernel32.dll")?;
        let load_library_addr = GetProcAddress(kernel_handle, s!("LoadLibraryA"))
            .context("failed to get address of LoadLibraryA")?;

        // `FARPROC`, the return type of `GetProcAddress`,
        // indicates a pointer to a function of any signature.
        // The actual type of `LoadLibraryA` is `fn(PCSTR) -> *mut c_void`.
        // However, the function pointer for `LoadLibraryA` must have the type
        // `fn(*mut c_void) -> u32` when using it with `CreateRemoteThread`.
        // For our purpose of compiling to a 32-bit target, these types are ABI compatible:
        // - The same calling convention is used.
        // - Inputs: both `PCSTR` and `*mut c_void` are 32-bit pointers,
        //   have the same alignment requirements, and
        //   get interpreted as raw memory addresses.
        // - Outputs: `u32` and `*mut c_void` are both 32 bits.
        // So, it is reasonable to transmute here.
        let thread_process: ThreadStartRoutine = transmute(load_library_addr);

        // Create a remote thread to call `LoadLibraryA` with our DLL path
        let thread_handle = CreateRemoteThread(
            process,
            None,
            0,
            Some(thread_process),
            Some(remote_buffer),
            0,
            None,
        )
        .context("failed to create DLL injection thread")?;

        // Wait for the remote thread to complete
        if WaitForSingleObject(thread_handle, INFINITE) != WAIT_OBJECT_0 {
            CloseHandle(thread_handle)?;
            bail!("failed to wait for DLL injection thread");
        }

        // Check if `LoadLibraryA` succeeded
        let mut exit_code = 0u32;
        GetExitCodeThread(thread_handle, &raw mut exit_code)
            .context("failed to get DLL injection thread exit code")?;

        CloseHandle(thread_handle).context("failed to clean up DLL injection thread handle")?;

        if exit_code == 0 {
            GetLastError()
                .to_hresult()
                .ok()
                .context("failed to inject DLL; LoadLibraryA returned error")?;
            bail!("failed to inject DLL; LoadLibraryA returned error");
        }

        drop(cleanup);

        Ok(())
    }
}
