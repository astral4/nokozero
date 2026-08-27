//! IAT hooking utilities.

use crate::patch::with_writable;
use std::ffi::{CStr, c_char};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::mem::offset_of;
use std::process::abort;
use std::ptr::{NonNull, read_unaligned, write_unaligned};
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::Diagnostics::Debug::{
    IMAGE_DATA_DIRECTORY, IMAGE_DIRECTORY_ENTRY_IMPORT, IMAGE_NT_HEADERS32, IMAGE_OPTIONAL_HEADER32,
};
use windows_sys::Win32::System::SystemServices::{
    IMAGE_DOS_HEADER, IMAGE_DOS_SIGNATURE, IMAGE_IMPORT_BY_NAME, IMAGE_IMPORT_DESCRIPTOR,
    IMAGE_NT_SIGNATURE, IMAGE_ORDINAL_FLAG32,
};
use windows_sys::Win32::System::WindowsProgramming::IMAGE_THUNK_DATA32;

/// How an import is identified in a module's import directory.
#[derive(Clone, Copy)]
pub(crate) enum ImportRef {
    /// By-name import matched case-insensitively across every descriptor.
    Name(&'static str),
    /// By-ordinal import matched within the named DLL's descriptor.
    Ordinal { dll: &'static str, ordinal: u16 },
}

impl Display for ImportRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Name(name) => f.write_str(name),
            Self::Ordinal { dll, ordinal } => write!(f, "{dll}!#{ordinal}"),
        }
    }
}

/// Rewrites the import's IAT slot to `hook`, aborting the process on any failure.
///
/// # Safety
///
/// `game` must be a loaded module handle. `hook` must point to a function whose signature matches the import's.
/// No other thread may call through the import concurrently with the plain slot rewrite.
/// In practice, this means this function should be called during `DLL_PROCESS_ATTACH`.
pub(crate) unsafe fn hook_import(game: HMODULE, import: ImportRef, hook: *mut ()) {
    let Some(slot_ptr) = (unsafe { find_iat_slot(game, import) }) else {
        eprintln!("nokozero_hook: iat: import not found: {import}");
        abort();
    };
    let slot_raw = slot_ptr.as_ptr();
    let current = unsafe { read_unaligned(slot_raw) };
    if current.is_null() {
        eprintln!("nokozero_hook: iat: null slot: {import}");
        abort();
    }
    let written = unsafe {
        with_writable(slot_raw.cast(), size_of::<*mut ()>(), |_| {
            write_unaligned(slot_raw, hook);
        })
    };
    if written.is_none() {
        eprintln!("nokozero_hook: iat: slot unwritable: {import}");
        abort();
    }
}

/// Returns the import directory's mapped base, or `None` for a malformed image.
unsafe fn import_directory(module: HMODULE) -> Option<*const u8> {
    unsafe {
        let base = module.cast::<u8>().cast_const();
        let e_magic: u16 = read_unaligned(base.add(offset_of!(IMAGE_DOS_HEADER, e_magic)).cast());
        if e_magic != IMAGE_DOS_SIGNATURE {
            return None;
        }
        // We treat negative `e_lfanew` as malformed instead of wrapping.
        let e_lfanew: i32 = read_unaligned(base.add(offset_of!(IMAGE_DOS_HEADER, e_lfanew)).cast());
        let nt_base = base.add(usize::try_from(e_lfanew).ok()?);
        let signature: u32 = read_unaligned(
            nt_base
                .add(offset_of!(IMAGE_NT_HEADERS32, Signature))
                .cast(),
        );
        if signature != IMAGE_NT_SIGNATURE {
            return None;
        }
        // `offset_of!` only takes literal paths, so the array index is manual.
        let dd_offset = offset_of!(IMAGE_NT_HEADERS32, OptionalHeader)
            + offset_of!(IMAGE_OPTIONAL_HEADER32, DataDirectory)
            + IMAGE_DIRECTORY_ENTRY_IMPORT as usize * size_of::<IMAGE_DATA_DIRECTORY>();
        let va: u32 = read_unaligned(
            nt_base
                .add(dd_offset + offset_of!(IMAGE_DATA_DIRECTORY, VirtualAddress))
                .cast(),
        );
        let size: u32 = read_unaligned(
            nt_base
                .add(dd_offset + offset_of!(IMAGE_DATA_DIRECTORY, Size))
                .cast(),
        );
        if va == 0 || size == 0 {
            return None;
        }
        Some(base.add(va as usize))
    }
}

/// Finds the `FirstThunk` slot matching `import`. Note: `module` must be the game, not our own DLL.
unsafe fn find_iat_slot(module: HMODULE, import: ImportRef) -> Option<NonNull<*mut ()>> {
    unsafe {
        let imp_dir = import_directory(module)?;
        let base_mut = module.cast::<u8>();
        let base = base_mut.cast_const();

        let mut desc_offset = 0;
        loop {
            let dll_name_rva: u32 = read_unaligned(
                imp_dir
                    .add(desc_offset + offset_of!(IMAGE_IMPORT_DESCRIPTOR, Name))
                    .cast(),
            );
            if dll_name_rva == 0 {
                return None;
            }

            // Ordinals are scoped to their exporting DLL, so mismatched descriptors are skipped wholesale.
            if let ImportRef::Ordinal { dll, .. } = import {
                let dll_name_ptr = base.add(dll_name_rva as usize).cast::<c_char>();
                let dll_name = CStr::from_ptr(dll_name_ptr).to_bytes();
                if !dll_name.eq_ignore_ascii_case(dll.as_bytes()) {
                    desc_offset += size_of::<IMAGE_IMPORT_DESCRIPTOR>();
                    continue;
                }
            }

            let oft: u32 = read_unaligned(
                imp_dir
                    .add(desc_offset + offset_of!(IMAGE_IMPORT_DESCRIPTOR, Anonymous))
                    .cast(),
            );
            let ft: u32 = read_unaligned(
                imp_dir
                    .add(desc_offset + offset_of!(IMAGE_IMPORT_DESCRIPTOR, FirstThunk))
                    .cast(),
            );

            if oft == 0 || ft == 0 {
                desc_offset += size_of::<IMAGE_IMPORT_DESCRIPTOR>();
                continue;
            }

            let mut i = 0;
            loop {
                let entry: u32 = read_unaligned(
                    base.add(oft as usize + i * size_of::<IMAGE_THUNK_DATA32>())
                        .cast(),
                );
                if entry == 0 {
                    break;
                }
                let hit = match import {
                    ImportRef::Name(want) if entry & IMAGE_ORDINAL_FLAG32 == 0 => {
                        let name_offset = entry as usize + offset_of!(IMAGE_IMPORT_BY_NAME, Name);
                        let name_ptr = base.add(name_offset).cast::<c_char>();
                        let imp_name = CStr::from_ptr(name_ptr).to_bytes();
                        imp_name.eq_ignore_ascii_case(want.as_bytes())
                    }
                    ImportRef::Ordinal { ordinal, .. } if entry & IMAGE_ORDINAL_FLAG32 != 0 => {
                        entry & 0xFFFF == u32::from(ordinal)
                    }
                    ImportRef::Name(_) | ImportRef::Ordinal { .. } => false,
                };
                if hit {
                    let slot_offset = ft as usize + i * size_of::<IMAGE_THUNK_DATA32>();
                    // Every access through this pointer is unaligned.
                    #[expect(clippy::cast_ptr_alignment)]
                    let slot = base_mut.add(slot_offset).cast::<*mut ()>();
                    return NonNull::new(slot);
                }
                i += 1;
            }
            desc_offset += size_of::<IMAGE_IMPORT_DESCRIPTOR>();
        }
    }
}
