//! In-memory ECL patching.

use crate::addrs::ENEMY_MANAGER_PTR_VA;
use crate::mem::read_ptr;
use crate::thread::MainToken;
use std::process::abort;
use std::ptr::{NonNull, copy_nonoverlapping, with_exposed_provenance_mut};

const ECL_FILE_MANAGER_OFFSET: usize = 0x17c;
const ECL_FILE_DATA_POINTERS_OFFSET: usize = 0xc;
// The instruction size is `0x18` and the opcode is `0xC`.
const ECL_JUMP_HEADER: u32 = 0x0018_000C;
// There are 2 parameters and the rank mask is `0xFF`.
const ECL_JUMP_DESCRIPTOR: u32 = 0x02FF_0000;

const MAX_UNDO_WRITE: usize = 8;

/// Returns `None` if no stage is loaded.
///
/// # Safety
///
/// This must only be called while the enemy manager pointer is valid to dereference
/// (e.g. in a loaded stage or once the loader has published).
unsafe fn resolve_file_array_offset() -> Option<usize> {
    let enemy_manager = unsafe { read_ptr(ENEMY_MANAGER_PTR_VA) }?;
    let file_manager = unsafe { read_ptr(enemy_manager + ECL_FILE_MANAGER_OFFSET) }?;
    Some(file_manager + ECL_FILE_DATA_POINTERS_OFFSET)
}

/// Reads ECL file slot `ordinal`. Returns `None` if the slot is null.
///
/// # Safety
///
/// `file_array_offset` must refer to a live ECL file-pointer array in which slot `ordinal` is in bounds.
unsafe fn read_slot(file_array_offset: usize, ordinal: usize) -> Option<NonNull<u8>> {
    let ptr = unsafe { read_ptr(file_array_offset + ordinal * 4) }?;
    NonNull::new(with_exposed_provenance_mut(ptr))
}

fn null_file_slot(ordinal: usize) -> ! {
    eprintln!(
        "nokozero_hook: ecl: file slot {ordinal} is null; this stage has fewer ECL files than expected"
    );
    abort();
}

fn id_mismatch(ordinal: usize, pos: u32, expect: u16, found: u16) -> ! {
    eprintln!(
        "nokozero_hook: ecl: file {ordinal} offset {pos:#x}: expected instruction id {expect:#X}, found {found:#X}"
    );
    abort();
}

struct UndoEntry {
    /// The file slot index.
    ordinal: usize,
    /// The base address of the file buffer when the write occurred.
    base: usize,
    offset: u32,
    /// The bytes before the write occurred. The first `len` are valid.
    original: [u8; MAX_UNDO_WRITE],
    /// The bytes after the write occurred. The first `len` are valid.
    patched: [u8; MAX_UNDO_WRITE],
    len: usize,
}

/// A warp's ECL patches in application order.
///
/// The game reuses a stage's in-memory ECL buffers when retrying the same stage,
/// so we need to prevent patches from persisting into the next episode or leaking into a vanilla reload.
#[derive(Default)]
pub(super) struct EclUndo {
    entries: Vec<UndoEntry>,
}

impl EclUndo {
    /// Restores each overwritten span to its original bytes. Returns whether the file array was resolved at all.
    /// `false` means nothing was attempted.
    ///
    /// # Safety
    ///
    /// This must only be called while the enemy manager pointer is valid to dereference
    /// (e.g. in a loaded stage or once the loader has published).
    #[must_use]
    pub(super) unsafe fn revert(&self, _token: MainToken) -> bool {
        let Some(file_array_offset) = (unsafe { resolve_file_array_offset() }) else {
            return false;
        };
        for entry in self.entries.iter().rev() {
            let len = entry.len;
            let current = unsafe { read_ptr(file_array_offset + entry.ordinal * 4) };
            if current != Some(entry.base) {
                continue;
            }
            let dst = with_exposed_provenance_mut(entry.base + entry.offset as usize);
            let mut live = [0u8; MAX_UNDO_WRITE];
            // SAFETY: `base` still backs slot `ordinal`, so this span is mapped.
            unsafe { copy_nonoverlapping(dst, live.as_mut_ptr(), len) };
            if live[..len] != entry.patched[..len] {
                continue;
            }
            unsafe { copy_nonoverlapping(entry.original.as_ptr(), dst, len) };
        }
        true
    }
}

/// A cursor over a set of loaded ECL files.
pub(super) struct Ecl {
    file_array_offset: usize,
    file: NonNull<u8>,
    /// The slot index of the current file.
    file_ordinal: usize,
    pos: u32,
    undo: EclUndo,
}

impl Ecl {
    /// Returns `None` if no stage is loaded or if the stage's ECL files are not published yet.
    ///
    /// # Safety
    ///
    /// This must only be called while the enemy manager pointer is valid to dereference
    /// (e.g. in a loaded stage or once the loader has published). The stage must stay loaded for the returned cursor's lifetime.
    #[must_use]
    pub(super) unsafe fn new(_token: MainToken) -> Option<Self> {
        let file_array_offset = unsafe { resolve_file_array_offset() }?;
        let file = unsafe { read_slot(file_array_offset, 0) }?;
        Some(Self {
            file_array_offset,
            file,
            file_ordinal: 0,
            pos: 0,
            undo: EclUndo {
                // The largest warp (`ST7_S10`) records ~80 entries; reserving up front
                // keeps the guarded first tick free of reallocation churn.
                entries: Vec::with_capacity(96),
            },
        })
    }

    #[must_use]
    pub(super) fn take_undo(self) -> EclUndo {
        self.undo
    }

    /// Selects file `ordinal` and seeks to its start.
    pub(super) fn set_file(&mut self, ordinal: usize) {
        // SAFETY: The array resolved in `Ecl::new`, the stage stays loaded for the cursor's lifetime,
        // and every ordinal passed by the warp tables is a small constant within the file manager's fixed-capacity array.
        let Some(file) = (unsafe { read_slot(self.file_array_offset, ordinal) }) else {
            null_file_slot(ordinal);
        };
        self.file = file;
        self.file_ordinal = ordinal;
        self.pos = 0;
    }

    /// Seeks within the current file.
    pub(super) fn set_pos(&mut self, pos: u32) {
        self.pos = pos;
    }

    /// Reads the ID of the instruction at `pos` and aborts unless it is `expect`.
    ///
    /// # Safety
    ///
    /// The current file must have at least six readable bytes at `pos`.
    unsafe fn expect_ins(&self, pos: u32, expect: u16) {
        let id = unsafe {
            self.file
                .as_ptr()
                .add(pos as usize + 4)
                .cast::<u16>()
                .read_unaligned()
        };
        if id != expect {
            id_mismatch(self.file_ordinal, pos, expect, id);
        }
    }

    /// # Safety
    ///
    /// The current file must have at least `size_of::<T>()` bytes at the cursor.
    unsafe fn write<T: Copy>(&mut self, value: T) {
        let len = size_of::<T>();
        assert!(
            len <= MAX_UNDO_WRITE,
            "ECL write of {len} bytes exceeds undo buffer"
        );
        let dst = unsafe { self.file.as_ptr().add(self.pos as usize) };
        let mut original = [0u8; MAX_UNDO_WRITE];
        let mut patched = [0u8; MAX_UNDO_WRITE];
        unsafe {
            copy_nonoverlapping(dst, original.as_mut_ptr(), len);
            copy_nonoverlapping((&raw const value).cast(), patched.as_mut_ptr(), len);
        }

        self.undo.entries.push(UndoEntry {
            ordinal: self.file_ordinal,
            base: self.file.as_ptr().expose_provenance(),
            offset: self.pos,
            original,
            patched,
            len,
        });

        unsafe { dst.cast::<T>().write_unaligned(value) };
        #[expect(clippy::cast_possible_truncation)]
        {
            self.pos += len as u32;
        }
    }

    /// # Safety
    ///
    /// The current file must have at least `size_of::<T>()` bytes at `pos`.
    pub(super) unsafe fn write_at<T: Copy>(&mut self, pos: u32, value: T) {
        self.set_pos(pos);
        unsafe { self.write(value) };
    }

    /// # Safety
    ///
    /// The current file must have room for `words.len() * 4` bytes at the cursor.
    pub(super) unsafe fn write_seq(&mut self, words: &[u32]) {
        for &word in words {
            unsafe { self.write(word) };
        }
    }

    /// Verifies the instruction at `pos` has ID `expect`, then writes `words` from there.
    ///
    /// # Safety
    ///
    /// The current file must have at least `max(6, words.len() * 4)` bytes at `pos`.
    pub(super) unsafe fn write_seq_at(&mut self, pos: u32, expect: u16, words: &[u32]) {
        unsafe { self.expect_ins(pos, expect) };
        self.set_pos(pos);
        unsafe { self.write_seq(words) };
    }

    /// Verifies the instruction at `start` has ID `expect`, then overwrites it with a 24-byte ECL jump to absolute file offset `dest`.
    /// The jump is taken at frame number `at_frame`.
    ///
    /// # Safety
    ///
    /// `start` must be an instruction boundary with at least 24 writable bytes before the end of its sub.
    /// `dest` must be an instruction boundary.
    pub(super) unsafe fn jump(
        &mut self,
        start: u32,
        expect: u16,
        dest: u32,
        at_frame: i32,
        ecl_time: i32,
    ) {
        unsafe { self.expect_ins(start, expect) };
        self.set_pos(start);
        unsafe {
            self.write(ecl_time);
            self.write(ECL_JUMP_HEADER);
            self.write(ECL_JUMP_DESCRIPTOR);
            self.write(0u32);
            self.write(dest.wrapping_sub(start));
            self.write(at_frame);
        }
    }
}
