//! A stable host-memory allocation handed to the game via `Lock` / `LockRect`.

use crate::log::fatal;
use std::cmp::max;
use std::ptr::NonNull;

pub(super) struct Backing(NonNull<[u8]>);

impl Backing {
    pub(super) fn new(len: usize) -> Self {
        let boxed = vec![0u8; max(len, 1)].into_boxed_slice();
        // SAFETY: `Box::into_raw` never returns null.
        Self(unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) })
    }

    /// Always returns at least 1, even when [`Backing::new`] was asked for 0 bytes.
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a pointer `offset` bytes into the allocation, valid for `bytes` bytes.
    pub(super) fn ptr_at(&self, offset: usize, bytes: usize) -> *mut u8 {
        let in_range = offset < self.len()
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.len());
        if !in_range {
            fatal!(
                "{bytes} bytes at offset {offset} out of range (allocation is {} bytes)",
                self.len()
            );
        }
        // SAFETY: `offset..offset + bytes` was just checked to be inside the allocation.
        unsafe { self.0.cast::<u8>().as_ptr().add(offset) }
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `Box::into_raw` and is reclaimed exactly once.
        unsafe { drop(Box::from_raw(self.0.as_ptr())) };
    }
}

// SAFETY: `Backing` is a plain heap allocation without thread affinity.
unsafe impl Send for Backing {}
unsafe impl Sync for Backing {}
