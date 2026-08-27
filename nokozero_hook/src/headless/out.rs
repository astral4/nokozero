//! COM out-parameter helpers.

use windows::core::Result;

/// # Safety
///
/// `out` must be null or point to a writable `T` for which the all-zeroes bit pattern is valid.
pub(super) unsafe fn get_zeroed<T>(out: *mut T) -> Result<()> {
    unsafe { get_zeroed_array(out, 1) }
}

/// # Safety
///
/// `out` must be null or point to `count` writable `T` for which the all-zeroes bit pattern is valid.
#[expect(clippy::unnecessary_wraps)]
pub(super) unsafe fn get_zeroed_array<T>(out: *mut T, count: usize) -> Result<()> {
    if !out.is_null() {
        unsafe { out.write_bytes(0, count) };
    }
    Ok(())
}

/// Writes `value` if `out` is non-null.
///
/// # Safety
///
/// `out` must be null or point to a writable `T`.
#[expect(clippy::unnecessary_wraps)]
pub(super) unsafe fn put<T>(out: *mut T, value: T) -> Result<()> {
    if !out.is_null() {
        unsafe { out.write(value) };
    }
    Ok(())
}
