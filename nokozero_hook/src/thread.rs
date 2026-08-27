//! Constructs for main-thread identity and access.

use crate::log::fatal;
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use windows_sys::Win32::System::Threading::GetCurrentThreadId;

static MAIN_TID: AtomicU32 = AtomicU32::new(0);

fn off_main_thread(current: u32) -> ! {
    let main = MAIN_TID.load(Ordering::Relaxed);
    fatal!("main-thread state touched from thread {current} (main thread: {main})");
}

/// A zero-sized, runtime-checked witness that the calling thread is the game's update ("main") thread. Required by [`MainCell`] accessors.
#[derive(Clone, Copy)]
pub(crate) struct MainThread(PhantomData<*const ()>);

impl MainThread {
    /// On the first call, claims the update thread for the calling thread. On subsequent calls, confirms the claim.
    /// This should only be called from the input hook, which is known to be on the update thread.
    pub(crate) fn claim() -> Self {
        let tid = unsafe { GetCurrentThreadId() };
        match MAIN_TID.load(Ordering::Relaxed) {
            0 => {
                if MAIN_TID
                    .compare_exchange(0, tid, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    off_main_thread(tid);
                }
                Self(PhantomData)
            }
            t if t == tid => Self(PhantomData),
            _ => off_main_thread(tid),
        }
    }

    /// Aborts if the update thread is unclaimed or the caller is not from the claimed thread.
    pub(crate) fn current() -> Self {
        let tid = unsafe { GetCurrentThreadId() };
        if MAIN_TID.load(Ordering::Relaxed) != tid {
            off_main_thread(tid);
        }
        Self(PhantomData)
    }
}

/// A zero-sized proof that game code is not concurrently reading or writing memory that the holder will modify.
#[derive(Clone, Copy)]
pub(crate) struct MainToken(MainThread);

impl MainToken {
    /// # Safety
    ///
    /// Game code must not be concurrently reading or writing memory that the holder will modify.
    pub(crate) unsafe fn new(thread: MainThread) -> Self {
        Self(thread)
    }

    /// Returns the claimed thread associated with this instance.
    pub(crate) fn thread(self) -> MainThread {
        self.0
    }
}

/// An interior-mutable cell for main-thread-only state in contexts that require `Sync`. This type should be preferred over atomic types
/// when there is no cross-thread sharing, as atomics would misleadingly signal lock-free synchronization that isn't present.
pub(crate) struct MainCell<T>(Cell<T>);

// SAFETY: Every access requires a `MainThread`, and `MainThread` is `!Send + !Sync`,
// so neither a witness nor a reference to one can reach another thread.
unsafe impl<T> Send for MainCell<T> {}
unsafe impl<T> Sync for MainCell<T> {}

impl<T> MainCell<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(Cell::new(value))
    }

    /// Drops the previous contents in place.
    pub(crate) fn set(&self, _thread: MainThread, value: T) {
        self.0.set(value);
    }

    /// Swaps in `value` and returns the previous contents.
    #[must_use]
    pub(crate) fn replace(&self, _thread: MainThread, value: T) -> T {
        self.0.replace(value)
    }
}

impl<T: Copy> MainCell<T> {
    pub(crate) fn get(&self, _thread: MainThread) -> T {
        self.0.get()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::MAIN_TID;
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// Temporary exclusive access to an unclaimed [`super::MAIN_TID`].
    ///
    /// A real process claims `MAIN_TID` once and never releases it, so any two tests that need a claim would race for it.
    /// Acquiring this guard serializes such tests against each other and hands the calling thread an unclaimed `MAIN_TID`,
    /// released again on drop. Tests that construct a [`super::MainThread`] should acquire this first
    /// and hold this for as long as any copy of the witness is in use.
    pub(crate) struct MainClaim(#[expect(dead_code)] MutexGuard<'static, ()>);

    impl MainClaim {
        pub(crate) fn acquire() -> Self {
            static LOCK: Mutex<()> = Mutex::new(());

            // Poisoning only records that an earlier holder panicked.
            // The state established by this guard is unconditional, so the lock is still usable afterwards.
            let guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
            MAIN_TID.store(0, Ordering::Relaxed);
            Self(guard)
        }
    }

    impl Drop for MainClaim {
        fn drop(&mut self) {
            MAIN_TID.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::MainClaim;
    use super::{MainCell, MainThread};

    #[test]
    fn thread_claim_idempotency() {
        static CELL: MainCell<u32> = MainCell::new(1);

        let _claim = MainClaim::acquire();

        let thread = MainThread::claim();
        let _confirmed = MainThread::claim();
        let _current = MainThread::current();

        assert_eq!(CELL.get(thread), 1);
        CELL.set(thread, 5);
        assert_eq!(CELL.replace(thread, 7), 5);
        assert_eq!(CELL.get(thread), 7);
    }
}
