//! Stage-load counter for tracking handoff between the loader thread and main thread.
//!
//! The counter starts even. The stage loader thread increments it to odd at its entry and back to even at its ret.
//! So, `generation = counter / 2` counts publishes, and an odd counter value means a load is between entry and publish.

use crate::thread::{MainCell, MainThread};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// A sample of the stage-load counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Generation(u32);

impl Generation {
    /// The counter's initial value when no load has ever published.
    pub(super) const PRE_LOAD: Self = Self(0);

    /// Returns the raw counter value for serialization into observations only.
    pub(crate) const fn to_wire(self) -> u32 {
        self.0
    }

    /// Testing utility for directly constructing an instance from a raw value.
    #[cfg(test)]
    pub(crate) const fn for_test(raw: u32) -> Self {
        Self(raw)
    }
}

/// What the main thread sees of the loader thread.
#[derive(Clone, Copy)]
pub(super) enum LoadStatus {
    /// A load is between loader entry and publish. In this state, arming should not be requested and ECL memory should not be touched.
    InFlight,
    /// No load is running.
    Settled {
        /// The number of publishes.
        generation: Generation,
        /// Whether the latest publish's load took the arm.
        armed: bool,
    },
}

/// Cross-thread atomics for handoff between the loader thread and main thread.
pub(super) struct LoadHandoff {
    /// The value is odd while a load is between loader entry and publish. `generation = counter / 2`.
    counter: AtomicU32,
    /// Whether the in-flight (or latest) load took the arm. Written at loader entry.
    armed: AtomicBool,
    /// Arm request set by the main thread. Taken at the next loader entry.
    arm_request: AtomicBool,
}

pub(super) static HANDOFF: LoadHandoff = LoadHandoff::new();

impl LoadHandoff {
    const fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
            armed: AtomicBool::new(false),
            arm_request: AtomicBool::new(false),
        }
    }

    /// Requests that the next load apply a warp on its first tick.
    ///
    /// This should be called from the main thread immediately before the `GAMEMODE_RETRY` write,
    /// and only while [`LoadStatus::Settled`] with every publish consumed.
    pub(super) fn arm(&self) {
        self.arm_request.store(true, Ordering::Relaxed);
    }

    /// Marks a load in flight and takes the arm request.
    ///
    /// This should be called from the loader thread at its entry.
    pub(super) fn enter(&self) {
        let armed = self.arm_request.swap(false, Ordering::Relaxed);
        self.armed.store(armed, Ordering::Relaxed);
        self.counter.fetch_add(1, Ordering::Release);
    }

    /// Publishes the load.
    ///
    /// This should be called from the loader thread at its ret.
    pub(super) fn publish(&self) {
        self.counter.fetch_add(1, Ordering::Release);
    }

    /// Samples the counter.
    ///
    /// This should be called from the main thread.
    pub(super) fn status(&self) -> LoadStatus {
        let counter = self.counter.load(Ordering::Acquire);
        if counter % 2 == 1 {
            LoadStatus::InFlight
        } else {
            LoadStatus::Settled {
                generation: Generation(counter / 2),
                armed: self.armed.load(Ordering::Relaxed),
            }
        }
    }
}

/// Returns the generation of the latest published load.
pub(super) fn load_generation() -> Generation {
    Generation(HANDOFF.counter.load(Ordering::Relaxed) / 2)
}

/// A [`MainCell`] where the value belongs within a single stage load. Writes indicate the value's load [`Generation`].
/// Reads indicate which generation is requesting the query, and mismatched-generation reads return the state of a brand-new load.
pub(super) struct PerLoad<T> {
    /// The state of a brand-new load.
    fresh: T,
    cell: MainCell<Option<(Generation, T)>>,
}

impl<T: Copy> PerLoad<T> {
    /// Constructs a new instance. `fresh` must be the state of a brand-new load.
    pub(super) const fn new(fresh: T) -> Self {
        Self {
            fresh,
            cell: MainCell::new(None),
        }
    }

    /// Returns the value if it was stored under `generation`. Otherwise, returns the fresh value.
    pub(super) fn get(&self, thread: MainThread, generation: Generation) -> T {
        match self.cell.get(thread) {
            Some((stamp, value)) if stamp == generation => value,
            _ => self.fresh,
        }
    }

    /// Stores `value` as belonging to `generation`.
    pub(super) fn set(&self, thread: MainThread, generation: Generation, value: T) {
        self.cell.set(thread, Some((generation, value)));
    }

    /// Runs `f` on the current value if it belongs to `generation`. Otherwise, runs `f` on the fresh value.
    /// If `f` returns `Some(_)`, then modifications to `f`'s input are stored back as the new cell value.
    /// If `f` returns `None`, then any modifications are discarded and the cell value is left untouched.
    pub(super) fn update<R>(
        &self,
        thread: MainThread,
        generation: Generation,
        f: impl FnOnce(&mut T) -> Option<R>,
    ) -> Option<R> {
        let mut value = self.get(thread, generation);
        let r = f(&mut value)?;
        self.set(thread, generation, value);
        Some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::{Generation, LoadHandoff, LoadStatus, PerLoad};
    use crate::thread::MainThread;
    use crate::thread::test_support::MainClaim;

    fn settled(handoff: &LoadHandoff) -> (Generation, bool) {
        match handoff.status() {
            LoadStatus::InFlight => panic!("expected a settled status"),
            LoadStatus::Settled { generation, armed } => (generation, armed),
        }
    }

    #[test]
    fn parity_track_load_window() {
        let handoff = LoadHandoff::new();
        assert_eq!(settled(&handoff), (Generation::for_test(0), false));
        handoff.enter();
        assert!(matches!(handoff.status(), LoadStatus::InFlight));
        handoff.publish();
        assert_eq!(settled(&handoff), (Generation::for_test(1), false));
    }

    #[test]
    fn arm_taken_by_next_entry() {
        let handoff = LoadHandoff::new();
        handoff.arm();
        handoff.enter();
        handoff.publish();
        assert_eq!(settled(&handoff), (Generation::for_test(1), true));
        // The arm was consumed, so the following load is unarmed.
        handoff.enter();
        handoff.publish();
        assert_eq!(settled(&handoff), (Generation::for_test(2), false));
    }

    #[test]
    fn load_enter_before_arm() {
        let handoff = LoadHandoff::new();
        handoff.enter();
        handoff.arm();
        handoff.publish();
        assert_eq!(settled(&handoff), (Generation::for_test(1), false));
        // The arm waits for the next load.
        handoff.enter();
        handoff.publish();
        assert_eq!(settled(&handoff), (Generation::for_test(2), true));
    }

    #[test]
    fn per_load_values_scoped_to_generation() {
        static VALUE: PerLoad<u32> = PerLoad::new(42);

        let _claim = MainClaim::acquire();

        let thread = MainThread::claim();
        let (g0, g1, g2, g3) = (
            Generation::for_test(0),
            Generation::for_test(1),
            Generation::for_test(2),
            Generation::for_test(3),
        );

        // Before any write has occurred, no generation matches, so the fresh value is read.
        assert_eq!(VALUE.get(thread, g0), 42);

        VALUE.set(thread, g1, 7);
        assert_eq!(VALUE.get(thread, g1), 7);
        assert_eq!(VALUE.get(thread, g2), 42);
        assert_eq!(VALUE.get(thread, g1), 7);

        let x = VALUE.update(thread, g1, |v| {
            *v *= 2;
            Some(*v)
        });
        assert_eq!(x, Some(14));
        assert_eq!(VALUE.get(thread, g1), 14);

        let x = VALUE.update(thread, g1, |v| {
            *v = 99;
            None::<()>
        });
        assert_eq!(x, None);
        assert_eq!(VALUE.get(thread, g1), 14);

        let x = VALUE.update(thread, g3, |v| {
            *v += 5;
            Some(*v)
        });
        assert_eq!(x, Some(47));
        assert_eq!(VALUE.get(thread, g3), 47);
        assert_eq!(VALUE.get(thread, g1), 42);
    }
}
