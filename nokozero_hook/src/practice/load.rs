//! The stage-load counter, plus handoff between the loader thread and main thread.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// The game is between resets, or the loader has enabled `GameThread::on_tick` but not yet reached the `thread_start` ret.
const LOAD_NOT_SEEN: u32 = 0;
/// The published load did not consume an arm. This occurs during an unrequested load
/// (menu entry, game-over re-entry, natural clear) or a reset with inactive params.
const LOAD_VANILLA: u32 = 1;
/// The published load consumed an arm, so its first tick applies a warp.
const LOAD_APPLY: u32 = 2;

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

/// Cross-thread atomics for handoff between the loader thread and main thread.
pub(super) struct LoadHandoff {
    state: AtomicU32,
    armed: AtomicBool,
    generation: AtomicU32,
}

pub(super) static HANDOFF: LoadHandoff = LoadHandoff {
    state: AtomicU32::new(LOAD_NOT_SEEN),
    armed: AtomicBool::new(false),
    generation: AtomicU32::new(0),
};

impl LoadHandoff {
    /// Forgets any unconsumed publish and records whether the upcoming load is managed.
    /// This is called on the main thread when starting a reset's reload.
    ///
    /// This must run before the `GAMEMODE_RETRY` write that lets the load start.
    pub(super) fn arm(&self, managed: bool) {
        self.state.store(LOAD_NOT_SEEN, Ordering::Relaxed);
        self.armed.store(managed, Ordering::Relaxed);
    }

    /// Consumes the armed flag, bumps the generation, and publishes the outcome. This is called on the loader thread at its ret.
    pub(super) fn publish(&self) {
        let armed = self.armed.swap(false, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
        let published = if armed { LOAD_APPLY } else { LOAD_VANILLA };
        self.state.store(published, Ordering::Release);
    }

    /// Takes an unconsumed publish, returning `(generation, managed)`, or `None` if nothing has published since the arm.
    /// This is called on the main thread.
    pub(super) fn consume(&self) -> Option<(Generation, bool)> {
        let published = self.state.swap(LOAD_NOT_SEEN, Ordering::Acquire);
        if published == LOAD_NOT_SEEN {
            return None;
        }
        Some((
            Generation(self.generation.load(Ordering::Relaxed)),
            published == LOAD_APPLY,
        ))
    }
}

/// Returns the live load counter.
pub(crate) fn load_generation() -> Generation {
    Generation(HANDOFF.generation.load(Ordering::Relaxed))
}
