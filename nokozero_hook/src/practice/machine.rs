//! Reset/load state machine logic.

use super::data::section_stage;
use super::load::Generation;
use super::{LiveWarp, PracticeParams};

/// Wire codes for the observation's `reset_outcome` word describing the reset from `reset_seq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    Idle = 0,
    Pending = 1,
    Applied = 2,
    Vanilla = 3,
    /// The `(section, phase)` pair is in range for a stage but isn't listed in the warp catalog.
    FailedUnmapped = 4,
    FailedStageMismatch = 5,
    /// The stage's ECL files were not resolvable when the warp was attempted.
    FailedNoEcl = 6,
}

#[derive(Clone, Copy)]
enum ResetState {
    /// No reset has ever been accepted.
    Idle,
    /// Reset accepted; waiting for a stable in-stage frame without an unpublished load.
    Requested {
        seq: u32,
        params: Option<PracticeParams>,
    },
    /// The reload has been triggered (teardown queued, tick guard holding); waiting for it to publish.
    Reloading {
        seq: u32,
        params: Option<PracticeParams>,
    },
    /// This is reported until replaced by the next command.
    Done { seq: u32, outcome: Outcome },
}

struct CurrentLoad {
    generation: Generation,
    section: u32,
}

/// Everything that the main thread knows about the reset/load lifecycle.
/// Transitions are methods that take the environment as parameters and return writes to be performed by the caller.
pub(super) struct Lifecycle {
    reset: ResetState,
    /// The last consumed load.
    current_load: CurrentLoad,
    /// Whether a load whose publish is not yet consumed has been observed running.
    load_in_flight: bool,
    /// The live warp's undo record.
    last_warp: Option<LiveWarp>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReloadPlan {
    /// `STAGE_SELECT` value when the reset involves a warp with a decodable stage.
    pub(super) stage_select: Option<u32>,
    /// `DIFFICULTY` value. `None` = leave as-is.
    pub(super) difficulty: Option<u32>,
    /// `CHARACTER` value. `None` = leave as-is.
    pub(super) character: Option<u32>,
    /// Whether the upcoming load is managed (i.e. whether its first tick applies a warp).
    pub(super) managed: bool,
}

impl Lifecycle {
    /// The state before claiming. No reset has been accepted, the generation is 0 and unconsumed, and there are no ECL patches to undo.
    pub(super) const INIT: Self = Self {
        reset: ResetState::Idle,
        current_load: CurrentLoad {
            generation: Generation::PRE_LOAD,
            section: 0,
        },
        load_in_flight: false,
        last_warp: None,
    };

    /// Accepts a decoded RESET command. Returns `false` if a previous reset is still in flight (which means a protocol violation).
    pub(super) fn accept(&mut self, seq: u32, params: PracticeParams) -> bool {
        if matches!(
            self.reset,
            ResetState::Requested { .. } | ResetState::Reloading { .. }
        ) {
            return false;
        }
        self.reset = ResetState::Requested {
            seq,
            params: Some(params).filter(|p| p.active),
        };
        true
    }

    /// Observes one supervisor frame by consuming an unmanaged generation bump.
    pub(super) fn observe(&mut self, generation: Generation, loader_running: bool) {
        if generation != self.current_load.generation {
            self.consume_load(generation, 0);
            if !matches!(self.reset, ResetState::Reloading { .. }) {
                self.last_warp = None;
            }
        }
        if loader_running {
            self.load_in_flight = true;
        }
    }

    /// Transitions from [`ResetState::Requested`] to [`ResetState::Reloading`] if `stable_ingame` is true
    /// (i.e. no other scene switch is queued). Returns the writes to perform, or `None` to keep the state at [`ResetState::Requested`].
    pub(super) fn try_start_reload(&mut self, stable_ingame: bool) -> Option<ReloadPlan> {
        let ResetState::Requested { seq, params } = self.reset else {
            return None;
        };
        if !stable_ingame || self.load_in_flight {
            return None;
        }
        self.reset = ResetState::Reloading { seq, params };
        Some(ReloadPlan {
            stage_select: params.and_then(|p| section_stage(p.section)),
            difficulty: params.and_then(|p| u32::try_from(p.difficulty).ok()),
            character: params.and_then(|p| u32::try_from(p.character).ok()),
            managed: params.is_some(),
        })
    }

    /// Returns whether the tick guard should be holding ticks (i.e. whether the current state is [`ResetState::Reloading`]).
    pub(super) fn reset_reloading(&self) -> bool {
        matches!(self.reset, ResetState::Reloading { .. })
    }

    /// Returns the warp params of the reset whose reload is in progress, or `None` for vanilla behavior.
    pub(super) fn reloading_params(&self) -> Option<PracticeParams> {
        match self.reset {
            ResetState::Reloading { params, .. } => params,
            ResetState::Idle | ResetState::Requested { .. } | ResetState::Done { .. } => None,
        }
    }

    /// Returns the live warp's undo slot for reverting and refilling ECL in place.
    pub(super) fn last_warp_mut(&mut self) -> &mut Option<LiveWarp> {
        &mut self.last_warp
    }

    /// Transitions from [`ResetState::Reloading`] to [`ResetState::Done`], consuming the new stage load parameters.
    pub(super) fn finish_managed(
        &mut self,
        generation: Generation,
        outcome: Outcome,
        section: u32,
    ) {
        self.consume_load(generation, section);
        if let ResetState::Reloading { seq, .. } = self.reset {
            self.reset = ResetState::Done { seq, outcome };
        }
    }

    fn consume_load(&mut self, generation: Generation, section: u32) {
        self.current_load = CurrentLoad {
            generation,
            section,
        };
        self.load_in_flight = false;
    }

    /// Returns the `(load_generation, reset_seq, reset_outcome, applied_section)` wire values.
    pub(super) fn wire(&self) -> (Generation, u32, Outcome, u32) {
        let (reset_seq, reset_outcome) = match self.reset {
            ResetState::Idle => (0, Outcome::Idle),
            ResetState::Requested { seq, .. } | ResetState::Reloading { seq, .. } => {
                (seq, Outcome::Pending)
            }
            ResetState::Done { seq, outcome } => (seq, outcome),
        };
        (
            self.current_load.generation,
            reset_seq,
            reset_outcome,
            self.current_load.section,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::ecl::EclUndo;
    use super::super::test_support::params;
    use super::{Generation, Lifecycle, LiveWarp, Outcome, ReloadPlan};

    #[test]
    fn wire_reset_phases() {
        let mut lc = Lifecycle::INIT;
        assert_eq!(lc.wire(), (Generation::for_test(0), 0, Outcome::Idle, 0));
        assert!(lc.accept(5, params(1202, 1, 0)));
        assert_eq!(lc.wire(), (Generation::for_test(0), 5, Outcome::Pending, 0));
        assert!(lc.try_start_reload(true).is_some());
        assert_eq!(lc.wire(), (Generation::for_test(0), 5, Outcome::Pending, 0));
        lc.finish_managed(Generation::for_test(1), Outcome::Applied, 1202);
        assert_eq!(
            lc.wire(),
            (Generation::for_test(1), 5, Outcome::Applied, 1202)
        );
    }

    #[test]
    fn refuse_reset_command_while_pending() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(!lc.accept(2, params(1202, 1, 0)));
        assert!(lc.try_start_reload(true).is_some());
        assert!(!lc.accept(3, params(1202, 1, 0)));
        lc.finish_managed(Generation::for_test(1), Outcome::Applied, 1202);
        assert!(lc.accept(4, params(1202, 1, 0)));
    }

    #[test]
    fn try_start_reload_environment_gate() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(lc.try_start_reload(false).is_none());
        lc.load_in_flight = true;
        assert!(lc.try_start_reload(true).is_none());
        lc.load_in_flight = false;
        let plan = lc.try_start_reload(true).expect("reload starts");
        assert_eq!(
            plan,
            ReloadPlan {
                stage_select: Some(1),
                difficulty: Some(3),
                character: Some(0),
                managed: true,
            }
        );
        assert!(lc.reset_reloading());
    }

    #[test]
    fn vanilla_reload_is_unmanaged() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(0, 0, 0)));
        let plan = lc.try_start_reload(true).expect("reload starts");
        assert_eq!(
            plan,
            ReloadPlan {
                stage_select: None,
                difficulty: None,
                character: None,
                managed: false,
            }
        );
    }

    #[test]
    fn managed_consume_report_reloading_seq() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(7, params(1202, 1, 0)));
        assert!(lc.try_start_reload(true).is_some());
        lc.last_warp = Some(LiveWarp {
            stage: 1,
            undo: EclUndo::default(),
        });
        lc.finish_managed(Generation::for_test(3), Outcome::Applied, 1202);
        assert_eq!(
            lc.wire(),
            (Generation::for_test(3), 7, Outcome::Applied, 1202)
        );
        assert!(!lc.load_in_flight);
        assert!(!lc.reset_reloading());
        assert!(lc.last_warp.is_some());
    }

    #[test]
    fn unmanaged_consume_drop_undo() {
        let mut lc = Lifecycle::INIT;
        lc.last_warp = Some(LiveWarp {
            stage: 1,
            undo: EclUndo::default(),
        });
        lc.observe(Generation::for_test(1), false);
        assert!(lc.last_warp.is_none());
        assert_eq!(lc.wire(), (Generation::for_test(1), 0, Outcome::Idle, 0));
    }

    #[test]
    fn unmanaged_bump_keep_undo_while_reloading() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(lc.try_start_reload(true).is_some());
        lc.last_warp = Some(LiveWarp {
            stage: 1,
            undo: EclUndo::default(),
        });
        lc.observe(Generation::for_test(1), false);
        assert!(lc.last_warp.is_some());
        assert_eq!(lc.wire(), (Generation::for_test(1), 1, Outcome::Pending, 0));
    }
}
