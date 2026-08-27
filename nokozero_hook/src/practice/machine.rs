//! Reset/load state machine logic.

use super::data::{rank_matches_stage, section_stage};
use super::load::{Generation, LoadStatus};
use super::{LiveWarp, PracticeParams};

/// Wire codes for the observation's `reset_outcome` word describing the reset from `reset_seq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    Idle = 0,
    Pending = 1,
    Applied = 2,
    Vanilla = 3,
    /// The reload landed on a stage other than the requested section's.
    FailedStageMismatch = 4,
    /// The stage's ECL files were not resolvable when the warp was attempted.
    FailedNoEcl = 5,
    /// A reset asked to restart the live stage at an invalid difficulty.
    /// (For example, Extra stage segments only exist on the Extra difficulty.)
    FailedDifficultyMismatch = 6,
    /// A reset specified a character other than the one currently being used.
    FailedCharacterMismatch = 7,
}

#[derive(Clone, Copy)]
enum ResetState {
    /// No reset has ever been accepted.
    Idle,
    /// Reset accepted; waiting for a stable in-stage frame without an unpublished load.
    Requested { seq: u32, params: PracticeParams },
    /// The reload has been triggered (teardown queued, tick guard holding); waiting for it to publish.
    Reloading { seq: u32, params: PracticeParams },
    /// This is reported until replaced by the next command.
    Done { seq: u32, outcome: Outcome },
}

struct CurrentLoad {
    generation: Generation,
    section: u32,
}

/// What the tick guard should do with a tick.
pub(super) enum TickDecision {
    /// There is no reload in progress. The tick is untouched.
    Run,
    /// A reload has not published yet, or the loader thread is managing the ECL buffers. The tick is held.
    Hold,
    /// A reload has been published. The tick is consumed, the previous warp is reverted, and the new warp is applied iff `params.active`.
    Consume {
        params: PracticeParams,
        generation: Generation,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReloadPlan {
    /// `STAGE_SELECT` value when the reset involves a warp with a decodable stage.
    pub(super) stage_select: Option<u32>,
    /// `DIFFICULTY` value.
    pub(super) difficulty: u32,
    /// Whether the upcoming load should take an arm (i.e. whether its first tick applies a warp).
    pub(super) arms_load: bool,
}

/// Everything that the main thread knows about the reset/load lifecycle.
/// Transitions are methods that take the environment as parameters and return writes to be performed by the caller.
pub(super) struct Lifecycle {
    reset: ResetState,
    /// The last consumed load.
    current_load: CurrentLoad,
    /// The live warp's undo record.
    last_warp: Option<LiveWarp>,
}

impl Lifecycle {
    /// The initial state. No reset has been accepted, the generation is 0 and consumed, and there are no ECL patches to undo.
    pub(super) const INIT: Self = Self {
        reset: ResetState::Idle,
        current_load: CurrentLoad {
            generation: Generation::PRE_LOAD,
            section: 0,
        },
        last_warp: None,
    };

    /// Accepts a decoded RESET command. Returns `false` if a previous reset is still pending (which means a protocol violation).
    pub(super) fn accept(&mut self, seq: u32, params: PracticeParams) -> bool {
        if matches!(
            self.reset,
            ResetState::Requested { .. } | ResetState::Reloading { .. }
        ) {
            return false;
        }
        self.reset = ResetState::Requested { seq, params };
        true
    }

    /// Observes one supervisor frame, consuming a settled publish unless a reset reload is in progress.
    pub(super) fn observe(&mut self, status: LoadStatus) {
        let LoadStatus::Settled { generation, .. } = status else {
            return;
        };
        if generation != self.current_load.generation
            && !matches!(self.reset, ResetState::Reloading { .. })
        {
            self.consume_load(generation, 0);
        }
    }

    /// Transitions from [`ResetState::Requested`] to [`ResetState::Reloading`] if `stable_ingame` is `true`
    /// (i.e. no other scene switch is queued). Returns the writes to perform, or `None` to keep the state at [`ResetState::Requested`].
    pub(super) fn try_start_reload(
        &mut self,
        stable_ingame: bool,
        status: LoadStatus,
        loader_running: bool,
        current_stage: u32,
        current_character: u32,
    ) -> Option<ReloadPlan> {
        let ResetState::Requested { seq, params } = self.reset else {
            return None;
        };

        let load_window_open = loader_running
            || match status {
                LoadStatus::InFlight => true,
                LoadStatus::Settled { generation, .. } => {
                    generation != self.current_load.generation
                }
            };
        if !stable_ingame || load_window_open {
            return None;
        }

        if params.character != current_character {
            self.reset = ResetState::Done {
                seq,
                outcome: Outcome::FailedCharacterMismatch,
            };
            return None;
        }

        if !params.active && !rank_matches_stage(current_stage, params.difficulty) {
            self.reset = ResetState::Done {
                seq,
                outcome: Outcome::FailedDifficultyMismatch,
            };
            return None;
        }

        self.reset = ResetState::Reloading { seq, params };
        Some(ReloadPlan {
            stage_select: params
                .active
                .then(|| section_stage(params.section))
                .flatten(),
            difficulty: params.difficulty,
            arms_load: params.active,
        })
    }

    /// Decides the tick guard's action.
    pub(super) fn tick_decision(&self, status: LoadStatus) -> TickDecision {
        let ResetState::Reloading { params, .. } = self.reset else {
            return TickDecision::Run;
        };
        match status {
            // Between loader entry and publish, the loader thread manages the ECL buffers.
            LoadStatus::InFlight => TickDecision::Hold,
            LoadStatus::Settled { generation, armed } => {
                if generation == self.current_load.generation {
                    // The reload has not published yet.
                    TickDecision::Hold
                } else {
                    assert!(
                        armed == params.active,
                        "load handoff and reset request diverge"
                    );
                    TickDecision::Consume { params, generation }
                }
            }
        }
    }

    /// Transitions from [`ResetState::Reloading`] to [`ResetState::Done`], consuming the new stage load parameters.
    pub(super) fn finish_reload(&mut self, generation: Generation, outcome: Outcome, section: u32) {
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
    }

    /// Returns the live warp's undo slot for reverting and refilling ECL in place.
    pub(super) fn last_warp_mut(&mut self) -> &mut Option<LiveWarp> {
        &mut self.last_warp
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
    use super::super::data::{EXTRA_DIFFICULTY, EXTRA_STAGE};
    use super::super::ecl::EclUndo;
    use super::super::test_support::{params, params_at};
    use super::{Generation, Lifecycle, LiveWarp, LoadStatus, Outcome, ReloadPlan, TickDecision};

    const LIVE_STAGE: u32 = 1;
    const LIVE_CHARACTER: u32 = 0;

    fn settled(generation: u32) -> LoadStatus {
        LoadStatus::Settled {
            generation: Generation::for_test(generation),
            armed: false,
        }
    }

    fn settled_armed(generation: u32) -> LoadStatus {
        LoadStatus::Settled {
            generation: Generation::for_test(generation),
            armed: true,
        }
    }

    #[test]
    fn wire_reset_phases() {
        let mut lc = Lifecycle::INIT;
        assert_eq!(lc.wire(), (Generation::for_test(0), 0, Outcome::Idle, 0));
        assert!(lc.accept(5, params(1202, 1, 0)));
        assert_eq!(lc.wire(), (Generation::for_test(0), 5, Outcome::Pending, 0));
        assert!(
            lc.try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_some()
        );
        assert_eq!(lc.wire(), (Generation::for_test(0), 5, Outcome::Pending, 0));
        lc.finish_reload(Generation::for_test(1), Outcome::Applied, 1202);
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
        assert!(
            lc.try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_some()
        );
        assert!(!lc.accept(3, params(1202, 1, 0)));
        lc.finish_reload(Generation::for_test(1), Outcome::Applied, 1202);
        assert!(lc.accept(4, params(1202, 1, 0)));
    }

    #[test]
    fn try_start_reload_environment_gate() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(
            lc.try_start_reload(false, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_none()
        );
        assert!(
            lc.try_start_reload(
                true,
                LoadStatus::InFlight,
                false,
                LIVE_STAGE,
                LIVE_CHARACTER
            )
            .is_none()
        );
        assert!(
            lc.try_start_reload(true, settled(0), true, LIVE_STAGE, LIVE_CHARACTER)
                .is_none()
        );
        assert!(
            lc.try_start_reload(true, settled(1), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_none()
        );
        let plan = lc
            .try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
            .expect("reload starts");
        assert_eq!(
            plan,
            ReloadPlan {
                stage_select: Some(1),
                difficulty: 3,
                arms_load: true,
            }
        );
    }

    #[test]
    fn vanilla_reload_spec() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(0, 0, 0)));
        let plan = lc
            .try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
            .expect("reload starts");
        assert_eq!(
            plan,
            ReloadPlan {
                stage_select: None,
                difficulty: 3,
                arms_load: false,
            }
        );
        let TickDecision::Consume { params, .. } = lc.tick_decision(settled(1)) else {
            panic!("publish is for the reload");
        };
        assert!(!params.active);
    }

    #[test]
    fn observe_consume_unrequested_publishes() {
        let mut lc = Lifecycle::INIT;
        lc.observe(settled(1));
        assert_eq!(lc.wire(), (Generation::for_test(1), 0, Outcome::Idle, 0));
        assert!(matches!(lc.tick_decision(settled(1)), TickDecision::Run));
    }

    #[test]
    fn observe_leave_reload_publish() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(
            lc.try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_some()
        );
        lc.observe(settled_armed(1));
        assert_eq!(lc.wire(), (Generation::for_test(0), 1, Outcome::Pending, 0));
        assert!(matches!(
            lc.tick_decision(settled_armed(1)),
            TickDecision::Consume { .. }
        ));
        lc.finish_reload(Generation::for_test(1), Outcome::Applied, 1202);
        assert_eq!(
            lc.wire(),
            (Generation::for_test(1), 1, Outcome::Applied, 1202)
        );
        assert!(matches!(
            lc.tick_decision(settled_armed(1)),
            TickDecision::Run
        ));
    }

    #[test]
    fn lifecycle_undo_record() {
        let mut lc = Lifecycle::INIT;
        lc.last_warp = Some(LiveWarp {
            stage: 1,
            undo: EclUndo::default(),
        });

        lc.observe(settled(1));
        assert!(lc.last_warp.is_some());

        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(
            lc.try_start_reload(true, settled(1), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_some()
        );
        lc.observe(settled(2));
        assert!(lc.last_warp.is_some());

        lc.finish_reload(Generation::for_test(2), Outcome::Applied, 1202);
        assert!(lc.last_warp.is_some());
    }

    #[test]
    fn tick_decision_hold_until_publish() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(1202, 1, 0)));
        assert!(
            lc.try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_some()
        );
        assert!(matches!(
            lc.tick_decision(LoadStatus::InFlight),
            TickDecision::Hold
        ));
        assert!(matches!(lc.tick_decision(settled(0)), TickDecision::Hold));
        let TickDecision::Consume { params, generation } = lc.tick_decision(settled_armed(1))
        else {
            panic!("the reload's publish is consumed");
        };
        assert!(params.active);
        assert_eq!(generation, Generation::for_test(1));
    }

    #[test]
    fn inactive_reset_refuse_incompatible_difficulty() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params(0, 0, 0)));
        assert!(
            lc.try_start_reload(true, settled(0), false, EXTRA_STAGE, LIVE_CHARACTER)
                .is_none()
        );
        assert_eq!(
            lc.wire(),
            (
                Generation::for_test(0),
                1,
                Outcome::FailedDifficultyMismatch,
                0
            )
        );
        assert!(matches!(lc.tick_decision(settled(0)), TickDecision::Run));
        assert!(lc.accept(2, params_at(0, 0, 0, EXTRA_DIFFICULTY.cast_signed())));
        assert!(
            lc.try_start_reload(true, settled(0), false, EXTRA_STAGE, LIVE_CHARACTER)
                .is_some()
        );

        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(1, params_at(0, 0, 0, EXTRA_DIFFICULTY.cast_signed())));
        assert!(
            lc.try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER)
                .is_none()
        );
        assert_eq!(lc.wire().2, Outcome::FailedDifficultyMismatch);
    }

    #[test]
    fn refuse_reload_for_invalid_character() {
        let mut lc = Lifecycle::INIT;
        assert!(lc.accept(9, params(1202, 1, 0)));
        assert!(
            lc.try_start_reload(true, settled(0), false, LIVE_STAGE, LIVE_CHARACTER + 1)
                .is_none()
        );

        let mut early = Lifecycle::INIT;
        assert!(early.accept(9, params(1202, 1, 0)));
        assert!(
            early
                .try_start_reload(false, settled(0), false, LIVE_STAGE, LIVE_CHARACTER + 1)
                .is_none()
        );
        assert_eq!(
            early.wire(),
            (Generation::for_test(0), 9, Outcome::Pending, 0)
        );
        assert_eq!(
            lc.wire(),
            (
                Generation::for_test(0),
                9,
                Outcome::FailedCharacterMismatch,
                0
            )
        );
    }
}
