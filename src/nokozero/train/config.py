"""Training run configuration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nokozero import wire
from nokozero.train.curriculum import CurriculumConfig
from nokozero.train.rollout import (
    SEGMENT_MAX_STEPS,
    TRAINING_POWER,
    Spec,
    Target,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Everything a training run is parameterized by."""

    target: Target | None = None
    sections: tuple[Spec, ...] = ()
    replay_weight: int = 1
    prefix_weight: int = 1
    warp_weight: int = 1
    out: Path
    seed: int = 0
    k: int = 96
    features: str = "v1"
    model_dim: int = 256
    model_heads: int = 8
    model_depth: int = 4
    #: Game frames per decision. Gamma and `n_step` are per decision, so they should be rescaled with this
    # to maintain the same time horizon (see `TrainConfig.horizon_for`).
    step_interval: int = wire.DEFAULT_STEP_INTERVAL
    actions: str = "hold"
    total_steps: int = 5_000_000
    #: Wall-clock budget in seconds. `None` runs to `total_steps`.
    max_seconds: float | None = None
    warmup_steps: int = 20_000
    max_episode_steps: int = SEGMENT_MAX_STEPS
    capacity: int = 400_000
    archive_capacity: int = 0
    n_step: int = 5
    #: Survival discount per step.
    gamma: float = 0.995
    batch_size: int = 256
    #: Stored transitions between gradient steps.
    update_every: int = 64
    #: Decisions collected before updates begin.
    learning_starts: int = 10_000
    learning_rate: float = 3e-4
    decay_updates: int | None = None
    tau: float = 0.005
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000
    log_every: float = 30.0
    checkpoint_every: float = 600.0
    #: Replay store to draw curriculum starts from. `None` trains on seed episodes alone.
    store: Path | None = None
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    buffer_seed_fraction: float = 0.5
    #: Cloned policy to explore with instead of using random actions.
    explore_with: Path | None = None
    #: Checkpoint weights. Continues training of a saved policy with a fresh optimizer and buffer.
    init_from: Path | None = None
    #: A checkpoint whose greedy value of a completed segment's terminal state stands in for the terminal value 1.
    handoff: Path | None = None
    #: End every episode as "complete" once the chapter's time counter reaches this many frames.
    #: `None` plays to the boundary.
    until_tic: int | None = None
    #: The next segment's expert, which plays on from every completed episode's end state until its own segment ends.
    #: Its outcome (1 or 0) becomes the episode's terminal value. Exclusive with `handoff`.
    suffix: Path | None = None
    #: A harvest whose segments enter the replay buffer as episodes before the first update.
    demos: Path | None = None
    #: Router spec whose policy plays a prefix from the target's stage opening until it reaches the target.
    #: Episodes then train the segment from full-run states.
    prefix_router: Path | None = None
    seed_pool: int | None = None
    #: Add this much of the boss HP ratio an n-step window takes to its target, capped at 1.
    #: 0 keeps the plain survival target.
    damage_bonus: float = 0.0
    #: Draw each warp's landing position from `rollout.JITTER_X` x `JITTER_Y` instead of
    #: the warp's fixed start position.
    start_jitter: bool = False
    start_power: int = TRAINING_POWER
    #: When this is set, starting power is uniformly drawn from `start_power` to this.
    start_power_max: int | None = None

    def __post_init__(self) -> None:
        """Reject an invalid run before the fleet boots."""
        rows = self.max_episode_steps + 1
        for name, capacity in (
            ("capacity", self.capacity),
            ("archive_capacity", self.archive_capacity),
        ):
            if capacity and capacity < rows:
                msg = f"{name} {capacity} cannot hold an episode of max_episode_steps ({rows} rows)"
                raise ValueError(msg)
        self._check_sources()
        if self.until_tic is not None and self.until_tic < 1:
            msg = f"until_tic {self.until_tic} must be at least 1"
            raise ValueError(msg)
        if self.epsilon_decay_steps < 1:
            msg = f"epsilon_decay_steps {self.epsilon_decay_steps} must be at least 1"
            raise ValueError(msg)
        if self.damage_bonus < 0:
            msg = f"damage_bonus {self.damage_bonus} cannot be negative"
            raise ValueError(msg)
        if not 0 <= self.start_power <= wire.MAX_POWER:
            msg = f"start_power {self.start_power} outside the game's 0-{wire.MAX_POWER}"
            raise ValueError(msg)
        if self.start_power_max is not None and not (self.start_power <= self.start_power_max <= wire.MAX_POWER):
            msg = (
                f"start_power_max {self.start_power_max} must be between start_power "
                f"{self.start_power} and {wire.MAX_POWER}"
            )
            raise ValueError(msg)
        if self.handoff is not None and self.suffix is not None:
            msg = "handoff (estimate) is mutually exclusive with suffix (measurement)"
            raise ValueError(msg)

    def _check_sources(self) -> None:
        """Reject a run without a valid start source."""
        if not self.sections and self.store is None and self.prefix_router is None:
            msg = "a run needs a start source (warp sections, a store, or a prefix router)"
            raise ValueError(msg)
        for source, given in (("a store", self.store), ("a prefix router", self.prefix_router)):
            if given is not None and self.target is None:
                msg = f"{source} draws its starts in the target segment; specify --target"
                raise ValueError(msg)
        if self.prefix_router is not None and not self.seed_pool:
            msg = "a prefix router reuses each seed's recorded prefix; specify a seed_pool"
            raise ValueError(msg)
        if self.start_jitter and not self.sections:
            msg = "start_jitter moves warps' landings; the run has no warp sections"
            raise ValueError(msg)
        for name, weight in (
            ("replay_weight", self.replay_weight),
            ("prefix_weight", self.prefix_weight),
            ("warp_weight", self.warp_weight),
        ):
            if weight < 1:
                msg = f"{name} {weight} must be at least 1"
                raise ValueError(msg)

    @staticmethod
    def horizon_for(step_interval: int) -> tuple[float, int]:
        """Return `(gamma, n_step)` at `step_interval` keeping the defaults' horizon."""
        base = wire.DEFAULT_STEP_INTERVAL
        gamma = TrainConfig.gamma ** (step_interval / base)
        n_step = max(1, round(TrainConfig.n_step * base / step_interval))
        return gamma, n_step

    def epsilon(self, step: int) -> float:
        """Return the exploration rate at `step`."""
        frac = min(1.0, max(0.0, step - self.warmup_steps) / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
