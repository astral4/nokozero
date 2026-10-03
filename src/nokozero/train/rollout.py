"""Per-instance segment episodes over a fleet."""

import hashlib
import struct
import time
from collections import deque
from collections.abc import Callable, Container, Hashable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from itertools import repeat
from typing import TYPE_CHECKING, Any, NamedTuple, Self, override

import numpy as np
import numpy.typing as npt

from nokozero import catalog, segments, wire
from nokozero.train.features import HOLD_ACTIONS, NUM_ACTIONS, ActionSet, Features, Featurizer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nokozero.env import Fleet, Fleets

M = wire.Meta

MAX_LANDING_RETRIES = 5
WINDOW_DECIDE_SHARE = 0.8
DECIDE_SMOOTHING = 0.1
WAIT_TAPE_SLACK = 4
STAGE_END_CHAPTER = 81
SEGMENT_MAX_STEPS = 2400
FULL_STAGE_MAX_STEPS = 12000


class Outcome(Enum):
    """The result of an episode."""

    HIT = "hit"
    COMPLETE = "complete"
    TRUNCATED = "truncated"


class Kind(Enum):
    """The source of an episode's origin state."""

    SEED = "seed"
    REPLAY = "replay"


class Target(NamedTuple):
    """A segment reached from the beginning of the stage it's in."""

    stage: int
    word: int
    occurrence: int = 1

    @override
    def __str__(self) -> str:
        return f"{self.opening}>{WordKey.entry(self.word, self.occurrence)}"

    @property
    def opening(self) -> int:
        """The section that the segment is counted from."""
        return catalog.opening(self.stage)

    def reads(self, obs: wire.Observation) -> bool:
        """Whether `obs` is in this segment."""
        return (obs.meta(M.CHAPTER), obs.meta(M.ENTRY_COUNT)) == (self.word, self.occurrence)


class WordKey(NamedTuple):
    """Specification for a chapter word (`word`, `word#n` or `word@tic`)."""

    word: int
    occurrence: int | None = None
    tic: int | None = None

    @classmethod
    def entry(cls, word: int, occurrence: int) -> WordKey:
        """Return the key for `word`'s `occurrence`-th entry."""
        return cls(word, occurrence if occurrence > 1 else None)

    @override
    def __str__(self) -> str:
        """Return the key formatted for display."""
        entry = "" if self.occurrence is None else f"#{self.occurrence}"
        split = "" if self.tic is None else f"@{self.tic}"
        return f"{self.word}{entry}{split}"


def parse_word_key(text: str) -> WordKey:
    """Parse a string into a `WordKey`."""
    head, at, tic = text.partition("@")
    word, hash_, nth = head.partition("#")
    parts = [word, *([nth] if hash_ else []), *([tic] if at else [])]
    if not all(part.isdigit() for part in parts) or (hash_ and at):
        msg = f"word key {text!r}: expected word, word#n or word@tic"
        raise ValueError(msg)
    key = WordKey(int(word), int(nth) if hash_ else None, int(tic) if at else None)
    if key.occurrence == 0:
        msg = f"word key {text!r}: occurrence starts from 1"
        raise ValueError(msg)
    return key


def parse_target(spec: str) -> Target:
    """Parse a segment spec string into a `Target`."""
    section, arrow, rest = spec.partition(">")
    if not arrow or not section.isdigit():
        msg = f"segment spec {spec!r}: expected opening>word or opening>word#n"
        raise ValueError(msg)
    stage = wire.stage_of(int(section))
    if int(section) != catalog.opening(stage):
        msg = f"segment spec {spec!r}: a segment starts from its stage's opening ({catalog.opening(stage)})"
        raise ValueError(msg)
    try:
        key = parse_word_key(rest)
    except ValueError as error:
        msg = f"segment spec {spec!r}: {error}"
        raise ValueError(msg) from error
    if key.tic is not None:
        msg = f"segment spec {spec!r}: a target specifies a word's entry, not a tic within it"
        raise ValueError(msg)
    return Target(stage, key.word, key.occurrence or 1)


class StartKey(NamedTuple):
    """Start information."""

    seed: int
    power: int


class Recording(NamedTuple):
    """A run to replay as a tape."""

    tape: npt.NDArray[np.uint16]
    landing: npt.NDArray[np.uint32]

    def reached(self, obs: wire.Observation) -> bool:
        """Whether `obs`, the observation after the tape, matches the expected/recorded state."""
        return bool(np.array_equal(obs.sync_state(), self.landing))


@dataclass
class Prefix:
    """A prefix played by a router."""

    policy: Policy
    #: The segment to reach.
    target: Target
    cache: dict[StartKey, Recording] = field(default_factory=dict)
    failed: set[StartKey] = field(default_factory=set)
    uncached: set[StartKey] = field(default_factory=set)
    max_steps: int = FULL_STAGE_MAX_STEPS


@dataclass(frozen=True)
class PrefixPlan:
    """Prefix policy information."""

    prefix: Prefix
    key: StartKey


@dataclass(frozen=True)
class Start:
    """The beginning of an episode."""

    params: wire.ResetParams
    recording: Recording | None = field(default=None, repr=False)
    await_chapter: bool = True
    kind: Kind = Kind.SEED
    target: Target | None = None
    label: str = ""
    plan: PrefixPlan | None = None
    origin: Hashable | None = None
    failed: set[Any] | None = field(default=None, compare=False, repr=False)

    def fail(self) -> None:
        """Mark the starting point as failed."""
        if self.failed is not None and self.origin is not None:
            self.failed.add(self.origin)

    def __post_init__(self) -> None:
        if self.recording is not None and len(self.recording.tape) % self.params.step_interval:
            msg = (
                f"tape of {len(self.recording.tape)} frames is not a multiple of the step interval "
                f"{self.params.step_interval}"
            )
            raise ValueError(msg)

    @property
    def key(self) -> Spec | Target:
        """The segment associated with the start's episodes."""
        return self.target if self.target is not None else Spec(self.params.section, self.await_chapter)

    @property
    def seed(self) -> int:
        """The episode's RNG seed."""
        return self.params.rng_seed

    @cached_property
    def fingerprint(self) -> tuple[int, int]:
        """A digest of what the start plays before its first "actual" decision."""
        digest = hashlib.blake2b(digest_size=8)
        digest.update(wire.reset_frame(0, self.params))
        target = self.plan.prefix.target if self.plan is not None else Target(0, 0)
        word, occurrence = target.word, target.occurrence
        digest.update(struct.pack("<?II", self.await_chapter, word, occurrence))
        if self.recording is not None:
            digest.update(self.recording.tape.astype("<u2", copy=False).tobytes())
        high, low = struct.unpack("<II", digest.digest())
        return high, low


TRAINING_POWER = wire.MAX_POWER
GAME_START_POWER = 100
type Powers = int | Iterator[int]


def reset_params(
    section: int, character: int, seed: int, step_interval: int, *, power: int = TRAINING_POWER
) -> wire.ResetParams:
    """Return the reset for a seed episode (i.e. a warp to `section` with specific resource amounts)."""
    return wire.ResetParams(
        active=True,
        section=section,
        difficulty=catalog.difficulty_for(section),
        character=character,
        rng_seed=seed,
        power=power,
        step_interval=step_interval,
    )


class Spec(NamedTuple):
    """A warp section spec."""

    section: int
    await_chapter: bool = True

    @override
    def __str__(self) -> str:
        return str(self.section) if self.await_chapter else f"{self.section}@0"


def parse_spec(spec: str | int) -> Spec:
    """Parse a section spec string into a `Spec`."""
    text = str(spec)
    section, _, restarts = text.partition("@")
    if restarts not in ("", "0", "1"):
        msg = f"section spec {text!r}: only @0 (the landing) and @1 (the next boundary) are valid"
        raise ValueError(msg)
    return Spec(int(section), restarts != "0")


def seed_start(  # noqa: PLR0913
    section: int,
    character: int,
    seed: int,
    step_interval: int = wire.DEFAULT_STEP_INTERVAL,
    *,
    await_chapter: bool = True,
    power: int = TRAINING_POWER,
) -> Start:
    """Build the start of an episode for a seed."""
    params = reset_params(section, character, seed, step_interval, power=power)
    label = f"seed:{Spec(section, await_chapter)}:{seed}"
    return Start(params, await_chapter=await_chapter, label=label)


def seed_starts(  # noqa: PLR0913
    section: int,
    character: int,
    seeds: Iterator[int],
    step_interval: int = wire.DEFAULT_STEP_INTERVAL,
    *,
    await_chapter: bool = True,
    power: Powers = TRAINING_POWER,
) -> Iterator[Start]:
    """Yield starts of episodes for a seed, for each seed drawn."""
    spec = Spec(section, await_chapter)
    return multi_seed_starts((spec,), character, start_keys(seeds, power), step_interval)


def start_keys(seeds: Iterator[int], power: Powers = TRAINING_POWER) -> Iterator[StartKey]:
    """Pair each seed drawn with the next of `power`."""
    powers = repeat(power) if isinstance(power, int) else power
    for seed in seeds:
        yield StartKey(seed, next(powers))


def pool_keys(
    rng: np.random.Generator,
    size: int,
    power: Powers = TRAINING_POWER,
    failed: Container[StartKey] = (),
) -> Iterator[StartKey]:
    """Yield keys drawn uniformly from a pool of `size` fixed on the first draw."""
    seeds = rng.choice(wire.MAX_RNG_SEED + 1, size=size, replace=False)
    pool = list(start_keys((int(seed) for seed in seeds), power))
    while pool:
        index = int(rng.integers(len(pool)))
        if pool[index] in failed:
            pool[index] = pool[-1]
            pool.pop()
            continue
        yield pool[index]
    msg = f"every key of the pool of {size} has failed its prefix"
    raise RuntimeError(msg)


def prefix_starts(
    prefix: Prefix,
    character: int,
    keys: Iterator[StartKey],
    step_interval: int = wire.DEFAULT_STEP_INTERVAL,
) -> Iterator[Start]:
    """Yield starts that reach `prefix.target` from its stage's opening, one per key drawn."""
    if prefix.max_steps * step_interval > wire.MAX_TAPE_FRAMES:
        msg = (
            f"a prefix of up to {prefix.max_steps} decisions at {step_interval} frames each "
            f"outgrows a tape of {wire.MAX_TAPE_FRAMES} frames"
        )
        raise ValueError(msg)
    return _prefix_starts(prefix, character, keys, step_interval)


def _prefix_starts(prefix: Prefix, character: int, keys: Iterator[StartKey], step_interval: int) -> Iterator[Start]:
    target = prefix.target
    for key in keys:
        params = reset_params(target.opening, character, key.seed, step_interval, power=key.power)
        yield Start(
            params,
            await_chapter=False,
            target=target,
            label=f"seed:{target}:{key.seed}",
            plan=PrefixPlan(prefix, key),
            origin=key,
            failed=prefix.failed,
        )


JITTER_X = (-176.0, 176.0)
JITTER_Y = (352.0, 432.0)


def jitter_starts(starts: Iterator[Start], rng: np.random.Generator) -> Iterator[Start]:
    """Yield warp `starts` with each landing position drawn from the jitter band."""
    for start in starts:
        params = replace(
            start.params,
            player_x=float(rng.uniform(*JITTER_X)),
            player_y=float(rng.uniform(*JITTER_Y)),
        )
        yield replace(start, params=params)


def power_stream(rng: np.random.Generator, low: int, high: int) -> Iterator[int]:
    """Yield power values uniformly drawn over `[low, high]`."""
    while True:
        yield int(rng.integers(low, high + 1))


def multi_seed_starts(
    specs: Sequence[Spec],
    character: int,
    keys: Iterator[StartKey],
    step_interval: int = wire.DEFAULT_STEP_INTERVAL,
) -> Iterator[Start]:
    """Yield seed starts cycling through the section specs, one key drawn per start."""
    for i, (seed, power) in enumerate(keys):
        section, await_chapter = specs[i % len(specs)]
        yield seed_start(section, character, seed, step_interval, await_chapter=await_chapter, power=power)


@dataclass
class Episode:
    """A finished episode."""

    start: Start
    outcome: Outcome
    features: Features | None
    actions: npt.NDArray[np.int32]
    instance: int = 0
    continuation: Outcome | None = None
    end: wire.Observation | None = None
    inputs: npt.NDArray[np.uint32] | None = None
    segments: dict[Target, Outcome | None] | None = None

    @property
    def length(self) -> int:
        """The number of decisions taken."""
        return len(self.actions)

    @property
    def states(self) -> Features:
        """Features of the episode's `T + 1` states."""
        if self.features is None:
            msg = f"{self.label}: the episode runner did not keep features"
            raise ValueError(msg)
        return self.features

    @property
    def seed(self) -> int:
        """The episode's RNG seed."""
        return self.start.seed

    @property
    def label(self) -> str:
        """The start's display label."""
        return self.start.label


def segment_end(*, hit: bool, done: bool, steps: int, cap: int) -> Outcome | None:
    """Classify the ending of a segment based on an observation. Returns `None` if the segment continues."""
    if hit:
        return Outcome.HIT
    if done:
        return Outcome.COMPLETE
    if steps >= cap:
        return Outcome.TRUNCATED
    return None


class _Phase(Enum):
    NEED_RESET = 0
    RESETTING = 1
    TAPING = 2
    WAITING = 3
    RUNNING = 4
    DONE = 5
    PREFIXING = 6
    SUFFIXING = 7


_DECIDING: dict[_Phase, int] = {_Phase.RUNNING: 0, _Phase.PREFIXING: 1, _Phase.SUFFIXING: 2}
_STEPPING: frozenset[_Phase] = frozenset({*_DECIDING, _Phase.WAITING})


@dataclass
class _Slot:
    start: Start | None = None
    last: wire.Observation | None = None
    phase: _Phase = _Phase.NEED_RESET
    seq: int = 0
    hits: int = 0
    time_in_chapter: int = 0
    waited: int = 0
    wait: int = 0
    steps: int = 0
    feats: list[Features] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    inputs: list[npt.NDArray[np.uint32]] = field(default_factory=list)
    landing_retries: int = 0
    ending: bool = False
    last_in_stage: wire.Observation | None = None
    segments: dict[Target, Outcome | None] | None = None
    passing: dict[Target, int] = field(default_factory=dict)
    stage: int = 0
    taping: Recording | None = None
    pending: Episode | None = None

    def boundary(self, time_in_chapter: int) -> bool:
        """Record the counter's new value. Returns whether the step was a segment boundary."""
        crossed = segments.crossed(self.time_in_chapter, time_in_chapter)
        self.time_in_chapter = time_in_chapter
        return crossed


@dataclass(frozen=True)
class Inputs:
    """Inputs that a policy chooses from."""

    features: Features
    featurizer: Featurizer
    prev: npt.NDArray[np.int32]
    observations: tuple[wire.Observation, ...]
    keys: tuple[tuple[int, ...], ...]

    def __len__(self) -> int:
        return len(self.features)


Pending = Callable[[], npt.NDArray[np.integer]]
Policy = Callable[[Inputs], Pending]


@dataclass(frozen=True, kw_only=True)
class RunnerConfig:
    """Episode configuration for a `SegmentRunner`."""

    max_steps: int | None = None
    max_wait: int = 1200
    until_stage_end: bool = False
    until_tic: int | None = None
    suffix: Policy | None = None
    suffix_max_steps: int = SEGMENT_MAX_STEPS
    keep_features: bool = True

    def __post_init__(self) -> None:
        if self.until_stage_end and self.suffix is not None:
            msg = "a suffix plays the next segment, but a full-stage episode doesn't have any"
            raise ValueError(msg)
        if self.until_stage_end and self.until_tic is not None:
            msg = "a cut at a tic ends a segment, but a full-stage episode plays through them"
            raise ValueError(msg)

    @property
    def step_cap(self) -> int:
        """The number of decisions before an episode is truncated."""
        if self.max_steps is not None:
            return self.max_steps
        return FULL_STAGE_MAX_STEPS if self.until_stage_end else SEGMENT_MAX_STEPS


class SegmentRunner:
    """Drives a fleet's instances through independent episodes drawn from `starts`."""

    def __init__(
        self,
        starts: Iterator[Start],
        featurizer: Featurizer,
        config: RunnerConfig | None = None,
        actions: ActionSet = HOLD_ACTIONS,
        *,
        gave_up: Callable[[Start], None] | None = None,
    ) -> None:
        self.starts = starts
        self.featurizer = featurizer
        self.config = config or RunnerConfig()
        self.actions = actions
        self.gave_up = gave_up
        frames, interval = actions.frames, featurizer.step_interval
        if frames not in (1, interval):
            msg = f"action set {actions.name} spans {frames} frames on a {interval}-frame step"
            raise ValueError(msg)
        self.env: Fleet | None = None
        self._slots: list[_Slot] = []
        self._ready: list[int] = []
        self._again: deque[Start] = deque()
        self.landing_mismatches = 0
        self.instance_steps = 0
        self.exchanges = 0
        self.decide_seconds = self.wait_seconds = self.collect_seconds = 0.0
        self._decide_time = 0.0
        self.prefixes_recorded = 0
        self.prefixes_failed = 0
        self.prefix_mismatches = 0
        self.replay_mismatches = 0
        self._wait_frames: dict[Spec | Target, int] = {}
        self.wait_misses = 0

    def attach(self, env: Fleet) -> Self:
        """Step `env` from this point onward. Episodes in flight on the fleet before (if there was one) are lost."""
        self._again.extend(slot.start for slot in self._slots if slot.start is not None)
        self.env = env
        self._slots = [_Slot() for _ in env.instances]
        self._ready = list(range(len(env.instances)))
        for slot in self._slots:
            self._next_episode(slot)
        return self

    @property
    def done(self) -> bool:
        """Whether every slot of the attached fleet has exhausted the start source."""
        return bool(self._slots) and all(slot.phase is _Phase.DONE for slot in self._slots)

    def counts(self) -> dict[str, int]:
        """Runner results."""
        return {
            "landing_mismatches": self.landing_mismatches,
            "wait_misses": self.wait_misses,
            "prefixes_recorded": self.prefixes_recorded,
            "prefixes_failed": self.prefixes_failed,
            "prefix_mismatches": self.prefix_mismatches,
            "replay_mismatches": self.replay_mismatches,
        }

    def timings(self) -> dict[str, float]:
        """Runner timings."""
        return {
            "exchanges": self.exchanges,
            "decide_s": round(self.decide_seconds, 1),
            "wait_s": round(self.wait_seconds, 1),
            "collect_s": round(self.collect_seconds, 1),
        }

    def _fleet(self) -> Fleet:
        env = self.env
        if env is None:
            msg = "the runner doesn't have a fleet"
            raise RuntimeError(msg)
        return env

    def step(self, policy: Policy) -> list[Episode]:
        """Send commands to every ready instance and handle the observations that arrive."""
        observations = self._exchange(policy)
        self._ready = sorted(observations)
        self.instance_steps += len(observations)
        finished: list[Episode] = []
        for i, obs in observations.items():
            slot = self._slots[i]
            slot.last = obs
            if slot.phase is _Phase.RESETTING:
                self._on_resetting(i, slot, obs)
            elif slot.phase is _Phase.TAPING:
                self._on_landed(slot, obs)
            elif slot.phase is _Phase.PREFIXING:
                self._on_prefixing(slot, obs)
            elif slot.phase is _Phase.SUFFIXING:
                episode = self._on_suffixing(slot, obs)
                if episode is not None:
                    finished.append(episode)
            elif slot.phase is _Phase.WAITING:
                self._on_waiting(slot, obs)
            elif slot.phase is _Phase.RUNNING:
                episode = self._on_running(i, slot, obs)
                if episode is not None:
                    finished.append(episode)
        return finished

    def _exchange(self, policy: Policy) -> dict[int, wire.Observation]:
        """Send commands to every ready instance and collect responses."""
        began = time.monotonic()
        decided_any = self._decide(policy)
        decided = time.monotonic()
        if decided_any:
            spent, average = decided - began, self._decide_time
            self._decide_time = average + DECIDE_SMOOTHING * (spent - average) if average else spent
        commands = {i: self._command(self._slots[i]) for i in self._ready if self._slots[i].phase is not _Phase.DONE}
        fleet = self._fleet()
        observations = fleet.exchange(commands)
        waited = time.monotonic()
        window = min(fleet.collect_window, WINDOW_DECIDE_SHARE * self._decide_time)
        while (left := waited + window - time.monotonic()) > 0 and self._owed(observations):
            observations |= fleet.exchange({}, timeout=left)
        collected = time.monotonic()
        self.exchanges += 1
        self.decide_seconds += decided - began
        self.wait_seconds += waited - decided
        self.collect_seconds += collected - waited
        return observations

    def _owed(self, arrived: Mapping[int, wire.Observation]) -> bool:
        """Whether an instance still needs to respond."""
        return any(slot.phase in _STEPPING and i not in arrived for i, slot in enumerate(self._slots))

    def _decide(self, policy: Policy) -> bool:
        """Featurize every ready instance that owes a decision and record each one's action."""
        deciding = [i for i in self._ready if self._decides(self._slots[i])]
        if not deciding:
            return False
        observations = [self._observed(i) for i in deciding]
        feats = self.featurizer.batch(observations)
        prev = np.array([self._previous(self._slots[i]) for i in deciding], dtype=np.int32)
        keys = tuple(self._decision_key(self._slots[i]) for i in deciding)
        groups: dict[int, tuple[Policy, list[int]]] = {}
        for row, i in enumerate(deciding):
            chosen = self._policy_of(self._slots[i], policy)
            groups.setdefault(id(chosen), (chosen, []))[1].append(row)
        pending: list[tuple[list[int], Pending]] = []
        for chosen, rows in groups.values():
            if len(rows) == len(deciding):
                inputs = Inputs(feats, self.featurizer, prev, tuple(observations), keys)
            else:
                index = np.array(rows)
                inputs = Inputs(
                    feats[index],
                    self.featurizer,
                    prev[index],
                    tuple(observations[r] for r in rows),
                    tuple(keys[r] for r in rows),
                )
            pending.append((rows, chosen(inputs)))
        for rows, collect in pending:
            for row, action in zip(rows, collect(), strict=True):
                slot = self._slots[deciding[row]]
                slot.actions.append(int(action))
                if slot.phase is _Phase.RUNNING and self.config.keep_features:
                    slot.feats.append(feats[row])
        return True

    @staticmethod
    def _decides(slot: _Slot) -> bool:
        """Whether the slot's last observation needs a policy decision."""
        return slot.phase in _DECIDING and not (slot.phase is _Phase.RUNNING and slot.ending)

    def _decision_key(self, slot: _Slot) -> tuple[int, ...]:
        """Specify the slot's pending decision."""
        start = self._start_of(slot)
        return (*start.fingerprint, _DECIDING[slot.phase], len(slot.actions))

    @staticmethod
    def _previous(slot: _Slot) -> int:
        """Return the slot's previous action in its phase, or -1 at the first."""
        return slot.actions[-1] if slot.actions else -1

    @staticmethod
    def _enter(slot: _Slot, phase: _Phase) -> None:
        """Enter a "deciding" phase."""
        slot.phase = phase
        slot.actions = []

    def _policy_of(self, slot: _Slot, policy: Policy) -> Policy:
        if slot.phase is _Phase.PREFIXING:
            return self._plan_of(slot).prefix.policy
        if slot.phase is _Phase.SUFFIXING:
            suffix = self.config.suffix
            assert suffix is not None  # noqa: S101
            return suffix
        return policy

    def _command(self, slot: _Slot) -> bytes:
        if slot.phase is _Phase.NEED_RESET:
            start = self._start_of(slot)
            slot.seq = self._fleet().next_reset_seq()
            slot.phase = _Phase.RESETTING
            return wire.reset_frame(slot.seq, start.params)
        if slot.phase is _Phase.TAPING:
            if slot.taping is None:
                return wire.tape_frame(np.zeros(slot.wait, dtype=np.uint16))
            return wire.tape_frame(slot.taping.tape, raw=True)
        if slot.phase is _Phase.RUNNING and slot.ending:
            return wire.act_frame(0 if slot.steps % 2 else int(wire.Action.SHOOT))
        if slot.phase in _DECIDING:
            return self.actions.command(slot.actions[-1])
        return wire.NEUTRAL

    @staticmethod
    def _start_of(slot: _Slot) -> Start:
        start = slot.start
        assert start is not None  # noqa: S101
        return start

    def _plan_of(self, slot: _Slot) -> PrefixPlan:
        plan = self._start_of(slot).plan
        assert plan is not None  # noqa: S101
        return plan

    def _observed(self, index: int) -> wire.Observation:
        obs = self._slots[index].last
        if obs is None:
            msg = f"instance {index} is running without an observation"
            raise RuntimeError(msg)
        return obs

    def _begin(self, slot: _Slot, obs: wire.Observation) -> None:
        self._enter(slot, _Phase.RUNNING)
        slot.hits = obs.meta(M.HITS)
        slot.time_in_chapter = obs.meta(M.TIME_IN_CHAPTER)
        slot.steps = 0
        slot.ending = False
        slot.last_in_stage = obs
        slot.feats = []
        slot.inputs = []
        section = self._start_of(slot).params.section
        slot.stage = wire.stage_of(section)
        opening = catalog.opening(slot.stage) == section
        slot.segments = {} if self.config.until_stage_end and opening else None
        slot.passing = {}
        self._pass(slot, obs, restarted=False, hit=False)

    def _on_resetting(self, index: int, slot: _Slot, obs: wire.Observation) -> None:
        start = self._start_of(slot)
        if not wire.reset_landed(obs, slot.seq, start.label):
            return
        signature = obs.landing_signature
        if signature != wire.CANONICAL_LANDING:
            self.landing_mismatches += 1
            slot.landing_retries += 1
            if slot.landing_retries > MAX_LANDING_RETRIES:
                tick, rng_count = signature
                msg = f"instance {index}: repeated non-canonical landings (tick {tick}, RNG count {rng_count})"
                raise RuntimeError(msg)
            slot.phase = _Phase.NEED_RESET
            return
        slot.landing_retries = 0
        plan = start.plan
        slot.taping = start.recording if plan is None else plan.prefix.cache.get(plan.key)
        slot.wait = 0
        wait = self._wait_frames.get(start.key) if start.await_chapter else None
        if slot.taping is not None:
            slot.phase = _Phase.TAPING
        elif plan is not None:
            self._enter(slot, _Phase.PREFIXING)
            slot.inputs = []
            slot.hits = obs.meta(M.HITS)
        elif wait:
            slot.wait = wait
            slot.phase = _Phase.TAPING
        else:
            self._on_landed(slot, obs)

    def _on_landed(self, slot: _Slot, obs: wire.Observation) -> None:
        """Handle the episode's origin being reached."""
        start = self._start_of(slot)
        recording, slot.taping = slot.taping, None
        if recording is not None and not recording.reached(obs):
            plan = start.plan
            if plan is None:
                self.replay_mismatches += 1
                self._give_up(slot)
                return
            self.prefix_mismatches += 1
            plan.prefix.cache.pop(plan.key, None)
            plan.prefix.uncached.add(plan.key)
            slot.phase = _Phase.NEED_RESET
            return
        if start.await_chapter:
            slot.time_in_chapter = obs.meta(M.TIME_IN_CHAPTER)
            slot.waited = 0
            slot.phase = _Phase.WAITING
        else:
            self._begin(slot, obs)

    def _on_prefixing(self, slot: _Slot, obs: wire.Observation) -> None:
        """Advance a prefix."""
        plan = self._plan_of(slot)
        prefix, key = plan.prefix, plan.key
        slot.inputs.append(obs.inputs.copy())
        failed = obs.meta(M.HITS) > slot.hits or len(slot.actions) >= prefix.max_steps or not obs.in_stage
        if failed:
            self.prefixes_failed += 1
            self._give_up(slot)
            return
        if prefix.target.reads(obs):
            if key not in prefix.uncached:
                prefix.cache[key] = Recording(recorded_keys(slot.inputs), obs.sync_state())
            self.prefixes_recorded += 1
            self._begin(slot, obs)

    def _on_waiting(self, slot: _Slot, obs: wire.Observation) -> None:
        restarted = slot.boundary(obs.meta(M.TIME_IN_CHAPTER))
        slot.waited += 1
        start = self._start_of(slot)
        if restarted:
            if start.recording is None and start.key not in self._wait_frames:
                interval = start.params.step_interval
                wait = (obs.meta(M.GAME_TICK) - 2 * interval) // interval * interval
                if wait >= interval:
                    self._wait_frames[start.key] = wait
            self._begin(slot, obs)
        elif slot.wait and slot.waited > WAIT_TAPE_SLACK:
            self._wait_frames.pop(start.key, None)
            self.wait_misses += 1
            slot.phase = _Phase.NEED_RESET
        elif slot.waited >= self.config.max_wait:
            msg = f"{start.label}: no chapter start within {self.config.max_wait} steps"
            raise RuntimeError(msg)

    def _on_running(self, index: int, slot: _Slot, obs: wire.Observation) -> Episode | None:
        slot.steps += 1
        slot.inputs.append(obs.inputs.copy())
        restarted = slot.boundary(obs.meta(M.TIME_IN_CHAPTER))
        hit = obs.meta(M.HITS) > slot.hits
        until_end = self.config.until_stage_end
        self._pass(slot, obs, restarted=restarted, hit=hit)
        slot.ending = until_end and Target(slot.stage, STAGE_END_CHAPTER).reads(obs)
        if obs.in_stage:
            slot.last_in_stage = obs
        until_tic = self.config.until_tic
        at_cut = until_tic is not None and obs.meta(M.TIME_IN_CHAPTER) >= until_tic
        done = not obs.in_stage or (not until_end and (restarted or at_cut))
        outcome = segment_end(hit=hit, done=done, steps=slot.steps, cap=self.config.step_cap)
        if outcome is None:
            return None
        feats, actions, inputs, segments = slot.feats, slot.actions, slot.inputs, slot.segments
        slot.feats, slot.actions, slot.inputs, slot.segments = [], [], [], None
        if segments is not None:
            segments.update(dict.fromkeys(slot.passing, None))
        episode = Episode(
            start=self._start_of(slot),
            outcome=outcome,
            features=Features.concatenate([*feats, self.featurizer(obs)]) if self.config.keep_features else None,
            actions=np.array(actions, dtype=np.int32),
            instance=index,
            end=slot.last_in_stage,
            inputs=np.concatenate(inputs),
            segments=segments,
        )
        if outcome is Outcome.COMPLETE and self.config.suffix is not None:
            slot.pending = episode
            self._enter(slot, _Phase.SUFFIXING)
            return None
        self._next_episode(slot)
        return episode

    def _on_suffixing(self, slot: _Slot, obs: wire.Observation) -> Episode | None:
        """Classify the suffix's observation. Returns the pending episode once the suffix ends."""
        restarted = slot.boundary(obs.meta(M.TIME_IN_CHAPTER))
        continuation = segment_end(
            hit=obs.meta(M.HITS) > slot.hits,
            done=restarted or not obs.in_stage,
            steps=len(slot.actions),
            cap=self.config.suffix_max_steps,
        )
        if continuation is None:
            return None
        pending = slot.pending
        assert pending is not None  # noqa: S101
        slot.pending = None
        self._next_episode(slot)
        return replace(pending, continuation=continuation)

    @staticmethod
    def _pass(slot: _Slot, obs: wire.Observation, *, restarted: bool, hit: bool) -> None:
        """Follow a full-stage run through its segments on `obs` (`Episode.segments`)."""
        segments = slot.segments
        if segments is None:
            return
        for target, entry in list(slot.passing.items()):
            outcome = segment_end(
                hit=hit,
                done=restarted or not obs.in_stage,
                steps=slot.steps - entry,
                cap=SEGMENT_MAX_STEPS,
            )
            if outcome is not None:
                segments[target] = outcome
                del slot.passing[target]
        occurrence = obs.meta(M.ENTRY_COUNT)
        if hit or not obs.in_stage or occurrence == 0:
            return
        target = Target(slot.stage, obs.meta(M.CHAPTER), occurrence)
        if target not in segments and target not in slot.passing:
            slot.passing[target] = slot.steps

    def _give_up(self, slot: _Slot) -> None:
        """Give up the slot's start for its source (`Start.fail`) and draw the next one."""
        start = self._start_of(slot)
        start.fail()
        if self.gave_up is not None:
            self.gave_up(start)
        self._next_episode(slot)

    def _next_episode(self, slot: _Slot) -> None:
        """Draw the slot's next start, or park the slot when the source is exhausted."""
        slot.start = self._again.popleft() if self._again else next(self.starts, None)
        slot.phase = _Phase.DONE if slot.start is None else _Phase.NEED_RESET
        if slot.start is not None:
            interval, run = slot.start.params.step_interval, self.featurizer.step_interval
            if interval != run:
                msg = f"{slot.start.label}: a {interval}-frame step on a {run}-frame run"
                raise ValueError(msg)


def play_out(fleets: Fleets[Fleet], runner: SegmentRunner, policy: Policy, each: Callable[[Episode], None]) -> None:
    """Play every start of `runner` to its end with `policy`, handing `each` every episode."""

    def play(env: Fleet) -> None:
        runner.attach(env)
        while not runner.done:
            for episode in runner.step(policy):
                each(episode)

    fleets.run(play)


def recorded_keys(inputs: Sequence[npt.NDArray[np.uint32]]) -> npt.NDArray[np.uint16]:
    """Return the key words of consecutive observations' input logs, one per frame."""
    rows = np.concatenate(inputs) if inputs else np.zeros((0, 2), dtype=np.uint32)
    frames = rows[:, 0].astype(np.int64)
    if len(frames) and np.any(frames != frames[0] + np.arange(len(frames))):
        msg = f"the input logs skip frames between {frames[0]} and {frames[-1]}"
        raise ValueError(msg)
    return rows[:, 1].astype(np.uint16)


def random_policy(rng: np.random.Generator, persistence: float = 0.75, num_actions: int = NUM_ACTIONS) -> Policy:
    """Return a policy that holds each row's previous action with probability `persistence`."""

    def policy(inputs: Inputs) -> Pending:
        n = len(inputs)
        fresh = rng.integers(0, num_actions, size=n)
        keep = (inputs.prev >= 0) & (rng.random(n) < persistence)
        actions = np.where(keep, inputs.prev, fresh)
        return lambda: actions

    return policy


def interleave[T](weighted: Sequence[tuple[Iterator[T], int]]) -> Iterator[T]:
    """Yield from the given streams, with proportions based on each one's weight over the sum of all weights."""
    if not weighted or any(weight < 1 for _, weight in weighted):
        msg = f"interleave needs streams of weight at least 1, got {[w for _, w in weighted]}"
        raise ValueError(msg)
    total = sum(weight for _, weight in weighted)
    credit = [0] * len(weighted)
    while True:
        credit = [c + weight for c, (_, weight) in zip(credit, weighted, strict=True)]
        turn = credit.index(max(credit))
        credit[turn] -= total
        yield next(weighted[turn][0])


def seed_stream(rng: np.random.Generator) -> Iterator[int]:
    """Yield uniform 16-bit seeds drawn from `rng`."""
    while True:
        yield int(rng.integers(0, wire.MAX_RNG_SEED + 1))
