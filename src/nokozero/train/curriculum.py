"""Backward curriculum over replay states."""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nokozero import wire
from nokozero.replay import ingest, rpy
from nokozero.train.rollout import Episode, Kind, Outcome, Recording, Start, Target

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

    import numpy.typing as npt


@dataclass(frozen=True)
class Candidate:
    """A stage play segment that episodes can start in."""

    key: str
    stage: rpy.Stage
    #: The segment's stage frames `[start, end)` in the stage play.
    start: int
    end: int
    #: Frames before this one are usable starting points.
    limit: int
    trace: npt.NDArray[np.uint32] = field(compare=False, repr=False)

    @property
    def length(self) -> int:
        """The segment's length in frames."""
        return self.end - self.start

    def first_decision(self, step_interval: int) -> int:
        """Return the segment's first frame aligning with the step interval."""
        return -(-self.start // step_interval) * step_interval

    def landing(self, frame: int) -> npt.NDArray[np.uint32]:
        """Return the state that a tape of `frame` frames ends on (row `frame - 1`)."""
        return self.trace[frame - 1]

    def synced(self, frame: int, obs: wire.Observation) -> bool:
        """Whether `obs` is the recorded state at `frame` frames into a playback of the keys."""
        if frame == 0:
            return obs.landing_signature == wire.CANONICAL_LANDING
        return self.recording(frame).reached(obs)

    def recording(self, frame: int) -> Recording:
        """Return the tape of the first `frame` frames and its ending state."""
        return Recording(self.stage.actions[:frame], self.landing(frame))

    def frames(self, depth: float, min_tail: int, step_interval: int, end_tic: int | None = None) -> range:
        """Return the available start frames for this segment at `depth`."""
        end = self.end if end_tic is None else min(self.end, self.start + end_tic)
        lo = max(self.start, end - int(min(depth, end - self.start)), step_interval)
        hi = min(self.limit, end - min_tail, wire.MAX_TAPE_FRAMES + 1)
        first = -(-lo // step_interval)
        last = -(-hi // step_interval)
        return range(first * step_interval, last * step_interval, step_interval)


def build_pool(store: ingest.Store, target: Target, character: int) -> list[Candidate]:
    """Collect every usable `target` segment from the store's verified plays by `character`."""
    pool: list[Candidate] = []
    for row, derived in store.plays(target.stage, character):
        segments = derived.segments
        hits = [s.first_hit for s in segments if s.first_hit is not None]
        first_hit = min(hits) if hits else None
        wanted = [s for s in segments if (s.chapter, s.entry_count) == (target.word, target.occurrence)]
        if not wanted:
            continue
        replay = store.replay(str(row["path"]))
        stage_index = int(row["stage_index"])
        played = replay.stages[stage_index]
        words = store.load_trace(str(row["key"]))
        for segment in wanted:
            start, end = segment.start, segment.end
            limit = min(end, first_hit) if first_hit is not None else end
            if limit <= start:
                continue
            pool.append(
                Candidate(
                    key=str(row["key"]),
                    stage=played,
                    start=start,
                    end=end,
                    limit=limit,
                    trace=ingest.sync_columns(words[:limit]),
                )
            )
    return pool


@dataclass(frozen=True, kw_only=True)
class CurriculumConfig:
    """Configuration for the backward schedule."""

    initial_depth: int = 300
    uniform: bool = False
    growth: float = 1.5
    threshold: float = 0.7
    window: int = 100
    min_tail: int = 30


class Curriculum:
    """Draws replay starting points from inside a target segment."""

    def __init__(  # noqa: PLR0913
        self,
        store: ingest.Store,
        target: Target,
        character: int,
        *,
        rng: np.random.Generator,
        config: CurriculumConfig | None = None,
        step_interval: int = wire.DEFAULT_STEP_INTERVAL,
        end_tic: int | None = None,
    ) -> None:
        self.store = store
        self.target = target
        self.character = character
        self.rng = rng
        self.config = config or CurriculumConfig()
        self.step_interval = step_interval
        self.end_tic = end_tic
        self.pool: list[Candidate] = []
        self.depth = math.inf if self.config.uniform else float(self.config.initial_depth)
        self.max_depth = 0
        self.recent: deque[bool] = deque(maxlen=self.config.window)
        self.growths = 0
        self.unreachable = 0
        self.failed: set[Hashable] = set()
        self.plays = self.verified = 0

    def warm(self) -> None:
        """Build the pool."""
        pool = build_pool(self.store, self.target, self.character)
        self.pool = [c for c in pool if self._frames(c, math.inf)]
        stage = self.target.stage
        rows = self.store.rows()
        self.plays = sum(r["stage"] == stage and r["character"] == self.character for r in rows)
        self.verified = sum(1 for _ in self.store.plays(stage, self.character))
        if not self.pool:
            msg = (
                f"no verified play of stage {self.target.stage} by character "
                f"{self.character} in {self.store.root} offers a start in {self.target} "
                f"({len(pool)} segments, none of which have room for a start)"
            )
            raise ValueError(msg)
        self.max_depth = max(c.length for c in self.pool)
        if self.config.uniform:
            self.depth = float(self.max_depth)

    def starts(self) -> Iterator[Start]:
        """Yield replay starts (`warm` first)."""
        while True:
            usable = self._usable()
            while not usable and self.depth < self.max_depth:
                self._grow()
                self.unreachable += 1
                usable = self._usable()
            if not usable:
                msg = f"every start in {self.target} has been given up ({len(self.failed)})"
                raise RuntimeError(msg)
            yield self._replay_start(usable[self.rng.integers(len(usable))])

    def _usable(self) -> list[Candidate]:
        """Return the candidates offering a start at the current depth."""
        return [c for c in self.pool if next(self._open_frames(c), None) is not None]

    def _frames(self, candidate: Candidate, depth: float) -> range:
        """Return the start frames offered by `candidate` at `depth`."""
        return candidate.frames(depth, self.config.min_tail, self.step_interval, self.end_tic)

    def _open_frames(self, candidate: Candidate) -> Iterator[int]:
        """Yield the start frames offered by `candidate` at the current depth."""
        frames = self._frames(candidate, self.depth)
        return (f for f in frames if (candidate.key, f) not in self.failed)

    def _grow(self) -> None:
        """Advance the depth by the growth factor up to the longest segment."""
        self.depth = min(self.depth * self.config.growth, float(self.max_depth))
        self.growths += 1

    def _replay_start(self, candidate: Candidate) -> Start:
        frames = list(self._open_frames(candidate))
        frame = frames[self.rng.integers(len(frames))]
        params = ingest.record_reset(candidate.stage, self.character, self.step_interval, faithful=False)
        return Start(
            params,
            candidate.recording(frame),
            await_chapter=False,
            kind=Kind.REPLAY,
            target=self.target,
            label=f"replay:{self.target}:{candidate.key}@{frame}",
            origin=(candidate.key, frame),
            failed=self.failed,
        )

    def observe(self, episode: Episode) -> None:
        """Feed back a finished replay episode."""
        if episode.start.kind is not Kind.REPLAY or episode.outcome is Outcome.TRUNCATED:
            return
        self.recent.append(episode.outcome is Outcome.COMPLETE)
        if (
            len(self.recent) == self.config.window
            and self.depth < self.max_depth
            and float(np.mean(self.recent)) >= self.config.threshold
        ):
            self._grow()
            self.recent.clear()

    def row(self) -> dict[str, float | int | None]:
        """Return the log row's curriculum fields."""
        return {
            "plays": self.plays,
            "verified": self.verified,
            "pool": len(self.pool),
            "usable": len(self._usable()),
            "unreachable": self.unreachable,
            "failed_starts": len(self.failed),
            "depth": int(self.depth) if math.isfinite(self.depth) else None,
            "max_depth": self.max_depth,
            "growths": self.growths,
            "recent_curriculum_complete": round(float(np.mean(self.recent)), 3) if self.recent else None,
        }
