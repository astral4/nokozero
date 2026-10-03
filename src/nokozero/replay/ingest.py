"""Replay ingestion."""

import io
import json
import shutil
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt

from nokozero import catalog, segments, wire
from nokozero.replay import rpy
from nokozero.utils import atomic_write

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from nokozero.env import Fleet

M = wire.Meta

TRACE_WORDS: Final = (
    M.GAME_TICK,
    M.CHAPTER,
    M.TIME_IN_CHAPTER,
    M.MISS_COUNT,
    M.BOMBS,
    M.HITS,
    M.SCORE_DIV10,
    M.GRAZE,
    M.POWER,
    M.LIVES,
    M.VALUE_X100,
    M.RNG_COUNT,
    M.SPELL_FLAGS,
    M.ENTRY_COUNT,
    M.PLAYER_X,
    M.PLAYER_Y,
)
_TRACE_INDEX: Final = np.array([int(word) for word in TRACE_WORDS], dtype=np.intp)
TRACE_VERSION: Final = 3

LIFE_BONUS_DIV10: Final = 1_000_000
BOMB_BONUS_DIV10: Final = 300_000

_CHECKS: Final = {
    "score_div10": M.SCORE_DIV10,
    "graze": M.GRAZE,
    "misses": M.MISS_COUNT,
    "power": M.POWER,
    "lives": M.LIVES,
    "piv_x100": M.VALUE_X100,
}


@dataclass(frozen=True)
class Job:
    """One stage of a decoded replay."""

    path: Path
    replay: rpy.Replay
    stage_index: int

    @property
    def stage(self) -> rpy.Stage:
        """The stage record to play."""
        return self.replay.stages[self.stage_index]

    @property
    def key(self) -> str:
        """The stage-play's identity in the store."""
        return f"{self.path.stem}:{self.stage.number}"


def record_reset(stage: rpy.Stage, character: int, step_interval: int, *, faithful: bool) -> wire.ResetParams:
    """Return the RESET that starts `stage` from its record.
    
    `faithful` plays the entire episode as the recorded run.
    """
    return wire.ResetParams(
        active=True,
        section=catalog.opening(stage.number),
        difficulty=stage.globals["difficulty"],
        character=character,
        rng_seed=stage.seed,
        step_interval=step_interval,
        real_deaths=faithful,
        record=stage.record,
    )


def jobs_for(
    paths: Iterator[Path] | list[Path],
    *,
    character: int,
    difficulty: int = catalog.LUNATIC,
    skip: Callable[[str], bool] = lambda _key: False,
) -> Iterator[Job]:
    """Yield the jobs of every stage 1-6 of `difficulty` played by `character` in `paths`."""
    for path in paths:
        try:
            info = rpy.read_info(path)
            if info.character != character or info.difficulty != difficulty:
                continue
            replay = rpy.load(path)
        except (rpy.ReplayError, OSError) as error:
            print(f"{path}: skipped ({error})", flush=True)  # noqa: T201
            continue
        for index, stage in enumerate(replay.stages):
            if stage.number not in wire.MAIN_STAGES:
                continue
            if stage.globals["difficulty"] != difficulty:
                continue
            job = Job(path, replay, index)
            if not skip(job.key):
                yield job


@dataclass(frozen=True)
class Segment:
    """A span between time-in-chapter restarts."""

    chapter: int
    entry_count: int
    #: Stage frames `[start, end)` whose observations lie in the segment.
    start: int
    end: int
    first_hit: int | None


def column(words: npt.NDArray[np.uint32], word: wire.Meta) -> npt.NDArray[np.int64]:
    """Read an integer trace column."""
    return words[:, TRACE_WORDS.index(word)].view(wire.meta_dtype(word)).astype(np.int64)


def sync_columns(words: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """Return the `wire.SYNC_WORDS` columns of a trace. Frame `r + 1` reads row `r`."""
    columns = [TRACE_WORDS.index(word) for word in wire.SYNC_WORDS]
    return np.ascontiguousarray(words[:, columns])


def label(words: npt.NDArray[np.uint32]) -> list[Segment]:
    """Split a trace into segments."""
    n = len(words)
    if n == 0:
        return []
    tic = column(words, M.TIME_IN_CHAPTER)
    chapter = column(words, M.CHAPTER)
    entry_count = column(words, M.ENTRY_COUNT)
    hits = column(words, M.HITS)
    starts = np.concatenate(([0], segments.boundaries(tic), [n]))
    hit_frames = np.flatnonzero(hits[1:] > hits[:-1]) + 1
    labeled: list[Segment] = []
    for index, (start, end) in enumerate(pairwise(starts)):
        counted = np.flatnonzero(entry_count[start:end])
        named = int(start + counted[0]) if len(counted) else int(end) - 1
        inside_hit = hit_frames[(hit_frames >= start) & (hit_frames < end)]
        labeled.append(
            Segment(
                chapter=int(chapter[named]),
                entry_count=int(entry_count[named]),
                start=int(start) + 1 if index else 0,
                end=int(end) + 1,
                first_hit=int(inside_hit[0]) + 1 if len(inside_hit) else None,
            )
        )
    return labeled


@dataclass(frozen=True)
class Derived:
    """Stage-play information derived from its trace."""

    #: The number of live frames observed.
    played: int
    # {checked field : (observed value, reference from the next record)}
    checks: dict[str, tuple[int, int]]
    verified: bool
    segments: list[Segment]


def expected_end(replay: rpy.Replay, stage_index: int) -> tuple[dict[str, int], bool]:
    """Return the expected data for a stage play's last live frame, plus whether a bonus was achieved."""
    if stage_index + 1 < len(replay.stages):
        reference = replay.stages[stage_index + 1].globals
        return {name: reference[name] for name in _CHECKS}, False
    return {"score_div10": replay.score_div10}, replay.cleared


@dataclass(frozen=True)
class Result:
    """A finished stage play."""

    job: Job
    landed: bool
    words: npt.NDArray[np.uint32]

    @cached_property
    def facts(self) -> Facts:
        """Judging criteria for the play's trace."""
        expected, all_clear = expected_end(self.job.replay, self.job.stage_index)
        return Facts(expected, all_clear, self.landed)

    @cached_property
    def derived(self) -> Derived:
        """Stage-play information derived from its trace."""
        return derive(self.words, self.facts)

    def row(self) -> dict[str, Any]:
        """Return the index line."""
        job = self.job
        return {
            "key": job.key,
            "path": str(job.path),
            "stage": job.stage.number,
            "stage_index": job.stage_index,
            "character": job.replay.character,
            "cleared": job.replay.cleared,
            "seed": job.stage.seed,
            "frames": job.stage.frames,
            **self.facts._asdict(),
        }


def all_clear_bonus(lives: int, bombs: int) -> int:
    """Return the all-clear bonus (in score/10 units) for a cleared run ending with these resources."""
    return lives * LIFE_BONUS_DIV10 + bombs * BOMB_BONUS_DIV10


def verify(
    words: npt.NDArray[np.uint32], expected: Mapping[str, int], *, all_clear: bool = False
) -> tuple[dict[str, tuple[int, int]], bool]:
    """Compare a stage play's last live frame with its expected data."""

    def last(word: wire.Meta) -> int:
        return int(column(words[-1:], word)[0])

    checks = {name: (last(_CHECKS[name]), want) for name, want in expected.items()}
    if all_clear:
        got, want = checks["score_div10"]
        checks["score_div10"] = (got, want - all_clear_bonus(last(M.LIVES), last(M.BOMBS)))
    return checks, all(got == want for got, want in checks.values())


class Facts(NamedTuple):
    """What a stage-play's trace is judged by, as its index line records them (`derive`)."""

    #: What the play's last live frame must read (`expected_end`).
    expected: dict[str, int]
    #: Whether the run's final score owes the all-clear bonus (`expected_end`).
    all_clear: bool
    #: Whether the landing signature was canonical (`wire.CANONICAL_LANDING`).
    landed: bool

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> Facts:
        """Read the facts an index line records (`Result.row`)."""
        return cls(dict(row["expected"]), bool(row["all_clear"]), bool(row["landed"]))


def derive(words: npt.NDArray[np.uint32], facts: Facts) -> Derived:
    """Compute stage play information from its trace."""
    checks, matched = verify(words, facts.expected, all_clear=facts.all_clear)
    return Derived(len(words), checks, matched and facts.landed, label(words))


class _Phase(Enum):
    NEED_RESET = 0
    RESETTING = 1
    PLAYING = 2
    DONE = 3


@dataclass
class _Slot:
    job: Job | None = None
    phase: _Phase = _Phase.NEED_RESET
    seq: int = 0
    landed: bool = False
    #: Frames sent so far.
    frame: int = 0
    rows: list[npt.NDArray[np.uint32]] = field(default_factory=list)


class Ingester:
    """Plays `jobs` over `env`, one stage play per instance at a time."""

    def __init__(self, env: Fleet, jobs: Iterator[Job]) -> None:
        self.env = env
        self.jobs = jobs
        self._slots = [_Slot() for _ in env.instances]
        self._ready = list(range(len(env.instances)))
        for slot in self._slots:
            self._next_job(slot)

    @property
    def done(self) -> bool:
        """Whether every slot has run out of jobs."""
        return all(slot.phase is _Phase.DONE for slot in self._slots)

    @staticmethod
    def _job_of(slot: _Slot) -> Job:
        job = slot.job
        assert job is not None  # noqa: S101
        return job

    def step(self) -> list[Result]:
        """Advance every ready instance by one exchange. Returns the stage plays that finished."""
        commands = {i: self._command(self._slots[i]) for i in self._ready if self._slots[i].phase is not _Phase.DONE}
        observations = self.env.exchange(commands)
        self._ready = sorted(observations)
        finished: list[Result] = []
        for i, obs in observations.items():
            slot = self._slots[i]
            if slot.phase is _Phase.RESETTING:
                self._on_resetting(slot, obs)
            elif slot.phase is _Phase.PLAYING:
                result = self._on_playing(slot, obs)
                if result is not None:
                    finished.append(result)
        return finished

    def _command(self, slot: _Slot) -> bytes:
        job = self._job_of(slot)
        if slot.phase is _Phase.NEED_RESET:
            slot.seq = self.env.next_reset_seq()
            slot.phase = _Phase.RESETTING
            params = record_reset(job.stage, self.env.character, 1, faithful=True)
            return wire.reset_frame(slot.seq, params)
        if slot.phase is _Phase.PLAYING:
            keys = int(job.stage.actions[slot.frame])
            slot.frame += 1
            return wire.act_frame(keys, raw=True)
        return wire.NEUTRAL

    def _on_resetting(self, slot: _Slot, obs: wire.Observation) -> None:
        job = self._job_of(slot)
        if not wire.reset_landed(obs, slot.seq, job.key):
            return
        slot.landed = obs.landing_signature == wire.CANONICAL_LANDING
        slot.frame = 0
        slot.phase = _Phase.PLAYING

    def _on_playing(self, slot: _Slot, obs: wire.Observation) -> Result | None:
        """Keep the words of a live frame. Finishes the play once the stage is over or the tape has ended."""
        job = self._job_of(slot)
        if obs.in_stage:
            slot.rows.append(obs.words[_TRACE_INDEX])
            if slot.frame < job.stage.frames:
                return None
        if not slot.rows:
            msg = f"{job.key}: the stage was over on its first frame"
            raise RuntimeError(msg)
        words = np.stack(slot.rows)
        result = Result(job, slot.landed, words)
        self._next_job(slot)
        return result

    def _next_job(self, slot: _Slot) -> None:
        slot.job = next(self.jobs, None)
        slot.phase = _Phase.DONE if slot.job is None else _Phase.NEED_RESET
        slot.rows = []


class Store:
    """The on-disk sink. Includes `traces/<key>.npz` for stage plays and an appended `stages.ndjson`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.traces = root / "traces"
        self.replays = root / "replays"
        self.index = root / "stages.ndjson"
        self.traces.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] | None = None
        self._plays: dict[str, Derived] = {}

    def replay(self, path: str | Path) -> rpy.Replay:
        """Return the decoded replay specified by an index row's path."""
        return rpy.load(self.root / path)

    def import_replay(self, path: Path) -> str:
        """Keep the replay at `path` under the root. Returns the path that the index uses for it.

        The file is copied to `replays/<file>` (unless it is already that file) and specified relative to the root.
        """
        target = self.replays / path.name
        if not (target.is_file() and target.samefile(path)):
            self.replays.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
        return str(target.relative_to(self.root))

    def plays(self, stage: int, character: int) -> Iterator[tuple[dict[str, Any], Derived]]:
        """Yield every verified stage play of `stage` by `character`."""
        for row in self.rows():
            if row["character"] == character and row["stage"] == stage:
                derived = self.play(row)
                if derived.verified:
                    yield row, derived

    def play(self, row: Mapping[str, Any]) -> Derived:
        """Return stage play information indicated by its corresponding index line's trace."""
        key = str(row["key"])
        if key not in self._plays:
            self._plays[key] = derive(self.load_trace(key), Facts.of(row))
        return self._plays[key]

    def keys(self) -> set[str]:
        """Return the stage plays already stored by the index."""
        return {str(row["key"]) for row in self.rows()}

    def rows(self) -> list[dict[str, Any]]:
        """Return every index line in order."""
        if self._rows is None:
            self._rows = self._read_rows()
        return self._rows

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.index.is_file():
            return []
        with self.index.open() as file:
            return [json.loads(line) for line in file if line.strip()]

    @staticmethod
    def trace_name(key: str) -> str:
        """Return the trace file name of a stage play key."""
        return key.replace(":", "_") + ".npz"

    def add(self, result: Result) -> None:
        """Write the trace file and keep the replay, then append the index line."""
        self._write_trace(result.job.key, result.words)
        row = result.row()
        row["path"] = self.import_replay(result.job.path)
        with self.index.open("a") as file:
            file.write(json.dumps(row) + "\n")
        self._rows = None

    def load_trace(self, key: str) -> npt.NDArray[np.uint32]:
        """Return a stage play's trace words."""
        with np.load(self.traces / self.trace_name(key)) as data:
            version = int(data["version"])
            words = data["words"]
        if version != TRACE_VERSION:
            msg = f"{key}: trace version {version}, expected {TRACE_VERSION}"
            raise ValueError(msg)
        return words

    def _write_trace(self, key: str, words: npt.NDArray[np.uint32]) -> None:
        """Atomically write a trace file."""
        buffer = io.BytesIO()
        np.savez_compressed(buffer, version=np.int32(TRACE_VERSION), words=words)
        atomic_write(self.traces / self.trace_name(key), buffer.getvalue())


def run(env: Fleet, jobs: Iterator[Job], store: Store, *, log_every: int = 25) -> int:
    """Play every job over `env` into `store`. Returns the number of stage plays verified."""
    ingester = Ingester(env, jobs)
    finished = verified = 0
    while not ingester.done:
        for result in ingester.step():
            store.add(result)
            finished += 1
            ok = result.derived.verified
            verified += ok
            if finished % log_every == 0 or not ok:
                print(  # noqa: T201
                    f"{result.job.key}: {'ok' if ok else 'DESYNC'} "
                    f"({result.derived.played}/{result.job.stage.frames} frames, {finished} done, "
                    f"{verified} verified)",
                    flush=True,
                )
    return verified
