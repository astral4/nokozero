"""Logic for harvesting demonstrations."""

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from nokozero import wire
from nokozero.replay import ingest
from nokozero.train.features import Features, Featurizer
from nokozero.utils import atomic_write

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from nokozero.env import Fleet
    from nokozero.train.curriculum import Candidate


@dataclass(frozen=True)
class Demonstration:
    """Decisions for a segment."""

    key: str
    features: Features
    keys: npt.NDArray[np.uint16]
    frames: npt.NDArray[np.int32]
    completed: bool = True


FEATURIZE_BATCH = 256


class _Phase(Enum):
    NEED_RESET = 0
    RESETTING = 1
    TAPING = 2
    PLAYING = 3
    DONE = 4


@dataclass
class _Slot:
    candidate: Candidate | None = None
    phase: _Phase = _Phase.NEED_RESET
    seq: int = 0
    frame: int = 0
    feats: list[Features] = field(default_factory=list)
    observations: list[wire.Observation] = field(default_factory=list)


class Harvester:
    """Plays every candidate's segment over `env`, one per instance at a time."""

    def __init__(self, env: Fleet, candidates: Iterator[Candidate], featurizer: Featurizer) -> None:
        self.env = env
        self.candidates = candidates
        self.featurizer = featurizer
        self._slots = [_Slot() for _ in env.instances]
        self._ready = list(range(len(env.instances)))
        self.desynced = 0
        self.unreachable = 0
        for slot in self._slots:
            self._next(slot)

    @property
    def step_interval(self) -> int:
        """Frames per recorded state."""
        return self.featurizer.step_interval

    @property
    def done(self) -> bool:
        """Whether every slot has run out of candidates."""
        return all(slot.phase is _Phase.DONE for slot in self._slots)

    def _candidate_of(self, slot: _Slot) -> Candidate:
        candidate = slot.candidate
        assert candidate is not None  # noqa: S101
        return candidate

    def step(self) -> list[Demonstration]:
        """Advance every ready instance. Returns the segments that finished."""
        commands = {i: self._command(self._slots[i]) for i in self._ready if self._slots[i].phase is not _Phase.DONE}
        observations = self.env.exchange(commands)
        self._ready = sorted(observations)
        finished: list[Demonstration] = []
        for i, obs in observations.items():
            slot = self._slots[i]
            if slot.phase is _Phase.RESETTING:
                self._on_resetting(slot, obs)
            elif slot.phase is _Phase.TAPING:
                slot.phase = _Phase.PLAYING
            if slot.phase is _Phase.PLAYING:
                demonstration = self._on_playing(slot, obs)
                if demonstration is not None:
                    finished.append(demonstration)
        return finished

    def _command(self, slot: _Slot) -> bytes:
        candidate = self._candidate_of(slot)
        if slot.phase is _Phase.NEED_RESET:
            slot.seq = self.env.next_reset_seq()
            slot.phase = _Phase.RESETTING
            params = ingest.record_reset(candidate.stage, self.env.character, self.step_interval, faithful=True)
            return wire.reset_frame(slot.seq, params)
        if slot.phase is _Phase.RESETTING:
            return wire.NEUTRAL
        keys = candidate.stage.actions
        if slot.phase is _Phase.TAPING:
            return wire.tape_frame(keys[: slot.frame], raw=True)
        step = keys[slot.frame : slot.frame + self.step_interval]
        slot.frame += self.step_interval
        return wire.tape_frame(step, raw=True)

    def _on_resetting(self, slot: _Slot, obs: wire.Observation) -> None:
        candidate = self._candidate_of(slot)
        if not wire.reset_landed(obs, slot.seq, candidate.key):
            return
        slot.frame = candidate.first_decision(self.step_interval)
        slot.feats = []
        slot.observations = []
        slot.phase = _Phase.PLAYING if slot.frame == 0 else _Phase.TAPING

    def _on_playing(self, slot: _Slot, obs: wire.Observation) -> Demonstration | None:
        """Classify and record the observation at `slot.frame`. Returns the segment once it ends."""
        candidate = self._candidate_of(slot)
        limit = min(candidate.limit, len(candidate.stage.keys))
        if slot.frame < limit and not candidate.synced(slot.frame, obs):
            self.desynced += 1
            self._next(slot)
            return None
        if slot.frame + self.step_interval <= limit:
            slot.observations.append(obs)
            if len(slot.observations) == FEATURIZE_BATCH:
                self._featurize(slot)
            return None
        demonstration = None
        self._featurize(slot)
        if slot.feats:
            features = Features.concatenate(slot.feats)
            n, first = len(features), candidate.first_decision(self.step_interval)
            end = first + n * self.step_interval
            demonstration = Demonstration(
                key=candidate.key,
                features=features,
                keys=candidate.stage.actions[first:end].reshape(n, self.step_interval),
                frames=np.arange(first, end, self.step_interval, dtype=np.int32),
                completed=limit == candidate.end,
            )
        self._next(slot)
        return demonstration

    def _featurize(self, slot: _Slot) -> None:
        """Featurize the slot's pending observations as a single batch."""
        if slot.observations:
            slot.feats.append(self.featurizer.batch(slot.observations))
            slot.observations = []

    def _next(self, slot: _Slot) -> None:
        slot.candidate = self._draw()
        slot.phase = _Phase.DONE if slot.candidate is None else _Phase.NEED_RESET

    def _draw(self) -> Candidate | None:
        """Return the next candidate whose first decision is reached by a tape."""
        for candidate in self.candidates:
            if candidate.first_decision(self.step_interval) <= wire.MAX_TAPE_FRAMES:
                return candidate
            self.unreachable += 1
        return None


def save(path: Path, demonstrations: list[Demonstration], featurizer: Featurizer) -> None:
    """Write the demonstrations as a compressed file of concatenated arrays."""
    if not demonstrations:
        msg = "no demonstrations to save"
        raise ValueError(msg)
    feats = Features.concatenate([d.features for d in demonstrations])
    bounds = np.cumsum([0, *(len(d.features) for d in demonstrations)])
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **featurizer.header(),
        tokens=feats.tokens.astype(np.float16),
        mask=feats.mask,
        globals=feats.globals,
        keys=np.concatenate([d.keys for d in demonstrations]),
        frames=np.concatenate([d.frames for d in demonstrations]),
        bounds=bounds,
        keys_of=np.array([d.key for d in demonstrations]),
        completed=np.array([d.completed for d in demonstrations]),
    )
    atomic_write(path, buffer.getvalue())
