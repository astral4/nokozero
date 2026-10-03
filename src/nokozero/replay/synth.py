"""Logic for creating `.rpy` replay files from game runs."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from nokozero import wire
from nokozero.replay import rpy
from nokozero.replay.ingest import all_clear_bonus
from nokozero.train.rollout import Outcome, recorded_keys

if TYPE_CHECKING:
    from nokozero.train.game import StageTrace

STAGES: Final = wire.MAIN_STAGES
TAIL_FRAMES = 120
NAME = "nokozero"


class SynthError(ValueError):
    """A cleared run could not be written as a replay file."""


@dataclass(frozen=True)
class Template:
    """A played replay for filling out fields not set by a synthesized run."""

    replay: rpy.Replay
    character: int
    difficulty: int

    def __post_init__(self) -> None:
        if self.replay.character != self.character:
            msg = f"the template is a run of character {self.replay.character}, not {self.character}"
            raise SynthError(msg)
        numbers = {s.number for s in self.replay.stages}
        missing = [n for n in STAGES if n not in numbers]
        if missing:
            msg = f"the template has no record of stages {missing}"
            raise SynthError(msg)
        other = sorted(
            {s.number for s in self.replay.stages if s.number in STAGES and s.globals["difficulty"] != self.difficulty}
        )
        if other:
            msg = f"the template played stages {other} at another difficulty than {self.difficulty}"
            raise SynthError(msg)

    def record(self, stage: int) -> bytes:
        """Return the template's record of `stage`."""
        return next(s.record[: rpy.RECORD_SIZE] for s in self.replay.stages if s.number == stage)


def stage_keys(trace: StageTrace) -> np.ndarray:
    """Return the key words of `trace`'s stage (one per frame)."""
    inputs = trace.inputs
    if not len(inputs) or int(inputs[0, 0]) != 0:
        first = int(inputs[0, 0]) if len(inputs) else None
        msg = f"stage {trace.stage} of run {trace.seed}: no inputs from its first frame ({first})"
        raise SynthError(msg)
    try:
        keys = recorded_keys([inputs])
    except ValueError as e:
        msg = f"stage {trace.stage} of run {trace.seed}: {e}"
        raise SynthError(msg) from e
    return np.concatenate([keys, np.zeros(TAIL_FRAMES, dtype=np.uint16)])


def stage_globals(trace: StageTrace) -> dict[str, int]:
    """Return the record's globals for `trace`'s start."""
    start = trace.params
    return {
        "chapter": 0,
        "score_div10": start.score // 10,
        "graze": start.graze,
        "spell_id": -1,
        "misses": 0,
        "point_items": 0,
        "piv_x100": start.value * 100,
        "power": start.power,
        "lives": start.lives,
        "life_fragments": start.life_fragments,
        "extends": 0,
        "bombs": start.bombs,
        "bomb_fragments": start.bomb_fragments,
    }


def synthesize(
    traces: list[StageTrace],
    template: Template,
    *,
    name: str = NAME,
    timestamp: int | None = None,
) -> bytes:
    """Return a `.rpy` file for the cleared run described by `traces`."""
    ordered = sorted(traces, key=lambda t: t.stage)
    if [t.stage for t in ordered] != list(STAGES):
        msg = f"the traces are stages {[t.stage for t in ordered]}, not {STAGES[0]} to {STAGES[-1]}"
        raise SynthError(msg)
    if any(t.outcome is not Outcome.COMPLETE for t in ordered):
        fates = {t.stage: t.outcome.value for t in ordered}
        msg = f"the run did not clear every stage: {fates}"
        raise SynthError(msg)
    character, difficulty = template.character, template.difficulty
    specs = [
        rpy.StageSpec(
            number=trace.stage,
            seed=trace.params.rng_seed,
            keys=stage_keys(trace),
            player_x=trace.params.player_x,
            player_y=trace.params.player_y,
            globals={**stage_globals(trace), "difficulty": difficulty},
            template=template.record(trace.stage),
        )
        for trace in ordered
    ]
    last = ordered[-1]
    score = last.end.score // 10 + all_clear_bonus(last.end.lives, last.end.bombs)
    stamp = int(time.time()) if timestamp is None else timestamp
    text = (
        "東方紺珠伝 リプレイファイル情報\r\nVersion 1.00b\r\n"
        f"Name {name:<8}\r\nDate {time.strftime('%y/%m/%d %H:%M', time.localtime(stamp))}\r\n"
        f"Chara {_CHARACTERS.get(character, '?'):<7}\r\nRank {_RANKS.get(difficulty, '?'):<7}\r\n"
        f"Stage All Clear\r\nScore {score * 10}\r\nSlow Rate 0.00\r\n"
    )
    return rpy.encode(
        specs,
        name=name,
        timestamp=stamp,
        score_div10=score,
        character=character,
        difficulty=difficulty,
        template_info=template.replay.raw_info,
        user_text=text,
    )


_CHARACTERS = {0: "Reimu", 1: "Marisa", 2: "Sanae", 3: "Reisen"}
_RANKS = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Lunatic"}
