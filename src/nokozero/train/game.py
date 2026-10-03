"""Logic for chaining stage routers to play entire games/credits."""

import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nokozero import catalog, wire
from nokozero.env import Env, Fleets
from nokozero.train.rollout import (
    FULL_STAGE_MAX_STEPS,
    GAME_START_POWER,
    Outcome,
    RunnerConfig,
    SegmentRunner,
    Start,
    WordKey,
    play_out,
    reset_params,
)
from nokozero.train.router import StagePolicy
from nokozero.utils import print_row, write_result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from nokozero.env import EnvConfig, Fleet
    from nokozero.train.rollout import Episode, Target

__all__ = [
    "GAME_START",
    "Carry",
    "GameResult",
    "StagePolicy",
    "StageResult",
    "StageTrace",
    "carry",
    "check_runs",
    "play_game",
    "report",
    "run_of",
    "stage_seed",
    "stage_start",
]

M = wire.Meta


@dataclass(frozen=True, kw_only=True)
class Carry:
    """Data carried between stages."""

    score: int = 0
    graze: int = 0
    value: int = 0
    power: int = GAME_START_POWER
    lives: int = 2
    life_fragments: int = 0
    bombs: int = 3
    bomb_fragments: int = 0
    player_x: float = 0.0
    player_y: float = 400.0


GAME_START = Carry()


def carry(obs: wire.Observation) -> Carry:
    """State for the next stage based on a stage ending on `obs`."""
    return Carry(
        score=obs.meta(M.SCORE_DIV10) * 10,
        graze=obs.meta(M.GRAZE),
        value=obs.meta(M.VALUE_X100) // 100,
        power=obs.meta(M.POWER),
        lives=obs.meta(M.LIVES),
        life_fragments=obs.meta(M.LIFE_FRAGMENTS),
        bombs=obs.meta(M.BOMBS),
        bomb_fragments=obs.meta(M.BOMB_FRAGMENTS),
        player_x=obs.meta_f32(M.PLAYER_X),
        player_y=obs.meta_f32(M.PLAYER_Y),
    )


def stage_seed(seed: int, stage: int) -> int:
    """Return the RNG seed for stage `stage` of run `seed`."""
    if stage == 1:
        return seed
    return int(np.random.default_rng([seed, stage]).integers(0, wire.MAX_RNG_SEED + 1))


def stage_start(stage: int, seed: int, carried: Carry, character: int, step_interval: int) -> Start:
    """Build the start of run `seed`'s stage `stage`."""
    seeded = stage_seed(seed, stage)
    params = replace(reset_params(catalog.opening(stage), character, seeded, step_interval), **asdict(carried))
    return Start(
        params,
        await_chapter=False,
        label=f"game:{stage}:{seed}",
        origin=seed,
    )


def ended_in(segments: Mapping[Target, Outcome | None] | None, end: wire.Observation) -> WordKey:
    """Specify the segment that a full-stage run ended within."""
    hits = [target for target, outcome in (segments or {}).items() if outcome is Outcome.HIT]
    if hits:
        target = hits[-1]
        return WordKey.entry(target.word, target.occurrence)
    return WordKey.entry(end.meta(M.CHAPTER), end.meta(M.ENTRY_COUNT))


def run_of(start: Start) -> int:
    """Return the run seed that a stage start was built for."""
    run = start.origin
    if not isinstance(run, int):
        msg = f"{start.label} is not a game's stage start"
        raise TypeError(msg)
    return run


@dataclass(frozen=True)
class StageTrace:
    """One stage of one run."""

    stage: int
    seed: int
    outcome: Outcome
    params: wire.ResetParams
    inputs: npt.NDArray[np.uint32]
    end: Carry


@dataclass(frozen=True)
class StageResult:
    """The result of runs that entered a stage."""

    stage: int
    entered: int
    cleared: int
    deaths: dict[str, int] = field(default_factory=dict)
    lost: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class GameResult:
    """The result of stages chained together."""

    start_power: int
    stages: tuple[StageResult, ...]
    per_seed: dict[int, str] = field(default_factory=dict)

    @property
    def seeds(self) -> int:
        """The runs started."""
        return len(self.per_seed)

    @property
    def clears(self) -> int:
        """Runs that cleared the last stage played."""
        return self.stages[-1].cleared if self.stages else 0

    @property
    def clear_rate(self) -> float:
        """Proportion of runs that cleared the last stage."""
        return self.clears / max(1, self.seeds)


def check_runs(
    stages: Mapping[int, StagePolicy],
    seeds: Iterable[int],
    *,
    start_power: int = GAME_START.power,
) -> int:
    """Refuse an invalid game."""
    order = sorted(stages)
    if not order or order != list(range(order[0], order[-1] + 1)):
        msg = f"stages must be consecutive, not {order}"
        raise ValueError(msg)
    first = stages[order[0]]
    character = first.character
    for stage, policy in stages.items():
        if policy.character != character:
            msg = f"stage {stage}'s policy plays character {policy.character}, stage {order[0]}'s {character}"
            raise ValueError(msg)
    interval = first.featurizer.step_interval
    carried = replace(GAME_START, power=start_power)
    for seed in seeds:
        try:
            stage_start(order[0], seed, carried, character, interval)
        except ValueError as error:
            msg = f"run seed {seed}: {error}"
            raise ValueError(msg) from error
    return character


def play_game(  # noqa: PLR0913
    stages: Mapping[int, StagePolicy],
    seeds: Iterable[int],
    env_config: EnvConfig,
    *,
    make: Callable[[EnvConfig], Fleet] = Env,
    start_power: int = GAME_START.power,
    max_steps: int = FULL_STAGE_MAX_STEPS,
    log: Callable[[Mapping[str, object]], None] = lambda _row: None,
    traces: dict[int, list[StageTrace]] | None = None,
) -> GameResult:
    """Play every seed's run through `stages` in order."""
    seeds = [int(seed) for seed in seeds]
    character = check_runs(stages, seeds, start_power=start_power)
    order = sorted(stages)
    alive: dict[int, Carry] = {seed: replace(GAME_START, power=start_power) for seed in seeds}
    per_seed: dict[int, str] = {}
    results: list[StageResult] = []

    with Fleets(replace(env_config, character=character), make, log) as fleets:
        for stage in order:
            played = stages[stage]
            interval = played.featurizer.step_interval
            starts = [stage_start(stage, seed, carried, character, interval) for seed, carried in alive.items()]
            config = RunnerConfig(max_steps=max_steps, until_stage_end=True, keep_features=False)
            runner = SegmentRunner(iter(starts), played.featurizer, config, played.actions)
            began = time.time()
            episodes: list[Episode] = []
            play_out(fleets, runner, played.policy, episodes.append)
            survivors: dict[int, Carry] = {}
            deaths: Counter[WordKey] = Counter()
            lost = 0
            for episode in episodes:
                seed = run_of(episode.start)
                end, inputs = episode.end, episode.inputs
                assert end is not None  # noqa: S101
                assert inputs is not None  # noqa: S101
                ended = carry(end)
                if traces is not None:
                    traces.setdefault(seed, []).append(
                        StageTrace(stage, seed, episode.outcome, episode.start.params, inputs, ended)
                    )
                if episode.outcome is Outcome.COMPLETE:
                    survivors[seed] = ended
                    continue
                where = ended_in(episode.segments, end)
                if episode.outcome is Outcome.HIT:
                    deaths[where] += 1
                else:
                    lost += 1
                per_seed[seed] = f"{stage}:{episode.outcome.value}@{where}"
            ordered = sorted(deaths.items(), key=lambda death: (death[0].word, death[0].occurrence or 1))
            result = StageResult(
                stage,
                len(starts),
                len(survivors),
                {str(where): count for where, count in ordered},
                lost,
                round(time.time() - began, 1),
            )
            results.append(result)
            log(vars(result))
            alive = survivors
    per_seed.update(dict.fromkeys(alive, "clear"))
    return GameResult(start_power, tuple(results), per_seed)


def report(result: GameResult, out: Path | None) -> None:
    """Print the game's result as a JSON line. Writes results to `out`."""
    row: dict[str, Any] = {
        "seeds": result.seeds,
        "start_power": result.start_power,
        "clears": result.clears,
        "clear_rate": round(result.clear_rate, 4),
        "stages": [vars(stage) for stage in result.stages],
    }
    print_row(row)
    if out is not None:
        write_result(out, row, result.per_seed)
