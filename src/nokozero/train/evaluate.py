"""Greedy evaluation of saved models over a fixed set of seeds."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np

from nokozero import wire
from nokozero.env import Env, Fleets
from nokozero.train.results import CUT, PREFIX_FAILED, Recorder, Result, open_result
from nokozero.train.rollout import (
    TRAINING_POWER,
    Prefix,
    RunnerConfig,
    SegmentRunner,
    Spec,
    StartKey,
    Target,
    multi_seed_starts,
    play_out,
    prefix_starts,
    start_keys,
)
from nokozero.train.router import Router, StagePolicy
from nokozero.utils import files_digest, print_row

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from nokozero.env import EnvConfig, Fleet
    from nokozero.train.rollout import Episode, Start

M = wire.Meta

SEED_SAMPLE_RNG = 310514


@dataclass(frozen=True)
class Evaluation:
    """Specification of plays for an evaluation."""

    where: Spec | Target
    prefix_router: Path | None = None
    full_stage: bool = False
    until_tic: int | None = None
    max_episode_steps: int | None = None
    start_power: int = TRAINING_POWER
    dither: float = 0.0
    noise: float = 0.0

    def __post_init__(self) -> None:
        prefixed = isinstance(self.where, Target)
        if prefixed != (self.prefix_router is not None):
            msg = "a prefix evaluation needs a prefix router and a target"
            raise ValueError(msg)
        if prefixed and self.full_stage:
            msg = "a full-stage evaluation plays from a section's landing, not from a prefix"
            raise ValueError(msg)
        if self.full_stage and isinstance(self.where, Spec):
            object.__setattr__(self, "where", self.where._replace(await_chapter=False))
        self.rules()

    def rules(self) -> RunnerConfig:
        """Return the episode runner configuration."""
        return RunnerConfig(
            max_steps=self.max_episode_steps,
            until_stage_end=self.full_stage,
            until_tic=self.until_tic,
            keep_features=False,
        )

    def row(self) -> dict[str, Any]:
        """Return the evaluation's fields."""
        prefix_router = None if self.prefix_router is None else str(self.prefix_router)
        return {**asdict(self), "where": str(self.where), "prefix_router": prefix_router}


def sample_seeds(count: int) -> list[int]:
    """Return `count` distinct seeds."""
    drawn = np.random.default_rng(SEED_SAMPLE_RNG).choice(wire.MAX_RNG_SEED + 1, size=count, replace=False)
    return [int(seed) for seed in drawn]


def result_path(out: Path, model: Path) -> Path:
    """Build the path for a model's result from `out`."""
    return out / f"{model.parent.name}-{model.stem}.jsonl"


@dataclass(frozen=True)
class _Model:
    """A model used by an evaluation."""

    played: StagePolicy
    recorder: Recorder


def evaluate(  # noqa: PLR0913
    models: Sequence[Path],
    evaluation: Evaluation,
    seeds: Sequence[int],
    env_config: EnvConfig,
    *,
    out: Path | None = None,
    make: Callable[[EnvConfig], Fleet] = Env,
) -> list[Result]:
    """Play every seed once with each model in turn on a single fleet. Returns the result of each one."""
    seeds = [int(seed) for seed in seeds]
    with ExitStack() as stack:
        character, prefix, planned = _plan(models, evaluation, seeds, out, stack)
        with Fleets(replace(env_config, character=character), make, print_row) as fleets:
            for model in planned:
                _play(fleets, model, evaluation, seeds, prefix)
        return [model.recorder.result for model in planned]


def _plan(
    models: Sequence[Path],
    evaluation: Evaluation,
    seeds: Sequence[int],
    out: Path | None,
    stack: ExitStack,
) -> tuple[int, Prefix | None, list[_Model]]:
    """Load and check the segments to play during an evaluation. Returns the character, the prefix, and the models."""
    paths = [None if out is None else result_path(out, model) for model in models]
    if out is not None and len(set(paths)) < len(paths):
        msg = f"two models would write to one result file: {[str(path) for path in paths]}"
        raise ValueError(msg)
    router = None if evaluation.prefix_router is None else Router.load(evaluation.prefix_router)
    loaded = [StagePolicy.load(model, dither=evaluation.dither, noise=evaluation.noise) for model in models]
    characters = sorted(
        {played.character for played in loaded} | ({router.featurizer.character} if router is not None else set())
    )
    if len(characters) != 1:
        msg = f"the models play more than one character: {characters}"
        raise ValueError(msg)
    if router is not None:
        for model, played in zip(models, loaded, strict=True):
            router.fallback.provenance.require_control(played.provenance, str(model))
    shared = {
        "prefix_router_digest": None if router is None else files_digest(router.files),
        **evaluation.row(),
        "seeds": len(seeds),
    }
    planned: list[_Model] = []
    for model, played, path in zip(models, loaded, paths, strict=True):
        header = {"model": str(model), "model_digest": files_digest(played.files), **shared}
        planned.append(_Model(played, open_result(path, header, stack)))
    prefix = None
    if router is not None:
        assert isinstance(evaluation.where, Target)  # noqa: S101
        prefix = Prefix(router, evaluation.where)
        results = [model.recorder.result for model in planned]
        if out is not None:
            results += [Result.load(other) for other in out.glob("*.jsonl")]
        prefix.failed.update(
            StartKey(seed, evaluation.start_power)
            for result in results
            if all(result.header.get(name) == value for name, value in shared.items())
            for seed, outcome in result.outcomes.items()
            if outcome == PREFIX_FAILED
        )
    (character,) = characters
    return character, prefix, planned


def _play(
    fleets: Fleets[Fleet],
    model: _Model,
    evaluation: Evaluation,
    seeds: Sequence[int],
    prefix: Prefix | None,
) -> None:
    """Play on seeds not covered by the result of a model. Prints the result."""
    recorder, played = model.recorder, model.played
    result = recorder.result
    if not result.missing(seeds):
        print_row({**result.row(), "kept": str(result.path)})
        return
    recorder.session(_device())
    _prefix_failures(recorder, prefix)
    missing = result.missing(seeds)
    character = fleets.config.character
    interval = played.featurizer.step_interval
    keys = start_keys(iter(missing), evaluation.start_power)
    starts: Iterator[Start]
    if prefix is not None:
        playable = (key for key in keys if key not in prefix.failed)
        starts = prefix_starts(prefix, character, playable, interval)
    else:
        assert isinstance(evaluation.where, Spec)  # noqa: S101
        starts = multi_seed_starts((evaluation.where,), character, keys, interval)
    runner = SegmentRunner(
        starts,
        played.featurizer,
        evaluation.rules(),
        played.actions,
        gave_up=lambda start: _prefix_failed(recorder, start.seed),
    )
    if missing:
        play_out(fleets, runner, played.policy, lambda episode: recorder.seed(_facts(episode)))
    print_row({**result.row(), **runner.timings()})


def _device() -> str:
    """Specify the device that the models play on."""
    device = cast("Any", jax.devices()[0])  # pyright: ignore[reportUnknownMemberType]
    return f"{jax.default_backend()}:{device.device_kind}"


def _prefix_failures(recorder: Recorder, prefix: Prefix | None) -> None:
    """Write down every seed known to fail the prefix that isn't recorded yet."""
    if prefix is None:
        return
    for key in sorted(prefix.failed):
        _prefix_failed(recorder, key.seed)


def _prefix_failed(recorder: Recorder, seed: int) -> None:
    """Write down a seed whose prefix failed."""
    if seed not in recorder.result.seeds:
        recorder.seed({"seed": seed, "outcome": PREFIX_FAILED})


def _facts(episode: Episode) -> dict[str, Any]:
    """Return a seed's line."""
    row: dict[str, Any] = {
        "seed": episode.seed,
        "outcome": episode.outcome.value,
        "length": episode.length,
    }
    end = episode.end
    if end is not None:
        row["end"] = {
            "chapter": end.meta(M.CHAPTER),
            "entry_count": end.meta(M.ENTRY_COUNT),
            "tic": end.meta(M.TIME_IN_CHAPTER),
            "tick": end.meta(M.GAME_TICK),
            "x": round(end.meta_f32(M.PLAYER_X), 2),
            "y": round(end.meta_f32(M.PLAYER_Y), 2),
        }
    if episode.segments is not None:
        row["segments"] = {
            str(target): CUT if outcome is None else outcome.value for target, outcome in episode.segments.items()
        }
    return row
