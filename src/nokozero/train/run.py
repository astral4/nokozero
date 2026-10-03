"""The training loop."""

import json
import time
from collections import deque
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Protocol

import jax
import numpy as np

from nokozero.env import Env, Fleets
from nokozero.replay import ingest
from nokozero.train import imitation
from nokozero.train.agent import Agent, Checkpoint, Explorer, Metrics, Provenance
from nokozero.train.buffer import ReplayBuffer, TieredBuffer
from nokozero.train.config import TrainConfig
from nokozero.train.curriculum import Curriculum
from nokozero.train.features import ACTION_SETS, Features, Featurizer
from nokozero.train.model import QNet
from nokozero.train.rollout import (
    Kind,
    Outcome,
    Powers,
    RunnerConfig,
    SegmentRunner,
    Start,
    interleave,
    jitter_starts,
    multi_seed_starts,
    pool_keys,
    power_stream,
    prefix_starts,
    random_policy,
    seed_stream,
    start_keys,
)
from nokozero.train.router import StagePolicy, load_prefix
from nokozero.utils import code_version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from nokozero.env import EnvConfig, Fleet
    from nokozero.train.rollout import Episode, Inputs, Pending, Policy, Spec, Target

__all__ = ["TrainConfig", "Trainer", "train"]

HIT_WINDOW_FRAMES = 60
STALL_SECONDS = 5.0

type Row = dict[str, float | int | None]


class StartSource(Protocol):
    """The source of a run's starts."""

    def warm(self) -> None:
        """Go through slow setup parts."""
        ...

    def starts(self) -> Iterator[Start]:
        """Yield starts."""
        ...

    def observe(self, episode: Episode) -> None:
        """Learn from a finished episode of one of this source's starts."""
        ...

    def row(self) -> Row:
        """Return the log row associated with this source."""
        ...


class Stream:
    def __init__(self, starts: Iterator[Start], row: Callable[[], Row] = dict) -> None:
        self._starts = starts
        self._row = row

    def warm(self) -> None:
        pass

    def starts(self) -> Iterator[Start]:
        return self._starts

    def observe(self, episode: Episode) -> None:=
        del episode

    def row(self) -> Row:
        return self._row()


class Mix:
    def __init__(self, sources: Sequence[tuple[StartSource, int]]) -> None:
        self.sources = sources

    def warm(self) -> None:
        for source, _ in self.sources:
            source.warm()

    def starts(self) -> Iterator[Start]:
        return interleave([(source.starts(), weight) for source, weight in self.sources])

    def observe(self, episode: Episode) -> None:
        for source, _ in self.sources:
            source.observe(episode)

    def row(self) -> Row:
        return {key: value for source, _ in self.sources for key, value in source.row().items()}


def _rate(recent: deque[tuple[Outcome, int]], outcome: Outcome) -> float:
    """Return the proportion of `recent` episodes that ended in `outcome`."""
    return sum(o is outcome for o, _ in recent) / max(1, len(recent))


def _mean(values: Sequence[float], digits: int) -> float | None:
    """Return the mean of `values` rounded to `digits`."""
    return round(float(np.mean(values)), digits) if values else None


def _explorer_policy(explorer: Explorer) -> Policy:
    def policy(inputs: Inputs) -> Pending:
        return explorer(inputs.features)

    return policy


class Trainer:
    """Training run state."""

    def __init__(self, config: TrainConfig, character: int, started: float | None = None) -> None:
        self.config = config
        self.character = character
        self.started = time.time() if started is None else started
        self.rng = np.random.default_rng(config.seed)
        self.featurizer = Featurizer(
            k=config.k,
            variant=config.features,
            step_interval=config.step_interval,
            character=character,
        )
        self.provenance = Provenance(self.featurizer, config.actions)
        self.actions = ACTION_SETS[config.actions]
        self.recent_by_key: dict[Spec | Target, deque[tuple[Outcome, int]]] = {
            spec: deque(maxlen=200) for spec in config.sections
        }
        if config.target is not None and config.prefix_router is not None:
            self.recent_by_key[config.target] = deque(maxlen=200)
        self.source = self.make_source()
        self.source.warm()
        self.handoff = self.make_handoff()
        self.handoff_values: deque[float] = deque(maxlen=200)
        self.suffix = self.make_suffix()
        self.suffix_outcomes: deque[bool] = deque(maxlen=200)
        self.runner = SegmentRunner(
            self.source.starts(),
            self.featurizer,
            RunnerConfig(max_steps=config.max_episode_steps, until_tic=config.until_tic, suffix=self.suffix),
            self.actions,
        )
        self.buffer = self.make_buffer()
        self.demo_rows = self.make_demos()
        self.agent = self.make_agent()
        self.explorer = self.make_explorer()
        self.explore = (
            random_policy(self.rng, num_actions=len(self.actions))
            if self.explorer is None
            else _explorer_policy(self.explorer)
        )
        self.recent: deque[tuple[Outcome, int]] = deque(maxlen=200)
        self.metrics: Metrics | None = None
        self.transitions = self.episodes = 0
        self.updates = 0
        self.update_seconds = 0.0
        self._learning = False

    def make_source(self) -> Mix:
        """Build the run's start sources."""
        config, character = self.config, self.character
        power: Powers = (
            config.start_power
            if config.start_power_max is None
            else power_stream(self.rng, config.start_power, config.start_power_max)
        )
        sources: list[tuple[StartSource, int]] = []
        target = config.target
        if config.store is not None and target is not None:
            curriculum = Curriculum(
                ingest.Store(config.store),
                target,
                character,
                rng=self.rng,
                config=config.curriculum,
                step_interval=config.step_interval,
                end_tic=config.until_tic,
            )
            sources.append((curriculum, config.replay_weight))
        prefix = None
        if config.prefix_router is not None and target is not None and config.seed_pool:
            prefix = load_prefix(config.prefix_router, self.provenance, target)
            pool = pool_keys(self.rng, config.seed_pool, power, failed=prefix.failed)
            starts = prefix_starts(prefix, character, pool, config.step_interval)
            row: Callable[[], Row] = lambda: {"prefix_cache": len(prefix.cache)}  # noqa: E731
            sources.append((Stream(starts, row), config.prefix_weight))
        if config.sections:
            keys = (
                pool_keys(self.rng, config.seed_pool, power)
                if config.seed_pool and prefix is None
                else start_keys(seed_stream(self.rng), power)
            )
            warps = multi_seed_starts(config.sections, character, keys, config.step_interval)
            if config.start_jitter:
                warps = jitter_starts(warps, self.rng)
            sources.append((Stream(warps), config.warp_weight))
        return Mix(sources)

    def make_buffer(self) -> ReplayBuffer | TieredBuffer:
        """Build the replay buffer."""
        config = self.config
        tier = partial(
            ReplayBuffer,
            k=config.k,
            token_dim=self.featurizer.token_dim,
            global_dim=self.featurizer.global_dim,
            n_step=config.n_step,
            hit_window=max(1, round(HIT_WINDOW_FRAMES / config.step_interval)),
            gamma=config.gamma,
            damage_bonus=config.damage_bonus,
        )
        if config.archive_capacity:
            return TieredBuffer(tier, config.capacity, config.archive_capacity, rng=self.rng)
        return tier(config.capacity)

    def make_demos(self) -> int:
        """Put the demonstrations' segments in the buffer as episodes. Returns the rows."""
        config = self.config
        if config.demos is None:
            return 0
        demos = imitation.Demonstrations.load(config.demos)
        self.provenance.require(Provenance(demos.featurizer, config.actions), f"the demonstrations at {config.demos}")
        rows = 0
        for episode in imitation.demonstration_episodes(demos, self.actions):
            value = None
            if episode.outcome is Outcome.COMPLETE:
                value = self.completion_value(episode.states[episode.length])
            self.buffer.add(episode, value)
            rows += episode.length
        return rows

    def make_agent(self) -> Agent:
        """Build the agent."""
        config = self.config
        if config.init_from is not None:
            loaded = Agent.load(
                config.init_from,
                learning_rate=config.learning_rate,
                tau=config.tau,
                decay_updates=config.decay_updates,
            )
            self.provenance.require(loaded.provenance, str(config.init_from))
            return loaded.agent
        network = QNet(
            self.featurizer.token_dim,
            self.featurizer.global_dim,
            len(self.actions),
            dim=config.model_dim,
            heads=config.model_heads,
            depth=config.model_depth,
            key=jax.random.PRNGKey(config.seed),
        )
        return Agent(
            network,
            learning_rate=config.learning_rate,
            tau=config.tau,
            decay_updates=config.decay_updates,
        )

    def make_explorer(self) -> Explorer | None:
        """Load the policy to explore with, if any."""
        config = self.config
        if config.explore_with is None:
            return None
        cloned = imitation.load(config.explore_with)
        self.provenance.require(cloned.provenance, str(config.explore_with))
        return imitation.sampler(cloned, seed=config.seed)

    def make_handoff(self) -> Checkpoint | None:
        """Load the downstream expert, if any."""
        config = self.config
        if config.handoff is None:
            return None
        loaded = Agent.load(config.handoff, trainable=False)
        self.provenance.require(loaded.provenance, str(config.handoff))
        return loaded

    def make_suffix(self) -> Policy | None:
        """Load the next segment's expert that plays on from every completion, if any."""
        config = self.config
        if config.suffix is None:
            return None
        return StagePolicy.load(config.suffix, self.provenance).policy

    def terminal_value(self, episode: Episode) -> float | None:
        """Return a completed episode's terminal value from its suffix or handoff, or `None` for the default."""
        if episode.outcome is not Outcome.COMPLETE:
            return None
        if episode.continuation is not None:
            survived = episode.continuation is not Outcome.HIT
            self.suffix_outcomes.append(survived)
            return 1.0 if survived else 0.0
        value = self.completion_value(episode.states[episode.length])
        if value is not None:
            self.handoff_values.append(value)
        return value

    def completion_value(self, last: Features) -> float | None:
        """Return the completion value for a completion ending on `last`, or `None` for 1."""
        if self.handoff is None:
            return None
        return float(jax.nn.sigmoid(self.handoff.agent.dispatch(last).read()[0].max()))

    @property
    def steps(self) -> int:
        """The number of observations handled by the runner."""
        return self.runner.instance_steps

    def within_budget(self) -> bool:
        """Whether the run has steps and time left."""
        config = self.config
        return self.steps < config.total_steps and (
            config.max_seconds is None or time.time() - self.started < config.max_seconds
        )

    def iterate(self) -> None:
        """Run one exchange on the runner's fleet, plus a gradient step if due."""
        runner = self.runner
        config = self.config
        epsilon = config.epsilon(self.steps)
        policy = (
            self.explore if self.steps < config.warmup_steps else self.agent.policy(epsilon, self.rng, self.explorer)
        )
        for episode in runner.step(policy):
            self.buffer.add(episode, self.terminal_value(episode))
            self.source.observe(episode)
            if episode.start.kind is Kind.SEED:
                self.recent.append((episode.outcome, episode.length))
                recent = self.recent_by_key.get(episode.start.key)
                if recent is not None:
                    recent.append((episode.outcome, episode.length))
            self.transitions += episode.length
            self.episodes += 1
        if self.transitions >= config.learning_starts:
            if not self._learning:
                self._learning = True
                self.updates = self.transitions // config.update_every
            seed_fraction = config.buffer_seed_fraction if config.store is not None else None
            began = time.monotonic()
            while self.updates < self.transitions // config.update_every:
                batch = self.buffer.sample(config.batch_size, self.rng, seed_fraction=seed_fraction)
                self.metrics = self.agent.update(batch)
                self.updates += 1
            self.update_seconds += time.monotonic() - began

    def row(self) -> Row:
        """Return a log row summarizing progress so far."""
        elapsed = time.time() - self.started
        m = self.metrics
        return {
            "steps": self.steps,
            "transitions": self.transitions,
            "episodes": self.episodes,
            "steps_per_s": round(self.steps / max(elapsed, 1e-9), 1),
            "updates": self.updates,
            "epsilon": round(self.config.epsilon(self.steps), 4),
            "buffer": self.buffer.size,
            **({"demo_rows": self.demo_rows} if self.config.demos is not None else {}),
            "recent_complete": round(_rate(self.recent, Outcome.COMPLETE), 3),
            "recent_hit": round(_rate(self.recent, Outcome.HIT), 3),
            "recent_length": _mean([length for _, length in self.recent], 1),
            **self.runner.counts(),
            **self.runner.timings(),
            "update_s": round(self.update_seconds, 1),
            **({"handoff_value": _mean(self.handoff_values, 4)} if self.handoff is not None else {}),
            **({"suffix_complete": _mean(self.suffix_outcomes, 3)} if self.suffix is not None else {}),
            "loss": round(float(m.loss), 4) if m else None,
            "mean_q": round(float(m.mean_q), 4) if m else None,
            "mean_target": round(float(m.mean_target), 4) if m else None,
            **self.source.row(),
            **{
                f"recent_complete_{spec}": round(_rate(recent, Outcome.COMPLETE), 3)
                for spec, recent in self.recent_by_key.items()
                if len(self.recent_by_key) > 1
            },
        }


def train(config: TrainConfig, env_config: EnvConfig, make: Callable[[EnvConfig], Fleet] = Env) -> None:
    """Run training to `config.total_steps` instance steps, logging under `config.out`."""
    started = time.time()
    config.out.mkdir(parents=True, exist_ok=True)
    row = {
        **asdict(config),
        "target": None if config.target is None else str(config.target),
        "sections": list(map(str, config.sections)),
        "fleet": {
            "instances": env_config.num_instances,
            "collect_window": env_config.collect_window,
        },
        "code": code_version(),
    }
    (config.out / "config.json").write_text(json.dumps(row, default=str, indent=1))
    with (config.out / "log.jsonl").open("a") as log:

        def emit(row: Mapping[str, object]) -> None:
            """Write one JSON line to the console and log file."""
            line = json.dumps({"time": round(time.time() - started, 1), **row})
            print(line, flush=True)  # noqa: T201
            log.write(line + "\n")
            log.flush()

        fleets = Fleets(env_config, make, emit)
        trainer: Trainer | None = None
        try:
            trainer = Trainer(config, env_config.character, started)
            with fleets:
                _loop(trainer, fleets, emit)
        except BaseException as failure:
            emit({"exit": type(failure).__name__, "reason": str(failure)[:200]})
            raise
        finally:
            if trainer is not None and fleets.booted:
                _save(trainer, "model.eqx")
                _save(trainer, "ema.eqx", averaged=True)
        emit({"exit": "budget"})


def _save(trainer: Trainer, name: str, *, averaged: bool = False) -> None:
    trainer.agent.save(trainer.config.out / name, trainer.provenance, averaged=averaged)


def _loop(trainer: Trainer, fleets: Fleets[Fleet], emit: Callable[[Mapping[str, object]], None]) -> None:
    """Iterate until the step or time budget is exhausted."""
    config = trainer.config
    last_log = last_checkpoint = time.time()

    def drive(env: Fleet) -> None:
        nonlocal last_log, last_checkpoint
        trainer.runner.attach(env)
        while trainer.within_budget():
            before = time.time()
            trainer.iterate()
            now = time.time()
            if now - before > STALL_SECONDS:
                emit({"stall_s": round(now - before, 1), "steps": trainer.steps})
            if now - last_log >= config.log_every:
                emit({**trainer.row(), "fleet_deaths": fleets.deaths})
                last_log = now
            if now - last_checkpoint >= config.checkpoint_every:
                _checkpoint(trainer, now)
                last_checkpoint = now

    fleets.run(drive, again=trainer.within_budget)


def _checkpoint(trainer: Trainer, now: float) -> None:
    """Write the live checkpoint."""
    minutes = int((now - trainer.started) // 60)
    for name in ("model.eqx", f"model-{minutes:03d}m.eqx"):
        _save(trainer, name)
    _save(trainer, f"ema-{minutes:03d}m.eqx", averaged=True)
