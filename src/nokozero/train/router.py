"""A policy that routes each state to a per-segment expert."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nokozero import wire
from nokozero.train.agent import Agent, Checkpoint, Provenance
from nokozero.train.rollout import Inputs, Prefix, Target, parse_word_key

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import numpy.typing as npt

    from nokozero.train.agent import Logits
    from nokozero.train.features import ActionSet, Features, Featurizer
    from nokozero.train.rollout import Pending, Policy

M = wire.Meta


class Router:
    """Dispatches states to experts by chapter word."""

    def __init__(  # noqa: PLR0913
        self,
        experts: dict[int, Checkpoint],
        fallback: Checkpoint,
        splits: dict[int, list[tuple[int, Checkpoint]]],
        occurrences: dict[int, dict[int, Checkpoint]],
        *,
        dither: float = 0.0,
        noise: float = 0.0,
        files: tuple[Path, ...] = (),
    ) -> None:
        self.experts = experts
        self.fallback = fallback
        self.dither = dither
        self.noise = noise
        self.occurrences = occurrences
        self.splits = {word: sorted(pairs, key=lambda p: p[0]) for word, pairs in splits.items()}
        self.files = files

    def expert_for(self, word: int, tic: int, occurrence: int = 1) -> Checkpoint:
        """Return the checkpoint playing `word` at `tic` frames in, on its n-th `occurrence`."""
        by_n = self.occurrences.get(word)
        if by_n is not None and occurrence in by_n:
            return by_n[occurrence]
        chosen = self.experts.get(word, self.fallback)
        for at, expert in self.splits.get(word, ()):
            if tic >= at:
                chosen = expert
        return chosen

    @property
    def featurizer(self) -> Featurizer:
        """Fallback featurizer."""
        return self.fallback.featurizer

    @property
    def actions(self) -> ActionSet:
        """Fallback action set."""
        return self.fallback.actions

    @classmethod
    def load(cls, spec: Path, *, dither: float = 0.0, noise: float = 0.0) -> Router:
        """Read a JSON spec of per-word expert checkpoints and a fallback."""
        data = json.loads(spec.read_text())
        unknown = sorted(set(data) - {"experts", "fallback"})
        if unknown:
            msg = f"{spec}: unknown router keys {unknown} (see `Router.load`)"
            raise ValueError(msg)
        base = spec.parent

        loaded: dict[Path, Checkpoint] = {}

        def checkpoint(path: str) -> Checkpoint:
            resolved = (base / path).resolve()
            if resolved not in loaded:
                loaded[resolved] = Agent.load(resolved, trainable=False)
            return loaded[resolved]

        fallback = checkpoint(data["fallback"])
        experts: dict[int, Checkpoint] = {}
        splits: dict[int, list[tuple[int, Checkpoint]]] = {}
        occurrences: dict[int, dict[int, Checkpoint]] = {}
        for key, path in data["experts"].items():
            try:
                word, nth, tic = parse_word_key(str(key))
            except ValueError as error:
                msg = f"{spec}: {error}"
                raise ValueError(msg) from error
            expert = checkpoint(path)
            fallback.provenance.require_control(expert.provenance, f"the expert for word {key}")
            if tic is not None:
                splits.setdefault(word, []).append((tic, expert))
            elif nth is not None:
                occurrences.setdefault(word, {})[nth] = expert
            else:
                experts[word] = expert
        return cls(
            experts,
            fallback,
            splits,
            occurrences,
            dither=dither,
            noise=noise,
            files=(spec, *sorted(loaded)),
        )

    def __call__(self, inputs: Inputs) -> Pending:
        """Start the experts for each row. Returns the pending actions."""
        actions = np.zeros(len(inputs), dtype=np.int32)
        groups: dict[int, tuple[Checkpoint, list[int]]] = {}
        for row, obs in enumerate(inputs.observations):
            chosen = self.expert_for(obs.meta(M.CHAPTER), obs.meta(M.TIME_IN_CHAPTER), obs.meta(M.ENTRY_COUNT))
            groups.setdefault(id(chosen), (chosen, []))[1].append(row)
        featurized = self._featurize(inputs, groups.values())
        pending: list[tuple[Checkpoint, npt.NDArray[np.intp], Logits]] = []
        for chosen, rows in groups.values():
            index = np.array(rows, dtype=np.intp)
            features, position = featurized[chosen.featurizer]
            pending.append((chosen, index, chosen.agent.dispatch(features[position[index]])))

        def collect() -> npt.NDArray[np.integer]:
            for chosen, index, logits in pending:
                keys = [inputs.keys[int(row)] for row in index]
                actions[index] = chosen.agent.choose(logits, dither=self.dither, noise=self.noise, keys=keys)
            return actions

        return collect

    def _featurize(
        self, inputs: Inputs, groups: Iterable[tuple[Checkpoint, list[int]]]
    ) -> dict[Featurizer, tuple[Features, npt.NDArray[np.intp]]]:
        """Return each featurizer's features and the position of every input row within them."""
        wanted: dict[Featurizer, list[int]] = {}
        for chosen, rows in groups:
            if chosen.featurizer != inputs.featurizer:
                wanted.setdefault(chosen.featurizer, []).extend(rows)
        featurized = {inputs.featurizer: (inputs.features, np.arange(len(inputs)))}
        for featurizer, rows in wanted.items():
            position = np.full(len(inputs), -1, dtype=np.intp)
            position[rows] = np.arange(len(rows))
            observations = [inputs.observations[row] for row in rows]
            featurized[featurizer] = (featurizer.batch(observations), position)
        return featurized


def load_router(spec: Path, provenance: Provenance) -> Router:
    """Load the router at `spec` to play inside a run of `provenance`."""
    router = Router.load(spec)
    provenance.require_control(router.fallback.provenance, str(spec))
    return router


def load_prefix(spec: Path, provenance: Provenance, target: Target) -> Prefix:
    """Load the router at `spec` as a `rollout.Prefix` reaching `target`."""
    return Prefix(load_router(spec, provenance), target)


@dataclass(frozen=True)
class StagePolicy:
    """Policy for a stage or a segment."""

    policy: Policy
    featurizer: Featurizer
    actions: ActionSet
    files: tuple[Path, ...] = ()

    @property
    def provenance(self) -> Provenance:
        """The policy's featurizer and action set."""
        return Provenance(self.featurizer, self.actions.name)

    @property
    def character(self) -> int:
        """The character used by the policy."""
        return self.featurizer.character

    @classmethod
    def load(
        cls,
        path: Path,
        provenance: Provenance | None = None,
        *,
        dither: float = 0.0,
        noise: float = 0.0,
    ) -> StagePolicy:
        """Load a router spec (`.json`) or a checkpoint (`.eqx`) to play."""
        if path.suffix == ".json":
            router = (
                Router.load(path, dither=dither, noise=noise) if provenance is None else load_router(path, provenance)
            )
            return cls(router, router.featurizer, router.actions, router.files)
        loaded = Agent.load(path, trainable=False)
        if provenance is not None:
            provenance.require(loaded.provenance, str(path))
        policy = loaded.agent.greedy(dither=dither, noise=noise)
        return cls(policy, loaded.featurizer, loaded.actions, (path,))
