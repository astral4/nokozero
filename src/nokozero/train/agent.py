"""Double DQN with n-step Bernoulli targets, trained with binary cross-entropy."""

# JAX, Equinox, and Optax only come with partial annotations.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false

import io
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxtyping import Array, Float  # noqa: TC002

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from nokozero.train.rollout import Inputs, Pending, Policy

# ruff: noqa: F722  # jaxtyping shape strings are not forward references

from nokozero.train.buffer import Batch, BoolArray, FloatArray  # noqa: TC001
from nokozero.train.features import (
    ACTION_SETS,
    ActionSet,
    Features,
    Featurizer,
)
from nokozero.train.model import QNet
from nokozero.utils import atomic_write, code_version

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "nokozero" / "jax"))

jax.config.update("jax_default_matmul_precision", "highest")

ACT_BATCH = 8
GRAD_CLIP = 10.0


def padded(features: Features, multiple: int = ACT_BATCH) -> Features:
    """Return `features` with its last row repeated up to a multiple of `multiple` rows."""
    n = len(features)
    pad = (-n) % multiple
    return features[np.minimum(np.arange(n + pad), n - 1)] if pad else features


class Logits(NamedTuple):
    """The in-flight logits of a batch."""

    chunks: tuple[Array, ...]
    rows: int

    def read(self) -> npt.NDArray[np.float32]:
        """Wait for the network and return the logits of each real row."""
        return np.concatenate([np.asarray(chunk) for chunk in self.chunks])[: self.rows]


#: A behavior policy for exploration which starts on a batch of states and returns the pending actions.
type Explorer = Callable[[Features], Pending]


@dataclass(frozen=True)
class Provenance:
    """Featurization and action set for a network."""

    featurizer: Featurizer
    actions: str

    def header(self) -> dict[str, dict[str, Any]]:
        """Return the header blocks recorded by a saved network."""
        fields_ = self.featurizer.header()
        return {
            "featurizer": {name: fields_[name] for name in ("k", "horizon", "variant")},
            "control": {
                "step_interval": fields_["step_interval"],
                "actions": self.actions,
                "character": fields_["character"],
            },
        }

    @classmethod
    def from_header(cls, header: dict[str, Any], what: str) -> Provenance:
        """Read the blocks written by `header`."""
        try:
            control = header["control"]
            featurizer = Featurizer.from_header(
                {
                    **header["featurizer"],
                    "step_interval": control["step_interval"],
                    "character": control["character"],
                }
            )
            return cls(featurizer, str(control["actions"]))
        except KeyError as missing:
            msg = f"{what} does not record {missing.args[0]} in its provenance"
            raise ValueError(msg) from missing

    def require(self, other: Provenance, what: str) -> None:
        """Raise unless `other` matches."""
        pairs = [
            (f.name, getattr(self.featurizer, f.name), getattr(other.featurizer, f.name))
            for f in fields(self.featurizer)
        ]
        pairs.append(("actions", self.actions, other.actions))
        self._require(pairs, what)

    def require_control(self, other: Provenance, what: str) -> None:
        """Raise unless `other` has the same control block."""
        self._require(
            [
                ("step_interval", self.featurizer.step_interval, other.featurizer.step_interval),
                ("character", self.featurizer.character, other.featurizer.character),
                ("actions", self.actions, other.actions),
            ],
            what,
        )

    @staticmethod
    def _require(pairs: list[tuple[str, Any, Any]], what: str) -> None:
        differing = [f"{name} {theirs!r} (the run's is {mine!r})" for name, mine, theirs in pairs if mine != theirs]
        if differing:
            msg = f"{what} was trained under another {', '.join(differing)}"
            raise ValueError(msg)


class Checkpoint(NamedTuple):
    """A model checkpoint."""

    agent: Agent
    provenance: Provenance

    @property
    def featurizer(self) -> Featurizer:
        """The featurizer that the network was trained on."""
        return self.provenance.featurizer

    @property
    def actions(self) -> ActionSet:
        """The action set corresponding to the network's outputs."""
        return ACTION_SETS[self.provenance.actions]


def save_network(path: Path, net: QNet, provenance: Provenance) -> None:
    """Write `net` as a self-describing file (JSON header line, then the network's leaves)."""
    header = {"model": net.hyperparameters, **provenance.header(), "code": code_version()}
    buffer = io.BytesIO()
    buffer.write((json.dumps(header) + "\n").encode())
    eqx.tree_serialise_leaves(buffer, net)
    atomic_write(path, buffer.getvalue())


def load_network(path: Path) -> tuple[QNet, Provenance]:
    """Read a network file written by `save_network`."""
    with path.open("rb") as file:
        try:
            header: dict[str, Any] = json.loads(file.readline())
        except UnicodeDecodeError, ValueError:
            msg = f"{path} is not a valid network file: no header line"
            raise ValueError(msg) from None
        provenance = Provenance.from_header(header, str(path))
        like = eqx.filter_eval_shape(QNet, **header["model"], key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(file, like), provenance


@dataclass
class Metrics:
    """Scalars from an update."""

    loss: Array
    mean_q: Array
    mean_target: Array


def q_logits(model: QNet, tokens: FloatArray, masks: BoolArray, globs: FloatArray) -> Float[Array, "n a"]:
    """Batched action logits."""
    params, static = eqx.partition(model, eqx.is_array)
    return _programs(static).forward(params, tokens, masks, globs)


def logits(model: QNet, features: Features, *, multiple: int = ACT_BATCH) -> npt.NDArray[np.float32]:
    """Return each row's action logits."""
    rows = padded(features, multiple)
    return np.asarray(q_logits(model, rows.tokens, rows.mask, rows.globals))[: len(features)]


def targets(batch: Batch, bootstrap: FloatArray) -> Float[Array, " n"]:
    """Return the n-step Bernoulli targets `1 - gamma^m + gamma^m * z`.

    `z` is the window's terminal value if it reached a real terminal, else `bootstrap`.
    So, a completed segment is `1` and a hit after `m` steps is `1 - gamma^m`.
    """
    discount = jnp.asarray(batch.discount)
    z = jnp.where(batch.done, batch.value, bootstrap)
    return jnp.clip(1.0 - discount + discount * z + jnp.asarray(batch.bonus), 0.0, 1.0)


def _loss(model: QNet, target: QNet, batch: Batch) -> tuple[Array, tuple[Array, Array]]:
    """Return the cross-entropy against the double-DQN targets."""
    tokens, next_tokens = batch.tokens.astype(jnp.float32), batch.next_tokens.astype(jnp.float32)
    chosen = jnp.take_along_axis(q_logits(model, tokens, batch.masks, batch.globals), batch.actions[:, None], axis=-1)[
        :, 0
    ]
    a_star = jnp.argmax(q_logits(model, next_tokens, batch.next_masks, batch.next_globals), axis=-1)
    next_target = q_logits(target, next_tokens, batch.next_masks, batch.next_globals)
    bootstrap = jax.nn.sigmoid(jnp.take_along_axis(next_target, a_star[:, None], axis=-1)[:, 0])
    y = jax.lax.stop_gradient(targets(batch, bootstrap))
    loss = optax.sigmoid_binary_cross_entropy(chosen, y).mean()
    return loss, (jax.nn.sigmoid(chosen).mean(), y.mean())


def _polyak(slow: QNet, fast: QNet, rate: float) -> QNet:
    """Move the arrays `slow` toward `fast` by `rate`."""
    return jax.tree.map(lambda s, f: (1.0 - rate) * s + rate * f, slow, fast)


class _Programs:
    """JIT-compiled programs for the structure of a network."""

    def __init__(self, static: QNet) -> None:
        def forward(params: QNet, tokens: FloatArray, masks: BoolArray, globs: FloatArray) -> Float[Array, "n a"]:
            model = eqx.combine(params, static)
            return jax.vmap(model)(jnp.asarray(tokens), jnp.asarray(masks), jnp.asarray(globs))

        def update(  # noqa: PLR0913, PLR0917
            online: QNet,
            target: QNet,
            ema: QNet,
            opt_state: optax.OptState,
            batch: Batch,
            optimizer: optax.GradientTransformation,
            tau: float,
            ema_tau: float,
        ) -> tuple[QNet, QNet, QNet, optax.OptState, Array, Array, Array]:
            """One gradient step, then the target's and the average's Polyak steps."""
            (loss, (mean_q, mean_y)), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(
                eqx.combine(online, static), eqx.combine(target, static), batch
            )
            updates, opt_state = optimizer.update(grads, opt_state, online)  # pyright: ignore[reportArgumentType]
            online = eqx.apply_updates(online, updates)
            return (
                online,
                _polyak(target, online, tau),
                _polyak(ema, online, ema_tau),
                opt_state,
                loss,
                mean_q,
                mean_y,
            )

        self.forward = jax.jit(forward)
        self.update = jax.jit(update, static_argnums=(5, 6, 7))


_PROGRAMS: dict[tuple[Any, ...], _Programs] = {}


def _programs(static: QNet) -> _Programs:
    """Return the programs of the structure where `static` is the static portion."""
    leaves, treedef = jax.tree.flatten(static)
    key = (treedef, *leaves)
    programs = _PROGRAMS.get(key)
    if programs is None:
        programs = _PROGRAMS[key] = _Programs(static)
    return programs


class Agent:
    """Networks and optimizer state."""

    def __init__(  # noqa: PLR0913
        self,
        model: QNet,
        *,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        decay_updates: int | None = None,
        ema_tau: float = 0.0002,
        trainable: bool = True,
    ) -> None:
        """The learning rate is decayed linearly to 0.1 * lr over `decay_updates` updates.

        `ema_tau` is the rate of a slow moving average of the weights kept for evaluation (`ema`).
        """
        online, self._static = eqx.partition(model, eqx.is_array)
        self._programs = _programs(self._static)
        # online, target, averaged networks' arrays
        self._online = self._target = self._ema = online
        self.ema_tau = ema_tau
        self.tau = tau
        self.optimizer: optax.GradientTransformation | None = None
        self.opt_state: optax.OptState | None = None
        if trainable:
            schedule: float | optax.Schedule = learning_rate
            if decay_updates is not None:
                schedule = optax.linear_schedule(learning_rate, learning_rate / 10, decay_updates)
            self.optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(schedule))
            self.opt_state = self.optimizer.init(online)

    @property
    def model(self) -> QNet:
        """The online network."""
        return eqx.combine(self._online, self._static)

    @property
    def ema(self) -> QNet:
        """The slow moving average of the online network's weights (`ema_tau`)."""
        return eqx.combine(self._ema, self._static)

    def act(
        self,
        features: Features,
        epsilon: float,
        rng: np.random.Generator,
        explore: Explorer | None = None,
    ) -> npt.NDArray[np.integer]:
        """Epsilon-greedy actions for a batch of states."""
        return self.decide(features, epsilon, rng, explore)()

    def decide(
        self,
        features: Features,
        epsilon: float,
        rng: np.random.Generator,
        explore: Explorer | None = None,
    ) -> Pending:
        """Start the epsilon-greedy choice for a batch of states.

        A state's action comes from `explore` with probability `epsilon`.
        """
        pending = self.dispatch(features)
        n = len(features)
        exploring = rng.random(n) < epsilon if epsilon > 0 else np.zeros(n, dtype=np.bool_)
        alternatives: Pending | None = None
        if exploring.any():
            if explore is not None:
                alternatives = explore(features)
            else:
                drawn = rng.integers(0, self._static.num_actions, size=n)
                alternatives = lambda: drawn  # noqa: E731

        def collect() -> npt.NDArray[np.integer]:
            greedy = self.choose(pending)
            return greedy if alternatives is None else np.where(exploring, alternatives(), greedy)

        return collect

    def dispatch(self, features: Features) -> Logits:
        """Start the online network on `features`. Returns its logits on the device."""
        rows = padded(features)
        chunks = tuple(
            self._programs.forward(
                self._online,
                rows.tokens[at : at + ACT_BATCH],
                rows.mask[at : at + ACT_BATCH],
                rows.globals[at : at + ACT_BATCH],
            )
            for at in range(0, len(rows), ACT_BATCH)
        )
        return Logits(chunks, len(features))

    def choose(
        self,
        pending: Logits,
        *,
        dither: float = 0.0,
        noise: float = 0.0,
        keys: Sequence[Sequence[int]] = (),
    ) -> npt.NDArray[np.integer]:
        """Return the greedy choice from the logits returned by `dispatch`.

        If `dither` > 0, then each state's action is uniformly drawn from those scoring within `dither` of its best.
        If `noise` > 0, then the corresponding amount of Gaussian noise is added to each score before the choice.
        """
        scores = pending.read()
        if dither <= 0 and noise <= 0:
            return scores.argmax(axis=1).astype(np.int32)
        if len(keys) != len(scores):
            msg = f"draw selection is based on each row's key: {len(keys)} for {len(scores)}"
            raise ValueError(msg)
        rngs = [np.random.default_rng(list(k)) for k in keys]
        if noise > 0:
            scores = scores + np.stack([r.normal(0.0, noise, size=scores.shape[1]) for r in rngs])
        near = scores >= scores.max(axis=1, keepdims=True) - dither
        return np.array(
            [r.choice(np.flatnonzero(row)) for r, row in zip(rngs, near, strict=True)],
            dtype=np.int32,
        )

    def policy(self, epsilon: float, rng: np.random.Generator, explore: Explorer | None = None) -> Policy:
        """Return this agent as an epsilon-greedy `rollout.Policy`."""

        def policy(inputs: Inputs) -> Pending:
            return self.decide(inputs.features, epsilon, rng, explore)

        return policy

    def greedy(self, *, dither: float = 0.0, noise: float = 0.0) -> Policy:
        """Return this agent as a `rollout.Policy` playing its best action.

        See `choose` for documentation on `dither` and `noise`.
        """

        def policy(inputs: Inputs) -> Pending:
            pending = self.dispatch(inputs.features)
            return lambda: self.choose(pending, dither=dither, noise=noise, keys=inputs.keys)

        return policy

    def update(self, batch: Batch) -> Metrics:
        """Perform a gradient step on `batch`, then the Polyak updates of the target and the average."""
        if self.optimizer is None or self.opt_state is None:
            msg = "this agent was only built for actions (trainable=False) and cannot update"
            raise RuntimeError(msg)
        (self._online, self._target, self._ema, self.opt_state, loss, mean_q, mean_y) = self._programs.update(
            self._online,
            self._target,
            self._ema,
            self.opt_state,
            jax.device_put(batch),
            self.optimizer,
            self.tau,
            self.ema_tau,
        )
        return Metrics(loss=loss, mean_q=mean_q, mean_target=mean_y)

    def save(self, path: str | Path, provenance: Provenance, *, averaged: bool = False) -> None:
        """Write a self-describing checkpoint (JSON header line, then the network's leaves)."""
        save_network(Path(path), self.ema if averaged else self.model, provenance)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        decay_updates: int | None = None,
        trainable: bool = True,
    ) -> Checkpoint:
        """Rebuild the agent and the provenance described by a checkpoint."""
        model, provenance = load_network(Path(path))
        agent = cls(
            model,
            learning_rate=learning_rate,
            tau=tau,
            decay_updates=decay_updates,
            trainable=trainable,
        )
        return Checkpoint(agent, provenance)
