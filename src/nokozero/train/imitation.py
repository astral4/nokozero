"""Behavior cloning from harvested demonstrations."""

# JAX, Equinox, and Optax only come with partial annotations.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false, reportMissingSuperCall=false

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from jaxtyping import Array  # noqa: TC002

from nokozero import catalog, wire
from nokozero.train.agent import Agent, Provenance, load_network, logits, q_logits, save_network
from nokozero.train.features import ActionSet, Features, Featurizer
from nokozero.train.model import QNet
from nokozero.train.rollout import Episode, Kind, Outcome, Start

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nokozero.train.agent import Explorer
    from nokozero.train.rollout import Pending

_MOVE_MASK = int(wire.Action.UP | wire.Action.DOWN | wire.Action.LEFT | wire.Action.RIGHT)
_FOCUS = int(wire.Action.FOCUS)
_MIN_DEMO_ROWS = 2


@dataclass(frozen=True)
class Demonstrations:
    """A harvested dataset."""

    features: Features
    keys: npt.NDArray[np.uint16]
    featurizer: Featurizer
    bounds: npt.NDArray[np.int64]
    completed: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def load(cls, path: Path) -> Demonstrations:
        """Read a file written by `replay.harvest.save`."""
        with np.load(path) as data:
            feats = Features(data["tokens"].astype(np.float32), data["mask"], data["globals"].astype(np.float32))
            return cls(
                feats,
                data["keys"],
                Featurizer.from_header(data),
                data["bounds"].astype(np.int64),
                data["completed"].astype(bool),
            )


DEMO_START: Final = Start(
    wire.ResetParams(difficulty=catalog.LUNATIC, character=0, rng_seed=0),
    await_chapter=False,
    kind=Kind.REPLAY,
    label="demo",
)


def demonstration_episodes(demos: Demonstrations, actions: ActionSet) -> list[Episode]:
    """Return the demonstrations' segments as episodes for a replay buffer."""
    episodes: list[Episode] = []
    for (lo, hi), completed in zip(pairwise(demos.bounds), demos.completed, strict=True):
        if hi - lo < _MIN_DEMO_ROWS:
            continue
        rows = np.arange(int(lo), int(hi))
        episodes.append(
            Episode(
                DEMO_START,
                Outcome.COMPLETE if completed else Outcome.TRUNCATED,
                demos.features[rows],
                labels(demos.keys[rows[:-1]], actions),
            )
        )
    return episodes


def _pattern_key(pattern: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(w & (_MOVE_MASK | _FOCUS)) for w in pattern)


def labels(keys: npt.NDArray[np.uint16], actions: ActionSet) -> npt.NDArray[np.int32]:
    """Map each row of held keys (`[N, frames]`) to the nearest action of `actions`."""
    moves = keys & np.uint16(_MOVE_MASK)
    opposed = ((moves & wire.Action.UP) != 0) & ((moves & wire.Action.DOWN) != 0)
    opposed |= ((moves & wire.Action.LEFT) != 0) & ((moves & wire.Action.RIGHT) != 0)
    clean = np.where(opposed, keys & np.uint16(_FOCUS), keys & np.uint16(_MOVE_MASK | _FOCUS))
    clean = clean & np.uint16(actions.key_mask)
    table = np.array([_pattern_key(p) for p in actions.patterns], dtype=np.uint16)
    frames = min(clean.shape[1], table.shape[1])
    matches = (clean[:, :frames, None] == table.T[None, :frames]).sum(axis=1)
    return np.argmax(matches, axis=1).astype(np.int32)


class BCPolicy(NamedTuple):
    """A Q-network-like classifier over an action set."""

    net: QNet
    provenance: Provenance


def _loss(net: QNet, tokens: Array, masks: Array, globs: Array, y: Array) -> Array:
    scores = q_logits(net, tokens, masks, globs)
    return optax.softmax_cross_entropy_with_integer_labels(scores, y).mean()


def _stepper(
    static: QNet, optimizer: optax.GradientTransformation
) -> Callable[
    [QNet, optax.OptState, tuple[Array, Array, Array, Array]],
    tuple[QNet, optax.OptState, Array],
]:
    """Return the JIT-compiled training step over the arrays of a policy where `static` is the static part."""

    def step(
        params: QNet, opt_state: optax.OptState, batch: tuple[Array, Array, Array, Array]
    ) -> tuple[QNet, optax.OptState, Array]:
        loss, grads = eqx.filter_value_and_grad(_loss)(eqx.combine(params, static), *batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)  # pyright: ignore[reportArgumentType]
        return eqx.apply_updates(params, updates), opt_state, loss

    return jax.jit(step)


def accuracy(policy: BCPolicy, feats: Features, y: npt.NDArray[np.int32], chunk: int = 4096) -> float:
    """Top-1 agreement with the labels."""
    hits = 0
    for lo in range(0, len(y), chunk):
        scores = logits(policy.net, feats.view(lo, lo + chunk), multiple=chunk)
        hits += int((scores.argmax(axis=1) == y[lo : lo + chunk]).sum())
    return hits / max(1, len(y))


def train(  # noqa: PLR0913
    demos: Demonstrations,
    actions: ActionSet,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    holdout: float = 0.1,
    seed: int = 0,
    log: Callable[[dict[str, float | int]], None] | None = None,
) -> BCPolicy:
    """Fit a `BCPolicy` on `demos`. The last `holdout` fraction of states is held out."""
    y = labels(demos.keys, actions)
    n = len(y)
    split = int(n * (1 - holdout))
    if split < batch_size:
        msg = (
            f"{split} training states after the holdout can't be put into batches of {batch_size}; "
            "use a smaller --batch-size or harvest more"
        )
        raise ValueError(msg)
    train_f, val_f = demos.features.view(0, split), demos.features.view(split, n)
    train_y, val_y = y[:split], y[split:]
    rng = np.random.default_rng(seed)
    provenance = Provenance(demos.featurizer, actions.name)
    featurizer = demos.featurizer
    net = QNet(featurizer.token_dim, featurizer.global_dim, len(actions), key=jax.random.PRNGKey(seed))
    params, static = eqx.partition(net, eqx.is_array)
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)
    step = _stepper(static, optimizer)
    for epoch in range(epochs):
        order = rng.permutation(split)
        losses: list[Array] = []
        for lo in range(0, split - batch_size + 1, batch_size):
            idx = order[lo : lo + batch_size]
            batch = (
                jnp.asarray(train_f.tokens[idx]),
                jnp.asarray(train_f.mask[idx]),
                jnp.asarray(train_f.globals[idx]),
                jnp.asarray(train_y[idx]),
            )
            params, opt_state, loss = step(params, opt_state, batch)
            losses.append(loss)
        if log is not None:
            policy = BCPolicy(eqx.combine(params, static), provenance)
            log(
                {
                    "epoch": epoch,
                    "loss": float(jnp.stack(losses).mean()),
                    "train_acc": accuracy(policy, train_f, train_y),
                    "val_acc": accuracy(policy, val_f, val_y),
                }
            )
    return BCPolicy(eqx.combine(params, static), provenance)


def save(policy: BCPolicy, path: Path) -> None:
    """Write the policy."""
    save_network(path, policy.net, policy.provenance)


def load(path: Path) -> BCPolicy:
    """Read a policy written by `save`."""
    return BCPolicy(*load_network(path))


def sampler(policy: BCPolicy, seed: int = 0, temperature: float = 1.0) -> Explorer:
    """Return an explorer drawing actions from the policy's softmax for a batch of states."""
    network = Agent(policy.net, trainable=False)
    rng = np.random.default_rng(seed)

    def sample(features: Features) -> Pending:
        pending = network.dispatch(features)

        def draw() -> npt.NDArray[np.integer]:
            scores = pending.read() / temperature
            return np.argmax(scores + rng.gumbel(size=scores.shape), axis=1)

        return draw

    return sample
