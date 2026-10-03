"""A replay buffer of contiguous episodes with n-step Bernoulli targets.

Episodes are stored as contiguous rows (`T` decision rows followed by the terminal observation's row).
So, a transition's n-step successor is the row `n` ahead, clipped to the end of the episode.

Every target is of the form `1 - gamma^m + gamma^m * z`: the probability of outliving a geometric horizon
with mean `1 / (1 - gamma)`, which expires with probability `1 - gamma` at each step.
Completed segments win outright (`1`), and hit-terminated segments still win if the horizon expired
within its `m` steps (`1 - gamma^m`).
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Final

import jax
import numpy as np
import numpy.typing as npt

from nokozero.train.features import BOSS_HP_GLOBAL
from nokozero.train.rollout import Episode, Kind, Outcome

if TYPE_CHECKING:
    from collections.abc import Callable

#: Batch fields are numpy arrays when sampled and JAX arrays when on the device.
FloatArray = npt.NDArray[np.float32] | jax.Array
HalfArray = npt.NDArray[np.float16] | jax.Array
BoolArray = npt.NDArray[np.bool_] | jax.Array
IntArray = npt.NDArray[np.int32] | jax.Array


@dataclass(frozen=True)
class Batch:
    """A sampled batch of n-step transitions."""

    tokens: HalfArray
    masks: BoolArray
    globals: FloatArray
    actions: IntArray
    #: Whether the n-step window hit a real terminal, and its value if so.
    done: BoolArray
    value: FloatArray
    #: `gamma^m` for the `m` surviving steps of the window.
    discount: FloatArray
    #: The window's damage bonus; i.e. `damage_bonus` times the boss HP ratio taken.
    bonus: FloatArray
    next_tokens: HalfArray
    next_masks: BoolArray
    next_globals: FloatArray


jax.tree_util.register_dataclass(Batch, data_fields=[f.name for f in fields(Batch)], meta_fields=[])


class _IndexSet:
    """A set of row indices below `capacity` with O(1) insertion, deletion, and uniform sampling."""

    def __init__(self, capacity: int) -> None:
        self._rows = np.empty(capacity, dtype=np.int32)
        self._pos = np.full(capacity, -1, dtype=np.int32)
        self.size = 0

    def add(self, rows: npt.NDArray[np.integer]) -> None:
        """Add `rows` (distinct). Already-present members are left alone."""
        rows = rows[self._pos[rows] < 0]
        n = len(rows)
        self._rows[self.size : self.size + n] = rows
        self._pos[rows] = np.arange(self.size, self.size + n, dtype=np.int32)
        self.size += n

    def remove(self, rows: npt.NDArray[np.integer]) -> None:
        """Remove `rows` (distinct). Non-members are ignored."""
        rows = rows[self._pos[rows] >= 0]
        n = len(rows)
        if not n:
            return
        vacated = self._pos[rows]
        self._pos[rows] = -1
        tail = self._rows[self.size - n : self.size]
        movers = tail[self._pos[tail] >= 0]
        holes = vacated[vacated < self.size - n]
        self._rows[holes] = movers
        self._pos[movers] = holes
        self.size -= n

    def contains(self, rows: npt.NDArray[np.integer]) -> npt.NDArray[np.bool_]:
        """Query membership of each of `rows`."""
        return self._pos[rows] >= 0

    def sample(self, rng: np.random.Generator, n: int) -> npt.NDArray[np.int64]:
        """Draw `n` members uniformly with replacement."""
        if n <= 0 or not self.size:
            return np.empty(0, dtype=np.int64)
        return self._rows[rng.integers(self.size, size=n)].astype(np.int64)


class ReplayBuffer:
    """Ring buffer of episode rows."""

    def __init__(  # noqa: PLR0913
        self,
        capacity: int,
        k: int,
        token_dim: int,
        global_dim: int,
        *,
        n_step: int = 5,
        hit_window: int = 20,
        gamma: float = 0.995,
        damage_bonus: float = 0.0,
    ) -> None:
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        if damage_bonus and global_dim <= BOSS_HP_GLOBAL:
            msg = f"a damage bonus reads global {BOSS_HP_GLOBAL}; the globals are {global_dim} wide"
            raise ValueError(msg)
        self.damage_bonus = damage_bonus
        self.hit_window = hit_window
        self.tokens = np.zeros((capacity, k, token_dim), dtype=np.float16)
        self.masks = np.zeros((capacity, k), dtype=np.bool_)
        self.globals = np.zeros((capacity, global_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.episode_start = np.zeros(capacity, dtype=np.int64)
        self.episode_end = np.zeros(capacity, dtype=np.int64)
        self.terminal_value = np.zeros(capacity, dtype=np.float32)
        self.bootstrap_end = np.zeros(capacity, dtype=np.bool_)
        self.valid = np.zeros(capacity, dtype=np.bool_)
        self.seed_episode = np.zeros(capacity, dtype=np.bool_)
        self._seed = _IndexSet(capacity)
        self._other = _IndexSet(capacity)
        self._hit = _IndexSet(capacity)
        self.pos = 0

    @property
    def size(self) -> int:
        """The number of decision rows that can be sampled."""
        return self._seed.size + self._other.size

    def add(self, episode: Episode, terminal_value: float | None = None) -> None:
        """Append an episode. The episode must fit in the buffer."""
        states = episode.states
        rows = episode.length + 1
        if rows > self.capacity:
            msg = f"episode of {rows} rows exceeds capacity {self.capacity}"
            raise ValueError(msg)
        if self.pos + rows > self.capacity:
            self._invalidate(self.pos, self.capacity)
            self.pos = 0
        start, end = self.pos, self.pos + rows - 1
        self._invalidate(start, end + 1)
        self.tokens[start : end + 1] = states.tokens
        self.masks[start : end + 1] = states.mask
        self.globals[start : end + 1] = states.globals
        self.actions[start:end] = episode.actions
        self.actions[end] = -1
        self.episode_start[start : end + 1] = start
        self.episode_end[start : end + 1] = end
        if episode.outcome is Outcome.COMPLETE:
            self.terminal_value[start : end + 1] = 1.0 if terminal_value is None else terminal_value
        else:
            self.terminal_value[start : end + 1] = 0.0
        self.bootstrap_end[start : end + 1] = episode.outcome is Outcome.TRUNCATED
        self.valid[start:end] = True
        self.valid[end] = False
        seed = episode.start.kind is Kind.SEED
        self.seed_episode[start : end + 1] = seed
        decisions = np.arange(start, end)
        (self._seed if seed else self._other).add(decisions)
        if episode.outcome is Outcome.HIT:
            self._hit.add(np.arange(max(start, end - self.hit_window), end))
        self.pos = end + 1

    def _invalidate(self, lo: int, hi: int) -> None:
        """Invalidate rows `[lo, hi)` and every row of the episodes they belong to."""
        if hi <= lo:
            return
        stale = self.valid[lo:hi]
        if stale.any():
            starts = self.episode_start[lo:hi][stale]
            ends = self.episode_end[lo:hi][stale]
            lo = min(lo, int(starts.min()))
            hi = max(hi, int(ends.max()) + 1)
        self.valid[lo:hi] = False
        rows = np.arange(lo, hi)
        for stratum in (self._seed, self._other, self._hit):
            stratum.remove(rows)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        hit_fraction: float = 0.25,
        seed_fraction: float | None = None,
    ) -> Batch:
        """Sample transitions. `hit_fraction` determines the proportion of transitions from steps just before hits.

        `seed_fraction` determines the proportion of remaining transitions/rows that come from seed episodes
        versus curriculum episodes. If unset, the remaining rows are sampled uniformly.
        """
        n_valid = self.size
        if n_valid == 0:
            msg = "empty buffer"
            raise ValueError(msg)
        n_hit = min(int(batch_size * hit_fraction), self._hit.size)
        rest = batch_size - n_hit
        if seed_fraction is not None and self._seed.size and self._other.size:
            n_seed = int(rest * seed_fraction)
        else:
            n_seed = int(rng.binomial(rest, self._seed.size / n_valid))
        picks = [self._seed.sample(rng, n_seed), self._other.sample(rng, rest - n_seed)]
        if n_hit:
            picks.append(self._hit.sample(rng, n_hit))
        idx = np.concatenate(picks)
        end = self.episode_end[idx]
        nxt = np.minimum(idx + self.n_step, end)
        reaches_end = nxt == end
        done = reaches_end & ~self.bootstrap_end[idx]
        value = np.where(done, self.terminal_value[idx], 0.0).astype(np.float32)
        discount = (self.gamma ** (nxt - idx)).astype(np.float32)
        bonus = np.zeros(len(idx), dtype=np.float32)
        if self.damage_bonus:
            taken = self.globals[idx, BOSS_HP_GLOBAL] - self.globals[nxt, BOSS_HP_GLOBAL]
            bonus = (self.damage_bonus * np.clip(taken, 0.0, 1.0)).astype(np.float32)
        return Batch(
            tokens=self.tokens[idx],
            masks=self.masks[idx],
            globals=self.globals[idx],
            actions=self.actions[idx],
            done=done,
            value=value,
            discount=discount,
            bonus=bonus,
            next_tokens=self.tokens[nxt],
            next_masks=self.masks[nxt],
            next_globals=self.globals[nxt],
        )


ARCHIVE_FRACTION: Final = 0.5


class TieredBuffer:
    """A recent window plus an archive that keeps a subsample of the entire run.

    The archive adds each finished episode with probability `min(1, rate / rows_seen)`,
    where `rate` is half its capacity. `sample` draws `ARCHIVE_FRACTION` of every batch from the archive
    and the rest from the recent window, each with its own hit stratification.
    """

    def __init__(
        self,
        tier: Callable[[int], ReplayBuffer],
        capacity: int,
        archive_capacity: int,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.recent = tier(capacity)
        self.archive = tier(archive_capacity)
        self.rate = archive_capacity / 2
        self.rows_seen = 0
        self.admitted = 0
        self._rng = rng

    @property
    def size(self) -> int:
        """The number of rows across both tiers that can be sampled."""
        return self.recent.size + self.archive.size

    def admission_probability(self) -> float:
        """The probability that the next finished episode will enter the archive."""
        return min(1.0, self.rate / max(1, self.rows_seen))

    def add(self, episode: Episode, terminal_value: float | None = None) -> None:
        """Append to the recent window and possibly the archive."""
        self.recent.add(episode, terminal_value)
        if self._rng.random() < self.admission_probability():
            self.archive.add(episode, terminal_value)
            self.admitted += episode.length + 1
        self.rows_seen += episode.length + 1

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        hit_fraction: float = 0.25,
        seed_fraction: float | None = None,
    ) -> Batch:
        """Sample from both tiers and join the batches."""
        n_archive = int(batch_size * ARCHIVE_FRACTION) if self.archive.size else 0
        parts = [self.recent.sample(batch_size - n_archive, rng, hit_fraction, seed_fraction)]
        if n_archive:
            parts.append(self.archive.sample(n_archive, rng, hit_fraction, seed_fraction))
        if len(parts) == 1:
            return parts[0]
        joined: Batch = jax.tree.map(lambda *arrays: np.concatenate(arrays), *parts)
        return joined
