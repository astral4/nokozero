"""Features from observations.

An observation contains a variable number of entity rows. The policy sees at most `K` entity tokens
chosen by how soon and how closely each entity approaches the player under its current motion/trajectory,
plus a small vector of global features. Coordinates are relative to the player and scaled to the playfield,
which spans 384 x 448 units with the player starting at (0, 400).
"""

from dataclasses import asdict, dataclass
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt

from nokozero import wire

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

B, E, IT, M = wire.BulletCol, wire.EnemyCol, wire.ItemCol, wire.Meta
SL, RL, CN = wire.SegmentLaserCol, wire.RayLaserCol, wire.CurveNodeCol

POS_SCALE = 224.0
VEL_SCALE = 8.0
RADIUS_SCALE = 16.0


class TokenType(IntEnum):
    """One-hot token types."""

    BULLET = 0
    ENEMY = 1
    CURVE_NODE = 2
    SEGMENT_LASER = 3
    RAY_LASER = 4


_POS: Final = slice(len(TokenType), len(TokenType) + 2)
_VEL: Final = slice(len(TokenType) + 2, len(TokenType) + 4)
_RADIUS: Final = len(TokenType) + 4
_EXTRAS: Final = (len(TokenType) + 5, len(TokenType) + 6)
TOKEN_DIM = len(TokenType) + 7
GLOBAL_DIM = 12
BOSS_HP_GLOBAL: Final = 11
HALF_W, HALF_H = 192.0, 224.0


@dataclass(frozen=True)
class _Blocks:
    """Features added on top of `v1`'s tokens and globals."""

    lookahead: bool = False
    beams: bool = False
    items: bool = False
    power: bool = False


_VARIANTS: Final = {
    "v1": _Blocks(),
    "v2": _Blocks(lookahead=True),
    "v3": _Blocks(beams=True),
    "v4": _Blocks(lookahead=True, beams=True),
    "v5": _Blocks(lookahead=True, beams=True, items=True),
    "v6": _Blocks(lookahead=True, beams=True, items=True, power=True),
}
VARIANTS: Final = tuple(_VARIANTS)
POWER_SCALE: Final = 400.0
ITEM_SLOTS: Final = 4
ITEM_DIM: Final = ITEM_SLOTS * 3 + 1
_BEAMS: Final = (TokenType.SEGMENT_LASER, TokenType.RAY_LASER)
_APPROACH_DIM: Final = 2
DEFAULT_HORIZON: Final = 60.0
PLAYER_SPEED, PLAYER_SPEED_FOCUSED = 4.5, 2.0
LOOKAHEAD_CHARACTER: Final = 0
CLEARANCE_SCALE: Final = 32.0


@dataclass(frozen=True)
class Features:
    """A batch of `N` states: `tokens[N, K, TOKEN_DIM]`, `mask[N, K]`, `globals[N, GLOBAL_DIM]`."""

    tokens: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    globals: npt.NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int | npt.NDArray[np.integer]) -> Features:
        """Return the state at `index` (or the states at an array of indices) as a batch."""
        rows = np.atleast_1d(index)
        return Features(self.tokens[rows], self.mask[rows], self.globals[rows])

    def view(self, lo: int, hi: int) -> Features:
        """Return the states `[lo, hi)` as views into this batch."""
        return Features(self.tokens[lo:hi], self.mask[lo:hi], self.globals[lo:hi])

    @classmethod
    def concatenate(cls, parts: Sequence[Features]) -> Features:
        """Join the batches of `parts` end to end in order."""
        return cls(
            np.concatenate([part.tokens for part in parts]),
            np.concatenate([part.mask for part in parts]),
            np.concatenate([part.globals for part in parts]),
        )


@dataclass(frozen=True)
class _Spec:
    """Token mapping for an entity table."""

    kind: TokenType
    entity: wire.Entity
    pos: tuple[int, int]
    vel: tuple[int, int]
    radius: int
    radius_factor: float = 1.0
    extras: tuple[tuple[int, float], ...] = ()


_SPECS: Final = (
    _Spec(TokenType.BULLET, wire.Entity.BULLETS, (B.POS_X, B.POS_Y), (B.VEL_X, B.VEL_Y), B.SIZE_W),
    _Spec(
        TokenType.ENEMY,
        wire.Entity.ENEMIES,
        (E.POS_X, E.POS_Y),
        (E.VEL_X, E.VEL_Y),
        E.HITBOX_W,
        0.5,
        ((E.IS_LETHAL, 1.0), (E.IS_BOSS, 1.0)),
    ),
    _Spec(
        TokenType.CURVE_NODE,
        wire.Entity.CURVE_NODES,
        (CN.POS_X, CN.POS_Y),
        (CN.VEL_X, CN.VEL_Y),
        CN.WIDTH,
        0.5,
    ),
    _Spec(
        TokenType.SEGMENT_LASER,
        wire.Entity.SEGMENT_LASERS,
        (SL.HEAD_X, SL.HEAD_Y),
        (SL.VEL_X, SL.VEL_Y),
        SL.WIDTH,
        0.5,
        ((SL.LENGTH, 1.0 / POS_SCALE),),
    ),
    _Spec(
        TokenType.RAY_LASER,
        wire.Entity.RAY_LASERS,
        (RL.ORIGIN_X, RL.ORIGIN_Y),
        (RL.ORIGIN_VEL_X, RL.ORIGIN_VEL_Y),
        RL.WIDTH,
        0.5,
        ((RL.COS_ANGLE, 1.0), (RL.SIN_ANGLE, 1.0)),
    ),
)

_STATIONARY = 1e-9 / VEL_SCALE**2


def _dot(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Dot product over the last axis."""
    return 0.0 + a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _norm(a: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Lengths over the last axis."""
    return np.sqrt(_dot(a, a))


def _approach(
    rel: npt.NDArray[np.float32], vel: npt.NDArray[np.float32], horizon: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Distance of closest approach to the origin within `horizon` time units, plus its time."""
    vv = _dot(vel, vel)
    pv = _dot(rel, vel)
    t = np.where(vv > _STATIONARY, -pv / np.maximum(vv, _STATIONARY), 0.0)
    t = np.clip(t, 0.0, horizon).astype(np.float32)
    closest = rel + vel * t[:, None]
    return _norm(closest).astype(np.float32), t


class _Table(NamedTuple):
    """An entity table."""

    rows: npt.NDArray[np.float32]
    owner: npt.NDArray[np.intp]
    counts: npt.NDArray[np.intp]


def _gather(entity: wire.Entity, observations: Sequence[wire.Observation]) -> _Table:
    tables = [obs.sections[entity] for obs in observations]
    counts = np.fromiter((len(table) for table in tables), dtype=np.intp, count=len(tables))
    rows = tables[0] if len(tables) == 1 else np.concatenate(tables)
    return _Table(rows, np.repeat(np.arange(len(tables)), counts), counts)


@dataclass(frozen=True)
class Featurizer:
    """Builds `Features` with a fixed token budget `K`."""

    k: int = 96
    horizon: float = DEFAULT_HORIZON
    variant: str = "v1"
    step_interval: int = wire.DEFAULT_STEP_INTERVAL
    character: int = LOOKAHEAD_CHARACTER

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            msg = f"unknown feature set {self.variant!r}; one of {VARIANTS}"
            raise ValueError(msg)
        if self._blocks.lookahead and self.character != LOOKAHEAD_CHARACTER:
            msg = (
                f"the {self.variant} lookahead is built on character {LOOKAHEAD_CHARACTER}'s speed; "
                f"character {self.character} needs its own (see PLAYER_SPEED)"
            )
            raise ValueError(msg)

    def header(self) -> dict[str, Any]:
        """Return the fields for building this featurizer instance."""
        return asdict(self)

    @classmethod
    def from_header(cls, header: Mapping[str, Any]) -> Featurizer:
        """Rebuild a featurizer from `header`'s fields."""
        return cls(
            k=int(header["k"]),
            horizon=float(header["horizon"]),
            variant=str(header["variant"]),
            step_interval=int(header["step_interval"]),
            character=int(header["character"]),
        )

    @property
    def _horizon(self) -> float:
        return self.horizon * VEL_SCALE / POS_SCALE

    @property
    def _blocks(self) -> _Blocks:
        """Additions to the `v1` feature set."""
        return _VARIANTS[self.variant]

    @property
    def token_dim(self) -> int:
        """Features per token."""
        return TOKEN_DIM + (_APPROACH_DIM if self._blocks.lookahead else 0)

    @property
    def global_dim(self) -> int:
        """Global features."""
        return (
            GLOBAL_DIM
            + (len(DIRECTIONS) * 2 if self._blocks.lookahead else 0)
            + (ITEM_DIM if self._blocks.items else 0)
            + (1 if self._blocks.power else 0)
        )

    def __call__(self, obs: wire.Observation) -> Features:
        """Featurize one observation."""
        return self.batch([obs])

    def batch(self, observations: Sequence[wire.Observation]) -> Features:
        """Featurize every observation."""
        meta = np.stack([obs.words[: wire.META_WORDS] for obs in observations])
        players = meta.view(np.float32)[:, [M.PLAYER_X, M.PLAYER_Y]]
        tables = {spec.entity: _gather(spec.entity, observations) for spec in _SPECS}
        tokens, mask = self._tokens(tables, players)
        glob = _globals(meta, tables, players)
        if self._blocks.lookahead:
            radii = meta.view(np.float32)[:, M.PLAYER_HIT_RADIUS]
            look = _lookahead(tables, players, radii, self.step_interval, beams=self._blocks.beams)
            glob = np.concatenate([glob, look], axis=1)
        if self._blocks.items:
            items = _gather(wire.Entity.ITEMS, observations)
            glob = np.concatenate([glob, _item_globals(items, players)], axis=1)
        if self._blocks.power:
            power = (meta.view(np.int32)[:, M.POWER].astype(np.float32) / POWER_SCALE)[:, None]
            glob = np.concatenate([glob, power], axis=1)
        return Features(tokens, mask, glob)

    def _tokens(
        self, tables: dict[wire.Entity, _Table], players: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        n = len(players)
        parts: list[npt.NDArray[np.float32]] = []
        owners: list[npt.NDArray[np.intp]] = []
        for spec in _SPECS:
            rows, owner, _ = tables[spec.entity]
            if not len(rows):
                continue
            out = np.zeros((len(rows), TOKEN_DIM), dtype=np.float32)
            out[:, spec.kind] = 1.0
            if self._blocks.beams and spec.kind in _BEAMS:
                out[:, _POS] = _beam_offsets(spec.kind, rows, players[owner]) / POS_SCALE
            else:
                out[:, _POS] = (rows[:, list(spec.pos)] - players[owner]) / POS_SCALE
            out[:, _VEL] = rows[:, list(spec.vel)] / VEL_SCALE
            out[:, _RADIUS] = rows[:, spec.radius] * (spec.radius_factor / RADIUS_SCALE)
            for slot, (column, scale) in zip(_EXTRAS, spec.extras, strict=False):
                out[:, slot] = rows[:, column] * scale
            parts.append(out)
            owners.append(owner)

        tokens = np.zeros((n, self.k, self.token_dim), dtype=np.float32)
        mask = np.zeros((n, self.k), dtype=np.bool_)
        if not parts:
            return tokens, mask
        allrows = np.concatenate(parts)
        owner = np.concatenate(owners)
        rel, vel = allrows[:, _POS], allrows[:, _VEL]
        closest, when = _approach(rel, vel, self._horizon)
        score = closest + 0.1 * _norm(rel)
        approach = (
            np.stack([closest, when / self._horizon], axis=1).astype(np.float32) if self._blocks.lookahead else None
        )
        order = np.argsort(owner, kind="stable")
        bounds = np.searchsorted(owner[order], np.arange(n + 1))
        for i in range(n):
            mine = order[bounds[i] : bounds[i + 1]]
            if len(mine) > self.k:
                mine = mine[np.argpartition(score[mine], self.k)[: self.k]]
            tokens[i, : len(mine), :TOKEN_DIM] = allrows[mine]
            if approach is not None:
                tokens[i, : len(mine), TOKEN_DIM:] = approach[mine]
            mask[i, : len(mine)] = True
        return tokens, mask


def _beam_offsets(
    kind: TokenType, rows: npt.NDArray[np.float32], players: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Return the offset from each player to the closest point of its beam."""
    if kind is TokenType.SEGMENT_LASER:
        head = rows[:, [SL.HEAD_X, SL.HEAD_Y]]
        unit = _segment_units(rows[:, [SL.VEL_X, SL.VEL_Y]])
        back = np.clip(-_dot(players - head, unit), 0.0, rows[:, SL.LENGTH])
        return (head - unit * back[:, None] - players).astype(np.float32)
    origin = rows[:, [RL.ORIGIN_X, RL.ORIGIN_Y]]
    direction = rows[:, [RL.COS_ANGLE, RL.SIN_ANGLE]]
    along = np.maximum(_dot(players - origin, direction), 0.0)
    return (origin + direction * along[:, None] - players).astype(np.float32)


_STATIONARY_SPEED: Final = 1e-6


def _segment_units(vel: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Each segment laser's unit direction of travel, based on its velocity."""
    speed = _norm(vel)
    return np.where((speed > _STATIONARY_SPEED)[:, None], vel / np.maximum(speed, _STATIONARY_SPEED)[:, None], 0.0)


def _globals(
    meta: npt.NDArray[np.uint32],
    tables: dict[wire.Entity, _Table],
    players: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    n = len(meta)
    meta_f32, meta_i32 = meta.view(np.float32), meta.view(np.int32)
    enemies = tables[wire.Entity.ENEMIES]
    lethal = np.bincount(enemies.owner[enemies.rows[:, E.IS_LETHAL] > 0], minlength=n)
    glob = np.zeros((n, GLOBAL_DIM), dtype=np.float32)
    glob[:, 0] = players[:, 0] / HALF_W
    glob[:, 1] = (players[:, 1] - HALF_H) / HALF_H
    glob[:, 2] = meta_f32[:, M.PLAYER_FOCUSED]
    glob[:, 3] = meta_f32[:, M.PLAYER_HIT_RADIUS] / 4.0
    glob[:, 4] = meta[:, M.TIME_IN_CHAPTER] / 3600.0
    glob[:, 5] = meta[:, M.SPELL_FLAGS] != 0
    glob[:, 6] = meta_i32[:, M.SPELL_TIMER] / 3600.0
    glob[:, 7] = tables[wire.Entity.BULLETS].counts / 200.0
    glob[:, 8] = lethal / 20.0
    boss_rows = np.flatnonzero(enemies.rows[:, E.IS_BOSS] > 0)
    state, first = np.unique(enemies.owner[boss_rows], return_index=True)
    boss = enemies.rows[boss_rows[first]]
    glob[state, 9:11] = (boss[:, [E.POS_X, E.POS_Y]] - players[state]) / POS_SCALE
    glob[state, 11] = boss[:, E.HP_RATIO]
    return glob


def _group_starts(sorted_owner: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """Return where each state's run of rows starts in `sorted_owner`."""
    return np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])


def _item_globals(items: _Table, players: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Per state, the `ITEM_SLOTS` nearest items' offsets and presence, then the item count."""
    n = len(players)
    out = np.zeros((n, ITEM_DIM), dtype=np.float32)
    rows, owner, counts = items
    if len(rows):
        rel = rows[:, [IT.POS_X, IT.POS_Y]] - players[owner]
        order = np.lexsort(((rel**2).sum(axis=1), owner))
        sorted_owner = owner[order]
        starts = _group_starts(sorted_owner)
        rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
        keep = rank < ITEM_SLOTS
        chosen, slot = order[keep], rank[keep]
        out[owner[chosen], slot * 3] = rel[chosen, 0] / POS_SCALE
        out[owner[chosen], slot * 3 + 1] = rel[chosen, 1] / POS_SCALE
        out[owner[chosen], slot * 3 + 2] = 1.0
    out[:, ITEM_DIM - 1] = counts / 50.0
    return out


type _LookaheadPart = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.intp],
]


def _lookahead(
    tables: dict[wire.Entity, _Table],
    players: npt.NDArray[np.float32],
    hit_radii: npt.NDArray[np.float32],
    step_interval: int,
    *,
    beams: bool = False,
) -> npt.NDArray[np.float32]:
    """Per state and held action, the least clearance over one step from any lethal entity."""
    n = len(players)
    parts: list[_LookaheadPart] = []
    for spec in _SPECS:
        if spec.kind not in (TokenType.BULLET, TokenType.ENEMY, TokenType.CURVE_NODE):
            continue
        rows, owner, _ = tables[spec.entity]
        if not len(rows):
            continue
        if spec.kind is TokenType.ENEMY:
            keep = rows[:, E.IS_LETHAL] > 0
        elif spec.kind is TokenType.BULLET:
            keep = (rows[:, B.FLAGS_LO].astype(np.int32) & int(wire.BulletFlag.COLLIDES)) != 0
        else:
            keep = np.ones(len(rows), dtype=bool)
        rows, owner = rows[keep], owner[keep]
        if not len(rows):
            continue
        radius = rows[:, spec.radius] * spec.radius_factor
        parts.append(
            (
                (rows[:, list(spec.pos)] - players[owner]).astype(np.float32),
                rows[:, list(spec.vel)].astype(np.float32),
                np.sqrt(radius**2 + hit_radii[owner] ** 2).astype(np.float32),
                owner,
            )
        )
    columns = np.full((n, len(DIRECTIONS) * 2), np.inf, dtype=np.float32)
    frames = np.arange(1, step_interval + 1, dtype=np.float32)
    moved = _DISPLACEMENTS[:, None, :] * frames[None, :, None]
    if parts:
        rel = np.concatenate([p[0] for p in parts])
        vel = np.concatenate([p[1] for p in parts])
        collide = np.concatenate([p[2] for p in parts])
        owner = np.concatenate([p[3] for p in parts])
        path = rel[:, None, :] + vel[:, None, :] * frames[None, :, None]
        neutral = _norm(path).min(axis=1) - collide
        best = np.full(n, np.inf, dtype=np.float32)
        np.minimum.at(best, owner, neutral)
        reach = PLAYER_SPEED * step_interval
        near = neutral <= best[owner] + 2 * reach
        path, collide, owner = path[near], collide[near], owner[near]
        gap = path[:, None, :, :] - moved[None]
        clearance = _norm(gap).min(axis=2) - collide[:, None]
        order = np.argsort(owner, kind="stable")
        sorted_owner = owner[order]
        firsts = _group_starts(sorted_owner)
        columns[sorted_owner[firsts]] = np.minimum.reduceat(clearance[order], firsts, axis=0)
    if beams:
        columns = np.minimum(columns, _beam_clearances(tables, hit_radii, players, frames, moved))
    return np.clip(columns / CLEARANCE_SCALE, -1.0, 2.0)


def _beam_clearances(
    tables: dict[wire.Entity, _Table],
    hit_radii: npt.NDArray[np.float32],
    players: npt.NDArray[np.float32],
    frames: npt.NDArray[np.float32],
    moved: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Per state and held action, the least clearance over one step from any laser beam."""
    n = len(players)
    columns = np.full((n, len(DIRECTIONS) * 2), np.inf, dtype=np.float32)
    rows, owner, _ = tables[wire.Entity.SEGMENT_LASERS]
    if len(rows):
        vel = rows[:, [SL.VEL_X, SL.VEL_Y]]
        unit = _segment_units(vel)
        heads = (rows[:, [SL.HEAD_X, SL.HEAD_Y]] - players[owner])[:, None, :] + vel[:, None, :] * frames[None, :, None]
        rel = heads[:, None, :, :] - moved[None]
        back = np.clip(_dot(rel, unit[:, None, None]), 0.0, rows[:, SL.LENGTH][:, None, None])
        gap = rel - unit[:, None, None, :] * back[..., None]
        distance = _norm(gap).min(axis=2)
        np.minimum.at(columns, owner, distance - (0.5 * rows[:, SL.WIDTH] + hit_radii[owner])[:, None])
    rows, owner, _ = tables[wire.Entity.RAY_LASERS]
    if len(rows):
        origins = (rows[:, [RL.ORIGIN_X, RL.ORIGIN_Y]] - players[owner])[:, None, :] + rows[
            :, [RL.ORIGIN_VEL_X, RL.ORIGIN_VEL_Y]
        ][:, None, :] * frames[None, :, None]
        angle = (
            np.arctan2(rows[:, RL.SIN_ANGLE], rows[:, RL.COS_ANGLE])[:, None]
            + rows[:, RL.ANGULAR_VEL][:, None] * frames[None, :]
        )
        direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)
        rel = moved[None] - origins[:, None, :, :]
        along = np.maximum(_dot(rel, direction[:, None]), 0.0)
        gap = rel - direction[:, None, :, :] * along[..., None]
        distance = _norm(gap).min(axis=2)
        np.minimum.at(columns, owner, distance - (0.5 * rows[:, RL.WIDTH] + hit_radii[owner])[:, None])
    return columns


A = wire.Action
DIRECTIONS = (
    0,
    A.UP,
    A.DOWN,
    A.LEFT,
    A.RIGHT,
    A.UP | A.LEFT,
    A.UP | A.RIGHT,
    A.DOWN | A.LEFT,
    A.DOWN | A.RIGHT,
)
_R2 = 2**-0.5
_MOVES: Final = (
    (0.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (-_R2, -_R2),
    (_R2, -_R2),
    (-_R2, _R2),
    (_R2, _R2),
)
_DISPLACEMENTS: Final = np.array(
    [(mx * speed, my * speed) for speed in (PLAYER_SPEED, PLAYER_SPEED_FOCUSED) for mx, my in _MOVES],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ActionSet:
    """The discrete actions that a policy chooses from."""

    name: str
    patterns: tuple[tuple[int, ...], ...]
    key_mask: int = int(A.UP | A.DOWN | A.LEFT | A.RIGHT | A.FOCUS)

    def __len__(self) -> int:
        return len(self.patterns)

    @property
    def frames(self) -> int:
        """The frames spanned by every pattern (1 for plain holds)."""
        return len(self.patterns[0])

    def command(self, index: int) -> bytes:
        """Return the wire command for action `index`."""
        return self._commands[index]

    @cached_property
    def _commands(self) -> tuple[bytes, ...]:
        return tuple(wire.tape_frame(words) for words in self.patterns)


def _hold_set() -> ActionSet:
    words = tuple(int(d | A.SHOOT | (A.FOCUS if focus else 0)) for focus in (0, 1) for d in DIRECTIONS)
    return ActionSet("hold", tuple((w,) for w in words))


def _tap_set(frames: int = 3) -> ActionSet:
    """Every direction and focus as a hold, a 1-frame tap, and a 2-frame tap over `frames`."""
    patterns: list[tuple[int, ...]] = []
    for focus in (0, 1):
        base = int(A.SHOOT | (A.FOCUS if focus else 0))
        patterns.append((base,) * frames)
        for d in DIRECTIONS[1:]:
            move = int(base | d)
            patterns.append((move,) * frames)
            patterns.extend((move,) * held + (base,) * (frames - held) for held in range(1, frames))
    return ActionSet(f"tap{frames}", tuple(patterns))


def _vertical_set() -> ActionSet:
    """Build the set of actions without horizontal movement."""
    words = tuple(int(d | A.SHOOT | (A.FOCUS if focus else 0)) for focus in (0, 1) for d in (0, A.UP, A.DOWN))
    mask = int(A.UP | A.DOWN | A.FOCUS)
    return ActionSet("vertical", tuple((w,) for w in words), key_mask=mask)


ACTION_SETS: Final[dict[str, ActionSet]] = {
    "hold": _hold_set(),
    "tap3": _tap_set(3),
    "vertical": _vertical_set(),
}
HOLD_ACTIONS: Final = ACTION_SETS["hold"]
NUM_ACTIONS = len(HOLD_ACTIONS)
