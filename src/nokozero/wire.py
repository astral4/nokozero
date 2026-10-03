"""The wire format."""

import math
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache, reduce
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import socket
    from collections.abc import Sequence


CMD_RESET: Final = 0x02
CMD_TAPE: Final = 0x03

MAX_TAPE_FRAMES: Final = 36_000
_TAPE_HEADER: Final = 6

MAX_FRAME_LEN: Final = 1 << 20


class Action(IntEnum):
    """Action-word bits."""

    SHOOT = 0x1
    BOMB = 0x2
    FOCUS = 0x8
    UP = 0x10
    DOWN = 0x20
    LEFT = 0x40
    RIGHT = 0x80
    SKIP = 0x200


ACTION_MASK: Final = reduce(int.__or__, Action, 0)


RECORD_LEN: Final = 0x238 + 0xA4


class ResetOutcome(IntEnum):
    """The observation's `reset_outcome` word describing the reset specified by `reset_seq`."""

    IDLE = 0
    PENDING = 1
    APPLIED = 2
    VANILLA = 3
    FAILED_STAGE_MISMATCH = 4
    FAILED_NO_ECL = 5
    FAILED_DIFFICULTY_MISMATCH = 6
    FAILED_CHARACTER_MISMATCH = 7
    FAILED_UNMAPPED = 8


class Gamemode(IntEnum):
    """The normalized scene class reported by the hook."""

    OTHER = 0
    MENU = 1
    INGAME = 2


class Meta(IntEnum):
    """Word indices into the observation's meta block."""

    STEP = 0
    GAMEMODE = 1
    IN_STAGE = 2
    LOAD_GENERATION = 3
    RESET_SEQ = 4
    RESET_OUTCOME = 5
    APPLIED_SECTION = 6
    HITS = 7
    ENTRY_COUNT = 8
    GAME_TICK = 9
    SCORE_DIV10 = 10
    GRAZE = 11
    VALUE_X100 = 12
    POWER = 13
    LIVES = 14
    LIFE_FRAGMENTS = 15
    BOMBS = 16
    BOMB_FRAGMENTS = 17
    RNG_STATE = 18
    RNG_COUNT = 19
    CHAPTER = 20
    TIME_IN_CHAPTER = 21
    SPELL_ID = 22
    MISS_COUNT = 23
    SPELL_TIMER = 24
    SPELL_FLAGS = 25
    MODE_FLAGS = 26
    PLAYER_X = 27
    PLAYER_Y = 28
    PLAYER_FOCUSED = 29
    PLAYER_HITBOX = 30
    PLAYER_HIT_RADIUS = 31


META_WORDS: Final = 32

SIGNED_META: Final = frozenset(
    {
        Meta.GRAZE,
        Meta.VALUE_X100,
        Meta.POWER,
        Meta.LIVES,
        Meta.LIFE_FRAGMENTS,
        Meta.BOMBS,
        Meta.BOMB_FRAGMENTS,
        Meta.SPELL_ID,
        Meta.MISS_COUNT,
        Meta.SPELL_TIMER,
    }
)

F32_META: Final = frozenset(
    {
        Meta.PLAYER_X,
        Meta.PLAYER_Y,
        Meta.PLAYER_FOCUSED,
        Meta.PLAYER_HITBOX,
        Meta.PLAYER_HIT_RADIUS,
    }
)


def meta_dtype(word: Meta) -> type[np.int32 | np.float32 | np.uint32]:
    """Return the numpy type that the hook serializes `word` as."""
    if word in SIGNED_META:
        return np.int32
    if word in F32_META:
        return np.float32
    return np.uint32


SYNC_WORDS: Final = (
    Meta.GAME_TICK,
    Meta.CHAPTER,
    Meta.TIME_IN_CHAPTER,
    Meta.RNG_COUNT,
    Meta.PLAYER_X,
    Meta.PLAYER_Y,
    Meta.HITS,
)
_SYNC_INDEX: Final = np.array([int(word) for word in SYNC_WORDS], dtype=np.intp)


class Entity(IntEnum):
    """The entity tables of an observation."""

    BULLETS = 0
    ENEMIES = 1
    ITEMS = 2
    SEGMENT_LASERS = 3
    RAY_LASERS = 4
    CURVE_NODES = 5


class BulletCol(IntEnum):
    """Columns of a bullet row."""

    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3
    SIZE_W = 4
    SIZE_H = 5
    SCALE = 6
    FLAGS_LO = 7
    FLAGS_HI = 8


class BulletFlag(IntEnum):
    """Bullet flag bits relevant for player collision testing."""

    COLLIDES = 0x2
    CIRCLE = 0x10
    SCALED = 0x40


class EnemyCol(IntEnum):
    """Columns of an enemy row."""

    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3
    HITBOX_W = 4
    HITBOX_H = 5
    HURTBOX_W = 6
    HURTBOX_H = 7
    HP_RATIO = 8
    MAX_HP = 9
    IS_BOSS = 10
    IS_INVULNERABLE = 11
    INVULN_FRAMES = 12
    IS_LETHAL = 13
    NO_HITBOX_FRAMES = 14
    FLAGS_LO = 15
    FLAGS_HI = 16


class ItemCol(IntEnum):
    """Columns of an item row."""

    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3
    KIND = 4


class SegmentLaserCol(IntEnum):
    """Columns of a segment laser row."""

    HEAD_X = 0
    HEAD_Y = 1
    VEL_X = 2
    VEL_Y = 3
    LENGTH = 4
    WIDTH = 5


class RayLaserCol(IntEnum):
    """Columns of a ray laser row."""

    ORIGIN_X = 0
    ORIGIN_Y = 1
    ORIGIN_VEL_X = 2
    ORIGIN_VEL_Y = 3
    COS_ANGLE = 4
    SIN_ANGLE = 5
    ANGULAR_VEL = 6
    WIDTH = 7


class CurveNodeCol(IntEnum):
    """Columns of a curve-laser node row."""

    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3
    WIDTH = 4


COLUMNS: Final[tuple[type[IntEnum], ...]] = (
    BulletCol,
    EnemyCol,
    ItemCol,
    SegmentLaserCol,
    RayLaserCol,
    CurveNodeCol,
)
SECTION_WIDTHS: Final = tuple(len(columns) for columns in COLUMNS)


_PARAMS = struct.Struct("<IIq7iIiiIIIff")
PARAMS_LEN: Final = _PARAMS.size


CHAPTER_BASE: Final = 10000
CHAPTER_LIMIT: Final = 20000
STAGES: Final = range(1, 8)
EXTRA_STAGE: Final = 7
MAIN_STAGES: Final = range(1, EXTRA_STAGE)
EXTRA_DIFFICULTY: Final = 4
MAX_CHARACTER: Final = 3
MAX_RNG_SEED: Final = 0xFFFF
MAX_POWER: Final = 400
MAX_LIVES: Final = 8
CANONICAL_LANDING: Final = (0, 0)
DEFAULT_STEP_INTERVAL: Final = 3
MAX_STEP_INTERVAL: Final = 60


def stage_of(section: int) -> int:
    """Return the stage targeted by a section ID."""
    if CHAPTER_BASE <= section < CHAPTER_LIMIT:
        return (section - CHAPTER_BASE) // 100
    return section // 1000


@dataclass(frozen=True, kw_only=True)
class ResetParams:
    """The params block for a RESET command."""

    section: int = 0
    active: bool = False
    score: int = 0
    graze: int = 0
    value: int = 0
    power: int = 100
    lives: int = 2
    life_fragments: int = 0
    bombs: int = 3
    bomb_fragments: int = 0
    phase: int = 0
    difficulty: int
    character: int
    rng_seed: int
    step_interval: int = DEFAULT_STEP_INTERVAL
    real_deaths: bool = False
    player_x: float = 0.0
    player_y: float = 400.0
    record: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.active and stage_of(self.section) not in STAGES:
            msg = f"section {self.section} does not target a valid stage (stages are {STAGES.start}-{STAGES[-1]})"
            raise ValueError(msg)
        self._check_selectors()
        self._check_resources()
        self._check_record()
        try:
            self.pack()
        except struct.error as error:
            msg = f"field outside its wire type ({error}): {self!r}"
            raise ValueError(msg) from error

    def _check_selectors(self) -> None:
        """Refuse selectors that would be rejected by the hook."""
        for name, value, low, high, why in (
            ("difficulty", self.difficulty, 0, EXTRA_DIFFICULTY, "leave-as-is does not exist"),
            ("character", self.character, 0, MAX_CHARACTER, "leave-as-is does not exist"),
            ("rng_seed", self.rng_seed, 0, MAX_RNG_SEED, "the RNG state is a u16"),
            ("step_interval", self.step_interval, 1, MAX_STEP_INTERVAL, "game frames per step"),
        ):
            if not low <= value <= high:
                msg = f"{name} {value} out of range ({low}-{high}; {why})"
                raise ValueError(msg)
        if not (math.isfinite(self.player_x) and math.isfinite(self.player_y)):
            msg = f"player position ({self.player_x}, {self.player_y}) is not finite"
            raise ValueError(msg)
        if self.active and (stage_of(self.section) == EXTRA_STAGE) != (self.difficulty == EXTRA_DIFFICULTY):
            msg = (
                f"section {self.section} with difficulty {self.difficulty}: "
                "Extra sections require difficulty 4"
            )
            raise ValueError(msg)

    def _check_resources(self) -> None:
        """Refuse resource values that would be clamped by the hook."""
        for name, value, bound in (
            ("score", self.score, 9_999_999_990),
            ("graze", self.graze, 999_999),
            ("value", self.value, 999_990),
            ("power", self.power, MAX_POWER),
            ("lives", self.lives, MAX_LIVES),
            (
                "life_fragments",
                self.life_fragments,
                5 if self.difficulty == EXTRA_DIFFICULTY else 3,
            ),
            ("bombs", self.bombs, 8),
            ("bomb_fragments", self.bomb_fragments, 4),
        ):
            if not 0 <= value <= bound:
                msg = f"{name} {value} out of range (0-{bound})"
                raise ValueError(msg)
        if self.score % 10:
            msg = f"score {self.score} is not a multiple of 10"
            raise ValueError(msg)

    def _check_record(self) -> None:
        if self.record is None:
            return
        if len(self.record) != RECORD_LEN:
            msg = f"record must be {RECORD_LEN} bytes, got {len(self.record)}"
            raise ValueError(msg)
        if not self.active:
            msg = "a record needs a warp to its stage (active=True)"
            raise ValueError(msg)
        recorded = int.from_bytes(self.record[:2], "little")
        if recorded != stage_of(self.section):
            msg = f"record of stage {recorded} on a warp to stage {stage_of(self.section)}"
            raise ValueError(msg)

    def pack(self) -> bytes:
        """Serialize to the `PARAMS_LEN`-byte wire layout."""
        return _PARAMS.pack(
            self.section,
            int(self.active),
            self.score,
            self.graze,
            self.value,
            self.power,
            self.lives,
            self.life_fragments,
            self.bombs,
            self.bomb_fragments,
            self.phase,
            self.difficulty,
            self.character,
            self.rng_seed,
            self.step_interval,
            int(self.real_deaths),
            self.player_x,
            self.player_y,
        )


@dataclass(frozen=True, eq=False)
class Observation:
    """A parsed observation."""

    words: npt.NDArray[np.uint32]
    """The full payload as little-endian u32 words."""
    sections: tuple[npt.NDArray[np.float32], ...]
    """Per-table `(count, width)` f32 arrays indexed by `Entity`."""
    inputs: npt.NDArray[np.uint32]
    """Live stage frame indices of the current load since the previous observation."""

    def meta(self, index: Meta) -> int:
        """Read one integer meta word."""
        if index in SIGNED_META:
            return int(self.words.view(np.int32)[index])
        return int(self.words[index])

    def meta_f32(self, index: Meta) -> float:
        """Read one f32 meta word."""
        return float(self.words.view(np.float32)[index])

    @property
    def gamemode(self) -> Gamemode:
        """The normalized scene class."""
        return Gamemode(self.meta(Meta.GAMEMODE))

    @property
    def in_stage(self) -> bool:
        """Whether the entity walk ran for this frame."""
        return self.meta(Meta.IN_STAGE) == 1

    @property
    def reset_outcome(self) -> ResetOutcome:
        """The state of the reset specified by `reset_seq`."""
        return ResetOutcome(self.meta(Meta.RESET_OUTCOME))

    @property
    def landing_signature(self) -> tuple[int, int]:
        """`(GAME_TICK, RNG_COUNT)`. Used for comparing to `CANONICAL_LANDING` on a landing."""
        return self.meta(Meta.GAME_TICK), self.meta(Meta.RNG_COUNT)

    def sync_state(self) -> npt.NDArray[np.uint32]:
        """Return the raw `SYNC_WORDS`. Used for comparing with frames of a recorded run."""
        return self.words[_SYNC_INDEX]


_U32 = struct.Struct("<I")


def _frame(tag: int, payload: bytes) -> bytes:
    """Wrap a payload in the wire framing."""
    return _U32.pack(1 + len(payload)) + bytes((tag,)) + payload


def act_frame(bits: int, *, raw: bool = False) -> bytes:
    """Build the command that holds `bits` for a step."""
    return _act_frame(bits, raw=raw)


@cache
def _act_frame(bits: int, *, raw: bool) -> bytes:
    # `act_frame` is used as a wrapper because `functools.cache` erases signatures, defeating type-checking.
    return tape_frame((bits,), raw=raw)


def tape_frame(keys: Sequence[int] | npt.NDArray[np.integer], *, raw: bool = False) -> bytes:
    """Build a TAPE command frame."""
    arr = np.ascontiguousarray(keys, dtype=np.uint32)
    if arr.ndim != 1 or not 0 < len(arr) <= MAX_TAPE_FRAMES:
        msg = f"tape must hold 1-{MAX_TAPE_FRAMES} frames, got shape {arr.shape}"
        raise ValueError(msg)
    if np.any(arr & ~np.uint32(ACTION_MASK)):
        msg = f"tape holds action bits outside mask {ACTION_MASK:#x}"
        raise ValueError(msg)
    return _frame(CMD_TAPE, bytes((raw,)) + arr.astype("<u2").tobytes())


NEUTRAL: Final = act_frame(0)


def command_frames(frame: bytes) -> int:
    """Return the game frames requested by a command."""
    if frame[4] != CMD_TAPE:
        return 1
    return (len(frame) - _TAPE_HEADER) // 2


def reset_frame(seq: int, params: ResetParams) -> bytes:
    """Build a RESET command frame."""
    return _frame(CMD_RESET, _U32.pack(seq) + params.pack() + (params.record or b""))


class ProtocolError(RuntimeError):
    """The peer sent something unexpected."""


def resolves(obs: Observation, seq: int) -> bool:
    """Whether `obs` reports the reset `seq` as resolved."""
    return obs.meta(Meta.RESET_SEQ) == seq and obs.reset_outcome != ResetOutcome.PENDING


def reset_landed(obs: Observation, seq: int, what: str) -> bool:
    """Whether `obs` resolves the reset `seq`."""
    if not resolves(obs, seq):
        return False
    if obs.reset_outcome != ResetOutcome.APPLIED:
        msg = f"{what}: reset landed as {obs.reset_outcome.name}"
        raise RuntimeError(msg)
    return True


_INPUT_WIDTH: Final = 2

_VALIDATED_WORDS: Final = ((Meta.GAMEMODE, Gamemode), (Meta.RESET_OUTCOME, ResetOutcome))


def parse_observation(payload: bytes | bytearray) -> Observation:
    """Parse an observation payload into meta, section and input-log views."""
    # meta block, section counts, input log
    if len(payload) % 4 != 0 or len(payload) < 4 * (META_WORDS + len(SECTION_WIDTHS) + 1):
        msg = f"observation payload of {len(payload)} bytes is too short or misaligned"
        raise ProtocolError(msg)
    words: npt.NDArray[np.uint32] = np.frombuffer(payload, dtype=np.dtype("<u4"))
    words.flags.writeable = False
    floats: npt.NDArray[np.float32] = words.view(np.float32)

    for word, allowed in _VALIDATED_WORDS:
        if int(words[word]) not in allowed:
            msg = f"unknown {word.name.lower()} {int(words[word])} (stale hook DLL?)"
            raise ProtocolError(msg)

    sections: list[npt.NDArray[np.float32]] = []
    offset = META_WORDS
    for width in SECTION_WIDTHS:
        count = int(words[offset])
        offset += 1
        end = offset + count * width
        if end > len(words):
            msg = "section count exceeds payload"
            raise ProtocolError(msg)
        sections.append(floats[offset:end].reshape(count, width))
        offset = end
    count = int(words[offset]) if offset < len(words) else 0
    offset += 1
    end = offset + count * _INPUT_WIDTH
    if end != len(words):
        msg = "the input log does not end the payload"
        raise ProtocolError(msg)
    inputs = words[offset:end].reshape(count, _INPUT_WIDTH)
    return Observation(words=words, sections=tuple(sections), inputs=inputs)


TIMEOUT_FLOOR: Final = 0.001


def _recv_exact_into(sock: socket.socket, view: memoryview, deadline: float | None) -> None:
    """Fill `view` from the socket in place."""
    while view:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = "frame budget exhausted mid-frame"
                raise TimeoutError(msg)
            sock.settimeout(max(remaining, TIMEOUT_FLOOR))
        received = sock.recv_into(view)
        if received == 0:
            msg = "connection closed mid-frame"
            raise ProtocolError(msg)
        view = view[received:]


def recv_observation(sock: socket.socket) -> Observation:
    """Block for one observation frame and parse it.

    Raises `ProtocolError` on a malformed frame and `TimeoutError`/`OSError` based on
    the socket's timeout configuration.
    """
    timeout = sock.gettimeout()
    deadline = time.monotonic() + timeout if timeout else None
    try:
        header = bytearray(_U32.size)
        _recv_exact_into(sock, memoryview(header), deadline)
        (length,) = _U32.unpack(header)
        if not 0 < length <= MAX_FRAME_LEN:
            msg = f"bad frame length {length}"
            raise ProtocolError(msg)
        payload = bytearray(length)
        _recv_exact_into(sock, memoryview(payload), deadline)
        return parse_observation(payload)
    finally:
        if deadline is not None:
            sock.settimeout(timeout)
