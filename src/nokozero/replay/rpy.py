"""Decoder and encoder for th15 v1.00b `.rpy` replay files."""

import struct
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from nokozero.wire import ACTION_MASK

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MAGIC = b"t15r"
VERSION = 3
HEADER = struct.Struct("<4sIIIIIIII")
#: (block size, key, step).
_PASSES = ((0x400, 0x5C, 0xE1), (0x100, 0x7D, 0x3A))
_OFFSET_BITS = 13
_LENGTH_BITS = 4
_MIN_MATCH = 3
_MAX_MATCH = _MIN_MATCH + (1 << _LENGTH_BITS) - 1
_RING = 1 << _OFFSET_BITS
INFO_SIZE = 0xA4
_NAME_LEN = 8
_INFO_TIMESTAMP_OFFSET = 0xC
_INFO_SCORE_OFFSET = 0x14
STAGE_COUNT_OFFSET = 0x88
CHARACTER_OFFSET = 0x8C
_INFO_DIFFICULTY_OFFSET = 0x94
FLAGS_OFFSET = 0x0A
FLAG_STAGE_PRACTICE = 1
FLAG_SPELL_PRACTICE = 2
END_STAGE_OFFSET = 0x98
CLEARED_END_STAGE = 8
USER_MAGIC = b"USER"
USER_HEADER_SIZE = 12
_USER_KIND = 0
_USER_TEXT_PAD = 20
RECORD_SIZE = 0x238
FRAME_SIZE = 6
LOG_FRAMES_PER_BYTE = 30
FULL_FRAME_RATE = 60
GLOBALS_OFFSET = 0x14
_GLOBALS = {
    "chapter": 0x08,
    "score_div10": 0x1C,
    "difficulty": 0x20,
    "graze": 0x2C,
    "spell_id": 0x34,
    "misses": 0x3C,
    "point_items": 0x40,
    "piv_x100": 0x44,
    "power": 0x50,
    "lives": 0x60,
    "life_fragments": 0x64,
    "extends": 0x68,
    "bombs": 0x6C,
    "bomb_fragments": 0x70,
}
PLAYER_POS_OFFSET = 0xC


class ReplayError(ValueError):
    """The file is not a valid th15 replay file."""


@dataclass(frozen=True)
class Stage:
    """A stage of a replay."""

    number: int
    seed: int
    #: The player's position on the stage's first frame.
    player_x: float
    player_y: float
    globals: dict[str, int]
    #: Held keys per frame.
    keys: npt.NDArray[np.uint16]
    record: bytes

    @property
    def frames(self) -> int:
        """The stage's length in frames."""
        return len(self.keys)

    @cached_property
    def actions(self) -> npt.NDArray[np.uint16]:
        """The held keys within `wire.ACTION_MASK` per frame."""
        return self.keys & np.uint16(ACTION_MASK)


@dataclass(frozen=True)
class Replay:
    """A decoded replay."""

    name: str
    timestamp: int
    score_div10: int
    character: int
    cleared: bool
    flags: int
    stages: tuple[Stage, ...]
    user_text: str
    raw_info: bytes


class Info(NamedTuple):
    """Replay info block data."""

    character: int
    difficulty: int

    @classmethod
    def of(cls, info: bytes | bytearray) -> Info:
        """Read the fields from a decompressed body's first `INFO_SIZE` bytes."""
        return cls(info[CHARACTER_OFFSET], struct.unpack_from("<I", info, _INFO_DIFFICULTY_OFFSET)[0])


def _shuffle_plan(
    length: int, block_size: int, key: int, step: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]:
    """Return a single pass of the byte shuffle over `length` bytes; i.e. its order and key stream."""
    span = length - (length & 1)
    if length % block_size < block_size // 4:
        span -= length % block_size
    span = max(span, 0)

    def block(size: int) -> npt.NDArray[np.intp]:
        return np.concatenate([np.arange(size - 1, -1, -2), np.arange(size - 2, -1, -2)])

    full, last = divmod(span, block_size)
    starts = np.arange(full)[:, np.newaxis] * block_size
    order = np.concatenate([(starts + block(block_size)).ravel(), full * block_size + block(last)])
    keys = ((key + step * np.arange(span)) & 0xFF).astype(np.uint8)
    return order, keys


def descramble(buf: bytes, block_size: int, key: int, step: int) -> bytearray:
    """Undo a single pass of the byte shuffle."""
    order, keys = _shuffle_plan(len(buf), block_size, key, step)
    shuffled = np.frombuffer(buf, dtype=np.uint8)
    out = bytearray(buf)
    np.frombuffer(out, dtype=np.uint8)[order] = shuffled[: len(order)] ^ keys
    return out


def unlzss(  # noqa: C901
    buf: bytes, out_size: int, *, prefix: bool = False
) -> bytearray:
    """Decompress LZSS-compressed data to exactly `out_size` bytes.

    A `prefix` decode returns the first `out_size` bytes of a longer stream.
    """
    out = bytearray()
    acc = held = pos = 0
    total = len(buf)

    def bits(n: int) -> int:
        nonlocal acc, held, pos
        while held < n:
            if pos == total:
                return -1
            acc = (acc << 8) | buf[pos]
            pos += 1
            held += 8
        held -= n
        value = acc >> held
        acc &= (1 << held) - 1
        return value

    while len(out) < out_size:
        flag = bits(1)
        if flag < 0:
            break
        if flag:
            c = bits(8)
            if c < 0:
                break
            out.append(c)
            continue
        offset = bits(_OFFSET_BITS)
        if offset <= 0:
            break
        length = bits(_LENGTH_BITS) + _MIN_MATCH
        written = len(out)
        source = written - 1 - ((written - offset) % _RING)
        if source >= 0 and source + length <= written:
            out += out[source : source + length]
        else:
            for i in range(length):
                k = source + i
                out.append(out[k] if k >= 0 else 0)
    if prefix:
        del out[out_size:]
    if len(out) != out_size:
        msg = f"decompressed {len(out)} bytes, expected {out_size}"
        raise ReplayError(msg)
    return out


def _body(data: bytes, limit: int | None = None) -> bytearray:
    """Return a `.rpy` file's decompressed body, or only its first `limit` bytes."""
    if len(data) < HEADER.size or data[:4] != MAGIC:
        msg = "not a th15 replay (bad magic)"
        raise ReplayError(msg)
    _, version, _, _, _, _, _, comp_size, decomp_size = HEADER.unpack_from(data, 0)
    if version != VERSION:
        msg = f"unsupported replay version {version}"
        raise ReplayError(msg)
    if decomp_size < INFO_SIZE:
        msg = f"a {decomp_size}-byte body has no room for the {INFO_SIZE}-byte info block"
        raise ReplayError(msg)
    body = data[HEADER.size : HEADER.size + comp_size]
    for block_size, key, step in _PASSES:
        body = bytes(descramble(body, block_size, key, step))
    if limit is None:
        return unlzss(body, decomp_size)
    return unlzss(body, limit, prefix=True)


def decode(data: bytes) -> Replay:
    """Decode the bytes of a `.rpy` file. Raises `ReplayError` upon encountering unreadable bytes."""
    dec = _body(data)
    user_offset = HEADER.unpack_from(data, 0)[3]
    name = dec[:12].split(b"\0")[0].decode("shift_jis", errors="replace")
    timestamp = struct.unpack_from("<I", dec, _INFO_TIMESTAMP_OFFSET)[0]
    score = struct.unpack_from("<I", dec, _INFO_SCORE_OFFSET)[0]
    stage_count = struct.unpack_from("<I", dec, STAGE_COUNT_OFFSET)[0]
    character = Info.of(dec).character
    cleared = struct.unpack_from("<I", dec, END_STAGE_OFFSET)[0] == CLEARED_END_STAGE
    flags = dec[FLAGS_OFFSET]
    info = bytes(dec[:INFO_SIZE])
    stages: list[Stage] = []
    offset = INFO_SIZE
    for _ in range(stage_count):
        if offset + RECORD_SIZE > len(dec):
            msg = "truncated stage record"
            raise ReplayError(msg)
        number, seed, frames, size = struct.unpack_from("<HHII", dec, offset)
        start = offset + RECORD_SIZE
        if size < frames * FRAME_SIZE or start + size > len(dec):
            msg = f"stage {number}: {frames} frames in {size} bytes, {len(dec) - start} left"
            raise ReplayError(msg)
        px, py = struct.unpack_from("<ii", dec, offset + PLAYER_POS_OFFSET)
        glob = {
            name_: struct.unpack_from("<i", dec, offset + GLOBALS_OFFSET + off)[0] for name_, off in _GLOBALS.items()
        }
        raw = np.frombuffer(dec, dtype="<u2", count=frames * (FRAME_SIZE // 2), offset=start)
        keys = raw.reshape(frames, FRAME_SIZE // 2)[:, 0].copy()
        stages.append(
            Stage(
                number=number,
                seed=seed,
                player_x=px / 128.0,
                player_y=py / 128.0,
                globals=glob,
                keys=keys,
                record=bytes(dec[offset : offset + RECORD_SIZE]) + info,
            )
        )
        offset = start + size
    user = data[user_offset:]
    user_text = (
        user[USER_HEADER_SIZE:].split(b"\0")[0].decode("shift_jis", errors="replace")
        if user.startswith(USER_MAGIC)
        else ""
    )
    return Replay(
        name=name,
        timestamp=timestamp,
        score_div10=score,
        character=character,
        cleared=cleared,
        flags=flags,
        stages=tuple(stages),
        user_text=user_text,
        raw_info=info,
    )


def load(path: Path) -> Replay:
    """Decode the replay file at `path`."""
    return decode(path.read_bytes())


def read_info(path: Path) -> Info:
    """Decode only the info block of the replay file at `path`."""
    return Info.of(_body(path.read_bytes(), INFO_SIZE))


def scramble(buf: bytes, block_size: int, key: int, step: int) -> bytearray:
    """Apply a single pass of the byte shuffle."""
    order, keys = _shuffle_plan(len(buf), block_size, key, step)
    plain = np.frombuffer(buf, dtype=np.uint8)
    out = bytearray(buf)
    np.frombuffer(out, dtype=np.uint8)[: len(order)] = plain[order] ^ keys
    return out


class _BitWriter:
    """Utility for writing bits (most significant first)."""

    def __init__(self) -> None:
        self.out = bytearray()
        self._acc = 0
        self._held = 0

    def put(self, value: int, n: int) -> None:
        self._acc = (self._acc << n) | value
        self._held += n
        while self._held >= 8:  # noqa: PLR2004
            self._held -= 8
            self.out.append((self._acc >> self._held) & 0xFF)
            self._acc &= (1 << self._held) - 1

    def finish(self) -> bytes:
        if self._held:
            self.put(0, 8 - self._held)
        return bytes(self.out)


_CHAIN_DEPTH = 64


def _longest_match(data: bytes, i: int, heads: dict[bytes, list[int]]) -> tuple[int, int]:
    """Return the longest (length, distance) match for `data[i:]` within the ring's reach."""
    best_len = best_dist = 0
    key = data[i : i + _MIN_MATCH]
    if len(key) < _MIN_MATCH:
        return 0, 0
    limit = min(_MAX_MATCH, len(data) - i)
    for j in reversed(heads.get(key, ())):
        dist = i - j
        if dist >= _RING:
            break
        if (i - dist + 1) % _RING == 0:
            continue
        length = _MIN_MATCH
        while length < limit and data[j + length] == data[i + length]:
            length += 1
        if length > best_len:
            best_len, best_dist = length, dist
            if length == _MAX_MATCH:
                break
    return best_len, best_dist


def lzss(data: bytes) -> bytes:
    """Apply LZSS compression such that `unlzss` returns `data`."""
    bits = _BitWriter()
    heads: dict[bytes, list[int]] = {}
    n = len(data)
    i = 0
    while i < n:
        length, dist = _longest_match(data, i, heads)
        if length >= _MIN_MATCH:
            bits.put(0, 1)
            bits.put((i - dist + 1) % _RING, _OFFSET_BITS)
            bits.put(length - _MIN_MATCH, _LENGTH_BITS)
        else:
            bits.put(1, 1)
            bits.put(data[i], 8)
            length = 1
        for k in range(i, i + length):
            prefix = data[k : k + _MIN_MATCH]
            if len(prefix) == _MIN_MATCH:
                chain = heads.setdefault(prefix, [])
                chain.append(k)
                if len(chain) > _CHAIN_DEPTH:
                    del chain[0]
        i += length
    bits.put(0, 1)
    bits.put(0, _OFFSET_BITS)
    return bits.finish()


def frame_records(keys: npt.NDArray[np.uint16]) -> bytes:
    """Return a stage's frame records for `keys` held per frame."""
    held = np.asarray(keys, dtype=np.uint16)
    previous = np.concatenate([np.zeros(1, dtype=np.uint16), held[:-1]])
    pressed = held & ~previous
    released = previous & ~held
    return np.stack([held, pressed, released], axis=1).astype("<u2").tobytes()


def frame_log(frames: int) -> bytes:
    """Return the framerate log specified by a stage of `frames` frames."""
    return bytes([FULL_FRAME_RATE]) * -(-frames // LOG_FRAMES_PER_BYTE)


@dataclass(frozen=True)
class StageSpec:
    """Synthesized stage information."""

    number: int
    seed: int
    #: Held keys per frame.
    keys: npt.NDArray[np.uint16]
    player_x: float
    player_y: float
    #: `_GLOBALS` fields to write into the record's globals copy.
    globals: dict[str, int]
    template: bytes


def stage_record(spec: StageSpec) -> bytes:
    """Return `spec`'s 0x238-byte record."""
    if len(spec.template) != RECORD_SIZE:
        msg = f"a stage record is {RECORD_SIZE} bytes, not {len(spec.template)}"
        raise ReplayError(msg)
    record = bytearray(spec.template)
    frames = len(spec.keys)
    size = frames * FRAME_SIZE + len(frame_log(frames))
    struct.pack_into("<HHII", record, 0, spec.number, spec.seed, frames, size)
    struct.pack_into("<ii", record, PLAYER_POS_OFFSET, round(spec.player_x * 128), round(spec.player_y * 128))
    for name, value in spec.globals.items():
        struct.pack_into("<i", record, GLOBALS_OFFSET + _GLOBALS[name], value)
    return bytes(record)


def pack(body: bytes, user: bytes = b"") -> bytes:
    """Return the `.rpy` file specifying the entirety of `body`, then the user section `user`."""
    packed = lzss(body)
    for block_size, key, step in reversed(_PASSES):
        packed = bytes(scramble(packed, block_size, key, step))
    header = HEADER.pack(MAGIC, VERSION, 0, HEADER.size + len(packed), 0x100, 0, 0, len(packed), len(body))
    return header + packed + user


def encode(  # noqa: PLR0913
    stages: Sequence[StageSpec],
    *,
    name: str,
    timestamp: int,
    score_div10: int,
    character: int,
    difficulty: int,
    template_info: bytes,
    user_text: str,
) -> bytes:
    """Return the bytes of a cleared run's `.rpy` file for `stages`."""
    if len(template_info) != INFO_SIZE:
        msg = f"an info block is {INFO_SIZE} bytes, not {len(template_info)}"
        raise ReplayError(msg)
    info = bytearray(template_info)
    info[:_NAME_LEN] = name.encode("shift_jis")[:_NAME_LEN].ljust(_NAME_LEN)
    info[FLAGS_OFFSET] = 0
    struct.pack_into("<I", info, _INFO_TIMESTAMP_OFFSET, timestamp)
    struct.pack_into("<I", info, _INFO_SCORE_OFFSET, score_div10)
    struct.pack_into("<I", info, STAGE_COUNT_OFFSET, len(stages))
    info[CHARACTER_OFFSET] = character
    struct.pack_into("<I", info, _INFO_DIFFICULTY_OFFSET, difficulty)
    struct.pack_into("<I", info, END_STAGE_OFFSET, CLEARED_END_STAGE)
    body = bytes(info)
    for spec in stages:
        body += stage_record(spec) + frame_records(spec.keys) + frame_log(len(spec.keys))
    text = user_text.encode("shift_jis") + b"\0" * _USER_TEXT_PAD
    user = USER_MAGIC + struct.pack("<II", USER_HEADER_SIZE + len(text), _USER_KIND) + text
    return pack(body, user)
