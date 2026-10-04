"""Logic for evaluation results."""

import fcntl
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any, Self

from nokozero.utils import code_version

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from contextlib import ExitStack
    from pathlib import Path

PREFIX_FAILED = "prefix_failed"
CUT = "cut"


def tally(outcomes: Iterable[str], unknown: str) -> dict[str, float]:
    """Count outcomes and clear rates."""
    counts = Counter(outcomes)
    known = sum(counts.values()) - counts[unknown]
    return {
        **{name: counts[name] for name in ("complete", "hit", "truncated", unknown)},
        "played_clear_rate": round(counts["complete"] / max(1, known), 4),
    }


@dataclass
class Result:
    """The result of a model."""

    path: Path | None
    header: dict[str, Any]
    seeds: dict[int, dict[str, Any]] = field(default_factory=dict)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def parse(cls, path: Path | None, text: str) -> Self:
        """Read the lines of a result."""
        header, *rows = map(json.loads, text.split("\n")[:-1])
        result = cls(path, header)
        for row in rows:
            result.add(row)
        return result

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read the result at `path`."""
        return cls.parse(path, path.read_text())

    def add(self, row: Mapping[str, Any]) -> None:
        """Add a line."""
        if "session" in row:
            self.sessions.append({**row["session"], "last": row["session"]["started"]})
            return
        seed = int(row["seed"])
        known = self.seeds.get(seed)
        if known is not None and known["outcome"] != row["outcome"]:
            msg = f"{self.path}: seed {seed} ended {known['outcome']} and {row['outcome']}"
            raise ValueError(msg)
        self.seeds[seed] = dict(row)
        if self.sessions:
            self.sessions[-1]["last"] = row["time"]

    @property
    def sample(self) -> int:
        """The number of seeds evaluated (i.e. the sample size)."""
        return int(self.header["seeds"])

    @property
    def complete(self) -> bool:
        """Whether every seed of the sample has a corresponding line."""
        return len(self.seeds) >= self.sample

    def missing(self, seeds: Sequence[int]) -> list[int]:
        """Return the seeds of `seeds` without a result line."""
        return [seed for seed in seeds if seed not in self.seeds]

    @property
    def outcomes(self) -> dict[int, str]:
        """The outcome of each seed (`complete`, `hit`, `truncated`, or `prefix_failed`)."""
        return {seed: row["outcome"] for seed, row in sorted(self.seeds.items())}

    @property
    def segments(self) -> dict[str, dict[int, str]]:
        """Per-segment completion rates and results for full-stage runs."""
        by_segment: dict[str, dict[int, str]] = {}
        for seed, row in sorted(self.seeds.items()):
            for target, outcome in row.get("segments", {}).items():
                by_segment.setdefault(target, {})[seed] = outcome
        return by_segment

    @property
    def seconds(self) -> float:
        """The number of seconds spent in a session across all seeds tested."""
        return round(sum(session["last"] - session["started"] for session in self.sessions), 1)

    @property
    def codes(self) -> list[dict[str, Any]]:
        """The code of each session."""
        codes: list[dict[str, Any]] = []
        for session in self.sessions:
            if session["code"] not in codes:
                codes.append(session["code"])
        return codes

    def row(self) -> dict[str, Any]:
        """Summarize the result as its header plus outcome tallies."""
        outcomes = self.outcomes.values()
        lengths = [row["length"] for row in self.seeds.values() if "length" in row]
        counts = tally(outcomes, PREFIX_FAILED)
        segments = self.segments
        return {
            **self.header,
            **({} if self.complete else {"played": len(self.seeds)}),
            **counts,
            "clear_rate": round(counts["complete"] / max(1, self.sample), 4),
            "mean_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            "seconds": self.seconds,
            **({"codes": len(self.codes)} if len(self.codes) > 1 else {}),
            **({"segments": {target: segment(by_seed) for target, by_seed in segments.items()}} if segments else {}),
        }


def segment(by_seed: Mapping[int, str]) -> dict[str, float]:
    """Tally a segment's outcomes over the seeds that reached it."""
    return {"reached": len(by_seed), **tally(by_seed.values(), CUT)}


class Recorder:
    """A result being played."""

    def __init__(self, result: Result, file: IO[str] | None) -> None:
        self.result = result
        self._file = file

    def session(self, device: str) -> None:
        """Begin a playthrough of the result."""
        row = {"session": {"started": time.time(), "code": code_version(), "device": device}}
        self.result.add(row)
        self._write(row)

    def seed(self, row: Mapping[str, Any]) -> None:
        """Add a line for a seed."""
        stamped = {**row, "time": time.time()}
        self.result.add(stamped)
        self._write(stamped)

    def _write(self, row: Mapping[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(row) + "\n")
            self._file.flush()


def open_result(path: Path | None, header: Mapping[str, Any], stack: ExitStack) -> Recorder:
    """Open a model's result to play into."""
    wanted = dict(header)
    if path is None:
        return Recorder(Result(None, wanted), None)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = stack.enter_context(path.open("a+"))
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        msg = f"{path} is being written by another process"
        raise ValueError(msg) from error
    file.seek(0)
    text = file.read()
    if not text:
        file.write(json.dumps(wanted) + "\n")
        file.flush()
        return Recorder(Result(path, wanted), file)
    result = Result.parse(path, text)
    differing = sorted(name for name in {*wanted, *result.header} if wanted.get(name) != result.header.get(name))
    if differing:
        msg = (
            f"{path} contains the result of another evaluation (its {', '.join(differing)} differ); "
            "this one should have its own unique --out value"
        )
        raise ValueError(msg)
    hook = code_version()["hook"]
    hooks = {session["code"].get("hook") for session in result.sessions} - {None, hook}
    if hook is not None and hooks and not result.complete:
        msg = (
            f"{path} was started with another hook DLL ({', '.join(sorted(hooks))}); "
            "this evaluation should have its own unique --out value"
        )
        raise ValueError(msg)
    os.ftruncate(file.fileno(), text.rfind("\n") + 1)
    return Recorder(result, file)
