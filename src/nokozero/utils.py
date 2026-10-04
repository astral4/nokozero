"""I/O and filesystem utilities."""

import hashlib
import json
import os
import subprocess
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


_TMP_NAME = "{name}.tmp-{pid}"


def print_row(row: Mapping[str, object]) -> None:
    """Print `row` as a JSON line."""
    print(json.dumps(row), flush=True)  # noqa: T201


def tmp_sibling(path: Path) -> Path:
    """Return the path of the temp file that `atomic_write` would use for `path`."""
    return path.with_name(_TMP_NAME.format(name=path.name, pid=os.getpid()))


def atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path`."""
    tmp = tmp_sibling(path)
    try:
        tmp.write_bytes(data)
        tmp.rename(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def check_writable(path: Path | None) -> None:
    """Fail fast if `path` cannot be written."""
    if path is None:
        return
    if path.is_dir():
        msg = f"{path} is a directory"
        raise IsADirectoryError(msg)
    probe = tmp_sibling(path)
    try:
        probe.write_bytes(b"")
    except OSError as error:
        msg = f"{path} is not writable: {error}"
        raise OSError(msg) from error
    finally:
        probe.unlink(missing_ok=True)


def clean_stale_tmp(path: Path) -> None:
    """Remove `atomic_write` temp files stranded beside `path` by a killed writer.

    The caller must hold whatever lock covers `path`'s directory.
    """
    for stale in path.parent.glob(_TMP_NAME.format(name=path.name, pid="[0-9]*")):
        with suppress(OSError):
            stale.unlink()


_PACKAGE = Path(__file__).parent
CHECKOUT = _PACKAGE.resolve().parents[1]
_hook: str | None = None


def record_hook(image: bytes) -> None:
    """Record the hook DLL deployed by this process."""
    global _hook  # noqa: PLW0603
    _hook = hashlib.blake2b(image, digest_size=8).hexdigest()


@cache
def _source_version() -> dict[str, Any]:
    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(_PACKAGE.rglob("*.py")):
        digest.update(path.relative_to(_PACKAGE).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    commit: str | None = None
    dirty: bool | None = None
    git = ["git", "-C", str(_PACKAGE)]
    with suppress(OSError, subprocess.SubprocessError):
        head, status = (
            subprocess.run(  # noqa: S603
                [*git, *args], capture_output=True, check=True, timeout=10
            ).stdout
            for args in (("rev-parse", "HEAD"), ("status", "--porcelain"))
        )
        commit, dirty = head.decode().strip(), bool(status.strip())
    return {"source": digest.hexdigest(), "commit": commit, "dirty": dirty}


def code_version() -> dict[str, Any]:
    """Specify the code used by this process."""
    return {**_source_version(), "hook": _hook}


def files_digest(paths: Iterable[Path]) -> str:
    """Compute a digest from the bytes of content from `paths`."""
    digest = hashlib.blake2b(digest_size=8)
    for path in paths:
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def write_result(path: Path, row: Mapping[str, Any], per_seed: Mapping[int, object]) -> None:
    """Write a result file."""
    per = {str(seed): value for seed, value in sorted(per_seed.items())}
    result = {**row, "code": code_version(), "per_seed": per}
    atomic_write(path, json.dumps(result, indent=1).encode())
