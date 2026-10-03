"""Wine process management."""

import ctypes
import fcntl
import os
import selectors
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final

from nokozero.utils import CHECKOUT, atomic_write, clean_stale_tmp, record_hook

if TYPE_CHECKING:
    import socket
    from collections.abc import Mapping

GAME_EXE: Final = "th15.exe"
HOOK_DLL: Final = "dinput8.dll"

_PDEATHSIG: Final = ("setpriv", "--pdeathsig", "KILL")


def default_hook_dll() -> Path:
    """The hook DLL path."""
    return CHECKOUT / "nokozero_hook/target/i686-pc-windows-gnu/release/nokozero_hook.dll"


def state_dir() -> Path:
    """Location of per-instance state such as Wine prefixes and stderr files."""
    data_home = Path(os.environ.get("XDG_DATA_HOME", ""))
    if not data_home.is_absolute():
        data_home = Path.home() / ".local/share"
    return data_home / "nokozero"


@cache
def _game_id(game_dir: Path) -> str:
    """Identify a game dir based on per-instance state."""
    real = game_dir.resolve()
    return f"{real.name}-{sha256(str(real).encode()).hexdigest()[:12]}"


def _instances_root(game_dir: Path) -> Path:
    """Location of `game_dir`'s per-instance state tree."""
    return state_dir() / "instances" / _game_id(game_dir)


def instance_state_dir(game_dir: Path, index: int) -> Path:
    """Location of instance `index`'s state such as the Wine prefix and stderr file."""
    return _instances_root(game_dir) / str(index)


_PREFIX_NAME: Final = "prefix"


def instance_prefix(game_dir: Path, index: int) -> Path:
    """Instance `index`'s Wine prefix (under `instance_state_dir`)."""
    return instance_state_dir(game_dir, index) / _PREFIX_NAME


def instance_stderr(game_dir: Path, index: int) -> Path:
    """Instance `index`'s stderr file (under `instance_state_dir`)."""
    return instance_state_dir(game_dir, index) / "stderr"


def check_tools() -> None:
    """Fail fast with a message if a required external binary is missing."""
    for tool, package in (
        ("wine", "wine"),
        ("wineserver", "wine"),
        ("setpriv", "util-linux"),
    ):
        if shutil.which(tool) is None:
            msg = f"`{tool}` (from {package}) not found in PATH"
            raise FileNotFoundError(msg)


_GAME_FILES: Final = (GAME_EXE, "th15.dat")


def check_game_dir(game_dir: Path) -> None:
    """Fail fast if `game_dir` does not contain the game."""
    missing = [name for name in _GAME_FILES if not (game_dir / name).is_file()]
    if missing:
        listed = ", ".join(missing)
        msg = f"{game_dir} is not a game directory: no {listed}"
        raise FileNotFoundError(msg)


def _hook_dll_problem(dll_path: Path, detail: str) -> str:
    """Description for a hook DLL problem."""
    ours = dll_path == default_hook_dll()
    fix = "build with `just build-hook`" if ours else "check --hook-dll"
    return f"hook DLL at {dll_path}: {detail} ({fix})"


def acquire_game_dir_lock(game_dir: Path) -> int:
    """Take the driver lock. The caller holds the fd for the Env's lifetime. The fd must remain non-inheritable."""
    fd = os.open(game_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        msg = f"another driver is already running on {game_dir}"
        raise RuntimeError(msg) from None
    except OSError:
        os.close(fd)
        raise
    return fd


_REAP_TIMEOUT: Final = 5.0


def reap_prefix(wine_prefix: Path) -> None:
    """SIGKILL every process of `wine_prefix` and wait until they are gone."""
    if not wine_prefix.is_dir():
        return
    env = {**os.environ, "WINEPREFIX": str(wine_prefix)}
    for flag in ("-k", "-w"):
        try:
            subprocess.run(  # noqa: S603
                ["wineserver", flag],  # noqa: S607
                env=env,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_REAP_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return


def reap_stale_instances(game_dir: Path) -> None:
    """Reap every per-instance prefix associated with `game_dir`."""
    root = _instances_root(game_dir)
    if not root.is_dir():
        return
    prefixes = [child / _PREFIX_NAME for child in sorted(root.iterdir())]
    if not prefixes:
        return
    with ThreadPoolExecutor(max_workers=len(prefixes)) as pool:
        list(pool.map(reap_prefix, prefixes))


_PROVISION_VERSION: Final = "1"
_PROVISION_STAMP: Final = ".nokozero-provisioned"
_PROVISION_TIMEOUT: Final = 120.0


def _provision_env(wine_prefix: Path) -> dict[str, str]:
    """Build the environment for provisioning `wine` calls."""
    env = dict(os.environ)
    env.update(
        WINEPREFIX=str(wine_prefix),
        WINEDLLOVERRIDES="mscoree=d;mshtml=d;winemenubuilder.exe=d",
        WINEDEBUG="-all",
    )
    return env


def provision_prefixes(game_dir: Path, count: int) -> None:
    """Apply one-time per-prefix setup to the first `count` instances."""
    pending: list[tuple[Path, subprocess.Popen[bytes]]] = []
    for index in range(count):
        prefix = instance_prefix(game_dir, index)
        try:
            if (prefix / _PROVISION_STAMP).read_text() == _PROVISION_VERSION:
                continue
        except OSError:
            pass
        prefix.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(  # noqa: S603
            [
                *_PDEATHSIG,
                *("wine", "reg", "add", r"HKCU\Software\Wine\WineDbg"),
                *("/v", "ShowCrashDialog", "/t", "REG_DWORD", "/d", "0", "/f"),
            ],
            env=_provision_env(prefix),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        pending.append((prefix, process))
    deadline = time.monotonic() + _PROVISION_TIMEOUT
    for prefix, process in pending:
        try:
            _, stderr = process.communicate(timeout=max(deadline - time.monotonic(), 1.0))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            msg = f"provisioning {prefix} timed out ({_PROVISION_TIMEOUT}s; is wine wedged?)"
            raise RuntimeError(msg) from None
        if process.returncode != 0:
            detail = stderr[-STDERR_TAIL_BYTES:].decode(errors="replace").strip()
            msg = f"provisioning {prefix} failed with exit {process.returncode}: {detail}"
            raise RuntimeError(msg)
        (prefix / _PROVISION_STAMP).write_text(_PROVISION_VERSION)


@cache
def hook_image(dll_path: Path) -> bytes:
    """Return the hook DLL's bytes."""
    try:
        image = dll_path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(_hook_dll_problem(dll_path, "not found")) from error
    except OSError as error:
        msg = _hook_dll_problem(dll_path, str(error))
        raise OSError(msg) from error
    record_hook(image)
    return image


def deploy_hook(game_dir: Path, dll_path: Path) -> None:
    """Install the hook into the game dir. Must be called under the game directory lock."""
    hook = hook_image(dll_path)
    dest = game_dir / HOOK_DLL
    clean_stale_tmp(dest)
    try:
        if dest.read_bytes() == hook:
            return
    except OSError:
        pass
    atomic_write(dest, hook)


def game_process_env(
    *,
    connect_addr: str,
    headed: bool,
    wine_prefix: Path,
    character: int,
) -> Mapping[str, str]:
    """Build the child's environment."""
    env = dict(os.environ)
    env.update(
        NOKOZERO_CONNECT=connect_addr,
        NOKOZERO_CHARACTER=str(character),
        WINEDLLOVERRIDES="dinput8=n,b;mscoree=d;mshtml=d;winemenubuilder.exe=d",
        WINEDEBUG="-all",
        EGL_LOG_LEVEL="fatal",
        WINEFSYNC="1",  # futex2-based sync; faster than esync
        WINE_LARGE_ADDRESS_AWARE="1",
        # threaded GL command submission
        mesa_glthread="true",
        __GL_THREADED_OPTIMIZATIONS="1",
        # no vsync
        vblank_mode="0",
        __GL_SYNC_TO_VBLANK="0",
    )
    env["NOKOZERO_HEADLESS"] = "0" if headed else "1"
    env["WINEPREFIX"] = str(wine_prefix)
    return env


def spawn_game(  # noqa: PLR0913
    *,
    game_dir: Path,
    connect_addr: str,
    headed: bool,
    wine_prefix: Path,
    stderr_path: Path,
    character: int,
) -> subprocess.Popen[bytes]:
    """Start one game instance under Wine. Must be called from a thread that lives as long as the driver."""
    exe = (game_dir / GAME_EXE).resolve()
    wine_prefix.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("wb") as stderr_file:
        return subprocess.Popen(  # noqa: S603
            [*_PDEATHSIG, "wine", str(exe)],
            cwd=game_dir,
            env=game_process_env(
                connect_addr=connect_addr,
                headed=headed,
                character=character,
                wine_prefix=wine_prefix,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )


_SYS_PIDFD_OPEN: Final = 434


def _pidfd_open(pid: int) -> int:
    """`os.pidfd_open` with a raw-syscall fallback."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # Standalone CPython builds (e.g. uv's python-build-standalone) are compiled against sufficiently old sysroots
    # where `os.pidfd_open` is absent, but the syscall exists on every kernel supported by this driver.
    libc = ctypes.CDLL(None, use_errno=True)
    fd: int = libc.syscall(_SYS_PIDFD_OPEN, pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _try_accept(listener: socket.socket) -> socket.socket | None:
    """Take a pending connection. Returns `None` if the connection wasn't actually ready."""
    listener.setblocking(False)  # noqa: FBT003
    try:
        sock, _addr = listener.accept()
    except BlockingIOError:
        return None
    return sock


def await_connection(
    listener: socket.socket,
    process: subprocess.Popen[bytes],
    deadline: float,
) -> socket.socket | None:
    """Accept a connection. Returns `None` if `process` dies or `deadline` passes."""
    try:
        pidfd = _pidfd_open(process.pid)
    except ProcessLookupError:
        return _try_accept(listener)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(listener, selectors.EVENT_READ)
            selector.register(pidfd, selectors.EVENT_READ)
            while True:
                ready = selector.select(timeout=max(deadline - time.monotonic(), 0.0))
                sock = _try_accept(listener)
                if sock is not None:
                    return sock
                if any(key.fd == pidfd for key, _ in ready):
                    return None
                if time.monotonic() >= deadline:
                    return None
    finally:
        os.close(pidfd)


STDERR_TAIL_BYTES: Final = 2048


def stderr_tail(stderr_path: Path) -> str:
    """Return the last `STDERR_TAIL_BYTES` of an instance's stderr."""
    try:
        with stderr_path.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - STDERR_TAIL_BYTES))
            data = f.read(STDERR_TAIL_BYTES)
    except OSError:
        return ""
    return data.decode(errors="replace")


def kill_and_reap(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL and reap an instance."""
    process.kill()
    try:
        process.wait(timeout=_REAP_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"pid {process.pid} survived SIGKILL (uninterruptible sleep?)", file=sys.stderr)  # noqa: T201
