"""Vectorized environment."""

import os
import select
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Self

from nokozero import spawn, wire

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Generator, Mapping, Sequence
    from pathlib import Path
    from typing import NoReturn


TAPE_BUDGET_BASE = 5.0
TAPE_BUDGET_PER_FRAME = 0.002


def command_budget(command: bytes, step_timeout: float) -> float:
    """Return the number of seconds that a command's answer is due within."""
    frames = wire.command_frames(command)
    return TAPE_BUDGET_BASE + TAPE_BUDGET_PER_FRAME * frames if frames > 1 else step_timeout


@dataclass(frozen=True, kw_only=True)
class EnvConfig:
    """Operator configuration for a driver run."""

    game_dir: Path
    hook_dll: Path = field(default_factory=spawn.default_hook_dll)
    num_instances: int = 1
    headed: bool = False
    character: int = 0
    step_timeout: float = 5.0
    startup_timeout: float = 60.0
    waiting_scene_delay: float = 0.003
    collect_window: float = 0.0
    reset_pending_floor: int = 4000
    reset_pending_deadline: float = 30.0
    unmanaged_floor: int = 600
    unmanaged_deadline: float = 10.0

    def __post_init__(self) -> None:
        """Reject an invalid configuration before anything is spawned."""
        if self.num_instances < 1:
            msg = f"num_instances must be at least 1, got {self.num_instances}"
            raise ValueError(msg)
        if not 0 <= self.character <= wire.MAX_CHARACTER:
            msg = f"character {self.character} out of range (0-{wire.MAX_CHARACTER})"
            raise ValueError(msg)
        for name, budget in (
            ("step_timeout", self.step_timeout),
            ("startup_timeout", self.startup_timeout),
            ("reset_pending_deadline", self.reset_pending_deadline),
            ("unmanaged_deadline", self.unmanaged_deadline),
        ):
            if budget <= 0:
                msg = f"{name} must be positive, got {budget}"
                raise ValueError(msg)
        for name, floor in (
            ("waiting_scene_delay", self.waiting_scene_delay),
            ("collect_window", self.collect_window),
            ("reset_pending_floor", self.reset_pending_floor),
            ("unmanaged_floor", self.unmanaged_floor),
        ):
            if floor < 0:
                msg = f"{name} must not be negative, got {floor}"
                raise ValueError(msg)


class Fleet(Protocol):
    """Per-instance `Env` information for a driver."""

    @property
    def instances(self) -> Sequence[object]:
        """One entry per instance."""
        ...

    @property
    def character(self) -> int:
        """The character used by the fleet."""
        ...

    @property
    def collect_window(self) -> float:
        """The number of seconds that a runner may keep collecting after the first exchange."""
        ...

    def next_reset_seq(self) -> int:
        """Return the next reset seq."""
        ...

    def exchange(self, commands: Mapping[int, bytes], *, timeout: float | None = None) -> dict[int, wire.Observation]:
        """Send commands to some instances and return the observations that have arrived."""
        ...

    def close(self) -> None:
        """Tear the fleet down. Idempotent."""
        ...


class InstanceDied(RuntimeError):  # noqa: N818
    """An instance violated the lockstep setup; the `Env` is closed and needs to be rebuilt."""

    def __init__(self, index: int, reason: str, tail: str) -> None:
        detail = f"instance {index}: {reason}"
        if tail:
            detail = f"{detail}\n--- stderr tail ---\n{tail}"
        super().__init__(detail)


@contextmanager
def _teardown_step(index: int, label: str) -> Generator[None]:
    """Attempt to tear down instance `index`."""
    try:
        yield
    except Exception as error:  # noqa: BLE001
        print(f"instance {index}: {label} failed: {error}", file=sys.stderr)  # noqa: T201


class Instance:
    """A game process. Use via `Env`."""

    def __init__(self, config: EnvConfig, index: int) -> None:
        self.index = index
        self._config = config
        self.stderr_path = spawn.instance_stderr(config.game_dir, index)
        self._prefix = spawn.instance_prefix(config.game_dir, index)
        self._deadline = time.monotonic() + config.startup_timeout
        self._started = False
        self._spawn_thread = threading.current_thread()
        self._listener = socket.create_server(("127.0.0.1", 0))
        host, port = self._listener.getsockname()
        try:
            self._process: subprocess.Popen[bytes] = spawn.spawn_game(
                game_dir=config.game_dir,
                connect_addr=f"{host}:{port}",
                headed=config.headed,
                character=config.character,
                wine_prefix=self._prefix,
                stderr_path=self.stderr_path,
            )
        except BaseException:
            self._listener.close()
            raise
        self._sock: socket.socket | None = None

    def _remaining(self) -> float:
        """Time remaining in the cold-start budget."""
        return max(self._deadline - time.monotonic(), wire.TIMEOUT_FLOOR)

    def accept(self) -> None:
        """Wait for the hook to dial in. Uses one connection per process lifetime."""
        try:
            sock = spawn.await_connection(self._listener, self._process, self._deadline)
        except OSError as error:
            self.fail(f"hook never connected ({error})")
        if sock is None:
            if self._process.poll() is None:
                budget = self._config.startup_timeout
                self.fail(f"hook never connected within the cold-start budget ({budget}s)")
            self.fail("hook never connected")
        self._listener.close()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def _connection(self) -> socket.socket:
        """Return the hook's connection. The instance must be connected."""
        if self._sock is None:
            self.fail("not connected")
        return self._sock

    def fileno(self) -> int:
        """Return the connection's file descriptor for `select`. The instance must be connected."""
        return self._connection().fileno()

    def recv_observation(self) -> wire.Observation:
        """Block for this instance's next observation."""
        sock = self._connection()
        if not self._started:
            sock.settimeout(self._remaining())
        try:
            observation = wire.recv_observation(sock)
        except (TimeoutError, OSError, wire.ProtocolError) as error:
            reason = str(error) or type(error).__name__
            if not self._started:
                budget = self._config.startup_timeout
                reason = f"no first observation within the cold-start budget ({budget}s): {reason}"
            self.fail(reason)
        if not self._started:
            self._started = True
            sock.settimeout(self._config.step_timeout)
        return observation

    def send(self, frame: bytes) -> None:
        """Send a prebuilt command frame (`wire.act_frame` / `wire.reset_frame`)."""
        sock = self._connection()
        try:
            sock.sendall(frame)
        except OSError as error:
            self.fail(str(error))

    def kill(self) -> None:
        """Signal the game process without waiting for it. Idempotent."""
        self._process.kill()

    def close(self) -> None:
        """Kill and reap the game process and its prefix. Idempotent."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._listener.close()
        with _teardown_step(self.index, "kill/reap"):
            spawn.kill_and_reap(self._process)
        with _teardown_step(self.index, "prefix reap"):
            spawn.reap_prefix(self._prefix)

    def fail(self, reason: str) -> NoReturn:
        """Declare the instance dead for `reason`. Closes it and raises `InstanceDied`."""
        exited = self._process.poll()
        if exited is not None:
            reason = f"{reason} (process exited with {exited})"
        if not self._spawn_thread.is_alive():
            reason = (
                f"{reason} [the thread that spawned this instance has exited, which kills instances "
                "via PR_SET_PDEATHSIG; construct Env on a thread that outlives it]"
            )
        self.close()
        raise InstanceDied(self.index, reason, spawn.stderr_tail(self.stderr_path))


class Env:
    """N instances driven by exchanges.

    Any `InstanceDied` or `WatchdogTripped` closes the entire `Env`. To continue, a new `Env` should be built.
    Closed `Env` instances will refuse further steps, but `close` remains callable.
    """

    def __init__(self, config: EnvConfig) -> None:
        self._config = config
        self._closed = False
        self._reset_seq = 0
        # {instance w/ in-flight command : observation deadline}
        self._inflight: dict[int, float] = {}
        #: Whether each instance's latest observation was in a stage.
        self._in_stage: list[bool] = [False] * config.num_instances
        self.instances: list[Instance] = []
        self._lock: int | None = None
        self._watchdogs = [_Watchdogs(config, index) for index in range(config.num_instances)]
        spawn.check_tools()
        spawn.check_game_dir(config.game_dir)
        spawn.hook_image(config.hook_dll)
        self._lock = spawn.acquire_game_dir_lock(config.game_dir)
        try:
            spawn.reap_stale_instances(config.game_dir)
            spawn.provision_prefixes(config.game_dir, config.num_instances)
            spawn.deploy_hook(config.game_dir, config.hook_dll)
            for i in range(config.num_instances):
                self.instances.append(Instance(config, i))
            for instance in self.instances:
                instance.accept()
            self._observe({index: instance.recv_observation() for index, instance in enumerate(self.instances)})
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def exchange(self, commands: Mapping[int, bytes], *, timeout: float | None = None) -> dict[int, wire.Observation]:
        """Send `commands` to corresponding instances and return the observations that have arrived.

        Blocks until `timeout` passes or at least one instance with an in-flight command has responded.
        """
        self._check_open()
        if not any(self._in_stage) and not self._inflight:
            time.sleep(self._config.waiting_scene_delay)
        observations: dict[int, wire.Observation] = {}
        try:
            now = time.monotonic()
            for index, command in commands.items():
                if index in self._inflight:
                    msg = f"instance {index}: a command is already in flight"
                    raise RuntimeError(msg)
                self.instances[index].send(command)
                self._inflight[index] = now + command_budget(command, self._config.step_timeout)
            if not self._inflight:
                return observations
            for index in self._wait_any(timeout):
                observations[index] = self.instances[index].recv_observation()
                del self._inflight[index]
            self._observe(observations)
        except InstanceDied, WatchdogTripped:
            self.close()
            raise
        return observations

    def _wait_any(self, timeout: float | None = None) -> list[int]:
        """Block until `timeout` passes or an in-flight instance has data."""
        give_up = None if timeout is None else time.monotonic() + timeout
        while True:
            now = time.monotonic()
            wait = max(min(self._inflight.values()) - now, 0.0)
            if give_up is not None:
                wait = min(wait, max(give_up - now, 0.0))
            by_fd = {self.instances[index].fileno(): index for index in self._inflight}
            readable, _, _ = select.select(list(by_fd), [], [], wait)
            if readable:
                return [by_fd[fd] for fd in readable]
            now = time.monotonic()
            for index, deadline in self._inflight.items():
                if now >= deadline:
                    self.instances[index].fail("no observation within the command's budget")
            if give_up is not None and now >= give_up:
                return []

    @property
    def character(self) -> int:
        """The character used by the fleet."""
        return self._config.character

    @property
    def collect_window(self) -> float:
        """The number of seconds that a runner may keep collecting after the first exchange."""
        return self._config.collect_window

    def next_reset_seq(self) -> int:
        """Return the next reset seq."""
        self._reset_seq += 1
        return self._reset_seq

    def _observe(self, observations: Mapping[int, wire.Observation]) -> None:
        """Send observations to corresponding instances' watchdog set."""
        for index, obs in observations.items():
            self._in_stage[index] = obs.in_stage
            self._watchdogs[index].observe(obs)

    def close(self) -> None:
        """Tear down every instance and release the driver lock. Idempotent."""
        self._closed = True
        try:
            for instance in self.instances:
                with _teardown_step(instance.index, "kill"):
                    instance.kill()
            if self.instances:
                with ThreadPoolExecutor(max_workers=len(self.instances)) as pool:
                    list(pool.map(_close_instance, self.instances))
        finally:
            if self._lock is not None:
                os.close(self._lock)  # closing the fd releases the flock
                self._lock = None

    def _check_open(self) -> None:
        if self._closed:
            msg = "Env is closed; build a new one"
            raise RuntimeError(msg)


def _close_instance(instance: Instance) -> None:
    with _teardown_step(instance.index, "close"):
        instance.close()


class WatchdogTripped(RuntimeError):  # noqa: N818
    """A watchdog gate tripped; the `Env` is closed and needs to be rebuilt."""


@dataclass
class _Gate:
    """A watchdog over a persistent condition."""

    name: str
    floor: int
    deadline: float

    _steps: int = field(default=0, init=False)
    _since: float | None = field(default=None, init=False)

    def feed(self, *, active: bool, now: float) -> str | None:
        """Advance one observation; returns the trip description once both limits pass."""
        if not active:
            self._steps = 0
            self._since = None
            return None
        if self._since is None:
            self._since = now
        self._steps += 1
        elapsed = now - self._since
        if self._steps > self.floor and elapsed > self.deadline:
            return (
                f"{self.name} for {self._steps} steps in {elapsed:.1f}s "
                f"({self._steps / elapsed:.0f} steps/s; floor {self.floor} steps, "
                f"deadline {self.deadline}s)"
            )
        return None


@dataclass
class _Watchdogs:
    """Watchdogs for an instance."""

    config: EnvConfig
    index: int
    clock: Callable[[], float] = time.monotonic

    _pending: _Gate = field(init=False)
    _unmanaged: _Gate = field(init=False)

    def __post_init__(self) -> None:
        self._pending = _Gate(
            "reset still pending",
            self.config.reset_pending_floor,
            self.config.reset_pending_deadline,
        )
        self._unmanaged = _Gate(
            "unmanaged scene with no reset pending (unhandled terminal screen?)",
            self.config.unmanaged_floor,
            self.config.unmanaged_deadline,
        )

    def observe(self, obs: wire.Observation) -> None:
        """Feed one observation. Raises `WatchdogTripped` if a gate trips."""
        now = self.clock()
        pending = obs.reset_outcome == wire.ResetOutcome.PENDING
        if trip := self._pending.feed(active=pending, now=now):
            self._trip(trip)

        unmanaged = obs.gamemode == wire.Gamemode.OTHER and not pending
        if trip := self._unmanaged.feed(active=unmanaged, now=now):
            self._trip(trip)

    def _trip(self, reason: str) -> NoReturn:
        msg = f"instance {self.index}: {reason}"
        raise WatchdogTripped(msg)


MAX_FLEET_DEATHS = 10


class Fleets[F: Fleet]:
    """Fleet for a command. Booted when first needed and replaced each time one dies."""

    def __init__(
        self,
        config: EnvConfig,
        make: Callable[[EnvConfig], F],
        log: Callable[[dict[str, object]], None],
    ) -> None:
        self.config = config
        self.make = make
        self.log = log
        self.deaths = 0
        self._env: F | None = None
        self._booted = False

    @property
    def booted(self) -> bool:
        """Whether a fleet has come up yet."""
        return self._booted

    def _live(self) -> F:
        """Return the live fleet, booting one if there is none."""
        while self._env is None:
            try:
                self._env = self.make(self.config)
            except (InstanceDied, WatchdogTripped) as death:
                if not self._booted:
                    raise
                self._lost(death, booting=True)
        self._booted = True
        return self._env

    def _lost(self, death: Exception, *, booting: bool = False) -> None:
        """Count the live fleet lost to `death` and log the event. Raises `death` past `MAX_FLEET_DEATHS`."""
        self._env = None
        self.deaths += 1
        self.log({"fleet_died": str(death)[:200], "fleet_deaths": self.deaths, "booting": booting})
        if self.deaths > MAX_FLEET_DEATHS:
            raise death

    def run(self, work: Callable[[F], object], *, again: Callable[[], bool] = lambda: True) -> None:
        """Invoke `work` on the live fleet (and on a replacement each time one dies)."""
        while True:
            env = self._live()
            try:
                work(env)
            except (InstanceDied, WatchdogTripped) as death:
                self._lost(death)
                if again():
                    continue
            return

    def close(self) -> None:
        """Close the live fleet if one exists."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
