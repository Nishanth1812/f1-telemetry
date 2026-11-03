"""Minimal localhost WebSocket transport for the P0 dashboard demo.

Scope, deliberately. PLAN.md section 8.4 asks for one WebSocket with three streams
(``live`` 30 Hz, ``trace`` 5 Hz, ``event``) of delta-encoded binary snapshots. That is
P8-T1 and it needs a real physics source to have anything to encode. P0 needs less and
gets it: a loopback-bound JSON server that pumps :mod:`f1telemetry.synthetic` frames at
30 Hz, which is exactly what ``web/src/telemetry/socket.ts`` parses today.

Design constraints this file honours:

* **Localhost only.** The bind address is validated against a loopback allowlist, so
  the demo cannot be started listening on a routable interface by a stray flag.
* **Simulated time, real pacing.** ``time_us`` comes from the source and is never
  derived from a clock. The monotonic loop clock is used only to decide *when to send*
  a frame, and each sleep targets an absolute deadline computed from the frame's own
  simulated timestamp, so a slow tick does not accumulate drift - it just sends the
  next frames back to back until it catches up.
* **Bounded, isolated delivery.** The pump never awaits a socket. Each viewer has its
  own bounded queue and sender task. A viewer whose queue overflows, or whose send
  stalls, is disconnected rather than silently skipped, so no sample or event is lost
  for a viewer that stays connected, and a slow viewer cannot hold up the others.
* **Clean shutdown.** :meth:`TelemetryServer.run` returns when its stop event is set;
  the ``serve`` context manager closes the listening socket and every open connection
  on the way out, and clients mid-frame are closed rather than left hanging. SIGINT
  and SIGTERM are wired to the same event where the platform allows it.

Run it with ``just serve`` (or ``uv run f1-serve``); the default endpoint is
``ws://localhost:8765/ws``, matching ``web/src/telemetry/config.ts``.

With ``--replay-parquet`` the source is a saved run instead of synthetic data
(P5-T9). The run's saved fault annotations are attached to the replay source, so
the dashboard's event view shows the injections along with the channel stream -
the frames themselves keep the live contract, annotations ride the optional
``events`` key.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, override
from urllib.parse import urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from f1telemetry.contracts.channels import load_channel_contract
from f1telemetry.live_car import DrivenCarSource
from f1telemetry.runs import RunStore, RunStoreError
from f1telemetry.synthetic import BROADCAST_HZ, Frame, SyntheticSource
from f1telemetry.telemetry.pipeline import read_annotations
from f1telemetry.testing.replay import ReplaySource
from f1telemetry.testing.scenario_lab import (
    _REQUEST_ID,  # pyright: ignore[reportPrivateUsage]
    ScenarioLabError,
    resting_car_values,
    run_lab,
)
from f1telemetry.web_site import WebSite

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PATH",
    "DEFAULT_PORT",
    "TelemetryServer",
    "TransportError",
    "main",
    "serve_forever",
]

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_PATH = "/ws"

_CHUNK_US = 1_000_000
"""Simulated microseconds pulled from the source per pass. Bounds generator lag only."""
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"})
_MAX_COMMAND_BYTES = 2_048
_QUEUE_LIMIT = 64
"""Messages a viewer may have waiting. Overflow disconnects that viewer; nothing is skipped."""
_SEND_TIMEOUT_S = 2.0
"""A send still pending after this long marks its viewer as stalled and disconnects it."""
_CLOSE_TIMEOUT_S = 2.0
_DRAIN_TIMEOUT_S = 5.0
"""How long a finished source waits for viewers to take their final queued messages."""
_SLOW_CLOSE = (1013, "viewer fell behind the live stream; reconnect to resume")
_LOGGER = logging.getLogger(__name__)


class TransportError(ValueError):
    """Raised for a transport configuration that must not be served."""


class FrameSource(Protocol):
    """Minimal source contract required by the WebSocket publisher."""

    @property
    def now_us(self) -> int: ...

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]: ...


@dataclass(eq=False)
class _Viewer:
    """One connected dashboard: its socket, its bounded output queue and the task sending it."""

    connection: ServerConnection
    queue: asyncio.Queue[str]
    detailed: bool = False
    sender: asyncio.Task[None] | None = None


class TelemetryServer:
    """Fan-out of source frames to every connected dashboard, one bounded queue per viewer.

    Args:
        source: the frame iterator's owner. One source per server; it holds the
            simulated clock, so two servers cannot share it.
        host: loopback name or address only.
        port: TCP port, 0 to let the OS pick (useful in tests).
        path: the single accepted URL path; anything else is closed with 1008.
        max_hz: upper bound on delivery rate, applied by the source's aggregation.
        run_store: where successful scenario runs are saved before they play; ``None``
            serves without saved runs.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        path: str = DEFAULT_PATH,
        max_hz: float = BROADCAST_HZ,
        run_store: RunStore | None = None,
        web_root: Path | None = None,
        read_only: bool = False,
        allowed_origins: Sequence[str] = (),
    ) -> None:
        origins = frozenset(_checked_origin(origin) for origin in allowed_origins)
        if host.lower() not in _LOOPBACK_HOSTS and not (read_only and origins):
            allowed = ", ".join(sorted(_LOOPBACK_HOSTS))
            raise TransportError(
                f"refusing to bind {host!r}: default is loopback-only ({allowed}); "
                "public binding requires read-only mode and explicit allowed origins"
            )
        if not 0 <= port <= 65535:
            raise TransportError(f"port must be in 0..65535, got {port}")
        self._source: FrameSource = source
        # One address keeps an ephemeral listener and its reported port on the same socket.
        self._host: str = "127.0.0.1" if host.lower() == "localhost" else host.lower()
        self._port: int = port
        self._path: str = path
        self._max_hz: float = max_hz
        self._run_store: RunStore | None = run_store
        self._viewers: dict[ServerConnection, _Viewer] = {}
        self._closing: set[asyncio.Task[None]] = set()
        self._stop: asyncio.Event | None = None
        self._ready: bool = False
        self._scenario_lock: asyncio.Lock = asyncio.Lock()
        self._frames_produced: int = 0
        self._messages_sent: int = 0
        self._bytes_sent: int = 0
        self._send_failures: int = 0
        self._slow_disconnects: int = 0
        self._scenario_failures: int = 0
        self._read_only: bool = read_only
        self._allowed_origins: frozenset[str] = origins
        self._last_result: dict[str, object] | None = None
        self._playback_started: float = 0.0
        self._site: WebSite = WebSite(web_root, lambda: self.metrics, read_only=read_only)

    @property
    def url(self) -> str:
        """The endpoint to point ``VITE_TELEMETRY_WS_URL`` at."""
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"ws://{host}:{self._port}{self._path}"

    @property
    def client_count(self) -> int:
        return len(self._viewers)

    @property
    def ready(self) -> bool:
        """True from the moment the listening socket accepts connections until shutdown."""
        return self._ready

    @property
    def metrics(self) -> dict[str, object]:
        """JSON-ready counters for operators: fixed keys, plain integers and booleans."""
        detailed = sum(1 for viewer in self._viewers.values() if viewer.detailed)
        return {
            "ready": self._ready,
            "clients": {"active": len(self._viewers), "detailed": detailed},
            "frames_produced": self._frames_produced,
            "messages_sent": self._messages_sent,
            "bytes_sent": self._bytes_sent,
            "send_failures": self._send_failures,
            "slow_disconnects": self._slow_disconnects,
            "scenario_failures": self._scenario_failures,
            "source": _source_state(self._source),
        }

    def stop(self) -> None:
        """Ask :meth:`run` to finish: no new frames, socket released."""
        if self._stop is not None:
            self._stop.set()

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Serve until ``stop`` is set (or a fresh :meth:`stop` is called).

        The event is created here when omitted, so :meth:`stop` works from another
        task. Everything the call opened is closed before this coroutine returns.
        """
        self._stop = asyncio.Event() if stop is None else stop
        loop = asyncio.get_running_loop()
        try:
            async with serve(
                self._handle,
                self._host,
                self._port,
                compression=None,
                max_size=_MAX_COMMAND_BYTES,
                process_request=self._site.process_request,
            ) as server:
                self._port = _bound_port(server, self._port)
                self._ready = True
                await self._pump(loop)
                # Close viewers before serve() exits, so its handler wait cannot outlast them.
                await self._close_clients()
        finally:
            self._ready = False
            self._stop = None
            await self._close_clients()

    async def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        """Broadcast the source's frames, paced to the wall clock, until stopped.

        The pacing deadline is absolute: ``start + frame.time_us``. Simulated time
        therefore tracks real time without drift, and falling behind compresses the
        gap instead of accumulating it.
        """
        start = loop.time() - self._source.now_us / 1_000_000
        while not self._stop_set():
            emitted = False
            for frame in self._source.iter_frames(_CHUNK_US, max_hz=self._max_hz):
                if self._stop_set():
                    return
                # Wait for the frame's own deadline first, so it is sent when due, not before.
                delay = (start + frame.time_us / 1_000_000) - loop.time()
                if delay > 0:
                    assert self._stop is not None
                    with contextlib.suppress(TimeoutError):
                        _ = await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    if self._stop_set():
                        return
                else:
                    # Behind schedule: yield so viewer senders run during a catch-up burst.
                    await asyncio.sleep(0)
                emitted = True
                self._broadcast(frame)
            if not emitted and getattr(self._source, "exhausted", False):
                await self._drain_viewers()
                return
            if not emitted:
                delay = (start + self._source.now_us / 1_000_000) - loop.time()
                if delay > 0 and self._stop is not None:
                    with contextlib.suppress(TimeoutError):
                        _ = await asyncio.wait_for(self._stop.wait(), timeout=delay)

    def _stop_set(self) -> bool:
        return self._stop is not None and self._stop.is_set()

    def _broadcast(self, frame: Frame) -> None:
        """Queue one frame for every viewer. Never awaits a socket; senders do the sending.

        The detailed snapshot is encoded only when a detailed viewer is connected, and the
        snapshot-only payload only when a summary viewer is - so the pump pays for at most
        one encode per distinct shape per frame. The viewer set is read once into a
        snapshot, so a viewer that arrives or drops mid-frame cannot change who this frame
        was queued for.
        """
        self._frames_produced += 1
        viewers = tuple(self._viewers.values())
        detailed = [viewer for viewer in viewers if viewer.detailed]
        summary = [viewer for viewer in viewers if not viewer.detailed]
        # allow_nan=False is a third guard: json would happily emit the invalid token
        # NaN, which the browser's JSON.parse rejects and the store counts as
        # malformed. Fail here instead of shipping a frame nobody can decode.
        if detailed:
            detailed_payload = _encode(frame.to_wire())
            for viewer in detailed:
                self._enqueue(viewer, detailed_payload)
        if summary:
            snapshot_payload = _encode(frame.to_wire(include_samples=False))
            for viewer in summary:
                self._enqueue(viewer, snapshot_payload)

    def _attach(self, connection: ServerConnection) -> _Viewer:
        """Register an accepted viewer and start the task that sends its queue in order."""
        viewer = _Viewer(connection, asyncio.Queue[str](_QUEUE_LIMIT))
        self._viewers[connection] = viewer
        viewer.sender = asyncio.get_running_loop().create_task(self._send_loop(viewer))
        return viewer

    def _enqueue(self, viewer: _Viewer, payload: str) -> None:
        """Queue one payload for one viewer; a full queue disconnects that viewer."""
        if self._viewers.get(viewer.connection) is not viewer:
            return
        try:
            viewer.queue.put_nowait(payload)
        except asyncio.QueueFull:
            self._slow_disconnects += 1
            self._drop(viewer, _SLOW_CLOSE)

    async def _send_loop(self, viewer: _Viewer) -> None:
        """Send one viewer's queue in order. A stalled or failed send removes only that viewer."""
        while True:
            payload = await viewer.queue.get()
            try:
                await asyncio.wait_for(viewer.connection.send(payload), _SEND_TIMEOUT_S)
            except TimeoutError:
                self._slow_disconnects += 1
                self._drop(viewer, _SLOW_CLOSE)
                return
            except Exception:
                self._send_failures += 1
                self._drop(viewer)
                return
            else:
                self._messages_sent += 1
                self._bytes_sent += len(payload)
            finally:
                viewer.queue.task_done()

    def _drop(self, viewer: _Viewer, close: tuple[int, str] | None = None) -> None:
        """Stop delivering to one viewer, discard what it has not taken, and optionally close it."""
        if self._viewers.get(viewer.connection) is viewer:
            del self._viewers[viewer.connection]
        _discard_pending(viewer.queue)
        sender = viewer.sender
        if sender is not None and sender is not asyncio.current_task():
            _ = sender.cancel()
        if close is not None:
            task = asyncio.get_running_loop().create_task(_close_quietly(viewer.connection, *close))
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)

    async def _drain_viewers(self) -> None:
        """Let each viewer take what is already queued before a finished source is torn down."""
        waiters = {
            viewer: asyncio.ensure_future(viewer.queue.join()) for viewer in self._viewers.values()
        }
        if not waiters:
            return
        _, pending = await asyncio.wait(waiters.values(), timeout=_DRAIN_TIMEOUT_S)
        for viewer, waiter in waiters.items():
            if waiter in pending:
                _ = waiter.cancel()
                self._slow_disconnects += 1
                self._drop(viewer, _SLOW_CLOSE)

    async def _handle(self, connection: ServerConnection) -> None:
        """Publish live frames and accept bounded requests from local browser pages."""
        request = connection.request
        if request is not None and request.path != self._path:
            await connection.close(code=1008, reason=f"this endpoint serves {self._path}")
            return
        origin = request.headers.get("Origin") if request is not None else None
        origin_allowed = (
            (origin is None or origin in self._allowed_origins)
            if self._allowed_origins
            else _loopback_origin(origin)
        )
        if not origin_allowed:
            await connection.close(code=1008, reason="browser origin must be localhost")
            return
        viewer = self._attach(connection)
        self._reply(
            viewer,
            {"type": "server_info", "read_only": self._read_only, "result": self._current_result()},
        )
        try:
            async for message in connection:
                await self._handle_command(viewer, message)
        except ConnectionClosed:
            pass
        finally:
            self._drop(viewer)
            if viewer.sender is not None:
                _ = await asyncio.gather(viewer.sender, return_exceptions=True)

    async def _handle_command(self, viewer: _Viewer, message: str | bytes) -> None:
        """Answer one small request. Storage and simulation run off the event loop."""
        if not isinstance(message, str) or len(message.encode("utf-8")) > _MAX_COMMAND_BYTES:
            self._reply_error(viewer, None, "Request is too large.")
            return
        try:
            request = json.loads(message)
        except json.JSONDecodeError:
            self._reply_error(viewer, None, "Request must be valid JSON.")
            return
        if not isinstance(request, dict):
            self._reply_error(viewer, None, "Request must be a JSON object.")
            return
        kind = request.get("type")
        if self._read_only and kind != "telemetry_detail":
            if (
                kind == "runs_list"
                and set(request) == {"type", "request_id"}
                and _echo_id(request.get("request_id")) is not None
            ):
                self._reply(
                    viewer,
                    {"type": "runs_list_result", "request_id": request["request_id"], "runs": []},
                )
            else:
                self._reply_error(
                    viewer, request.get("request_id"), "This is a read-only recorded demo."
                )
            return
        if kind == "telemetry_detail":
            if set(request) != {"type", "enabled"} or not isinstance(request.get("enabled"), bool):
                self._reply_error(viewer, None, "Invalid telemetry detail setting.")
                return
            viewer.detailed = request["enabled"] is True
        elif kind == "scenario_run":
            await self._scenario_run(viewer, request)
        elif kind == "run_replay":
            await self._run_replay(viewer, request)
        elif kind == "runs_list":
            await self._runs_list(viewer, request)
        else:
            self._reply_error(viewer, None, "Unknown dashboard request.")

    async def _scenario_run(self, viewer: _Viewer, request: dict[object, object]) -> None:
        """Compute a scenario, save it if a store is configured, then play it on the car."""
        request_id = request.get("request_id")
        try:
            car = self._free_car()
            # No await between the check above and taking the lock: the car is claimed here.
            async with self._scenario_lock:
                lab_run = await asyncio.to_thread(run_lab, request)
                result = lab_run.result
                if self._run_store is not None:
                    result = await asyncio.to_thread(self._run_store.save, lab_run)
                self._start_playback(car, lab_run.frames, result)
        except ScenarioLabError as error:
            self._fail_scenario(viewer, request_id, str(error))
        except RunStoreError:
            _LOGGER.exception("Scenario run could not be saved")
            self._fail_scenario(
                viewer, request_id, "The run could not be saved, so it was not played."
            )
        except Exception:
            _LOGGER.exception("Scenario Lab run failed")
            self._fail_scenario(
                viewer, request_id, "The run failed. Check that the server is still running."
            )

    async def _run_replay(self, viewer: _Viewer, request: dict[object, object]) -> None:
        """Load a saved run and play its original frames on the car."""
        request_id = request.get("request_id")
        try:
            store = self._require_store()
            if set(request) != {"type", "request_id", "run_id"}:
                raise ScenarioLabError("Request must include type, request_id, and run_id only.")
            checked_id = _checked_request_id(request_id)
            run_id = request.get("run_id")
            if not isinstance(run_id, str):
                raise ScenarioLabError("Run ID must be a saved run's ID.")
            car = self._free_car()
            async with self._scenario_lock:
                lab_run = await asyncio.to_thread(store.load, run_id)
                self._start_playback(
                    car, lab_run.frames, {**lab_run.result, "request_id": checked_id}
                )
        except (ScenarioLabError, RunStoreError) as error:
            self._fail_scenario(viewer, request_id, str(error))
        except Exception:
            _LOGGER.exception("Saved run could not be replayed")
            self._fail_scenario(viewer, request_id, "The saved run could not be replayed.")

    def _start_playback(
        self, car: DrivenCarSource, frames: tuple[Frame, ...], result: dict[str, object]
    ) -> None:
        car.start_run(frames)
        self._last_result = result
        self._playback_started = asyncio.get_running_loop().time()
        for viewer in tuple(self._viewers.values()):
            self._reply(viewer, result)

    def _current_result(self) -> dict[str, object] | None:
        if self._last_result is None:
            return None
        value = self._last_result.get("duration_s", 0.0)
        duration = float(value) if isinstance(value, (int, float)) else 0.0
        elapsed = asyncio.get_running_loop().time() - self._playback_started
        return {**self._last_result, "playback_elapsed_s": min(elapsed, float(duration))}

    async def _runs_list(self, viewer: _Viewer, request: dict[object, object]) -> None:
        """Summaries of every saved run, read off the event loop."""
        request_id = request.get("request_id")
        try:
            store = self._require_store()
            if set(request) != {"type", "request_id"}:
                raise ScenarioLabError("Request must include type and request_id only.")
            checked_id = _checked_request_id(request_id)
            runs = await asyncio.to_thread(store.list_runs)
        except (ScenarioLabError, RunStoreError) as error:
            self._reply_error(viewer, request_id, str(error))
        except Exception:
            _LOGGER.exception("Saved runs could not be listed")
            self._reply_error(viewer, request_id, "The saved runs could not be listed.")
        else:
            self._reply(
                viewer, {"type": "runs_list_result", "request_id": checked_id, "runs": runs}
            )

    def _free_car(self) -> DrivenCarSource:
        """The simulated car, if no scenario is computing or playing on it."""
        if not isinstance(self._source, DrivenCarSource):
            raise ScenarioLabError(
                "Scenarios need the simulated car. "
                "This server is streaming recorded or synthetic data."
            )
        if self._scenario_lock.locked() or self._source.running:
            raise ScenarioLabError("A scenario is already running. Wait for it to finish first.")
        return self._source

    def _require_store(self) -> RunStore:
        if self._run_store is None:
            raise ScenarioLabError("Saved runs are not enabled on this server.")
        return self._run_store

    def _reply(self, viewer: _Viewer, payload: dict[str, object]) -> None:
        self._enqueue(viewer, _encode(payload))

    def _reply_error(self, viewer: _Viewer, request_id: object, detail: str) -> None:
        self._reply(
            viewer,
            {"type": "scenario_error", "request_id": _echo_id(request_id), "error": detail},
        )

    def _fail_scenario(self, viewer: _Viewer, request_id: object, detail: str) -> None:
        self._scenario_failures += 1
        self._reply_error(viewer, request_id, detail)

    async def _close_clients(self) -> None:
        viewers = tuple(self._viewers.values())
        for viewer in viewers:
            self._drop(viewer, (1001, "server shutting down"))
        pending = (*(v.sender for v in viewers if v.sender is not None), *self._closing)
        if pending:
            _ = await asyncio.gather(*pending, return_exceptions=True)


def _encode(payload: dict[str, object]) -> str:
    """Compact wire JSON, with the NaN token refused rather than emitted.

    ``json`` happily writes the invalid token ``NaN``, which the browser's
    ``JSON.parse`` rejects and the store counts as a malformed frame, so the one
    encoder the transport uses never allows it.
    """
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _discard_pending(queue: asyncio.Queue[str]) -> None:
    while not queue.empty():
        _ = queue.get_nowait()
        queue.task_done()


async def _close_quietly(connection: ServerConnection, code: int, reason: str) -> None:
    """Close with a deadline, so a stalled peer cannot hold up shutdown or other viewers."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(connection.close(code=code, reason=reason), _CLOSE_TIMEOUT_S)


def _source_state(source: FrameSource) -> dict[str, object]:
    if isinstance(source, DrivenCarSource):
        return {"kind": "simulated_car", "running": source.running}
    if isinstance(getattr(source, "source", None), ReplaySource):
        return {"kind": "looping_replay"}
    return {"kind": "replay" if isinstance(source, ReplaySource) else "synthetic"}


def _echo_id(value: object) -> str | None:
    """A request ID we will repeat back: only one that matches the lab's own rule."""
    if isinstance(value, str) and _REQUEST_ID.fullmatch(value) is not None:
        return value
    return None


def _checked_request_id(value: object) -> str:
    checked = _echo_id(value)
    if checked is None:
        raise ScenarioLabError(
            "Request ID must be 1–64 letters, numbers, dots, dashes, or underscores."
        )
    return checked


def _bound_port(server: Server, fallback: int) -> int:
    """Resolve the real port after binding, so ``port=0`` is reportable."""
    sockets = getattr(server, "sockets", None)
    if not sockets:
        return fallback
    return int(sockets[0].getsockname()[1])


def _checked_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise TransportError(f"Invalid allowed origin: {origin}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise TransportError(f"Allowed origin must be an exact http(s) origin: {origin}")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def _loopback_origin(origin: str | None) -> bool:
    """Allow non-browser clients and pages served from a loopback host only."""
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        hostname = (parsed.hostname or "").lower()
        _ = parsed.port
        return (
            parsed.scheme in {"http", "https"}
            and parsed.username is None
            and parsed.password is None
            and hostname in _LOOPBACK_HOSTS
        )
    except ValueError:
        return False


def replay_source(path: Path) -> ReplaySource:
    """A replay source with the run's saved fault annotations attached.

    Legacy files without the annotation metadata key read as "no annotations"
    (:func:`f1telemetry.telemetry.pipeline.read_annotations`), so an old saved
    run still replays exactly as before.
    """
    return ReplaySource(path, annotations=read_annotations(path))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="f1-serve",
        description="Serve the simulated car's telemetry to the dashboard on loopback.",
    )
    _ = parser.add_argument(
        "--channels",
        type=Path,
        default=None,
        help="contract to drive (default: the repository channels.yaml)",
    )
    _ = parser.add_argument(
        "--replay-parquet",
        type=Path,
        default=None,
        help="replay saved frames from a Parquet file instead of synthetic data",
    )
    _ = parser.add_argument(
        "--noise",
        action="store_true",
        help="stream uniform random contract data instead of the simulated car",
    )
    _ = parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="directory where scenario runs are saved (simulated car only)",
    )
    _ = parser.add_argument("--host", default=DEFAULT_HOST, help="loopback host only")
    _ = parser.add_argument("--web-root", type=Path, help="built dashboard directory")
    _ = parser.add_argument(
        "--read-only", action="store_true", help="disable simulation and saved-run control"
    )
    _ = parser.add_argument(
        "--allowed-origin", action="append", default=[], help="exact permitted browser origin"
    )
    _ = parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="TCP port, 0 to auto-pick"
    )
    _ = parser.add_argument("--path", default=DEFAULT_PATH, help="URL path to serve")
    _ = parser.add_argument("--hz", type=float, default=BROADCAST_HZ, help="max delivery frames/s")
    _ = parser.add_argument("--seed", type=int, default=0, help="synthetic sample seed")
    return parser.parse_args(argv)


async def serve_forever(server: TelemetryServer) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            _ = loop.add_signal_handler(sig, stop.set)
    await server.run(stop)


class _DroppedBeforeHandshake(logging.Filter):
    """Hide the traceback websockets logs when a client hangs up mid-handshake.

    The dashboard's page setup opens a socket and closes it again before the first
    request is sent, so this is routine rather than a server fault.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "opening handshake failed" or record.exc_info is None:
            return True
        return not isinstance(record.exc_info[1], ConnectionClosed)


def main(argv: Sequence[str] | None = None) -> int:
    logging.getLogger("websockets.server").addFilter(_DroppedBeforeHandshake())
    args = _parse_args(argv)
    source: FrameSource
    if args.replay_parquet is not None:
        source = replay_source(args.replay_parquet)
    elif args.noise:
        source = SyntheticSource(load_channel_contract(args.channels), seed=args.seed)
    else:
        print("starting the simulated car...")
        source = DrivenCarSource(resting_car_values())
    run_store = RunStore(args.runs_dir) if isinstance(source, DrivenCarSource) else None
    server = TelemetryServer(
        source,
        host=args.host,
        port=args.port,
        path=args.path,
        max_hz=args.hz,
        run_store=run_store,
        web_root=args.web_root,
        read_only=args.read_only,
        allowed_origins=args.allowed_origin,
    )
    if isinstance(source, SyntheticSource):
        print(f"serving {len(source.contract)} channels at {server.url} (seed {args.seed})")
        rate = source.contract.samples_per_second()
        print(f"sampling {rate:.0f} channel-samples/s of contract rate")
    elif isinstance(source, DrivenCarSource):
        print(f"simulated car at rest, serving {server.url}")
        print("start a test from the dashboard to watch it drive")
        print(f"saving scenario runs to {args.runs_dir}")
    elif isinstance(source, ReplaySource):
        print(
            f"replaying {args.replay_parquet} at {server.url} "
            f"({len(source.annotations)} saved fault annotations)"
        )
    try:
        asyncio.run(serve_forever(server))
    except KeyboardInterrupt:
        # Not reachable where add_signal_handler works; needed on Windows consoles.
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
