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
* **Clean shutdown.** :meth:`TelemetryServer.run` returns when its stop event is set;
  the ``serve`` context manager closes the listening socket and every open connection
  on the way out, and clients mid-frame are closed rather than left hanging. SIGINT
  and SIGTERM are wired to the same event where the platform allows it.

Run it with ``just serve`` (or ``uv run f1-serve``); the default endpoint is
``ws://localhost:8765/ws``, matching ``web/src/telemetry/config.ts``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

from websockets.asyncio.server import Server, ServerConnection, serve

from f1telemetry.contracts.channels import load_channel_contract
from f1telemetry.synthetic import BROADCAST_HZ, Frame, SyntheticSource
from f1telemetry.testing.replay import ReplaySource

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PATH",
    "DEFAULT_PORT",
    "TelemetryServer",
    "TransportError",
    "main",
]

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_PATH = "/ws"

_CHUNK_US = 1_000_000
"""Simulated microseconds pulled from the source per pass. Bounds generator lag only."""
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"})


class TransportError(ValueError):
    """Raised for a transport configuration that must not be served."""


class FrameSource(Protocol):
    """Minimal source contract required by the WebSocket publisher."""

    @property
    def now_us(self) -> int: ...

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]: ...


class TelemetryServer:
    """Fan-out of synthetic frames to every connected dashboard.

    Args:
        source: the frame iterator's owner. One source per server; it holds the
            simulated clock, so two servers cannot share it.
        host: loopback name or address only.
        port: TCP port, 0 to let the OS pick (useful in tests).
        path: the single accepted URL path; anything else is closed with 1008.
        max_hz: upper bound on delivery rate, applied by the source's aggregation.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        path: str = DEFAULT_PATH,
        max_hz: float = BROADCAST_HZ,
    ) -> None:
        if host.lower() not in _LOOPBACK_HOSTS:
            allowed = ", ".join(sorted(_LOOPBACK_HOSTS))
            raise TransportError(
                f"refusing to bind {host!r}: this server is loopback-only ({allowed})"
            )
        if not 0 <= port <= 65535:
            raise TransportError(f"port must be in 0..65535, got {port}")
        self._source: FrameSource = source
        self._host: str = host.lower()
        self._port: int = port
        self._path: str = path
        self._max_hz: float = max_hz
        self._clients: set[ServerConnection] = set()
        self._stop: asyncio.Event | None = None

    @property
    def url(self) -> str:
        """The endpoint to point ``VITE_TELEMETRY_WS_URL`` at."""
        return f"ws://{self._host}:{self._port}{self._path}"

    @property
    def client_count(self) -> int:
        return len(self._clients)

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
            async with serve(self._handle, self._host, self._port, compression=None) as server:
                self._port = _bound_port(server, self._port)
                await self._pump(loop)
        finally:
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
                emitted = True
                await self._broadcast(frame)
                delay = (start + frame.time_us / 1_000_000) - loop.time()
                if delay <= 0:
                    continue
                assert self._stop is not None
                with contextlib.suppress(TimeoutError):
                    _ = await asyncio.wait_for(self._stop.wait(), timeout=delay)
            if not emitted and getattr(self._source, "exhausted", False):
                return
            if not emitted:
                delay = (start + self._source.now_us / 1_000_000) - loop.time()
                if delay > 0 and self._stop is not None:
                    with contextlib.suppress(TimeoutError):
                        _ = await asyncio.wait_for(self._stop.wait(), timeout=delay)

    def _stop_set(self) -> bool:
        return self._stop is not None and self._stop.is_set()

    async def _broadcast(self, frame: Frame) -> None:
        """Send one frame to every client, dropping any whose socket has failed."""
        if not self._clients:
            return
        # allow_nan=False is a third guard: json would happily emit the invalid token
        # NaN, which the browser's JSON.parse rejects and the store counts as
        # malformed. Fail here instead of shipping a frame nobody can decode.
        payload = json.dumps(frame.to_wire(), separators=(",", ":"), allow_nan=False)
        targets = tuple(self._clients)
        results = await asyncio.gather(
            *(connection.send(payload) for connection in targets),
            return_exceptions=True,
        )
        for connection, result in zip(targets, results, strict=True):
            if isinstance(result, BaseException):
                self._clients.discard(connection)

    async def _handle(self, connection: ServerConnection) -> None:
        """Hold the connection open until the client goes away."""
        request = connection.request
        if request is not None and request.path != self._path:
            await connection.close(code=1008, reason=f"this endpoint serves {self._path}")
            return
        self._clients.add(connection)
        try:
            await connection.wait_closed()
        finally:
            self._clients.discard(connection)

    async def _close_clients(self) -> None:
        connections = tuple(self._clients)
        self._clients.clear()
        if not connections:
            return
        _ = await asyncio.gather(
            *(
                connection.close(code=1001, reason="server shutting down")
                for connection in connections
            ),
            return_exceptions=True,
        )


def _bound_port(server: Server, fallback: int) -> int:
    """Resolve the real port after binding, so ``port=0`` is reportable."""
    sockets = getattr(server, "sockets", None)
    if not sockets:
        return fallback
    return int(sockets[0].getsockname()[1])


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="f1-serve",
        description="Serve synthetic contract telemetry to the dashboard on loopback.",
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
    _ = parser.add_argument("--host", default=DEFAULT_HOST, help="loopback host only")
    _ = parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="TCP port, 0 to auto-pick"
    )
    _ = parser.add_argument("--path", default=DEFAULT_PATH, help="URL path to serve")
    _ = parser.add_argument("--hz", type=float, default=BROADCAST_HZ, help="max delivery frames/s")
    _ = parser.add_argument("--seed", type=int, default=0, help="synthetic sample seed")
    return parser.parse_args(argv)


async def _serve_forever(server: TelemetryServer) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            _ = loop.add_signal_handler(sig, stop.set)
    await server.run(stop)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source: FrameSource
    if args.replay_parquet is None:
        source = SyntheticSource(load_channel_contract(args.channels), seed=args.seed)
    else:
        source = ReplaySource(args.replay_parquet)
    server = TelemetryServer(
        source,
        host=args.host,
        port=args.port,
        path=args.path,
        max_hz=args.hz,
    )
    if isinstance(source, SyntheticSource):
        print(f"serving {len(source.contract)} channels at {server.url} (seed {args.seed})")
        rate = source.contract.samples_per_second()
        print(f"sampling {rate:.0f} channel-samples/s of contract rate")
    else:
        print(f"replaying {args.replay_parquet} at {server.url}")
    try:
        asyncio.run(_serve_forever(server))
    except KeyboardInterrupt:
        # Not reachable where add_signal_handler works; needed on Windows consoles.
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
