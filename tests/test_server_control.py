import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from websockets.asyncio.server import ServerConnection

from f1telemetry import server as server_module
from f1telemetry.server import (
    TelemetryServer,
    _loopback_origin,  # pyright: ignore[reportPrivateUsage]
    _parse_args,  # pyright: ignore[reportPrivateUsage]
)
from f1telemetry.synthetic import Frame


class EmptySource:
    @property
    def now_us(self) -> int:
        return 0

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        return iter(())


class FakeConnection:
    """A socket stand-in: records delivered payloads, or never completes a send when stalled."""

    def __init__(self, *, stalled: bool = False) -> None:
        self.stalled: bool = stalled
        self.messages: list[dict[str, object]] = []
        self.closed: asyncio.Event = asyncio.Event()
        self.close_code: int | None = None

    async def send(self, message: str) -> None:
        if self.stalled:
            await asyncio.Event().wait()
        self.messages.append(json.loads(message))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.closed.set()


def _connection(fake: FakeConnection) -> ServerConnection:
    return cast("ServerConnection", cast("object", fake))


def _frame(index: int, *, samples: tuple[Frame, ...] = ()) -> Frame:
    return Frame(time_us=index * 1_000, channels={"speed": float(index)}, samples=samples)


def _speeds(connection: FakeConnection) -> list[float]:
    channels = [message["channels"] for message in connection.messages]
    return [cast("dict[str, float]", channel)["speed"] for channel in channels]


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (None, True),
        ("http://localhost:5173", True),
        ("http://127.0.0.1:3000", True),
        ("https://[::1]:5173", True),
        ("https://example.com", False),
        ("file://localhost", False),
        ("http://localhost.attacker.test", False),
        ("not a URL", False),
    ],
)
def test_only_local_browser_origins_can_use_the_dashboard_socket(
    origin: str | None, allowed: bool
) -> None:
    assert _loopback_origin(origin) is allowed


def test_full_rate_batches_are_sent_only_to_detailed_clients() -> None:
    async def broadcast() -> tuple[dict[str, object], dict[str, object]]:
        server = TelemetryServer(EmptySource())
        summary = FakeConnection()
        detailed = FakeConnection()
        summary_viewer = server._attach(_connection(summary))  # pyright: ignore[reportPrivateUsage]
        detailed_viewer = server._attach(_connection(detailed))  # pyright: ignore[reportPrivateUsage]
        detailed_viewer.detailed = True
        sample = Frame(time_us=0, channels={"speed": 1.0})
        frame = Frame(time_us=0, channels={"speed": 2.0}, samples=(sample,))
        server._broadcast(frame)  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(summary_viewer.queue.join(), 2)
        await asyncio.wait_for(detailed_viewer.queue.join(), 2)
        return summary.messages[0], detailed.messages[0]

    summary, detailed = asyncio.run(broadcast())

    assert "samples" not in summary
    assert detailed["samples"] == [{"time_us": 0, "channels": {"speed": 1.0}}]


def test_overflowing_viewer_is_disconnected_while_a_healthy_viewer_gets_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "_QUEUE_LIMIT", 4)

    async def scenario() -> tuple[TelemetryServer, FakeConnection, FakeConnection]:
        server = TelemetryServer(EmptySource())
        stalled = FakeConnection(stalled=True)
        healthy = FakeConnection()
        _ = server._attach(_connection(stalled))  # pyright: ignore[reportPrivateUsage]
        healthy_viewer = server._attach(_connection(healthy))  # pyright: ignore[reportPrivateUsage]
        for index in range(20):
            server._broadcast(_frame(index))  # pyright: ignore[reportPrivateUsage]
            await asyncio.sleep(0)  # the pump yields between frames; senders run then
        await asyncio.wait_for(healthy_viewer.queue.join(), 2)
        await asyncio.wait_for(stalled.closed.wait(), 2)
        return server, healthy, stalled

    server, healthy, stalled = asyncio.run(scenario())

    assert _speeds(healthy) == [float(index) for index in range(20)]
    assert stalled.close_code == 1013
    assert server.metrics["slow_disconnects"] == 1
    assert server.metrics["clients"] == {"active": 1, "detailed": 0}


def test_blocked_send_times_out_and_disconnects_only_that_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "_SEND_TIMEOUT_S", 0.05)

    async def scenario() -> tuple[TelemetryServer, FakeConnection, FakeConnection]:
        server = TelemetryServer(EmptySource())
        stalled = FakeConnection(stalled=True)
        healthy = FakeConnection()
        _ = server._attach(_connection(stalled))  # pyright: ignore[reportPrivateUsage]
        healthy_viewer = server._attach(_connection(healthy))  # pyright: ignore[reportPrivateUsage]
        server._broadcast(_frame(0))  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
        server._broadcast(_frame(1))  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(healthy_viewer.queue.join(), 2)
        await asyncio.wait_for(stalled.closed.wait(), 2)
        return server, healthy, stalled

    server, healthy, stalled = asyncio.run(scenario())

    assert _speeds(healthy) == [0.0, 1.0]
    assert stalled.close_code == 1013
    assert server.metrics["slow_disconnects"] == 1


def test_runs_directory_defaults_to_the_runs_folder() -> None:
    assert _parse_args([]).runs_dir == Path("runs")
