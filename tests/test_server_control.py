import asyncio
import json
from collections.abc import Iterator
from typing import cast

import pytest
from websockets.asyncio.server import ServerConnection

from f1telemetry.server import TelemetryServer, _loopback_origin
from f1telemetry.synthetic import Frame


class EmptySource:
    @property
    def now_us(self) -> int:
        return 0

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        return iter(())


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
    class Connection:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, message: str) -> None:
            self.messages.append(json.loads(message))

    async def broadcast() -> tuple[dict[str, object], dict[str, object]]:
        server = TelemetryServer(EmptySource())
        summary = Connection()
        detailed = Connection()
        server._clients.update(  # pyright: ignore[reportPrivateUsage]
            (cast("ServerConnection", summary), cast("ServerConnection", detailed))
        )
        server._detailed_clients.add(  # pyright: ignore[reportPrivateUsage]
            cast("ServerConnection", detailed)
        )
        sample = Frame(time_us=0, channels={"speed": 1.0})
        frame = Frame(time_us=0, channels={"speed": 2.0}, samples=(sample,))
        await server._broadcast(frame)  # pyright: ignore[reportPrivateUsage]
        return summary.messages[0], detailed.messages[0]

    summary, detailed = asyncio.run(broadcast())

    assert "samples" not in summary
    assert detailed["samples"] == [{"time_us": 0, "channels": {"speed": 1.0}}]
