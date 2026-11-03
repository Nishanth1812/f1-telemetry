import asyncio
import json
import urllib.request
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from f1telemetry.live_car import DrivenCarSource
from f1telemetry.server import TelemetryServer, TransportError


def test_public_binding_needs_readonly_and_explicit_origins() -> None:
    source = DrivenCarSource({"speed": 0.0})
    with pytest.raises(TransportError):
        TelemetryServer(source, host="0.0.0.0", read_only=True)
    server = TelemetryServer(
        source, host="0.0.0.0", read_only=True, allowed_origins=["https://demo.example"]
    )
    assert server.url.startswith("ws://0.0.0.0:")


def test_readonly_demo_serves_assets_health_and_rejects_mutation(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>Recorded demo</h1>", encoding="utf-8")

    async def run() -> None:
        server = TelemetryServer(
            DrivenCarSource({"speed": 0.0}),
            host="127.0.0.1",
            port=0,
            web_root=tmp_path,
            read_only=True,
            allowed_origins=["http://localhost:8765"],
        )
        task = asyncio.create_task(server.run())
        try:
            async with asyncio.timeout(5):
                while not server.ready:
                    await asyncio.sleep(0.001)
                root = server.url.replace("ws://", "http://").removesuffix("/ws")

                def get(path: str) -> bytes:
                    with urllib.request.urlopen(root + path, timeout=2) as response:
                        return response.read()

                assert b"Recorded demo" in await asyncio.to_thread(get, "/")
                health = json.loads(await asyncio.to_thread(get, "/healthz"))
                assert health["ready"] and health["read_only"]
                async with connect(server.url, origin=Origin("https://other.example")) as denied:
                    with pytest.raises(ConnectionClosed) as closed:
                        await denied.recv()
                    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
                async with connect(server.url, origin=Origin("http://localhost:8765")) as ws:
                    info = json.loads(await ws.recv())
                    assert info["type"] == "server_info" and info["read_only"] is True
                    await ws.send(
                        json.dumps(
                            {
                                "type": "scenario_run",
                                "request_id": "demo",
                                "scenario_id": "launch",
                                "setting": 80,
                            }
                        )
                    )
                    while True:
                        reply = json.loads(await ws.recv())
                        if reply.get("type") == "scenario_error":
                            assert "read-only" in reply["error"]
                            break
        finally:
            server.stop()
            await asyncio.wait_for(task, 5)

    asyncio.run(run())
