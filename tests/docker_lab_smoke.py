"""Exercise a running Compose lab, including its persistent volume."""

import asyncio
import json
import subprocess
import time
import urllib.request
from typing import cast

from websockets.asyncio.client import connect
from websockets.typing import Origin

URL = "ws://127.0.0.1:8766/ws"


async def request(payload: dict[str, object]) -> dict[str, object]:
    async with connect(URL, origin=Origin("http://localhost:8766"), max_size=None) as ws:
        await ws.send(json.dumps(payload))
        async with asyncio.timeout(180):
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("request_id") == payload["request_id"]:
                    if reply.get("type") == "scenario_error":
                        raise AssertionError(reply)
                    return reply


def wait_ready() -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8766/healthz", timeout=2) as response:
                if json.load(response)["ready"]:
                    return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("interactive container did not become ready")


async def main() -> None:
    wait_ready()
    original = await request(
        {"type": "scenario_run", "request_id": "smoke-run", "scenario_id": "launch", "setting": 80}
    )
    assert original["type"] == "scenario_result"
    run_id = original["run_id"]
    subprocess.run(["docker", "compose", "--profile", "lab", "restart", "lab"], check=True)
    wait_ready()
    listing = await request({"type": "runs_list", "request_id": "smoke-list"})
    assert any(run["run_id"] == run_id for run in cast("list[dict[str, object]]", listing["runs"]))
    replay = await request({"type": "run_replay", "request_id": "smoke-replay", "run_id": run_id})
    assert replay["type"] == "scenario_result"
    assert replay["run_id"] == run_id and replay["points"] == original["points"]
    print(f"Saved and replayed {run_id} after container restart")


if __name__ == "__main__":
    asyncio.run(main())
