"""Real-socket checks for the dashboard transport: ephemeral ports, real clients, saved runs.

One test runs the real Scenario Lab end to end, because a launch plays back for seven
seconds of wall time. The others use a fast stand-in for the lab so each stays quick.
"""

import asyncio
import json
import re
import socket
import threading
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from f1telemetry import server as server_module
from f1telemetry.live_car import DrivenCarSource
from f1telemetry.runs import RunStore
from f1telemetry.server import FrameSource, TelemetryServer
from f1telemetry.synthetic import Frame
from f1telemetry.testing.scenario_lab import LabRun, resting_car_values

_TIMEOUT_S: Final[float] = 15.0
_ORIGIN: Final[str] = "http://localhost:5173"
_RUN_ID: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")


class FiniteSource:
    """Plays a fixed list of frames, then reports itself exhausted, like a finished replay."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames: list[Frame] = frames
        self._done: bool = False

    @property
    def now_us(self) -> int:
        return 0

    @property
    def exhausted(self) -> bool:
        return self._done

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        if self._done:
            return
        yield from self._frames
        self._done = True


def _fake_lab(value: object) -> LabRun:
    """A two-frame stand-in with the same shape as a real lab result."""
    request = cast("dict[str, object]", value)
    result: dict[str, object] = {
        "type": "scenario_result",
        "request_id": request["request_id"],
        "scenario_id": request["scenario_id"],
        "points": [],
    }
    frames = (Frame(0, {"speed": 1.0}), Frame(1_000_000, {"speed": 2.0}))
    return LabRun(result=result, frames=frames, provenance=None)


@dataclass(frozen=True)
class Served:
    server: TelemetryServer
    task: asyncio.Task[None]


@asynccontextmanager
async def serving(source: FrameSource, runs_dir: Path | None = None) -> AsyncGenerator[Served]:
    store = None if runs_dir is None else RunStore(runs_dir)
    server = TelemetryServer(source, port=0, run_store=store)
    assert not server.ready
    task = asyncio.create_task(server.run())
    try:
        while not server.ready and not task.done():
            await asyncio.sleep(0.01)
        assert server.ready
        yield Served(server, task)
    finally:
        server.stop()
        await asyncio.wait_for(task, _TIMEOUT_S)


async def open_viewer(
    server: TelemetryServer, *, origin: str | None = _ORIGIN, path: str | None = None
) -> ClientConnection:
    url = server.url if path is None else server.url.removesuffix("/ws") + path
    return await connect(
        url,
        origin=Origin(origin) if origin is not None else None,
        open_timeout=_TIMEOUT_S,
        max_size=None,
    )


async def send_json(ws: ClientConnection, payload: dict[str, object]) -> None:
    await ws.send(json.dumps(payload))


async def receive_json(ws: ClientConnection) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(await asyncio.wait_for(ws.recv(), _TIMEOUT_S)))


async def receive_reply(ws: ClientConnection) -> dict[str, object]:
    """The next reply. Live frames carry no ``type`` key, so they are skipped."""
    while True:
        message = await receive_json(ws)
        if "type" in message:
            return message


def _channels(wire: object) -> dict[str, float]:
    return cast("dict[str, float]", cast("dict[str, object]", wire)["channels"])


def test_foreign_origin_and_unknown_path_are_closed_with_policy_violation() -> None:
    async def close_code(ws: ClientConnection) -> int | None:
        with pytest.raises(ConnectionClosed) as closed:
            _ = await asyncio.wait_for(ws.recv(), _TIMEOUT_S)
        return closed.value.rcvd.code if closed.value.rcvd else None

    async def scenario() -> tuple[int | None, int | None]:
        async with serving(DrivenCarSource(resting_car_values())) as running:
            foreign = await open_viewer(running.server, origin="https://example.com")
            wrong_path = await open_viewer(running.server, path="/other")
            return await close_code(foreign), await close_code(wrong_path)

    assert asyncio.run(scenario()) == (1008, 1008)


def test_malformed_requests_get_scoped_errors_and_keep_the_socket_open() -> None:
    async def scenario() -> list[dict[str, object]]:
        replies: list[dict[str, object]] = []
        async with serving(DrivenCarSource(resting_car_values())) as running:
            ws = await open_viewer(running.server)
            for raw in (
                "not json",
                "[1]",
                json.dumps({"type": "nope", "request_id": "u-1"}),
                json.dumps({"type": "telemetry_detail", "enabled": "yes"}),
                json.dumps(
                    {
                        "type": "scenario_run",
                        "request_id": "v-1",
                        "scenario_id": "nope",
                        "setting": 1,
                    }
                ),
                json.dumps(
                    {
                        "type": "scenario_run",
                        "request_id": "has space",
                        "scenario_id": "launch",
                        "setting": 100,
                    }
                ),
                json.dumps({"type": "runs_list", "request_id": "l-1"}),
                json.dumps({"type": "run_replay", "request_id": "r-1", "run_id": "x"}),
            ):
                await ws.send(raw)
                replies.append(await receive_reply(ws))
        return replies

    replies = asyncio.run(scenario())

    assert [reply["error"] for reply in replies] == [
        "Request must be valid JSON.",
        "Request must be a JSON object.",
        "Unknown dashboard request.",
        "Invalid telemetry detail setting.",
        "Choose launch, braking, or cornering.",
        replies[5]["error"],
        "Saved runs are not enabled on this server.",
        "Saved runs are not enabled on this server.",
    ]
    assert all(reply["type"] == "scenario_error" for reply in replies)
    assert [reply["request_id"] for reply in replies] == [
        None,
        None,
        None,
        None,
        "v-1",
        None,
        "l-1",
        "r-1",
    ]
    assert str(replies[5]["error"]).startswith("Request ID must be")


def test_oversized_command_closes_the_socket() -> None:
    async def scenario() -> int | None:
        async with serving(DrivenCarSource(resting_car_values())) as running:
            ws = await open_viewer(running.server)
            await ws.send("x" * 3_000)
            with pytest.raises(ConnectionClosed) as closed:
                _ = await asyncio.wait_for(ws.recv(), _TIMEOUT_S)
        return closed.value.rcvd.code if closed.value.rcvd else None

    assert asyncio.run(scenario()) == 1009


def test_hostile_run_ids_never_reach_the_filesystem(tmp_path: Path) -> None:
    async def scenario() -> list[dict[str, object]]:
        replies: list[dict[str, object]] = []
        async with serving(DrivenCarSource(resting_car_values()), tmp_path / "runs") as running:
            ws = await open_viewer(running.server)
            for run_id in ("../../escape", 12, "0" * 32):
                await send_json(ws, {"type": "run_replay", "request_id": "h-1", "run_id": run_id})
                replies.append(await receive_reply(ws))
        return replies

    replies = asyncio.run(scenario())

    assert all(reply["type"] == "scenario_error" for reply in replies)
    assert str(replies[0]["error"]).startswith("run id must be")
    assert str(replies[1]["error"]).startswith("Run ID must be")
    assert "not found" in str(replies[2]["error"])
    assert not (tmp_path / "escape.parquet").exists()


def test_real_scenario_is_saved_before_playback_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"

    async def first_session() -> tuple[dict[str, object], dict[str, object]]:
        async with serving(DrivenCarSource(resting_car_values()), runs_dir) as running:
            viewer = await open_viewer(running.server)
            await send_json(
                viewer,
                {
                    "type": "scenario_run",
                    "request_id": "launch-1",
                    "scenario_id": "launch",
                    "setting": 100,
                },
            )
            result = await receive_reply(viewer)
            assert running.server.metrics["source"] == {"kind": "simulated_car", "running": True}
            competitor = await open_viewer(running.server)
            await send_json(
                competitor,
                {
                    "type": "scenario_run",
                    "request_id": "launch-2",
                    "scenario_id": "launch",
                    "setting": 100,
                },
            )
            busy = await receive_reply(competitor)
            await send_json(
                competitor,
                {"type": "run_replay", "request_id": "replay-busy", "run_id": result["run_id"]},
            )
            busy_replay = await receive_reply(competitor)
            assert busy["error"] == busy_replay["error"]
            assert "already running" in str(busy["error"])
            return result, busy

    result, _ = asyncio.run(first_session())

    assert result["type"] == "scenario_result"
    assert result["request_id"] == "launch-1"
    run_id = str(result["run_id"])
    assert _RUN_ID.fullmatch(run_id)
    assert sorted(path.name for path in runs_dir.iterdir()) == [f"{run_id}.parquet"]
    original = RunStore(runs_dir).load(run_id)
    assert original.frames
    expected_points = result["points"]

    async def second_session() -> tuple[dict[str, object], list[dict[str, object]]]:
        async with serving(DrivenCarSource(resting_car_values()), runs_dir) as running:
            viewer = await open_viewer(running.server)
            await send_json(viewer, {"type": "runs_list", "request_id": "list-1"})
            listing = await receive_reply(viewer)
            assert listing["type"] == "runs_list_result"
            assert listing["request_id"] == "list-1"
            await send_json(viewer, {"type": "telemetry_detail", "enabled": True})
            await send_json(
                viewer, {"type": "run_replay", "request_id": "replay-1", "run_id": run_id}
            )
            replay = await receive_reply(viewer)
            wanted = [dict(frame.channels) for frame in original.frames[:40]]
            seen: list[dict[str, float]] = []
            while len(seen) < 120:
                message = await receive_json(viewer)
                for sample in cast("list[object]", message.get("samples", [])):
                    seen.append(_channels(sample))
            # Replayed samples must contain the saved frames, in order, with the same values.
            cursor = 0
            for channels in seen:
                if cursor < len(wanted) and channels == wanted[cursor]:
                    cursor += 1
            assert cursor == len(wanted)
            return replay, cast("list[dict[str, object]]", listing["runs"])

    replay, runs = asyncio.run(second_session())

    assert replay["type"] == "scenario_result"
    assert replay["request_id"] == "replay-1"
    assert replay["run_id"] == run_id
    assert replay["points"] == expected_points
    assert [entry["run_id"] for entry in runs] == [run_id]


def test_second_scenario_is_rejected_while_the_first_is_computing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = threading.Event()
    started = threading.Event()

    def slow_lab(value: object) -> LabRun:
        started.set()
        _ = gate.wait(_TIMEOUT_S)
        return _fake_lab(value)

    monkeypatch.setattr(server_module, "run_lab", slow_lab)

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with serving(DrivenCarSource(resting_car_values()), tmp_path / "runs") as running:
            first = await open_viewer(running.server)
            second = await open_viewer(running.server)
            try:
                await send_json(
                    first,
                    {
                        "type": "scenario_run",
                        "request_id": "first",
                        "scenario_id": "launch",
                        "setting": 100,
                    },
                )
                assert await asyncio.to_thread(started.wait, _TIMEOUT_S)
                await send_json(
                    second,
                    {
                        "type": "scenario_run",
                        "request_id": "second",
                        "scenario_id": "launch",
                        "setting": 100,
                    },
                )
                busy = await receive_reply(second)
            finally:
                gate.set()
            result = await receive_reply(first)
            return busy, result

    busy, result = asyncio.run(scenario())

    assert busy["type"] == "scenario_error"
    assert busy["request_id"] == "second"
    assert "already running" in str(busy["error"])
    assert result["type"] == "scenario_result"
    assert result["request_id"] == "first"


def test_graceful_shutdown_closes_viewers_and_releases_the_port() -> None:
    async def scenario() -> tuple[int | None, bool, int, int]:
        async with serving(DrivenCarSource(resting_car_values())) as running:
            viewer = await open_viewer(running.server)
            _ = await receive_json(viewer)
            port = int(running.server.url.rsplit(":", 1)[1].split("/", 1)[0])
            running.server.stop()
            await asyncio.wait_for(running.task, _TIMEOUT_S)
            # Frames sent before the stop may still be queued client-side; read until the close.
            with pytest.raises(ConnectionClosed) as closed:
                while True:
                    _ = await asyncio.wait_for(viewer.recv(), _TIMEOUT_S)
            ready = running.server.ready
            clients = running.server.client_count
        refused = False
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
        except OSError:
            refused = True
        return (closed.value.rcvd.code if closed.value.rcvd else None), ready, clients, int(refused)

    code, ready, clients, refused = asyncio.run(scenario())

    assert code == 1001
    assert ready is False
    assert clients == 0
    assert refused == 1


def test_finite_replay_delivers_its_final_frame_before_closing() -> None:
    frames = [
        Frame(time_us=1_000_000 + step * 100_000, channels={"speed": float(step)})
        for step in range(3)
    ]

    async def scenario() -> tuple[list[float], int | None]:
        async with serving(FiniteSource(frames)) as running:
            viewer = await open_viewer(running.server)
            speeds: list[float] = []
            with pytest.raises(ConnectionClosed) as closed:
                while True:
                    message = await receive_json(viewer)
                    speeds.append(_channels(message)["speed"])
        return speeds, closed.value.rcvd.code if closed.value.rcvd else None

    speeds, code = asyncio.run(scenario())

    assert speeds == [0.0, 1.0, 2.0]
    assert code == 1001


def test_metrics_are_json_ready_and_count_what_was_delivered() -> None:
    async def scenario() -> dict[str, object]:
        async with serving(DrivenCarSource(resting_car_values())) as running:
            viewer = await open_viewer(running.server)
            _ = await receive_json(viewer)
            _ = await receive_json(viewer)
            return running.server.metrics

    metrics = asyncio.run(scenario())

    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics
    assert metrics["ready"] is True
    assert cast("int", metrics["frames_produced"]) >= 2
    assert cast("int", metrics["messages_sent"]) >= 2
    assert cast("int", metrics["bytes_sent"]) > 0
    assert metrics["send_failures"] == 0
    assert metrics["slow_disconnects"] == 0
    assert metrics["scenario_failures"] == 0
    assert metrics["clients"] == {"active": 1, "detailed": 0}
