"""Loopback delivery measurements with one shared monotonic clock, no physics claims."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, override

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from f1telemetry.server import TelemetryServer
from f1telemetry.synthetic import BROADCAST_HZ, Frame, SyntheticSource, load_source


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linearly interpolated percentile; an empty receiver has no measurement."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class _MeasuredFrame(Frame):
    @override
    def to_wire(self, *, include_samples: bool = True) -> dict[str, object]:
        payload = super().to_wire(include_samples=include_samples)
        # Same-process clients share this clock; the marker is benchmark-only.
        payload["benchmark_sent_ns"] = time.perf_counter_ns()
        return payload


class _MeasuredSource:
    def __init__(self) -> None:
        self.source: SyntheticSource = load_source(seed=0)

    @property
    def now_us(self) -> int:
        return self.source.now_us

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        for frame in self.source.iter_frames(duration_us, max_hz=max_hz):
            yield _MeasuredFrame(frame.time_us, frame.channels, frame.events, frame.samples)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run_benchmark(
    *, duration_s: float = 10.0, clients: int = 1, detailed: bool = False, slow_delay_s: float = 0.0
) -> dict[str, Any]:
    """Measure the actual publisher and websocket clients in this Python process.

    Latency begins before JSON serialization and ends after recv, before JSON decode.
    CPU and traced Python memory cover server AND load clients, not native RSS.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    if isinstance(clients, bool) or not isinstance(clients, int) or clients < 1:
        raise ValueError("clients must be a positive integer")
    if not math.isfinite(slow_delay_s) or slow_delay_s < 0:
        raise ValueError("slow_delay_s must be non-negative and finite")
    source = _MeasuredSource()
    server = TelemetryServer(source, host="127.0.0.1", port=0)
    task = asyncio.create_task(server.run())
    own_trace = not tracemalloc.is_tracing()
    receivers: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(5):
            while ":0/" in server.url:
                if task.done():
                    await task
                await asyncio.sleep(0.001)
        async with contextlib.AsyncExitStack() as stack:
            connections = [
                await stack.enter_async_context(connect(server.url, max_queue=1))
                for _ in range(clients)
            ]
            if detailed:
                for connection in connections:
                    await connection.send(json.dumps({"type": "telemetry_detail", "enabled": True}))
            if own_trace:
                tracemalloc.start()
            started = time.perf_counter()
            cpu_started = time.process_time()
            deadline = started + duration_s

            async def receive(index: int) -> None:
                connection = connections[index]
                latencies: list[float] = []
                arrivals: list[float] = []
                total_bytes = 0
                close_code: int | None = None
                while time.perf_counter() < deadline:
                    try:
                        async with asyncio.timeout(max(0.001, deadline - time.perf_counter())):
                            message = await connection.recv()
                    except TimeoutError:
                        break
                    except ConnectionClosed as error:
                        close_code = error.rcvd.code if error.rcvd is not None else None
                        break
                    arrived_ns = time.perf_counter_ns()
                    payload = json.loads(message)
                    marker = payload.get("benchmark_sent_ns")
                    if marker is None:
                        continue
                    latencies.append((arrived_ns - marker) / 1_000_000)
                    arrivals.append(arrived_ns / 1_000_000_000)
                    total_bytes += len(
                        message.encode("utf-8") if isinstance(message, str) else message
                    )
                    if index == 0 and slow_delay_s:
                        await asyncio.sleep(
                            min(slow_delay_s, max(0.0, deadline - time.perf_counter()))
                        )
                span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
                receivers.append(
                    {
                        "client": index,
                        "slow_reader": index == 0 and slow_delay_s > 0,
                        "frames": len(arrivals),
                        "rate_hz": (len(arrivals) - 1) / span if span > 0 else 0.0,
                        "bytes": total_bytes,
                        "latency_ms": {f"p{p}": percentile(latencies, p) for p in (50, 95, 99)},
                        "close_code": close_code,
                    }
                )

            await asyncio.gather(*(receive(index) for index in range(clients)))
            cpu_s = time.process_time() - cpu_started
            elapsed_s = time.perf_counter() - started
            for receiver in receivers:
                receiver["bytes_per_second"] = receiver["bytes"] / elapsed_s
            _current, peak = tracemalloc.get_traced_memory()
    finally:
        if own_trace and tracemalloc.is_tracing():
            tracemalloc.stop()
        server.stop()
        await task
    return {
        "revision": _revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "clock": "shared-process perf_counter_ns",
        "latency_boundary": "before JSON encode to client recv, before JSON decode; loopback only",
        "workload": "seeded contract noise, no physics; includes Python allocation tracing",
        "memory_scope": "peak traced Python allocations during load; server and clients, not RSS",
        "duration_s": elapsed_s,
        "clients": clients,
        "detailed": detailed,
        "slow_delay_s": slow_delay_s,
        "target_hz": BROADCAST_HZ,
        "cpu_s": cpu_s,
        "peak_traced_python_bytes": peak,
        "receivers": sorted(receivers, key=lambda item: item["client"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--duration", type=float, default=10.0)
    _ = parser.add_argument("--clients", type=int, default=1)
    _ = parser.add_argument("--detailed", action="store_true")
    _ = parser.add_argument("--slow-delay", type=float, default=0.0)
    _ = parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(
            run_benchmark(
                duration_s=args.duration,
                clients=args.clients,
                detailed=args.detailed,
                slow_delay_s=args.slow_delay,
            )
        )
    except ValueError as error:
        parser.error(str(error))
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _ = args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
