"""Serve a read-only looping recorded demonstration with the built dashboard."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

from f1telemetry.contracts.channels import repo_root
from f1telemetry.runs import RunStore
from f1telemetry.server import TelemetryServer, replay_source, serve_forever
from f1telemetry.synthetic import Frame
from f1telemetry.testing.replay import ReplaySource
from f1telemetry.testing.scenario_lab import run_lab


def generate_sample(path: Path) -> None:
    """Publish a real bounded launch recording without replacing an existing file."""
    if path.exists():
        raise FileExistsError(f"Recording already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    run = run_lab(
        {"type": "scenario_run", "request_id": "sample", "scenario_id": "launch", "setting": 80}
    )
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        saved = RunStore(Path(temporary)).save(run)
        os.link(Path(temporary) / f"{saved['run_id']}.parquet", path)


class LoopingReplaySource:
    """Repeat a finite recording, rebasing samples/events onto a monotonic timeline."""

    def __init__(self, path: Path) -> None:
        self.source: ReplaySource = replay_source(path)
        self._offset_us: int = 0
        self._now_us: int = 0
        # An empty recording would spin forever without ever yielding data.
        if self.source.exhausted:
            raise ValueError("Demo recording must contain samples or events")

    @property
    def now_us(self) -> int:
        return self._now_us

    def iter_frames(self, duration_us: int, *, max_hz: float) -> Iterator[Frame]:
        if self.source.exhausted:
            self._offset_us = self._now_us
            self.source.reset()
        for frame in self.source.iter_frames(duration_us, max_hz=max_hz):
            offset = self._offset_us
            rebased = replace(
                frame,
                time_us=offset + frame.time_us,
                events=tuple(
                    replace(event, time_us=offset + event.time_us) for event in frame.events
                ),
                samples=tuple(
                    replace(sample, time_us=offset + sample.time_us) for sample in frame.samples
                ),
            )
            self._now_us = rebased.time_us
            yield rebased
        if self.source.exhausted:
            self._now_us += math.ceil(1_000_000 / max_hz)
        else:
            self._now_us = self._offset_us + self.source.now_us


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--recording", type=Path, default=repo_root() / "examples/standing-launch.parquet"
    )
    _ = parser.add_argument("--web-root", type=Path, default=repo_root() / "web/dist")
    _ = parser.add_argument("--host", default="localhost")
    _ = parser.add_argument("--port", type=int, default=8765)
    _ = parser.add_argument("--allowed-origin", action="append", default=[])
    _ = parser.add_argument(
        "--generate-only", action="store_true", help="create a launch recording and exit"
    )
    args = parser.parse_args(argv)
    if args.generate_only:
        generate_sample(args.recording)
        print(f"Saved sample recording: {args.recording}")
        return 0
    server = TelemetryServer(
        LoopingReplaySource(args.recording),
        host=args.host,
        port=args.port,
        read_only=True,
        allowed_origins=args.allowed_origin,
        web_root=args.web_root,
    )
    print(f"Read-only recorded demo: http://{args.host}:{args.port}/")
    try:
        asyncio.run(serve_forever(server))
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
