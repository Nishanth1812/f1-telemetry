# Run and demonstrate the project

## Local interactive application

From a checkout, use Python 3.12, uv, and Node.js 22.12 or newer:

```powershell
uv sync --frozen
npm ci --prefix web
npm run build --prefix web
uv run f1-serve --web-root web/dist
```

Open <http://localhost:8765>. One process serves both the built dashboard and its
WebSocket. With just installed, `just lab` builds and starts that application.
For frontend development, run `uv run f1-serve` and `npm --prefix web run dev` in
two terminals instead. The development page uses the local socket by default.

Run a standing launch at one throttle setting. Wait for playback to finish,
change the throttle and run again. The summary compares matching scenarios.
Use the two **Saved tests** selectors to compare earlier runs without simulation,
or press **Replay saved test** to play the selected recording. Restart the server
and refresh saved tests to verify persistence.

One server owns one shared car. Viewers see the same playback; a second command
that would replace a computing/playing run is rejected. A newly connected viewer
receives the current result and elapsed playback time. Recordings are published
before success is reported, including when an initiating viewer disconnects.

## Read-only replay

```powershell
uv run f1-demo
```

The same URL loops `examples/standing-launch.parquet`, a real synthetic launch
recording at 80% throttle. The demo rebases original samples/events onto a forward
timeline when it repeats. Interactive controls are hidden and the server also rejects
mutation commands; hiding a button is not the security boundary.

Generate a different included-style recording at a new path:

```powershell
uv run f1-demo --generate-only --recording reports/my-launch.parquet
uv run f1-demo --recording reports/my-launch.parquet
```

Generation refuses an existing destination. The sample's configuration snapshots
are in Parquet metadata; its source revision/dirty flag describes generation time,
not necessarily the commit from which the sample is being viewed.

## Container

```powershell
docker compose up --build
```

Open <http://localhost:8765>. Compose binds the published port to localhost,
runs as a non-root user with a read-only filesystem and temporary `/tmp`, drops
Linux capabilities and prevents privilege escalation. No recording directory is
mounted into the public demo. Stop it with `docker compose down`.

The multi-stage image builds locked web assets and installs locked Python
dependencies. It runs the Python server, not a development Vite server. To cite a
source revision in an image without Git, set `F1_TELEMETRY_REVISION` before building;
an absent revision remains unknown rather than being invented.

Public hosting is deliberately deferred by the project owner. For a later host,
keep the backend behind an HTTPS reverse proxy that forwards WebSocket upgrades,
and override the container command with `--host 0.0.0.0 --allowed-origin
https://your-domain.example`. Public binding is accepted only in read-only mode
with explicitly permitted origins. Add one flag for each exact browser origin.
This is a deployment mechanism, not proof of production uptime or internet-scale load.

## Stored files and failure policy

Normal scenario runs default to `runs/`; `--runs-dir PATH` selects another local
server-owned directory. Each opaque ID identifies one complete immutable Parquet
file. Temporary writes are fsynced before no-replacement publication. Filesystems
must support hard links; Windows does not support the directory fsync used on
POSIX. No guarantee against every power-loss/filesystem failure is implied.

A corrupt run makes listing fail with its identifier. Preserve that file for
inspection and move it out of the run directory to inspect the remaining runs;
the application never silently skips, deletes or replaces it. There is no
automatic deletion or retention policy, so back up and manage local recordings.

Slow viewers have a bounded queue and send deadline. Overflow or a blocked send
disconnects that viewer rather than dropping hidden samples/events. On reconnect,
the stream resumes live; the gap is not a claim of lossless catch-up. The saved
recording is the path for complete offline replay.

`/healthz` returns readiness and bounded delivery/source counters. It contains no
recording paths, configuration snapshots or saved-run listing.

## Browser verification and recording

The existing noise-rate CI job checks per-channel sample rates. A separate real
workflow uses the same native Chrome/CDP harness, starts its own backend and
temporary recording directory, and proves scenario/save/restart/replay/comparison.
Run it from the repository root after building the dashboard, with port 8765 free:

```powershell
$env:CHROME_BIN = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
$env:SMOKE_WORKFLOW = '1'
$env:DASHBOARD_URL = 'http://127.0.0.1:8765'
$env:SMOKE_ARTIFACT_DIR = 'reports/browser'
node web/tests/dashboard-smoke.mjs
```

On Linux, the harness finds `/usr/bin/google-chrome` automatically. For an optional
video, set `SMOKE_VIDEO=1` and install ffmpeg on PATH (or set `FFMPEG_BIN`). This
records the actual workflow via Chrome screencast frames and encodes `demo.webm`;
ffmpeg is not an application or CI-test dependency. The backend and temporary
browser profile are closed by the harness.

[The included video](assets/demo.webm) and [dashboard image](assets/dashboard.png)
show the application. [Release checks](release-checklist.md) distinguish tested
behavior from the research roadmap.
