# Engineering case study

## Problem and behavior

A useful telemetry system needs an agreed schema, preserved sampling times,
reliable delivery and evidence that can be inspected again. F1 Telemetry Lab
uses a synthetic car as a repeatable source for those engineering problems.

A user runs a bounded experiment, watches its samples, restarts the server,
reloads a saved recording, and compares two runs of the same scenario. Replay
uses the recorded values and timestamps; it does not rerun physics.

## Decisions

**One channel contract.** Python, TypeScript and Arrow metadata come from the
same YAML contract. Drift and rate-edit tests cross that boundary, reducing
the chance of producer, storage and frontend schemas quietly diverging.

**Separate sampling from delivery and display.** Network messages contain
original timestamped samples plus a latest-value snapshot. The browser retains
history without rerendering each sample. A refresh rate is not a sampling rate.

**Bound each viewer's output.** The shared publisher never waits on a socket.
Queues and send deadlines isolate stalled viewers. Disconnecting a lagging
viewer makes incompleteness visible instead of silently losing samples/events.

**Reuse Parquet.** One immutable self-contained file is enough for one complete
run. Atomic no-replacement publication avoids partial or replaced recordings.
The simulator parses the same captured configuration stored with the run.

**Keep provenance honest.** A deterministic lab has no RNG seed; an artifact
without Git has no known revision. Those facts stay explicit. A dirty flag
does not make a checkout reconstructable from its commit alone. Exact sample
replay remains possible because the observations are stored.

**Read-only public demonstration.** A looping recording is useful to visitors
without granting simulation control or recording-directory access. Static
assets and WebSockets share an origin; interactive controls stay local.

## Evidence and limits

[Release checks](release-checklist.md) record actual verification.
[Performance measurements](performance.md) specify workload, machine, revision,
timing boundaries and reproduction commands. The [demo](assets/demo.webm)
shows the visible application.

The most valuable tests cross boundaries: real commands over sockets,
persistence before success, restart/replay, competing controllers, slow-viewer
isolation and the browser workflow. Unit tests own numerical invariants and parsers.

This is software engineering using synthetic vehicle data. Numerical invariants
and reproducible files do not establish a calibrated F1 car, validated real-world
fault detection, internet-scale capacity or production operating history.

## Résumé wording

Built a Python/TypeScript telemetry platform with contract-generated schemas,
bounded WebSocket delivery, deterministic vehicle experiments, and immutable
Parquet recording/replay with captured configuration provenance.

Add performance numbers only with the workload and measurement boundaries from
the performance report. Link that report and this case study from the repository.
