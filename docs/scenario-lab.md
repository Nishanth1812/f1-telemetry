# Scenario Lab design

The dashboard opens with a 3-step guided flow: (1) Pick a run — choose from a standing launch,
braking run, or constant-radius corner; (2) Change one thing — adjust the single bounded control
(throttle, brake, or steering); (3) See the result — view peak speed, longitudinal and lateral
g-force, the setting you changed, plus a compact speed and acceleration trace. Each result also
shows a plain-English explanation for each metric (for example, "1 g equals your own body weight")
and, when a previous result exists for the same scenario, a "What changed" line comparing the new
launch to the previous one (e.g., "Peak speed 212.4 km/h, up 6.1 km/h from your last launch run").

The page labels these outputs from the project's uncalibrated simulation and includes a subtitle
explaining the dashboard purpose in plain language. The connection status appears in the header
as clear words: "Connected to the simulator" when open, or "Not connected - start the server"
when idle. The Scenario Lab section is placed first, before the collapsible Advanced telemetry
section.

The existing telemetry WebSocket is loopback-only and already belongs to the dashboard. It will
accept one small JSON request type on that connection and return a matching request ID with either
a compact result or a readable error. The server accepts only the three named scenarios and one
numeric setting in a fixed range; it constructs the scenario from the validated car specification
and the existing deterministic runner. The request cannot supply code, paths, coefficients, or an
arbitrary scenario document. Browser-origin checks remain loopback-only.

The current live telemetry and analysis panels move under a closed **Advanced telemetry** section.
They mount only while open. Until then, the server sends 30 Hz snapshots without the full-rate
sample batches, and the browser skips storing those snapshot-only frames in chart history. Opening
the section subscribes that browser to detailed batches. Its canvas charts draw at a bounded rate
and pause while offscreen or the document is hidden. The run view stays useful without the
high-volume raw channel grid.

The visible success case is straightforward: choose a run, change its one control, press Run, and
see the result; repeating the same choice yields the same simulation output. Invalid controls and
failed runs produce a visible error while live telemetry remains connected.
