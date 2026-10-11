# Virtual tests and the live dashboard

The dashboard is an engineer's view of one simulated Formula car. The **virtual test** bar at
the top starts a controlled test; everything below it shows the car's telemetry live.

## What happens when you press Run test

1. The browser sends one small JSON request over the existing loopback-only telemetry WebSocket:
   `{"type":"scenario_run","request_id","scenario_id","setting"}`. The server accepts only three
   named tests (launch, braking, cornering) and one numeric setting in a fixed range. The request
   cannot supply code, paths, coefficients, or an arbitrary scenario document.
2. The server builds the scenario from the validated car spec and runs the deterministic headless
   kernel (about 1-2 s of compute for a 7 s test). Repeating the same test gives the same output.
3. The server replies with a compact `scenario_result` summary (or a readable `scenario_error`)
   and queues the run's 100 Hz frames on `DrivenCarSource` (`src/f1telemetry/live_car.py`).
4. That source replays the frames in real time through the same wire format the dashboard already
   reads: a 30 Hz snapshot carrying the latest values, plus the full-rate samples the charts use.
   When the test ends the last values stay on screen so tyre and brake state can be inspected.

Until the first test the car is at rest: the source repeats the standstill values. The clock only
moves forward, so a second test follows the first. The browser clears its charts and event log
when a new test starts.

`f1-serve --noise` swaps in the old uniform-random contract generator, which only proves the
per-channel rates and has nothing to do with the car's behaviour.

## Page layout

1. **Virtual test**: test buttons, the single setting, Run, and a status chip
   ("Car at rest", "Driving: 3.2 / 7.0 s", "Test finished").
2. **Live car data**: RPM, speed, gear, throttle, steering, lateral / longitudinal G and yaw rate.
3. **Charts, tyres, brakes** (live while a test runs).
4. **Model analysis**: anomaly warnings from the dashboard's detectors.
5. **Last test result**: peak speed and g-forces, a comparison with the previous run of the same
   test, and speed / acceleration traces. The simulation is uncalibrated, and runs where the car
   slides sideways in a straight-line test are flagged as a model limit.
6. **Advanced**: the raw channel grid, laps, events and scrubber, for engineers.

Detailed full-rate batches are sent to the browser while the dashboard is open. Canvas charts draw
at a bounded rate and pause while offscreen or the document is hidden.
