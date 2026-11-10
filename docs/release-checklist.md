# Release evidence

The portfolio release demonstrates local synthetic telemetry experiments, saved
recordings, replay and comparison. It does not validate a real Formula car or
establish production operating history. See the [release scope](roadmap/release/specs/2026-11-03-portfolio-release-design.md),
[calibration notes](calibration.md) and [detection results](detection.md).

## Evidence

- [CI run 41655859505](https://github.com/Nishanth1812/f1-telemetry/actions/runs/41655859505)
  passed on `84d4650`, including fast Python checks, browser scenarios, contract
  rate checks and the read-only demo container. It predates later release changes.
- On 2026-11-09, local lint, formatting, type, contract, frontend tests and build
  checks were run; the browser workflow passed scenario, save, restart, replay
  and comparison. The recorded measurements and their limits are in
  [performance.md](performance.md).
- The [scheduled full suite](https://github.com/Nishanth1812/f1-telemetry/actions/runs/42604272587)
  on `main` failed an exact floating-point comparison in a simulated golden trace.
  Simulation traces are advisory and have a separate threshold-based report;
  fixed hand-authored fixtures retain exact checks.
- On 2026-11-09, the corrected baseline and advisory report checks passed on
  Windows, as did five focused golden-harness checks. The full Python suite
  then passed on Windows: 1,443 tests in 1:21:09. The subsequent history merge
  introduced no source or test changes; this evidence update is documentation only.

## Before publishing

- Review golden reports and rerun checks after any source or test changes. The
  latest four golden reports flagged no channels.
- Confirm CI passes on the release commit and that the public default branch
  contains the intended README, screenshot, demo and evidence.
- Choose a release tag only after the checks pass. Public hosting is optional;
  a deployed service needs its own operational and security assessment.
