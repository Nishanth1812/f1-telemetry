# Detection table (generated)

Generated from `src/f1telemetry/analytics/evaluate.py`; do not hand-edit. Regenerate with:

```text
uv run python -m f1telemetry.analytics.evaluate --regenerate docs/detection.md
```

## Method

Channels: `tyre_temp_fl, tyre_temp_fr` (multivariate pair); faulted channel: `tyre_temp_fl`. Records hold 60 frames at 0.05 s with fault onset at frame 20. Sweep: 10 fault types x 2 severities (0.5, 1.0) x 3 seeds (11, 22, 33). Calibration uses a clean six-frame window only; injected faults are never in the training set. Persistence rule: 2-of-3. Localisation is the Layer 1 ranking at the detection frame with a Layer 0 fallback; `swap` accepts either corner.

## Detection matrix (per fault x severity, seeds aggregated)

Latency is mean frames from onset over detected seeds; `--` means no seed in the group was detected (miss) or localised.

| fault | severity | seeds | l2 miss | l2 latency | l2 loc | l2+persistence miss | l2+persistence latency | l2+persistence loc | l2+persistence+isolationforest miss | l2+persistence+isolationforest latency | l2+persistence+isolationforest loc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dropout | 0.5 | 3 | 1.00 | -- | -- | 1.00 | -- | -- | 1.00 | -- | -- |
| dropout | 1.0 | 3 | 1.00 | -- | -- | 1.00 | -- | -- | 1.00 | -- | -- |
| freeze | 0.5 | 3 | 0.00 | 0.7 | 0.00 | 0.00 | 4.3 | 0.00 | 1.00 | -- | -- |
| freeze | 1.0 | 3 | 0.00 | 0.7 | 0.00 | 0.00 | 4.3 | 0.00 | 1.00 | -- | -- |
| spike | 0.5 | 3 | 0.00 | 0.0 | 1.00 | 1.00 | -- | -- | 1.00 | -- | -- |
| spike | 1.0 | 3 | 0.00 | 0.0 | 1.00 | 1.00 | -- | -- | 1.00 | -- | -- |
| step | 0.5 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| step | 1.0 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| gain | 0.5 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| gain | 1.0 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| noise | 0.5 | 3 | 0.00 | 0.3 | 0.33 | 0.00 | 1.3 | 0.67 | 1.00 | -- | -- |
| noise | 1.0 | 3 | 0.00 | 0.0 | 0.67 | 0.00 | 1.0 | 0.67 | 1.00 | -- | -- |
| quantise | 0.5 | 3 | 0.00 | 0.7 | 0.33 | 0.00 | 4.7 | 0.00 | 1.00 | -- | -- |
| quantise | 1.0 | 3 | 0.00 | 0.0 | 0.33 | 0.00 | 1.3 | 0.67 | 1.00 | -- | -- |
| swap | 0.5 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| swap | 1.0 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| saturate | 0.5 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| saturate | 1.0 | 3 | 0.00 | 0.0 | 1.00 | 0.00 | 1.0 | 1.00 | 1.00 | -- | -- |
| stale | 0.5 | 3 | 0.00 | 0.0 | 0.00 | 0.00 | 3.7 | 0.67 | 1.00 | -- | -- |
| stale | 1.0 | 3 | 0.00 | 0.7 | 0.67 | 0.00 | 3.0 | 0.33 | 1.00 | -- | -- |

## Baseline summaries

| baseline | miss rate | mean latency (frames) | localisation accuracy | clean false alarms | false alarms / hour |
|---|---|---|---|---|---|
| l2 | 0.100 | 0.17 | 0.685 | 0/180 | 0.0 |
| l2+persistence | 0.200 | 1.98 | 0.688 | 0/180 | 0.0 |
| l2+persistence+isolationforest | 1.000 | -- | -- | 0/180 | 0.0 |

## Negative control

Operational-variability record (slow 0.5 C warm-up ramp, seeded jitter, no injected fault, no genuine abnormality). The detector must stay silent.

| baseline | alarms | frames | silent |
|---|---|---|:---:|
| l2 | 0 | 60 | yes |
| l2+persistence | 0 | 60 | yes |
| l2+persistence+isolationforest | 0 | 60 | yes |

## Isolation Forest gap

The `l2+persistence+isolationforest` column is scikit-learn's IsolationForest (`n_estimators=100`, `random_state=0`), fit on the six clean calibration frames only. Its feature vector is the Layer 2 standardised pair, so the forest sees the same `(y - mu) / sigma` the manifold does. The alarm threshold is the calibration window's own 0.003 score quantile, the lower-tail mirror of Layer 2's upper-quantile threshold, and the same k-of-n persistence rule as the other columns applies.

Caveats. `max_samples` resolves to the calibration size, so each tree sees every calibration frame: a six-point fit is a thin basis, not a learned density. Attribution is the estimator-score gain from resetting one channel to its calibration mean. The hyperparameters were fixed before the sweep and not tuned on it. The sweep is simulated, so this is not a held-out real-data result.

## Reading notes

* Layer 2 scores every frame with data; frames with a missing or non-finite value are skipped by design (Layer 0 owns those values), so a sustained `dropout` is a Layer 2 miss and a Layer 0 catch.
* Single-frame `spike` trips the single-window baseline but cannot survive a k-of-n rule with k >= 2; that miss under persistence is the documented cost of false-alarm suppression.
* `freeze` and `stale` hold plausible values, so a marginal manifold often stays silent on them; the stuck-value rule in Layer 0 is the layer that owns `freeze`.

