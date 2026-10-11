"""P7 Layers 2+3 multivariate SPC + persistence: hand-derived fixtures.

The calibration window is six frames of front/rear-left... front-right tyre
temperature pairs whose z-scores (against the contract sigma, read off the
``contract`` fixture and never restated) are exactly::

    (1, 1), (-1, -1), (2, 2), (-2, -2), (0.1, -0.1), (-0.1, 0.1)

Column means are exactly zero, so the covariance is ``[[2.004, 1.996],
[1.996, 2.004]]`` with eigenvalues 4.0 and 0.008 on the exact ``(1,1)/sqrt(2)``
and ``(1,-1)/sqrt(2)`` axes. Calibration T2 is ``{0.5, 0.5, 2, 2, 0, 0}`` and
calibration Q is ``{0, 0, 0, 0, 0.02, 0.02}``, so the empirical 99.7% thresholds
are 2.0 and 0.02. Every expected number below follows from that by hand.
"""

from __future__ import annotations

import pytest

from f1telemetry.analytics.multivariate import (
    cusum,
    ewma,
    fit_multivariate,
    persistent_alarms,
    score_multivariate,
)
from f1telemetry.analytics.validity import check_validity
from f1telemetry.contracts.channels import ChannelContract
from f1telemetry.testing.records import SensorFrame

_CHANNELS = ("tyre_temp_fl", "tyre_temp_fr")
_DT_S = 0.05
_BASE_C = 90.0

# Per-channel quiet bound: a classic 3-sigma marginal gate. The asymmetry tests
# keep every |z| below it while the joint statistics trip.
_QUIET_Z = 3.0


def _zframes(
    contract: ChannelContract, zpairs: tuple[tuple[float, float], ...]
) -> tuple[SensorFrame, ...]:
    sigma_fl = contract.by_name("tyre_temp_fl").sigma
    sigma_fr = contract.by_name("tyre_temp_fr").sigma
    assert sigma_fl == sigma_fr
    return tuple(
        SensorFrame(
            t_s=index * _DT_S,
            values={
                "tyre_temp_fl": _BASE_C + zfl * sigma_fl,
                "tyre_temp_fr": _BASE_C + zfr * sigma_fr,
            },
        )
        for index, (zfl, zfr) in enumerate(zpairs)
    )


def _calibration(contract: ChannelContract) -> tuple[SensorFrame, ...]:
    return _zframes(
        contract,
        ((1.0, 1.0), (-1.0, -1.0), (2.0, 2.0), (-2.0, -2.0), (0.1, -0.1), (-0.1, 0.1)),
    )


def test_fit_thresholds_are_empirical_997_quantiles(contract: ChannelContract) -> None:
    model = fit_multivariate(_calibration(contract), _CHANNELS, contract)
    assert model.channels == _CHANNELS
    assert model.means == pytest.approx((_BASE_C, _BASE_C))
    assert model.eigenvalues == pytest.approx((4.0,))
    assert model.t2_threshold == pytest.approx(2.0)
    assert model.q_threshold == pytest.approx(0.02)


def test_antisymmetric_fault_trips_q_while_channels_stay_quiet(
    contract: ChannelContract,
) -> None:
    model = fit_multivariate(_calibration(contract), _CHANNELS, contract)
    frames = _zframes(contract, ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.0, -2.0)))
    assert check_validity(frames, contract) == ()
    (sample,) = [s for s in score_multivariate(frames, model) if s.index == 3]
    # On the exact anti-diagonal: projection onto (1,1)/sqrt(2) is zero, so all
    # of ||z||^2 = 8 lands in Q while T2 stays at zero.
    assert sample.t2 == pytest.approx(0.0, abs=1e-9)
    assert sample.q == pytest.approx(8.0)
    assert not sample.t2_exceeds
    assert sample.q_exceeds
    assert max(abs(value) for pair in ((2.0, -2.0),) for value in pair) < _QUIET_Z


def test_in_manifold_shift_trips_t2_while_channels_stay_quiet(
    contract: ChannelContract,
) -> None:
    model = fit_multivariate(_calibration(contract), _CHANNELS, contract)
    frames = _zframes(contract, ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.6, 2.6)))
    assert check_validity(frames, contract) == ()
    (sample,) = [s for s in score_multivariate(frames, model) if s.index == 3]
    # On the manifold axis: t = 2.6*sqrt(2), T2 = 2*2.6^2/4.0 = 3.38, Q is zero.
    assert sample.t2 == pytest.approx(3.38)
    assert sample.q == pytest.approx(0.0, abs=1e-9)
    assert sample.t2_exceeds
    assert not sample.q_exceeds
    assert max(abs(value) for pair in ((2.6, 2.6),) for value in pair) < _QUIET_Z


def test_slow_ramp_trips_cusum_and_persistence_but_not_static_threshold() -> None:
    series = tuple(0.5 + 0.1 * index for index in range(12))
    static_threshold = 2.0
    assert max(series) < static_threshold
    assert not [value for value in series if value > static_threshold]

    stats = cusum(series, drift=1.0, threshold=1.0)
    # Hand-derived: x - 1 is negative through index 4, zero at 5, then
    # 0.1, 0.2, ... accumulate: g = 0.1, 0.3, 0.6, 1.0, 1.5, 2.1.
    assert stats == pytest.approx((0.0,) * 6 + (0.1, 0.3, 0.6, 1.0, 1.5, 2.1))
    exceeds = tuple(stat > 1.0 for stat in stats)
    assert exceeds == (False,) * 10 + (True, True)

    alarms = persistent_alarms(exceeds, k=2, n=3)
    assert alarms == (False,) * 11 + (True,)


def test_ewma_matches_hand_derivation() -> None:
    assert ewma((2.0, 2.0, 2.0), alpha=0.5) == pytest.approx((2.0, 2.0, 2.0))
    # s0 = 0, s1 = 0.5*1 + 0.5*0 = 0.5, s2 = 0.5*2 + 0.5*0.5 = 1.25.
    assert ewma((0.0, 1.0, 2.0), alpha=0.5) == pytest.approx((0.0, 0.5, 1.25))


def test_clean_record_stays_silent(contract: ChannelContract) -> None:
    model = fit_multivariate(_calibration(contract), _CHANNELS, contract)
    frames = _zframes(
        contract,
        ((0.2, 0.2), (-0.2, -0.2), (0.4, 0.36), (-0.3, -0.34), (0.1, 0.06), (0.0, 0.0)),
    )
    samples = score_multivariate(frames, model)
    assert len(samples) == len(frames)
    assert not [s for s in samples if s.t2_exceeds or s.q_exceeds]
    # Worst case by hand: (0.4, 0.36) gives T2 = 0.2888/4 = 0.0722, Q = 0.0008.
    worst = max(samples, key=lambda s: s.t2)
    assert worst.t2 == pytest.approx(0.0722)
    assert worst.q == pytest.approx(0.0008)
    # Slow-drift layer stays silent too: every T2 is far below the drift, so the
    # CUSUM never leaves zero and persistence has nothing to confirm.
    stats = cusum(tuple(s.t2 for s in samples), drift=1.0, threshold=1.0)
    assert stats == pytest.approx((0.0,) * len(samples))
    flags = tuple(s.t2_exceeds or s.q_exceeds for s in samples)
    assert persistent_alarms(flags, k=2, n=3) == (False,) * len(samples)


def test_persistence_cuts_false_alarms_on_noisy_fixture() -> None:
    threshold = 2.0
    series = [0.5] * 20
    for index in (2, 7, 13, 15, 16, 17, 18):
        series[index] = 2.5
    raw = tuple(value > threshold for value in series)
    assert sum(raw) == 7

    alarms = persistent_alarms(tuple(raw), k=3, n=5)
    # Hand-derived windows: the isolated spikes at 2, 7, 13 never see 3 hits in
    # any 5-window; the sustained block 15-18 first reaches 3 hits at index 16
    # (13, 15, 16) and holds through 19 on trailing windows.
    assert alarms == tuple(index in (16, 17, 18, 19) for index in range(20))
    assert sum(alarms) == 4
    assert sum(alarms) < sum(raw)
    assert not [index for index in (2, 7, 13) if alarms[index]]


def test_score_skips_frame_with_missing_channel(contract: ChannelContract) -> None:
    model = fit_multivariate(_calibration(contract), _CHANNELS, contract)
    frames = _zframes(contract, ((0.1, 0.1), (0.2, 0.2)))
    frames = (*frames, SensorFrame(t_s=2 * _DT_S, values={"tyre_temp_fl": _BASE_C}))
    samples = score_multivariate(frames, model)
    assert [s.index for s in samples] == [0, 1]


def test_default_contract_loads_channels_yaml(contract: ChannelContract) -> None:
    assert fit_multivariate(_calibration(contract), _CHANNELS).t2_threshold == pytest.approx(
        fit_multivariate(_calibration(contract), _CHANNELS, contract).t2_threshold
    )


def test_fit_rejects_bad_inputs(contract: ChannelContract) -> None:
    clean = _calibration(contract)
    with pytest.raises(ValueError, match="at least 2 channels"):
        fit_multivariate(clean, ("tyre_temp_fl",), contract)
    with pytest.raises(ValueError, match="duplicate"):
        fit_multivariate(clean, ("tyre_temp_fl", "tyre_temp_fl"), contract)
    with pytest.raises(ValueError, match="unknown channel"):
        fit_multivariate(clean, ("tyre_temp_fl", "nope"), contract)
    with pytest.raises(ValueError, match="no noise scale"):
        fit_multivariate(clean, ("tyre_temp_fl", "gear"), contract)
    with pytest.raises(ValueError, match="n_components"):
        fit_multivariate(clean, _CHANNELS, contract, n_components=2)
    with pytest.raises(ValueError, match="level"):
        fit_multivariate(clean, _CHANNELS, contract, level=1.0)
    with pytest.raises(ValueError, match="at least 2 frames"):
        fit_multivariate(clean[:1], _CHANNELS, contract)
    dirty = (SensorFrame(t_s=0.0, values={"tyre_temp_fl": _BASE_C}),) * 2
    with pytest.raises(ValueError, match="no usable value"):
        fit_multivariate(dirty, _CHANNELS, contract)


def test_layer3_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="threshold"):
        cusum((1.0,), drift=0.0, threshold=0.0)
    with pytest.raises(ValueError, match="drift"):
        cusum((1.0,), drift=float("nan"), threshold=1.0)
    with pytest.raises(ValueError, match="finite"):
        cusum((float("inf"),), drift=0.0, threshold=1.0)
    with pytest.raises(ValueError, match="alpha"):
        ewma((1.0,), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ewma((1.0,), alpha=1.5)
    with pytest.raises(ValueError, match="1 <= k"):
        persistent_alarms((True,), k=2, n=1)
    with pytest.raises(ValueError, match="n must be"):
        persistent_alarms((), k=1, n=0)
    assert cusum((), drift=1.0, threshold=1.0) == ()
    assert ewma((), alpha=0.5) == ()
    assert persistent_alarms((), k=1, n=3) == ()
