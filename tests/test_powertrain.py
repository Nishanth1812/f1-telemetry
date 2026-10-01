"""P1-T4: the synthesised ICE torque curve and its turbo-lag multiplier.

``PHASES.md`` P1-T4 asks for an ICE torque curve that is **synthesised rather than sourced**, with
a turbo-lag multiplier collapsing below about 4 000 rpm. The eight knots and both lag numbers are
``car_spec.yaml`` data; this file covers the two decisions that turn that table into a torque for
any engine speed, and the boundary they are read through:

* **Interpolation.** At a knot the torque is the file's own number; between knots it is the
  linear interpolation - the rule Task 3 froze for ``Cl(v)`` and ``Cd(v)``, and a nearest-knot
  lookup cannot produce a midpoint.
* **The lag.** The multiplier applies strictly below the configured ``collapse_below_rpm`` and is
  exactly ``1.0`` at and above it, so the collapse is a property of the file and not of the code.
* **The configuration.** Every input is read off ``KernelConfig``, so replacing the curve, the
  threshold or the multiplier moves the torque it names. That is what makes this file a test of
  the model rather than a restatement of it.
* **The boundary.** ``KernelConfig`` is public and replaceable, so ``step_ice_torque`` checks
  what it reads before the values reach a ``boundscheck=False`` lookup, which on an empty, short
  or reversed curve returns a plausible-looking number rather than failing.

Every physical number comes from the loaded ``car_spec.yaml``, so this file tunes nothing.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from f1telemetry.physics import powertrain

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import CarSpec, KernelConfig

pytestmark = pytest.mark.powertrain


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the torque model is allowed to see."""
    return spec.kernel_config()


def _torque(config: KernelConfig, rpm: float) -> float:
    """``ice_torque_nm`` with the configured curve and lag, in the documented order."""
    return powertrain.ice_torque_nm(
        rpm,
        config.torque_rpm,
        config.torque_nm,
        config.turbo_lag_collapse_rpm,
        config.turbo_lag_multiplier,
    )


def test_the_curve_is_read_as_a_piecewise_linear_table(config: KernelConfig) -> None:
    """At a knot the file's own number, and halfway across a segment the mean of two.

    At a knot the interpolation is the identity, so the only ways to be wrong there are to shift
    the rpm axis, to read the wrong array, or to divide by a lag that must not apply. Between
    knots the answer is the mean of the two tabulated values, which a nearest-knot lookup cannot
    produce. Only the knots at or above the lag threshold are used: below it the file's number is
    deliberately not the answer, and that is the next test's business.
    """
    axis, values = config.torque_rpm, config.torque_nm
    threshold = config.turbo_lag_collapse_rpm
    assert float(axis[0]) < threshold <= float(axis[-1]), (
        "the committed curve must straddle the lag threshold, or the multiplier is dead code"
    )
    unreduced = [index for index, rpm in enumerate(axis) if rpm >= threshold]
    assert len(unreduced) >= 2, "a between-knot check needs two knots above the threshold"

    for index in unreduced:
        assert _torque(config, float(axis[index])) == float(values[index])

    for index in unreduced[:-1]:
        low_rpm, high_rpm = float(axis[index]), float(axis[index + 1])
        midpoint = 0.5 * (low_rpm + high_rpm)
        assert low_rpm < midpoint < high_rpm
        assert _torque(config, midpoint) == pytest.approx(
            0.5 * (float(values[index]) + float(values[index + 1])), rel=1e-12
        )


def test_the_turbo_multiplier_is_the_configured_value_below_the_threshold_and_one_at_or_above(
    config: KernelConfig,
) -> None:
    """The reduction is checked from both sides of the number the file holds.

    One ULP below the threshold is the smallest step a caller can ask about, and it is enough to
    catch an implementation that compares against the wrong side of it. The last two assertions
    are the same claim at the torque level, so the multiplier is shown to divide the curve rather
    than replace it.
    """
    threshold = config.turbo_lag_collapse_rpm
    multiplier = config.turbo_lag_multiplier
    assert 0.0 < multiplier < 1.0, "a lag multiplier is a reduction; 1.0 would mean there is none"
    factor = powertrain.turbo_lag_factor

    assert factor(threshold, threshold, multiplier) == 1.0
    assert factor(math.nextafter(threshold, 0.0), threshold, multiplier) == multiplier
    assert factor(threshold - 1.0, threshold, multiplier) == multiplier
    assert factor(threshold + 1.0, threshold, multiplier) == 1.0

    assert _torque(config, math.nextafter(threshold, 0.0)) == pytest.approx(
        _torque(config, threshold) * multiplier, rel=1e-12
    )


def test_the_torque_and_the_lag_are_read_off_the_kernel_config(config: KernelConfig) -> None:
    """Replacing a config field moves the torque that field names, and nothing else.

    A multiplier or a threshold written into the module would pass the two tests above and fail
    here, which is why this file can claim it tunes nothing. The composition is checked against
    the primitive first, so the five values it reads are pinned in order as well as present.
    """
    threshold = config.turbo_lag_collapse_rpm
    multiplier = config.turbo_lag_multiplier
    below, above = threshold - 1_000.0, threshold + 1_000.0
    for rpm in (below, threshold, above):
        assert powertrain.step_ice_torque(config, rpm) == _torque(config, rpm)

    weaker_lag = replace(config, turbo_lag_multiplier=multiplier / 2.0)
    assert _torque(weaker_lag, below) == pytest.approx(_torque(config, below) / 2.0, rel=1e-12)
    assert _torque(weaker_lag, above) == _torque(config, above)

    later_spool = replace(config, turbo_lag_collapse_rpm=threshold + 2_000.0)
    assert _torque(later_spool, above) == pytest.approx(
        _torque(config, above) * multiplier, rel=1e-12
    ), "raising the threshold puts a speed that was undivided under the reduction"

    doubled_top = config.torque_nm.copy()
    doubled_top[-1] *= 2.0
    assert _torque(
        replace(config, torque_nm=doubled_top), float(config.torque_rpm[-1])
    ) == pytest.approx(2.0 * float(config.torque_nm[-1]), rel=1e-12)


@pytest.mark.parametrize("rpm", [math.nan, math.inf, -math.inf, np.float32(4_000.0)])
def test_step_ice_torque_refuses_an_rpm_it_could_not_narrow_to_a_float(
    config: KernelConfig,
    rpm: float,
) -> None:
    """Two ways an rpm fails, neither of which is the torque: not finite, or not a plain number.

    A NaN rpm reaches the compiled lookup and comes back as a NaN torque: every comparison
    against it is false, so the segment search never advances and the interpolation is handed a
    NaN weight - a number rather than an error. The composed torque is what an integrator
    multiplies by, so a NaN here is a run that looks like it finished. An infinity is the same
    failure wearing a finite sign.

    A ``float32`` is refused for a different reason, and a valid one: it is not an ``int`` or a
    ``float``, so it would narrow to neither and compile a second Numba signature alongside the
    ``float`` one. The caller narrows at the Python boundary instead, which is the same rule the
    last test in this file pins from the other side.
    """
    with pytest.raises(ValueError, match="rpm"):
        powertrain.step_ice_torque(config, rpm)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("turbo_lag_collapse_rpm", 0.0),
        ("turbo_lag_collapse_rpm", -1.0),
        ("turbo_lag_collapse_rpm", math.nan),
        ("turbo_lag_multiplier", 0.0),
        ("turbo_lag_multiplier", -0.5),
        ("turbo_lag_multiplier", 1.5),
        ("turbo_lag_multiplier", math.inf),
    ],
)
def test_step_ice_torque_refuses_a_lag_configuration_it_could_not_use(
    config: KernelConfig,
    name: str,
    value: float,
) -> None:
    """The two lag numbers carry the ranges ``car_spec.yaml`` checks them against.

    ``multiplier_at_collapse`` is a reduction, so it has to sit in ``(0, 1]``: a zero would hand
    back no torque below the threshold and an above-one would call the collapsed engine the
    stronger one. ``collapse_below_rpm`` has to be positive because a nonpositive threshold sits
    below every engine speed and would collapse the whole curve.
    """
    with pytest.raises(ValueError, match=name):
        powertrain.step_ice_torque(replace(config, **{name: value}), 5_000.0)


def test_step_ice_torque_refuses_a_replaced_curve_it_could_not_read(
    config: KernelConfig,
) -> None:
    """A replaced ``KernelConfig`` cannot get an unlookupable curve past the boundary.

    ``ice_torque_nm`` is compiled with ``boundscheck=False``, so each of these returns a plausible
    number rather than raising: an empty curve reads uninitialised memory, a mismatched length
    reads past the end of ``torque_nm``, and a reversed axis finds a segment that is not there.
    None of them can be caught downstream, because a wrong torque is exactly what the model is
    asked to produce.
    """
    axis, values = config.torque_rpm, config.torque_nm
    repeated = axis.copy()
    repeated[2] = repeated[1]
    infinite = values.copy()
    infinite[3] = np.inf
    empty = np.array([], dtype=np.float64)
    strided = np.empty(axis.size * 2, dtype=np.float64)
    strided[::2] = axis
    malformed = (
        replace(config, torque_rpm=axis.astype(np.float32)),
        replace(config, torque_nm=values.astype(np.int64)),
        replace(config, torque_rpm=axis.tolist()),
        replace(config, torque_nm=values.reshape(2, -1)),
        replace(config, torque_rpm=strided[::2]),
        replace(config, torque_rpm=empty, torque_nm=empty),
        replace(config, torque_nm=values[:-1]),
        replace(config, torque_rpm=repeated),
        replace(config, torque_rpm=axis[::-1].copy()),
        replace(config, torque_nm=infinite),
    )
    for broken in malformed:
        with pytest.raises(ValueError, match="step_ice_torque: config"):
            powertrain.step_ice_torque(broken, 5_000.0)


@pytest.mark.parametrize("rpm", [0.0, -1_000.0])
def test_step_ice_torque_is_finite_at_zero_and_negative_rpm(
    config: KernelConfig,
    rpm: float,
) -> None:
    assert math.isfinite(powertrain.step_ice_torque(config, rpm))


def test_step_ice_torque_refuses_a_negative_torque_knot_and_accepts_a_zero(
    config: KernelConfig,
) -> None:
    """A negative knot is not a torque curve; zero is the other end of the range.

    The table is read as written, so a negative knot hands back a torque that pushes the car
    backwards at full request - engine braking wearing the ICE's sign, and a drivetrain that P1-T5
    gets from the clutch and the brakes instead. Nothing downstream can tell that apart from a
    real drive torque, which is what makes it worth a check at the boundary. Zero is the other end
    of the legal range and an engine legitimately makes no torque, so the rule is a sign rather
    than the strict positivity the aero curves carry: a curve allowed to fall to 0 must be allowed
    to sit at it.
    """
    top_rpm = float(config.torque_rpm[-1])
    zeroed = config.torque_nm.copy()
    zeroed[-1] = 0.0
    assert powertrain.step_ice_torque(replace(config, torque_nm=zeroed), top_rpm) == 0.0

    negative = config.torque_nm.copy()
    negative[-1] = -1.0
    with pytest.raises(ValueError, match=r"step_ice_torque: config\.torque_nm"):
        powertrain.step_ice_torque(replace(config, torque_nm=negative), top_rpm)


def test_step_ice_torque_narrows_numeric_scalars_before_numba(config: KernelConfig) -> None:
    """Equivalent int and float inputs use the same compiled specialisation.

    Without the narrowing, an int read off a config and a float read off one compile two
    signatures and every later rpm value has to be checked against a second copy of the curve.
    """
    powertrain.step_ice_torque(config, 5_000.0)
    compiled = len(powertrain.ice_torque_nm.nopython_signatures)
    assert compiled > 0
    integer_config = replace(config, turbo_lag_collapse_rpm=4_000, turbo_lag_multiplier=1)
    powertrain.step_ice_torque(integer_config, 5_000)
    powertrain.step_ice_torque(config, 5_000.0)
    assert len(powertrain.ice_torque_nm.nopython_signatures) == compiled
