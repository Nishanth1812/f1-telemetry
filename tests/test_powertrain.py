"""P1-T4's ICE torque curve, and the 2026 power-unit limits that bound it.

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
* **Nothing clips the curve on its way to the car.** One test reaches the drivetrain to check that
  the engine's own torque arrives at the rear axle: the synthetic ``clutch_torque_capacity_nm`` that
  used to cap a launch below what first gear offers is gone, and the C9.2.5 demand sits above it.

**The power-unit limits, which are the point of the second half of this file.** The regulations
never state an ICE power in kW; they bound the engine through *fuel energy flow* (C5.2.3, C5.2.4,
C5.2.5, page 64), so ``docs/calibration.md`` section 5 asks for the torque curve to be checked
against those limits rather than against the published 400 kW shorthand. ``step_ice_torque`` is
where that check is applied, and the MGU-K limits are applied to the motor that joins the same
crankshaft:

* **C5.2.11 is a torque limit at 500 Nm.** C5.18.4's 520 Nm is *not* another cap - it is the
  threshold above which an optional torque-limiting device may act - so it never reaches the
  physics and the applied cap is the smaller number.
* **C5.2.7 and C5.2.8 bound the motor by power,** and the conversion from shaft power to
  electrical DC power needs an efficiency the regulations do not publish, so
  ``mgu_k.motor_inverter_efficiency`` and ``ice.fuel_to_shaft_efficiency`` are declared synthetic
  and are read as data.
* **C5.2.12 blocks the motor below 50 km/h in a *declared* grid standing start,** and only there -
  the exception the article carries is a fact about the mandated launch, not a constant, so both are
  caller declarations rather than numbers.
* **C5.2.9 is a 4 MJ usable window, and C5.2.10 is an event-conditioned recharge,** so the store is
  a range the deployment and the recuperation are both bounded by, and the per-lap recharge limit
  is selected by the event rather than fixed at the 8.5 MJ baseline.

Every physical number comes from the loaded ``car_spec.yaml``, so this file tunes nothing.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from f1telemetry.physics import gearbox, powertrain
from f1telemetry.physics.powertrain import RechargeEvent

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


def _ice_rpm_for_km_h(config: KernelConfig, speed_km_h: float) -> float:
    """The crankshaft speed that puts the lowest gear on the road at ``speed_km_h``.

    The MGU-K limits are all speed- or rpm-referenced, so a test that wants to ask about one at a
    road speed has to get to it through the drivetrain rather than invent an engine speed. First
    gear is the only gear that reaches every speed, so it is the one used.
    """
    wheel_ratio = float(config.gear_ratios[0]) * config.final_drive
    return speed_km_h / 3.6 * wheel_ratio * 60.0 / (math.tau * config.rolling_radius_m)


def _fuel_flow_limit(config: KernelConfig, rpm: float, power_kw: float) -> float:
    """The C5.2.3/.4/.5 limit at one operating point, through the public primitive."""
    return powertrain.ice_fuel_energy_flow_limit_mj_h(
        rpm,
        power_kw,
        config.fuel_energy_flow_max_mj_h,
        config.fuel_energy_flow_low_rpm_limit_rpm,
        config.fuel_energy_flow_low_rpm_gain,
        config.fuel_energy_flow_low_rpm_offset_mj_h,
        config.fuel_energy_flow_partial_load_threshold_kw,
        config.fuel_energy_flow_partial_load_gain,
        config.fuel_energy_flow_partial_load_offset_mj_h,
        config.fuel_energy_flow_partial_load_min_mj_h,
    )


def _fuel_flow(config: KernelConfig, torque_nm: float, rpm: float) -> float:
    """The fuel energy flow a delivered torque costs, through the public primitives."""
    return powertrain.ice_fuel_energy_flow_mj_h(
        powertrain.ice_power_kw(torque_nm, rpm), config.fuel_to_shaft_efficiency
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


def test_the_ice_torque_reaches_the_rear_axle_without_the_synthetic_clutch_clamp(
    config: KernelConfig,
) -> None:
    """The engine's own torque is what drives the car, with nothing clipped out of it on the way.

    This is the powertrain's half of the C9.2.5 clutch slice. The committed synthetic
    ``clutch_torque_capacity_nm`` of 3 000 Nm used to sit below what first gear offers, so at the
    engine's peak the driveline got the clamp rather than the engine - a 330 Nm curve whose first
    gear delivered 3 000 Nm instead of 3 844 Nm. The clamp is gone, so the launch now delivers the
    engine's torque scaled by the configured reduction, and the C9.2.5 demand sits above it rather
    than in the way of it.

    The gearbox is reached through :func:`~f1telemetry.physics.gearbox.step_gearbox` on purpose: the
    claim is about the torque at the rear axle, which is the composition's output and not a number
    this module could produce on its own.
    """
    peak_rpm = float(config.torque_rpm[int(np.argmax(config.torque_nm))])
    peak_nm = powertrain.step_ice_torque(config, peak_rpm)
    first = float(config.gear_ratios[0]) * config.final_drive
    offered_nm = peak_nm * first

    assert offered_nm > config.clutch_torque_capacity_nm, (
        "this test needs a gear whose offer is above the old clamp, or the clamp would have been "
        "invisible here"
    )
    assert offered_nm < config.clutch_demand_torque_nm, (
        "the C9.2.5 demand has to exceed launch torque, or the paddle could not hold "
        "the clutch against the engine"
    )

    state = gearbox.initial_state(gear=1.0, clutch_engagement=1.0)
    assert gearbox.step_gearbox(config, state, peak_rpm, 1.0) == pytest.approx(
        offered_nm, rel=1e-12
    )

    # And the same at a fraction of the pedal: the engine decides the torque through a closed
    # clutch,
    # so a half-throttle launch is half of it rather than the clamp or the demand.
    assert gearbox.step_gearbox(config, state, peak_rpm, 0.5) == pytest.approx(
        0.5 * offered_nm, rel=1e-12
    )


def test_the_fuel_energy_flow_limit_is_the_smallest_clause_that_applies(
    config: KernelConfig,
) -> None:
    """Three caps, each stated as "must not exceed", so the binding one is the minimum.

    C5.2.3 caps the total at 3 000 MJ/h, C5.2.4 replaces it below 10 500 rpm with
    ``0.27*N + 165``, and C5.2.5 adds a two-arm limit in *engine power*: a flat 380 MJ/h at or
    below -50 kW and ``9.78*P + 869`` above it. Taking the smallest of the applicable arms is what
    "must not exceed" means when three articles state it, and it is the reading under which the
    published 400 kW shorthand never enters the physics at all.

    The boundaries are checked from both sides of each clause rather than at one interior point,
    because an arm that is off by one comparison silently doubles the limit at exactly the speed a
    standing start or a top-speed run sits on.
    """
    limit = powertrain.ice_fuel_energy_flow_limit_mj_h
    args = (
        config.fuel_energy_flow_max_mj_h,
        config.fuel_energy_flow_low_rpm_limit_rpm,
        config.fuel_energy_flow_low_rpm_gain,
        config.fuel_energy_flow_low_rpm_offset_mj_h,
        config.fuel_energy_flow_partial_load_threshold_kw,
        config.fuel_energy_flow_partial_load_gain,
        config.fuel_energy_flow_partial_load_offset_mj_h,
        config.fuel_energy_flow_partial_load_min_mj_h,
    )
    low_rpm = config.fuel_energy_flow_low_rpm_limit_rpm
    threshold = config.fuel_energy_flow_partial_load_threshold_kw

    def low_rpm_arm(rpm: float) -> float:
        return (
            config.fuel_energy_flow_low_rpm_gain * rpm + config.fuel_energy_flow_low_rpm_offset_mj_h
        )

    def partial_arm(power_kw: float) -> float:
        return (
            config.fuel_energy_flow_partial_load_gain * power_kw
            + config.fuel_energy_flow_partial_load_offset_mj_h
        )

    # C5.2.4's arm is the binding one through most of the rev range, because 0.27 * N + 165 only
    # reaches the 3 000 MJ/h total at the very rpm where the clause stops applying. 250 kW is
    # chosen so C5.2.5's power arm sits above it and cannot be mistaken for the binding one.
    assert limit(4_000.0, 250.0, *args) == pytest.approx(low_rpm_arm(4_000.0))
    assert limit(low_rpm - 1.0, 250.0, *args) == pytest.approx(low_rpm_arm(low_rpm - 1.0))
    assert limit(low_rpm, 250.0, *args) == config.fuel_energy_flow_max_mj_h, (
        "C5.2.4 applies *below* its crossover; at and above it the 3 000 MJ/h total binds"
    )
    assert partial_arm(250.0) > low_rpm_arm(low_rpm - 1.0), "the arms have to be the ones compared"

    # C5.2.5's two arms, from both sides of the -50 kW threshold. Neither reaches a 3 000 MJ/h
    # total at these powers, so the flat arm is pinned against a replaced total instead.
    assert limit(4_000.0, threshold, *args) == pytest.approx(
        config.fuel_energy_flow_partial_load_min_mj_h
    )
    assert limit(4_000.0, threshold - 1.0, *args) == pytest.approx(
        config.fuel_energy_flow_partial_load_min_mj_h
    )
    assert limit(4_000.0, threshold + 1.0, *args) == pytest.approx(partial_arm(threshold + 1.0))

    thin_total = (100.0, *args[1:])
    assert limit(4_000.0, threshold, *thin_total) == 100.0, (
        "the total cap must bind when it is the smallest of the three, or the model would read "
        "C5.2.5's flat arm as the only limit"
    )


def test_the_ice_torque_curve_is_bounded_by_the_fuel_energy_flow_and_not_by_the_400_kw_shorthand(
    config: KernelConfig,
) -> None:
    """Every engine speed on the curve delivers a fuel energy flow inside the limit.

    This is the check ``docs/calibration.md`` section 5 asks for: the ICE is bounded by fuel
    energy flow, because that is the only bound the regulations state. Sweeping the whole curve
    and re-deriving the fuel flow from the *returned* torque is what makes it a check on the
    physics rather than a restatement of the limit function - a curve that ignored the limit
    entirely would pass a test that only asked for the limit's value.

    The committed curve is not clipped by it at the configured efficiency, which the next test
    asserts out loud, so a later edit that quietly made the limit a dead clamp would be visible
    rather than absorbed into the knot assertions.
    """
    efficiency = config.fuel_to_shaft_efficiency
    for rpm in np.linspace(0.0, float(config.rev_limit_rpm), 61):
        rpm = float(rpm)
        torque = powertrain.step_ice_torque(config, rpm)
        power_kw = powertrain.ice_power_kw(torque, rpm)
        if power_kw <= 0.0:
            continue
        flow = powertrain.ice_fuel_energy_flow_mj_h(power_kw, efficiency)
        limit = _fuel_flow_limit(config, rpm, power_kw)
        assert flow <= limit + 1e-9, f"fuel energy flow {flow} MJ/h over {limit} MJ/h at {rpm} rpm"


def test_a_greedy_engine_is_clipped_to_the_fuel_energy_flow_limit(config: KernelConfig) -> None:
    """The limit is load-bearing: a curve that costs more fuel than C5.2.3 allows is cut back.

    ``replace`` on the efficiency is the honest way to make the committed curve expensive enough
    to hit the cap: a 400 kW peak at 50 % efficiency is 2 880 MJ/h, which fits under 3 000 MJ/h,
    and no edit to the torque table would be needed to prove the check is real. This is the same
    argument Task 3 used to keep a grip limit from being a branch that can never fire.
    """
    greedy = replace(config, fuel_to_shaft_efficiency=0.40)
    efficiency = greedy.fuel_to_shaft_efficiency
    top_rpm = float(config.torque_rpm[-1])
    torque = powertrain.step_ice_torque(greedy, top_rpm)
    unclipped = powertrain.step_ice_torque(config, top_rpm)

    assert torque < unclipped, (
        "a 40 % efficient engine at the limiter must be cut back, or the C5.2.3 cap is dead code"
    )
    assert (
        powertrain.ice_fuel_energy_flow_mj_h(
            powertrain.ice_power_kw(unclipped, top_rpm), efficiency
        )
        > config.fuel_energy_flow_max_mj_h
    ), (
        "the unclipped curve must be the one that breaks the limit, or the clip below it is "
        "being compared against nothing"
    )
    assert powertrain.ice_fuel_energy_flow_mj_h(
        powertrain.ice_power_kw(torque, top_rpm), efficiency
    ) == pytest.approx(config.fuel_energy_flow_max_mj_h, rel=1e-9)


def test_the_committed_ice_curve_is_not_clipped_by_the_fuel_energy_flow_limit(
    config: KernelConfig,
) -> None:
    """The whole curve survives the cap at the committed efficiency, so the cap is a bound and not
    a hidden calibration of the curve.

    A limit that quietly rewrote every torque the file holds would make this file's ICE tests
    meaningless: ``test_the_curve_is_read_as_a_piecewise_linear_table`` compares the knots against
    the file's own numbers, and a clip would pass that only if it never fired. Saying so here
    means a later efficiency edit that does start clipping shows up as *this* test failing rather
    than as every knot assertion quietly becoming about the clip.
    """
    top_rpm = float(config.torque_rpm[-1])
    assert powertrain.step_ice_torque(config, top_rpm) == _torque(config, top_rpm), (
        "the committed curve must survive the C5.2.3 fuel-energy-flow cap uncut; if it does not, "
        "the efficiency in car_spec.yaml is what is being calibrated, not the torque table"
    )
    assert (
        powertrain.ice_fuel_energy_flow_mj_h(
            powertrain.ice_power_kw(_torque(config, top_rpm), top_rpm),
            config.fuel_to_shaft_efficiency,
        )
        < config.fuel_energy_flow_max_mj_h
    )


def test_the_fuel_energy_flow_limit_sees_the_throttled_power_not_the_full_request(
    config: KernelConfig,
) -> None:
    """Half throttle is checked at half the power, because C5.2.5 is stated in engine power.

    The limit is a function of the power the engine is actually making, so a caller that asked for
    a full-throttle torque and was refused a pedal halfway has to be measured at the delivered
    figure. Measuring at the full request instead would clip torque that is legal at the pedal
    position the driver actually chose, and would do it silently.
    """
    rpm = float(config.torque_rpm[int(np.argmax(config.torque_nm))])
    greedy = replace(config, fuel_to_shaft_efficiency=0.40)
    requested = _torque(greedy, rpm)

    full = powertrain.step_ice_torque(greedy, rpm, 1.0)
    half = powertrain.step_ice_torque(greedy, rpm, 0.5)

    assert full < requested, "this test needs a point where the cap actually bites at full pedal"
    assert half == pytest.approx(0.5 * requested, rel=1e-12), (
        "at half the pedal the engine delivers half the power, and C5.2.5's arm rises with power - "
        "so the cap that binds at full request cannot bind here. Measuring at the full request "
        "would clip torque the driver is legally allowed to ask for"
    )
    full_kw = powertrain.ice_power_kw(full, rpm)
    half_kw = powertrain.ice_power_kw(half, rpm)
    assert _fuel_flow(greedy, full, rpm) == pytest.approx(
        _fuel_flow_limit(greedy, rpm, full_kw), rel=1e-9
    ), "the full-pedal case must sit exactly on its limit"
    assert _fuel_flow(greedy, half, rpm) == pytest.approx(0.5 * _fuel_flow(greedy, requested, rpm))
    assert _fuel_flow(greedy, half, rpm) < _fuel_flow_limit(greedy, rpm, half_kw), (
        "half the pedal costs half the fuel and is measured against the arm that goes with it, "
        "which is higher - so the cap that bit at full pedal is out of reach here"
    )


def test_the_mgu_k_never_delivers_more_dc_power_than_the_c52_7_cap_or_the_c52_8_speed_curve(
    config: KernelConfig,
) -> None:
    """The power bound is read off both clauses, and 350 kW is the ceiling at every speed.

    C5.2.7 caps absolute ERS-K electrical DC power at 350 kW and C5.2.8 gives a speed-dependent
    propulsion limit. The stored curves are already ``min(C5.2.7, C5.2.8)``, so asking the model
    for the limit at 250 km/h and getting the interpolated 350 -> 100 segment rather than the
    1800 kW C5.2.8's own formula would allow is the property under test.

    The check is on the *delivered* torque rather than on the limit function, so a model that
    returned the request unclipped would fail here: the electrical power is re-derived from the
    torque and speed the step actually applied.

    **C5.18.5's ceiling is relaxed for this test on purpose.** With the committed 3.0 ratio the
    60 000 rpm part speed arrives at about 233 km/h, below every speed at which C5.2.8's curve is
    interesting, so with both caps in force the speed-dependent one could never be seen. Raising
    the ceiling isolates the power bound, and the ceiling gets its own test rather than being
    quietly dropped from the model.
    """
    efficiency = config.mgu_k_motor_inverter_efficiency
    power_only = replace(config, mgu_k_relative_speed_limit_rpm=1.0e9)
    for speed_km_h in (0.0, 150.0, 290.0, 300.0, 330.0, 344.0, 400.0):
        ice_rpm = _ice_rpm_for_km_h(power_only, speed_km_h)
        state = powertrain.mgu_k_initial_state(power_only)
        torque = powertrain.step_mgu_k(
            power_only,
            state,
            request_nm=10_000.0,
            ice_rpm=ice_rpm,
            speed_m_s=speed_km_h / 3.6,
        )
        crank_omega = abs(ice_rpm) * math.tau / 60.0
        dc_kw = torque * crank_omega / efficiency / 1_000.0
        allowed = powertrain.mgu_k_power_limit_kw(
            speed_km_h, config.ers_speed_km_h, config.ers_limit_kw
        )
        assert dc_kw <= allowed + 1e-9, (
            f"{dc_kw} kW over the {allowed} kW limit at {speed_km_h} km/h"
        )
        assert allowed <= config.mgu_k_peak_power_kw + 1e-9, "C5.2.7's cap is the ceiling"
        if speed_km_h >= 345.0:
            assert torque == 0.0, "C5.2.8 permits no ERS-K power above 345 km/h"
        elif speed_km_h == 300.0:
            assert dc_kw == pytest.approx(300.0, rel=1e-6), (
                "the effective limit between 290 and 340 km/h is the 350 -> 100 segment, so 300"
            )

    assert config.mgu_k_peak_power_kw == 350.0
    assert config.ers_limit_kw.max() == 350.0


def test_the_c52_11_torque_cap_is_the_500_nm_crank_referenced_limit_and_not_the_520_nm_threshold(
    config: KernelConfig,
) -> None:
    """C5.2.11's 500 Nm is applied at the crankshaft, and C5.18.4's 520 Nm is nowhere in the path.

    C5.2.11 states the limit on the MGU-K's mechanical torque *referenced to the crankshaft*. A
    motor shaft spinning three times faster must carry one third the torque to deliver the same
    power, and its crankshaft-equivalent torque is multiplied by the ratio. The returned value is
    that equivalent torque, so a request above the cap is cut back to exactly 500 Nm, which makes
    the test able to
    distinguish 500 from 520 rather than merely to detect a cap.

    C5.18.4's 520 Nm is the threshold above which an *optional* torque-limiting device may act. It
    is not a second cap and is deliberately not read by the model at all: ``KernelConfig`` does not
    carry it, so a 520 Nm cap could not be enforced even by accident.
    """
    ratio = config.mgu_k_crankshaft_ratio
    assert config.mgu_k_torque_limit_nm == 500.0
    assert "transient_torque_limiter_threshold_nm" not in config.__dataclass_fields__

    state = powertrain.mgu_k_initial_state(config)
    torque = powertrain.step_mgu_k(
        config,
        state,
        request_nm=10_000.0,
        ice_rpm=500.0,
        speed_m_s=50.0,
    )
    assert torque == pytest.approx(config.mgu_k_torque_limit_nm, rel=1e-12)

    # A request under the cap is untouched, so the cap is not a flat 500 Nm the model always
    # returns - the dead-clamp failure the clutch capacity was removed for.
    roomy = powertrain.mgu_k_initial_state(config)
    assert powertrain.step_mgu_k(
        config, roomy, request_nm=100.0, ice_rpm=500.0, speed_m_s=50.0
    ) == pytest.approx(100.0 * ratio)


def test_the_relative_speed_cap_stops_the_mgu_k_at_60000_rpm(config: KernelConfig) -> None:
    """C5.18.5's ceiling is on the part's speed *relative to* the crankshaft.

    The part spins at ``crankshaft_ratio * ice_rpm``, so the ceiling is a cap on that product and
    not on either factor. The configured 3.0 reaches 60 000 rpm at 20 000 rpm of crankshaft
    speed, above the 13 000 rpm limiter, so the committed powertrain can never hit it - which is
    why the test lowers the ceiling with ``replace`` instead of the model's own numbers.
    """
    ratio = config.mgu_k_crankshaft_ratio
    ceiling = config.mgu_k_relative_speed_limit_rpm
    assert ceiling == 60_000.0

    tight = replace(config, mgu_k_relative_speed_limit_rpm=3_000.0)
    assert powertrain.step_mgu_k(
        tight,
        powertrain.mgu_k_initial_state(tight),
        request_nm=100.0,
        ice_rpm=1_000.0,
        speed_m_s=50.0,
    ) == pytest.approx(100.0 * ratio), (
        "at exactly 3 * 1000 rpm the part is at the ceiling, not over it"
    )
    assert (
        powertrain.step_mgu_k(
            tight,
            powertrain.mgu_k_initial_state(tight),
            request_nm=100.0,
            ice_rpm=1_000.001,
            speed_m_s=50.0,
        )
        == 0.0
    ), "over the ceiling the MGU-K transmits nothing at all, in either direction"
    assert powertrain.mgu_k_rpm(2_000.0, ratio) == pytest.approx(2_000.0 * ratio)
    assert ceiling / ratio > config.rev_limit_rpm, (
        "with the committed ratio the ceiling is above the limiter, so P1 never reaches it and "
        "this cap exists for a calibration that does"
    )


def test_the_standing_start_rule_applies_to_a_declared_start_and_yields_to_the_standard_ecu(
    config: KernelConfig,
) -> None:
    """C5.2.12's 50 km/h block is scoped to a grid standing start, and its exception is a flag.

    Both halves of that sentence are the test. The rule is about a *declared* standing start, so
    the same 40 km/h with the flag off has to deploy normally - enforcing it universally would
    break every mid-corner deployment under 50 km/h. The exception is a fact about whether the FIA
    Standard ECU mandates minimum acceleration, which is not a constant any car file can carry, so
    it is a caller declaration rather than a number.

    Only *positive* torque is blocked. Regeneration below the threshold is what a car does to get
    to 50 km/h in the first place, and the article restricts deployment, not recovery.
    """
    assert config.launch_speed_kmh == 50.0
    below = 40.0 / 3.6
    at = config.launch_speed_kmh / 3.6

    def deploy(*, standing_start: bool, ecu: bool, speed_m_s: float) -> float:
        return powertrain.step_mgu_k(
            config,
            powertrain.mgu_k_initial_state(config),
            request_nm=200.0,
            ice_rpm=5_000.0,
            speed_m_s=speed_m_s,
            grid_standing_start=standing_start,
            ecu_mandates_minimum_acceleration=ecu,
        )

    assert deploy(standing_start=True, ecu=False, speed_m_s=below) == 0.0
    assert deploy(standing_start=True, ecu=True, speed_m_s=below) > 0.0
    assert deploy(standing_start=False, ecu=False, speed_m_s=below) > 0.0
    assert deploy(standing_start=True, ecu=False, speed_m_s=at) > 0.0, (
        "the article says once the car has reached 50 km/h, so 50 itself is allowed"
    )

    regen = powertrain.step_mgu_k(
        config,
        powertrain.mgu_k_initial_state(config, soc_mj=1.0),
        request_nm=-200.0,
        ice_rpm=5_000.0,
        speed_m_s=below,
        grid_standing_start=True,
    )
    assert regen < 0.0, "regeneration below 50 km/h is what gets the car there in the first place"

    assert deploy(standing_start=True, ecu=False, speed_m_s=-below) == 0.0, (
        "the rule is about reaching 50 km/h, so it holds in either direction of travel"
    )


def test_the_energy_store_never_leaves_the_four_mj_window(config: KernelConfig) -> None:
    """C5.2.9 bounds the state-of-charge *swing*, which is a window and not a capacity.

    Deploying more than the store holds and recovering more than the space left are both refused
    by clamping the torque rather than by letting the number go negative or past the window, so the
    invariant holds after any number of steps rather than only while the caller is careful. The
    window is 4 MJ, not the 7 MJ that ``PLAN.md`` section 6 carried: 7 MJ is C5.2.10's reduced
    per-lap *recharge* figure, a different limit.
    """
    assert config.store_energy_mj == 4.0
    ice_rpm, speed_m_s = 6_000.0, 60.0
    # One second of full deployment rather than the configured 100 us: at 350 kW the 4 MJ window is
    # about twelve seconds of energy, so a 10 kHz step would need 10^5 Python calls to spend it.
    # Every number the step does is per-step arithmetic on an interval, so a coarser step exercises
    # the same clamp - and it makes the test read as the physical question it is.
    dt_s = 1.0
    crank_omega = ice_rpm * math.tau / 60.0

    def deliver(state: np.ndarray, request_nm: float) -> float:
        return powertrain.step_mgu_k(config, state, request_nm, ice_rpm, speed_m_s, dt_s=dt_s)

    # Drain: ask for far more than the store holds and watch it land on empty.
    drained = powertrain.mgu_k_initial_state(config, soc_mj=1.0e-3)
    torque = deliver(drained, 1_000.0)
    assert torque > 0.0, "a depleted store still has its last 1 mJ to give"
    assert drained[powertrain.SOC_INDEX] == pytest.approx(0.0, abs=1e-15)
    assert 1.0e-3 - drained[powertrain.SOC_INDEX] == pytest.approx(
        torque * crank_omega * dt_s / config.mgu_k_motor_inverter_efficiency / 1.0e6, rel=1e-9
    ), "what leaves the store is exactly the energy the delivered torque carried"

    # Fill: recover into a full store and watch it stop at the window's top.
    filled = powertrain.mgu_k_initial_state(config, soc_mj=config.store_energy_mj)
    assert deliver(filled, -1_000.0) == 0.0
    assert filled[powertrain.SOC_INDEX] == config.store_energy_mj

    # And a request inside the window is met exactly, which is what makes the clamp a bound
    # rather than a blanket refusal.
    part = powertrain.mgu_k_initial_state(config, soc_mj=1.0)
    used = deliver(part, 1_000.0)
    assert 0.0 < used < 1_000.0
    assert part[powertrain.SOC_INDEX] == pytest.approx(
        1.0
        - used * crank_omega * dt_s / config.mgu_k_motor_inverter_efficiency / 1.0e6,
        rel=1e-9,
    )


def test_the_per_lap_recharge_limit_is_chosen_by_the_event_and_the_allowance_is_conditional(
    config: KernelConfig,
) -> None:
    """C5.2.10's three figures are three *events*, and the 0.5 MJ is a separate declaration.

    Reading 8.5 MJ as one standing cap is the error this tests against: the same 30 MJ of recovery
    has to fit in 8.5 MJ on a race, 7 MJ under the reduction, and 4 MJ in qualifying, with the
    conditional 0.5 MJ allowance added only when the caller says the event grants it. Every limit
    is checked as an upper bound on what the step actually stores, because a budget the model
    tracks but does not enforce is bookkeeping, not a limit.
    """
    assert config.recharge_limit_mj_per_lap == 8.5
    assert config.recharge_limit_reduced_mj_per_lap == 7.0
    assert config.recharge_limit_qualifying_floor_mj_per_lap == 4.0
    assert config.recharge_allowance_mj_per_lap == 0.5

    ice_rpm, speed_m_s = 6_000.0, 60.0
    # One second of step, for the reason the energy-window test gives: the per-lap budget is tens
    # of seconds of full recovery, so at the configured 100 us it would take 10^5 steps to fill.
    dt_s = 1.0

    def harvest_one_lap(event: RechargeEvent, allowance: bool) -> float:
        """Sweep the store up and down until a recovery step stores nothing more.

        Cycling is what makes the lap budget the binding limit rather than the store: a 4 MJ
        window cannot fill an 8.5 MJ lap in one go, so a model that only ever recovered would
        stop at 4 MJ and this would measure the store twice instead of the article.
        """
        state = powertrain.mgu_k_initial_state(config)
        for _ in range(4_000):
            deploying = state[powertrain.SOC_INDEX] > 0.5 * config.store_energy_mj
            torque = powertrain.step_mgu_k(
                config,
                state,
                1_000.0 if deploying else -1_000.0,
                ice_rpm,
                speed_m_s,
                dt_s,
                recharge_event=event,
                recharge_allowance_applies=allowance,
            )
            if not deploying and torque == 0.0:
                return state[powertrain.LAP_RECHARGE_INDEX]
        raise AssertionError("the lap budget never filled, so the limit is not being applied")

    assert harvest_one_lap(powertrain.RechargeEvent.RACE, False) == pytest.approx(8.5, rel=1e-9)
    assert harvest_one_lap(powertrain.RechargeEvent.REDUCED, False) == pytest.approx(7.0, rel=1e-9)
    assert harvest_one_lap(powertrain.RechargeEvent.QUALIFYING, False) == pytest.approx(
        4.0, rel=1e-9
    )
    assert harvest_one_lap(powertrain.RechargeEvent.QUALIFYING, True) == pytest.approx(
        4.5, rel=1e-9
    ), "the conditional allowance adds to whichever figure the event selects"

    # A new lap starts the budget again, which is the difference between a per-lap limit and a
    # whole-race one.
    state = powertrain.mgu_k_initial_state(config)
    for _ in range(4_000):
        deploying = state[powertrain.SOC_INDEX] > 0.5 * config.store_energy_mj
        torque = powertrain.step_mgu_k(
            config, state, 1_000.0 if deploying else -1_000.0, ice_rpm, speed_m_s, dt_s
        )
        if not deploying and torque == 0.0:
            break
    assert state[powertrain.LAP_RECHARGE_INDEX] == pytest.approx(8.5, rel=1e-9)
    powertrain.begin_lap(state)
    assert state[powertrain.LAP_RECHARGE_INDEX] == 0.0
    assert state[powertrain.SOC_INDEX] > 0.0, "a lap boundary is not a store reset"


def test_step_mgu_k_refuses_the_state_and_the_numbers_it_could_not_use(
    config: KernelConfig,
) -> None:
    """The MGU-K buffer is validated before a ``boundscheck=False`` step writes it.

    The compiled step advances the state of charge and the lap accumulator in place, so a short,
    read-only or wrongly typed buffer is an out-of-bounds write, and a NaN in a request or a speed
    is a torque that reads as a finished run. Both are refused with a message naming what was
    wrong rather than surfacing as a plausible number.
    """
    good = powertrain.mgu_k_initial_state(config)
    assert len(powertrain.mgu_k_initial_state(config)) == powertrain.MGU_K_STATE_SIZE
    # The two calls below deliberately pass a wrong *type*, which is the point: the boundary has to
    # refuse them at runtime rather than trust the annotation, and pyright is right to complain.
    anything: object = math.nan
    flag: object = 1

    frozen = powertrain.mgu_k_initial_state(config)
    frozen.flags.writeable = False
    short = powertrain.mgu_k_initial_state(config)[:-1]
    for broken in (frozen, short, powertrain.mgu_k_initial_state(config).astype(np.int64)):
        with pytest.raises(ValueError, match="state"):
            powertrain.step_mgu_k(config, broken, request_nm=100.0, ice_rpm=5_000.0, speed_m_s=50.0)

    with pytest.raises(ValueError, match="request_nm"):
        powertrain.step_mgu_k(config, good, anything, 5_000.0, 50.0)
    with pytest.raises(ValueError, match="speed_m_s"):
        powertrain.step_mgu_k(config, good, 100.0, 5_000.0, math.inf)
    with pytest.raises(ValueError, match="crankshaft_ratio"):
        powertrain.step_mgu_k(
            replace(config, mgu_k_crankshaft_ratio=0.0), good, 100.0, 5_000.0, 50.0
        )
    with pytest.raises(ValueError, match="motor_inverter_efficiency"):
        powertrain.step_mgu_k(
            replace(config, mgu_k_motor_inverter_efficiency=0.0), good, 100.0, 5_000.0, 50.0
        )
    with pytest.raises(ValueError, match="dt_s"):
        powertrain.step_mgu_k(config, good, 100.0, 5_000.0, 50.0, dt_s=0.0)
    with pytest.raises(ValueError, match="grid_standing_start"):
        powertrain.step_mgu_k(
            config,
            good,
            100.0,
            5_000.0,
            50.0,
            grid_standing_start=flag,  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValueError, match="recharge_event"):
        powertrain.step_mgu_k(config, good, 100.0, 5_000.0, 50.0, recharge_event=9)

    # A malformed C5.2.8 curve is refused rather than looked up with `boundscheck=False`, and each
    # profile is checked against its own partner - the two have different lengths by design, so a
    # single shared bound would read past the end of the three-breakpoint Overtake curve.
    malformed = (
        (replace(config, ers_speed_km_h=config.ers_speed_km_h[:-1]), "ers_speed_km_h"),
        (replace(config, ers_limit_kw=np.empty(0, dtype=np.float64)), "ers_limit_kw"),
        (replace(config, ers_speed_km_h=config.ers_speed_km_h.tolist()), "ers_speed_km_h"),
        (
            replace(config, ers_overtake_speed_km_h=config.ers_overtake_speed_km_h[::-1].copy()),
            "ers_overtake_speed_km_h",
        ),
    )
    for broken_curve, matched in malformed:
        with pytest.raises(ValueError, match=matched):
            powertrain.step_mgu_k(
                broken_curve,
                good,
                100.0,
                5_000.0,
                50.0,
                overtake=matched.startswith("ers_overtake"),
            )


def test_the_mgu_k_joins_the_crankshaft_upstream_of_the_gearbox_and_the_clutch(
    config: KernelConfig,
) -> None:
    """The motor's torque arrives at the rear axle through the same reduction the engine's does.

    C5.18.2 permanently gears the MGU-K to the crankshaft at a fixed ratio, so this is the model's
    inferred boundary *before* the clutch and gearbox rather than a second path into the
    differential. The composed driveline torque is therefore
    ``(throttle * ice + mgu_k) * ratio * final_drive`` limited by the same C9.2.5 clutch demand,
    which is what makes the ratio of the two contributions at the axle equal to their ratio at the
    crankshaft.

    The throttle is the engine's alone. The MGU-K is its own actuator with its own request, and
    scaling it by the ICE pedal would make deployment a function of a control the driver does not
    have for it.
    """
    ice_rpm = 8_000.0
    ice = powertrain.step_ice_torque(config, ice_rpm)
    gear_state = gearbox.initial_state(gear=3.0, clutch_engagement=1.0)
    mgu_state = powertrain.mgu_k_initial_state(config)
    ratio = float(config.gear_ratios[2]) * config.final_drive

    # 10 Nm at the MGU-K shaft is inside every cap and keeps the combined torque below the clutch
    # demand: the returned crankshaft-equivalent contribution is 30 Nm.
    assert powertrain.step_mgu_k(
        config, mgu_state, request_nm=10.0, ice_rpm=ice_rpm, speed_m_s=70.0
    ) == pytest.approx(10.0 * config.mgu_k_crankshaft_ratio)

    driven = gearbox.step_drivetrain(
        config,
        gear_state,
        mgu_state,
        ice_rpm=ice_rpm,
        throttle=1.0,
        speed_m_s=70.0,
        mgu_k_request_nm=10.0,
    )
    expected = (ice + 10.0 * config.mgu_k_crankshaft_ratio) * ratio
    assert driven == pytest.approx(expected, rel=1e-12)

    # Half the ICE pedal scales the engine's half of the sum and leaves the motor's alone.
    part = gearbox.step_drivetrain(
        config,
        gear_state,
        mgu_state,
        ice_rpm=ice_rpm,
        throttle=0.5,
        speed_m_s=70.0,
        mgu_k_request_nm=10.0,
    )
    assert part == pytest.approx(
        (0.5 * ice + 10.0 * config.mgu_k_crankshaft_ratio) * ratio, rel=1e-12
    )

    # The same C9.2.5 clutch demand is the ceiling on both contributions together, which is the
    # property that makes the motor upstream rather than beside the clutch.
    demand = gearbox.clutch_demand_nm(
        0.5, config.clutch_demand_torque_nm, config.clutch_demand_travel_fraction
    )
    clamped = gearbox.step_drivetrain(
        config,
        gearbox.initial_state(gear=3.0, clutch_engagement=0.5),
        mgu_state,
        ice_rpm=ice_rpm,
        throttle=1.0,
        speed_m_s=70.0,
        mgu_k_request_nm=100.0,
    )
    assert demand < expected, "this test needs the demand to be the binding ceiling"
    assert clamped == pytest.approx(demand, rel=1e-12)


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
