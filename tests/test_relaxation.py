"""P2-T5: tyre relaxation - transient slip states that chase their targets.

``PHASES.md`` P2-T5 and the plan's Task 4 ask for caller-owned
longitudinal slip-ratio and lateral slip-angle state per wheel, relaxed
toward the steady targets with the configured relaxation lengths, the
actual patch speed, and a finite low-speed floor, integrated on the fixed
100 us step. The claims are checked here in the form that would actually
fail:

* **The exact update, once.** One step is ``s += (target - s) *
  (1 - exp(-v_eff dt / L))`` with ``v_eff = max(|v_patch|, v_min)``,
  compared for tight rather than bit equality against the same arithmetic
  in plain Python: ``atan``/``exp`` come from the platform's libm and this
  suite runs on both Windows and Linux.
* **Fraction of gap closed.** One step closes exactly ``1 - exp(-v dt /
  L)`` of the remaining gap, so a faster patch relaxes faster at the same
  configured length, and the state never overshoots its target.
* **Low-speed floor.** A stopped patch relaxes with ``v_min``, not zero
  or a division failure, and a negative patch speed is its magnitude.
* **Determinism and caller ownership.** Same inputs give the same bits,
  and the state arrays are the caller's - updated in place, never
  allocated inside the step.
* **Zero load is someone else's problem.** No function here reads a
  vertical load: a wheel in the air keeps relaxing its caller-owned slip
  state, and the load guard that zeroes its force lives in the tyre
  primitives. Relaxing slip while unloaded is not a failure.
* **Invalid input.** Nonfinite targets, a nonpositive step, a wrong-shaped
  or wrong-dtype state vector, and a configuration that cannot describe a
  relaxation - a zero length, a zero floor - are refused before any
  arithmetic happens.

Every physical number comes from the loaded ``car_spec.yaml``, so this
file contains no tuned constant. The ``forces`` marker keeps it with the
other physics-core tests.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from f1telemetry.contracts.car_spec import CarSpec

# The layer-isolation rule bans importing the physics core from layers 4-6
# (PLAN.md section 3). This file is the boundary test that proves the core
# itself, so the import is carved out the same way the kernel's is.
from f1telemetry.physics import relaxation  # noqa: TID251 -- the relaxation tests test the core

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

pytestmark = pytest.mark.forces

# Read off the dispatchers rather than copied out of the decorator text: a
# copy in a test can only agree with the decorator, which is the thing that
# needed checking. `cache` is absent on purpose - numba consumes it at
# decoration time and does not report it back on `targetoptions`.
EXPECTED_OPTIONS: dict[str, Any] = {
    "fastmath": False,
    "nopython": True,
    "nogil": True,
    "boundscheck": False,
    "error_model": "numpy",
}

# The fixed integrator step. Named here rather than repeated, and small
# enough that one step is genuinely transient.
DT_S = 100e-6


@pytest.fixture(scope="module")
def config(spec: CarSpec) -> KernelConfig:
    """The validated, flat configuration the relaxation model is allowed to see."""
    return spec.kernel_config()


def _reference_step(
    current: float,
    target: float,
    patch_speed_m_s: float,
    relaxation_length_m: float,
    dt_s: float,
    min_speed_m_s: float,
) -> float:
    """The documented one-step update in plain Python.

    A transcription of the module's documented arithmetic, not a captured
    output: a reference that copied a run's numbers could only prove the
    code still does what it did.
    """
    speed = max(abs(patch_speed_m_s), min_speed_m_s)
    return target + (current - target) * math.exp(-speed * dt_s / relaxation_length_m)


def _state_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Caller-owned FL/FR/RL/RR buffers with distinct values per corner."""
    kappa_state = np.array([0.05, -0.10, 0.20, 0.00], dtype=np.float64)
    kappa_target = np.array([0.15, 0.05, -0.30, 0.10], dtype=np.float64)
    alpha_state = np.array([3.0, -2.0, 5.0, 0.5], dtype=np.float64)
    alpha_target = np.array([-1.0, 4.0, 2.0, -3.0], dtype=np.float64)
    patch_speed = np.array([25.0, 24.0, 22.0, 0.0], dtype=np.float64)
    return kappa_state, kappa_target, alpha_state, alpha_target, patch_speed


def test_the_relaxation_primitives_are_compiled_kernels_with_the_project_options(
    config: KernelConfig,
) -> None:
    """A Python fallback here would be a silent step backwards, so the dispatchers are asked."""
    relaxation.relax_slip_ratio(
        0.05,
        0.10,
        20.0,
        config.relaxation_length_longitudinal_m,
        DT_S,
        config.relaxation_min_speed_m_s,
    )
    relaxation.decay_factor(
        20.0, config.relaxation_length_longitudinal_m, DT_S, config.relaxation_min_speed_m_s
    )
    kappa_state, kappa_target, alpha_state, alpha_target, patch_speed = _state_pair()
    relaxation.relax_slip_state(
        kappa_state,
        kappa_target,
        alpha_state,
        alpha_target,
        patch_speed,
        config.relaxation_length_longitudinal_m,
        config.relaxation_length_lateral_m,
        DT_S,
        config.relaxation_min_speed_m_s,
    )
    for function in (
        relaxation.decay_factor,
        relaxation.relax_slip_ratio,
        relaxation.relax_slip_state,
    ):
        name = function.py_func.__name__
        assert len(function.signatures) >= 1, f"{name} never compiled a signature"
        options = dict(function.targetoptions)
        for option, expected in EXPECTED_OPTIONS.items():
            assert options[option] == expected, (name, option)


def test_one_step_is_the_exact_exponential_update(config: KernelConfig) -> None:
    """``s += (target - s)(1 - exp(-v_eff dt / L))``, term for term."""
    for patch_speed in (0.0, 0.4, 18.0, -18.0, 120.0):
        for current, target in ((0.0, 0.2), (0.4, 0.1), (-0.3, 0.6)):
            expected = _reference_step(
                current,
                target,
                patch_speed,
                config.relaxation_length_longitudinal_m,
                DT_S,
                config.relaxation_min_speed_m_s,
            )
            actual = relaxation.relax_slip_ratio(
                current,
                target,
                patch_speed,
                config.relaxation_length_longitudinal_m,
                DT_S,
                config.relaxation_min_speed_m_s,
            )
            # Tight, not bit: exp is the platform libm on both operating systems.
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)
            # And the two axis lengths give genuinely different numbers when used.
            other = relaxation.relax_slip_ratio(
                current,
                target,
                patch_speed,
                config.relaxation_length_lateral_m,
                DT_S,
                config.relaxation_min_speed_m_s,
            )
            if (
                patch_speed != 0.0
                or config.relaxation_length_longitudinal_m != config.relaxation_length_lateral_m
            ):
                assert other == pytest.approx(
                    _reference_step(
                        current,
                        target,
                        patch_speed,
                        config.relaxation_length_lateral_m,
                        DT_S,
                        config.relaxation_min_speed_m_s,
                    ),
                    rel=1e-12,
                    abs=1e-15,
                )


def test_one_step_closes_the_documented_fraction_of_the_gap(config: KernelConfig) -> None:
    """The closed fraction is ``1 - exp(-v_eff dt / L)``, faster patchs faster."""
    current = 0.0
    target = 1.0
    for patch_speed in (2.0, 10.0, 60.0):
        actual = relaxation.relax_slip_ratio(
            current,
            target,
            patch_speed,
            config.relaxation_length_longitudinal_m,
            DT_S,
            config.relaxation_min_speed_m_s,
        )
        expected_fraction = 1.0 - math.exp(
            -patch_speed * DT_S / config.relaxation_length_longitudinal_m
        )
        assert actual == pytest.approx(expected_fraction, rel=1e-12, abs=1e-15)
    # Faster patch speed, same length, same step: strictly more of the gap closed.
    slow = relaxation.relax_slip_ratio(
        current,
        target,
        2.0,
        config.relaxation_length_longitudinal_m,
        DT_S,
        config.relaxation_min_speed_m_s,
    )
    fast = relaxation.relax_slip_ratio(
        current,
        target,
        60.0,
        config.relaxation_length_longitudinal_m,
        DT_S,
        config.relaxation_min_speed_m_s,
    )
    assert 0.0 < slow < fast < 1.0


def test_the_low_speed_floor_keeps_a_stopped_patch_finite(config: KernelConfig) -> None:
    """At zero or small patch speed the floor, not zero or a divide, sets the lag."""
    stopped = relaxation.decay_factor(
        0.0, config.relaxation_length_longitudinal_m, DT_S, config.relaxation_min_speed_m_s
    )
    floored = math.exp(
        -config.relaxation_min_speed_m_s * DT_S / config.relaxation_length_longitudinal_m
    )
    assert stopped == pytest.approx(floored, rel=1e-12, abs=1e-15)
    assert 0.0 < stopped < 1.0
    # Below the floor the decay is pinned to the floor; above it the magnitude wins.
    below = relaxation.decay_factor(
        0.3 * config.relaxation_min_speed_m_s,
        config.relaxation_length_longitudinal_m,
        DT_S,
        config.relaxation_min_speed_m_s,
    )
    assert below == pytest.approx(stopped, rel=0.0, abs=0.0)
    negative = relaxation.decay_factor(
        -20.0, config.relaxation_length_longitudinal_m, DT_S, config.relaxation_min_speed_m_s
    )
    positive = relaxation.decay_factor(
        20.0, config.relaxation_length_longitudinal_m, DT_S, config.relaxation_min_speed_m_s
    )
    assert negative == pytest.approx(positive, rel=0.0, abs=0.0)


def test_the_state_converges_to_a_constant_target_without_overshoot(config: KernelConfig) -> None:
    """First-order lag property: monotone approach, no overshoot, no oscillation."""
    state = -0.5
    target = 0.3
    previous = state
    for _ in range(200_000):  # 20 s at 100 us
        state = relaxation.relax_slip_ratio(
            state,
            target,
            30.0,
            config.relaxation_length_longitudinal_m,
            DT_S,
            config.relaxation_min_speed_m_s,
        )
        assert abs(state - target) <= abs(previous - target) + 1e-300
        previous = state
    # tau = L / v = 0.3 / 30 s = 10 ms; 20 s is 2000 taus, so the lag is exact to machine.
    assert state == pytest.approx(target, abs=1e-12)


def test_step_relaxation_updates_caller_owned_state_in_place(config: KernelConfig) -> None:
    """The four-wheel step mutates the caller's arrays and allocates nothing visible."""
    kappa_state, kappa_target, alpha_state, alpha_target, patch_speed = _state_pair()
    expected_kappa = kappa_state.copy()
    expected_alpha = alpha_state.copy()
    for wheel in range(4):
        expected_kappa[wheel] = _reference_step(
            expected_kappa[wheel],
            kappa_target[wheel],
            patch_speed[wheel],
            config.relaxation_length_longitudinal_m,
            DT_S,
            config.relaxation_min_speed_m_s,
        )
        expected_alpha[wheel] = _reference_step(
            expected_alpha[wheel],
            alpha_target[wheel],
            patch_speed[wheel],
            config.relaxation_length_lateral_m,
            DT_S,
            config.relaxation_min_speed_m_s,
        )
    before_kappa = kappa_state.copy()
    before_alpha = alpha_state.copy()
    relaxation.step_relaxation(
        config,
        kappa_state,
        kappa_target,
        alpha_state,
        alpha_target,
        patch_speed,
        DT_S,
    )
    np.testing.assert_allclose(kappa_state, expected_kappa, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(alpha_state, expected_alpha, rtol=1e-12, atol=1e-15)
    # The arrays are the same objects - the step owns no state of its own.
    assert not np.array_equal(kappa_state, before_kappa)
    assert not np.array_equal(alpha_state, before_alpha)


def test_relaxing_caller_owned_state_through_the_njit_vector_step_matches(
    config: KernelConfig,
) -> None:
    """The compiled vector step and the plain step give the same values."""
    kappa_state, kappa_target, alpha_state, alpha_target, patch_speed = _state_pair()
    relaxation.relax_slip_state(
        kappa_state,
        kappa_target,
        alpha_state,
        alpha_target,
        patch_speed,
        config.relaxation_length_longitudinal_m,
        config.relaxation_length_lateral_m,
        DT_S,
        config.relaxation_min_speed_m_s,
    )
    fresh_kappa, fresh_target, fresh_alpha, fresh_alpha_target, fresh_speed = _state_pair()
    relaxation.step_relaxation(
        config,
        fresh_kappa,
        fresh_target,
        fresh_alpha,
        fresh_alpha_target,
        fresh_speed,
        DT_S,
    )
    np.testing.assert_allclose(kappa_state, fresh_kappa, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(alpha_state, fresh_alpha, rtol=0.0, atol=0.0)


def test_determinism_holds_bit_for_bit(config: KernelConfig) -> None:
    """Same inputs, same bits: a relaxed run is replayable."""
    first_kappa, first_target, first_alpha, first_alpha_target, first_speed = _state_pair()
    second_kappa, second_target, second_alpha, second_alpha_target, second_speed = _state_pair()
    relaxation.step_relaxation(
        config, first_kappa, first_target, first_alpha, first_alpha_target, first_speed, DT_S
    )
    relaxation.step_relaxation(
        config, second_kappa, second_target, second_alpha, second_alpha_target, second_speed, DT_S
    )
    assert np.array_equal(first_kappa, second_kappa)
    assert np.array_equal(first_alpha, second_alpha)


def test_each_wheel_relaxes_at_its_own_patch_speed(config: KernelConfig) -> None:
    """The inside wheel, slower through the corner, lags the outside wheel."""
    alpha_state = np.zeros(4, dtype=np.float64)
    alpha_target = np.full(4, 0.1, dtype=np.float64)
    patch_speed = np.array([30.0, 10.0, 30.0, 10.0], dtype=np.float64)
    kappa_state = np.zeros(4, dtype=np.float64)
    kappa_target = np.zeros(4, dtype=np.float64)
    relaxation.step_relaxation(
        config, kappa_state, kappa_target, alpha_state, alpha_target, patch_speed, DT_S
    )
    fast = alpha_state[0]
    slow = alpha_state[1]
    assert 0.0 < slow < fast


def test_a_zero_step_leaves_the_state_untouched(config: KernelConfig) -> None:
    """``dt -> 0`` closes no fraction of the gap."""
    current = 0.25
    target = -0.15
    actual = relaxation.relax_slip_ratio(
        current,
        target,
        30.0,
        config.relaxation_length_longitudinal_m,
        0.0,
        config.relaxation_min_speed_m_s,
    )
    assert actual == current


def test_invalid_configuration_is_refused(config: KernelConfig) -> None:
    """A relaxation length or floor of zero makes the lag undefined or infinite."""
    kappa_state, kappa_target, alpha_state, alpha_target, patch_speed = _state_pair()
    for broken in (
        replace(config, relaxation_length_longitudinal_m=0.0),
        replace(config, relaxation_length_lateral_m=-0.5),
        replace(config, relaxation_min_speed_m_s=0.0),
        replace(config, relaxation_min_speed_m_s=math.nan),
    ):
        with pytest.raises(ValueError, match="relaxation"):
            relaxation.step_relaxation(
                broken,
                kappa_state.copy(),
                kappa_target.copy(),
                alpha_state.copy(),
                alpha_target.copy(),
                patch_speed.copy(),
                DT_S,
            )


def test_invalid_state_and_step_are_refused(config: KernelConfig) -> None:
    """Nonfinite inputs and a nonpositive step fail before they reach Numba."""
    kappa_state, kappa_target, alpha_state, alpha_target, patch_speed = _state_pair()
    with pytest.raises(ValueError, match="dt_s"):
        relaxation.step_relaxation(
            config,
            kappa_state.copy(),
            kappa_target.copy(),
            alpha_state.copy(),
            alpha_target.copy(),
            patch_speed.copy(),
            -DT_S,
        )
    with pytest.raises(ValueError, match="dt_s"):
        relaxation.step_relaxation(
            config,
            kappa_state.copy(),
            kappa_target.copy(),
            alpha_state.copy(),
            alpha_target.copy(),
            patch_speed.copy(),
            math.nan,
        )
    bad_target = kappa_target.copy()
    bad_target[2] = math.inf
    with pytest.raises(ValueError, match="target"):
        relaxation.step_relaxation(
            config,
            kappa_state.copy(),
            bad_target,
            alpha_state.copy(),
            alpha_target.copy(),
            patch_speed.copy(),
            DT_S,
        )
    with pytest.raises(ValueError, match="float64"):
        relaxation.step_relaxation(
            config,
            kappa_state.astype(np.float32),
            kappa_target.copy(),
            alpha_state.copy(),
            alpha_target.copy(),
            patch_speed.copy(),
            DT_S,
        )
    with pytest.raises(ValueError, match="float64"):
        relaxation.step_relaxation(
            config,
            kappa_state[:2].copy(),
            kappa_target.copy(),
            alpha_state.copy(),
            alpha_target.copy(),
            patch_speed.copy(),
            DT_S,
        )


def test_unloaded_state_still_relaxes_because_load_is_not_this_module(config: KernelConfig) -> None:
    """Zero load zeroes the tyre force elsewhere; the slip state keeps tracking."""
    alpha_state = np.zeros(4, dtype=np.float64)
    alpha_target = np.full(4, 0.05, dtype=np.float64)
    kappa_state = np.zeros(4, dtype=np.float64)
    kappa_target = np.zeros(4, dtype=np.float64)
    # Wheel 3 is a parked, unloaded corner: patch speed zero, target finite.
    patch_speed = np.array([20.0, 20.0, 20.0, 0.0], dtype=np.float64)
    relaxation.step_relaxation(
        config, kappa_state, kappa_target, alpha_state, alpha_target, patch_speed, DT_S
    )
    expected = _reference_step(
        0.0, 0.05, 0.0, config.relaxation_length_lateral_m, DT_S, config.relaxation_min_speed_m_s
    )
    assert alpha_state[3] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert math.isfinite(alpha_state[3])
