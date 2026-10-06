"""P3 thermal primitives: lumped heat flow, tyre/brake heat sources, tyre pressure.

This module is the pure numeric layer of the P3 thermal model. It answers four
questions with scalar/numpy functions and no state of its own:

* **Lumped heat flow.** One node per thermal mass (engine, gearbox, each tyre
  corner, each brake corner): ``m·c·dT/dt = Q_in − Q_conv − Q_rad``, integrated
  with an explicit Euler step over the caller's ``dt``.
* **Tyre heat from the contact patch.** ``Q = |F_patch · v_slip|`` — heat comes
  from patch force dotted with slip velocity, *not* from road speed. A tyre
  rolling at zero slip, or a wheel in the air at zero normal load, produces
  exactly zero heat; the same function gives zero mechanically, without a
  special case.
* **Brake heat.** ``Q = f_brake · τ_brake · ω_wheel`` — brake torque times
  wheel angular speed, scaled by an explicit, declared heat fraction.
* **Tyre pressure.** Ideal gas: ``p = m_gas · R_gas · T_K / V`` with a
  fixed/configured volume and a declared gas-mass state. A bounded leak
  reduces the mass (and hence the pressure) directly; it is *not* coupled
  into the thermal solver.

**Units and sign conventions.** Temperatures are stored and reported in
degrees Celsius everywhere in this module; Kelvin appears only inside the
radiation term and the ideal-gas law, via an explicit ``+ 273.15`` conversion.
Powers are watts, torques N·m, angular speeds rad/s, areas m², heat-transfer
coefficients W/(m²·K), masses kg, volumes m³, heat capacities J/K, pressures
Pa, and gas constants J/(kg·K). Convective and radiative flows are *signed
losses*: positive when the node is hotter than its ambient, negative when the
ambient is hotter, so ``Q_in − Q_conv − Q_rad`` is correct for both heating
and cooling.

**All runtime inputs are synthetic.** Heat capacities, heat fractions,
cooling areas/coefficients, tyre volumes, gas constants and initial
conditions live in ``car_spec.yaml`` / calibration data as labelled
placeholders; this module treats them as data to be calibrated, not numbers
with authority. Nothing here reads a clock, a seed, or a regulation.

**Validation shape.** Like the rest of :mod:`f1telemetry.physics`, every
entry point narrows its inputs and refuses non-finite values and non-positive
divisors *before* any arithmetic, so a NaN can never reach an integrator.
Functions accept Python scalars or numpy arrays; scalars come back as Python
floats, arrays as numpy arrays of the same shape. A lumped step that would
drive a node at or below absolute zero raises rather than clamping: that is a
physically declared boundary, and silently clamping it would hide an
insane energy imbalance in the caller's inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "ABSOLUTE_ZERO_C",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "ThermalTrace",
    "brake_heat_w",
    "convective_heat_flow_w",
    "lumped_temperature_step_c",
    "radiative_heat_flow_w",
    "simulate_thermal_trace",
    "tyre_gas_mass_step_kg",
    "tyre_pressure_pa",
    "tyre_slip_heat_w",
]

_ABSOLUTE_ZERO_OFFSET: Final[float] = 273.15

#: Celsius value of absolute zero; every temperature must stay above this.
ABSOLUTE_ZERO_C: Final[float] = -_ABSOLUTE_ZERO_OFFSET

#: Stefan–Boltzmann constant, W/(m²·K⁴). A physical constant, not a calibration.
STEFAN_BOLTZMANN_W_M2_K4: Final[float] = 5.670374419e-8


@dataclass(frozen=True, slots=True)
class ThermalTrace:
    """Computed ideal thermal channels, before the sensor pipeline.

    All parameters in :func:`simulate_thermal_trace` are provisional synthetic
    placeholders. The temperatures are scenario outputs, not calibrated car data.
    """

    tyre_temp_c: np.ndarray
    brake_temp_c: np.ndarray
    tyre_pressure_psi: np.ndarray
    engine_temp_c: np.ndarray
    gearbox_temp_c: np.ndarray


def simulate_thermal_trace(
    *,
    dt_s: float,
    speed_m_s: np.ndarray,
    tyre_slip_work_j: np.ndarray,
    brake_work_j: np.ndarray,
    engine_heat_j: np.ndarray,
    gearbox_heat_j: np.ndarray,
    leak_rate_kg_s: float = 0.0,
    ambient_temp_c: float = 25.0,
) -> ThermalTrace:
    """Advance synthetic lumped nodes from interval energy inputs.

    Inputs carry energy accumulated over each record interval. Heating and
    airflow cooling are therefore integrated at the scenario's 100 Hz boundary.
    The fixed parameters and initial states below are calibration placeholders;
    no thermal value in this trace claims a measurement or FIA coefficient.
    """
    speed = _finite_nonnegative("speed_m_s", speed_m_s)
    slip = _finite_nonnegative("tyre_slip_work_j", tyre_slip_work_j)
    brake = _finite_nonnegative("brake_work_j", brake_work_j)
    engine = _finite_nonnegative("engine_heat_j", engine_heat_j)
    gearbox = _finite_nonnegative("gearbox_heat_j", gearbox_heat_j)
    if slip.ndim != 2 or slip.shape[1] != 4 or brake.shape != slip.shape:
        raise ValueError("tyre_slip_work_j and brake_work_j must have shape (steps, 4)")
    count = slip.shape[0]
    if speed.shape != (count,) or engine.shape != (count,) or gearbox.shape != (count,):
        raise ValueError("speed_m_s, engine_heat_j, and gearbox_heat_j must match step count")
    dt = float(_finite("dt_s", dt_s))
    if dt <= 0.0:
        raise ValueError("dt_s must be > 0")
    leak = float(_finite_nonnegative("leak_rate_kg_s", leak_rate_kg_s))

    ambient = float(_celsius_to_kelvin("ambient_temp_c", ambient_temp_c) - _ABSOLUTE_ZERO_OFFSET)
    tyre_t = np.full(4, 25.0)
    brake_t = np.full(4, 25.0)
    engine_t = 90.0
    gearbox_t = 65.0
    tyre_capacity, brake_capacity = 9_000.0, 8_000.0
    engine_capacity, gearbox_capacity = 450_000.0, 65_000.0
    tyre_area, brake_area, engine_area, gearbox_area = 0.30, 0.10, 2.5, 0.6
    tyre_eps, brake_eps, engine_eps, gearbox_eps = 0.8, 0.8, 0.8, 0.8
    tyre_volume, air_r = 0.030, 287.05
    # Seed the gas mass to 22 psi gauge at the declared initial temperature.
    initial_gas_mass = (101_325.0 + 22.0 * 6_894.757) * tyre_volume / (air_r * 298.15)
    tyre_mass = np.full(4, initial_gas_mass)
    tyre_out = np.empty((count, 4))
    brake_out = np.empty((count, 4))
    pressure_out = np.empty((count, 4))
    engine_out = np.empty(count)
    gearbox_out = np.empty(count)
    for i in range(count):
        speed_i = float(speed[i])
        tyre_h, brake_h = 55.0 + 3.0 * speed_i, 90.0 + 2.0 * speed_i
        engine_h, gearbox_h = 90.0 + 6.0 * speed_i, 45.0 + 2.0 * speed_i
        tyre_q, brake_q = slip[i] / dt, brake[i] * 0.75 / dt
        tyre_t = lumped_temperature_step_c(
            tyre_t,
            tyre_capacity,
            tyre_q,
            convective_heat_flow_w(tyre_h, tyre_area, tyre_t, ambient),
            radiative_heat_flow_w(tyre_eps, tyre_area, tyre_t, ambient),
            dt,
        )
        brake_t = lumped_temperature_step_c(
            brake_t,
            brake_capacity,
            brake_q,
            convective_heat_flow_w(brake_h, brake_area, brake_t, ambient),
            radiative_heat_flow_w(brake_eps, brake_area, brake_t, ambient),
            dt,
        )
        engine_t = float(
            lumped_temperature_step_c(
                engine_t,
                engine_capacity,
                engine[i] / dt,
                convective_heat_flow_w(engine_h, engine_area, engine_t, ambient),
                radiative_heat_flow_w(engine_eps, engine_area, engine_t, ambient),
                dt,
            )
        )
        gearbox_t = float(
            lumped_temperature_step_c(
                gearbox_t,
                gearbox_capacity,
                gearbox[i] / dt,
                convective_heat_flow_w(gearbox_h, gearbox_area, gearbox_t, ambient),
                radiative_heat_flow_w(gearbox_eps, gearbox_area, gearbox_t, ambient),
                dt,
            )
        )
        tyre_mass = tyre_gas_mass_step_kg(tyre_mass, leak, dt)
        tyre_out[i] = tyre_t
        brake_out[i] = brake_t
        pressure_out[i] = (
            np.asarray(tyre_pressure_pa(tyre_mass, tyre_volume, tyre_t, air_r)) - 101_325.0
        ) / 6_894.757
        engine_out[i], gearbox_out[i] = engine_t, gearbox_t
    return ThermalTrace(tyre_out, brake_out, pressure_out, engine_out, gearbox_out)


def convective_heat_flow_w(
    heat_transfer_coefficient_w_m2_k: float | np.ndarray,
    area_m2: float | np.ndarray,
    body_temp_c: float | np.ndarray,
    ambient_temp_c: float | np.ndarray,
) -> float | np.ndarray:
    """Newton cooling loss, ``h·A·(T_body − T_ambient)`` in watts.

    Signed: positive when the node is hotter than the ambient (heat leaves),
    negative when the ambient is hotter (heat arrives). Zero-speed cases have
    zero airflow-dependent contribution only if the caller scales
    ``heat_transfer_coefficient_w_m2_k`` (or the area) to the airflow's value;
    this function applies whatever coefficient it is given. ``h`` and ``A``
    must be finite and non-negative; a zero area or zero ``h`` yields
    exactly zero. Both temperatures must be finite and above absolute zero.
    """
    h = _finite_nonnegative("heat_transfer_coefficient_w_m2_k", heat_transfer_coefficient_w_m2_k)
    area = _finite_nonnegative("area_m2", area_m2)
    body_k = _celsius_to_kelvin("body_temp_c", body_temp_c)
    ambient_k = _celsius_to_kelvin("ambient_temp_c", ambient_temp_c)
    return _maybe_scalar(h * area * (body_k - ambient_k))


def radiative_heat_flow_w(
    emissivity: float | np.ndarray,
    area_m2: float | np.ndarray,
    body_temp_c: float | np.ndarray,
    ambient_temp_c: float | np.ndarray,
) -> float | np.ndarray:
    """Net blackbody radiation loss, ``ε·σ·A·(T_body⁴ − T_ambient⁴)`` in watts.

    The one place Kelvin is mandatory: temperatures are converted from the
    stored Celsius state with an explicit ``+ 273.15`` before the fourth power.
    Signed like :func:`convective_heat_flow_w`. ``emissivity`` must lie in
    ``[0, 1]`` (a surface cannot emit more than a blackbody), ``area`` finite
    and non-negative, both temperatures finite and above absolute zero.
    """
    eps = _finite_nonnegative("emissivity", emissivity)
    if np.any(np.asarray(eps) > 1.0):
        raise ValueError(f"emissivity must be <= 1, got {emissivity!r}")
    area = _finite_nonnegative("area_m2", area_m2)
    body_k = _celsius_to_kelvin("body_temp_c", body_temp_c)
    ambient_k = _celsius_to_kelvin("ambient_temp_c", ambient_temp_c)
    return _maybe_scalar(eps * STEFAN_BOLTZMANN_W_M2_K4 * area * (body_k**4 - ambient_k**4))


def tyre_slip_heat_w(
    patch_force_n: float | np.ndarray,
    slip_velocity_m_s: float | np.ndarray,
) -> float | np.ndarray:
    """Tyre heating power, ``|F_patch · v_slip|`` in watts.

    The scalar form of the patch-force dot slip-velocity contact power. It is
    the magnitude of mechanical power dissipated at the contact patch, so it
    is non-negative by construction and identically zero when either the patch
    force or the slip velocity is zero — a free-rolling wheel (zero slip) or a
    wheel off the ground (zero force) makes no heat, and no road-speed term
    exists to invent any. Both inputs must be finite; neither is sign
    restricted, because slip force and slip velocity carry signs and only
    their product's magnitude heats the tyre.
    """
    force = _finite("patch_force_n", patch_force_n)
    velocity = _finite("slip_velocity_m_s", slip_velocity_m_s)
    return _maybe_scalar(np.abs(np.asarray(force) * np.asarray(velocity)))


def brake_heat_w(
    brake_torque_nm: float | np.ndarray,
    wheel_angular_speed_rad_s: float | np.ndarray,
    heat_fraction: float | np.ndarray,
) -> float | np.ndarray:
    """Brake heating power, ``heat_fraction · τ_brake · ω_wheel`` in watts.

    Both ``brake_torque_nm`` and ``wheel_angular_speed_rad_s`` are non-negative
    magnitudes in their braking convention (see the physics package docstring:
    a braking torque and a forward-rolling wheel), so the product needs no
    absolute value. ``heat_fraction`` is the caller-declared fraction of the
    dissipated brake power that lands in *this* node (rotor, caliper, pad),
    and the remainder is assumed to go to the other brake masses and the air —
    it must lie in ``[0, 1]``. The fraction and any split between disc and pad
    are synthetic calibration placeholders; the module only multiplies them.
    """
    torque = _finite_nonnegative("brake_torque_nm", brake_torque_nm)
    omega = _finite_nonnegative("wheel_angular_speed_rad_s", wheel_angular_speed_rad_s)
    fraction = _finite_nonnegative("heat_fraction", heat_fraction)
    if np.any(np.asarray(fraction) > 1.0):
        raise ValueError(f"heat_fraction must be <= 1, got {heat_fraction!r}")
    return _maybe_scalar(np.asarray(fraction) * np.asarray(torque) * np.asarray(omega))


def lumped_temperature_step_c(
    temp_c: float | np.ndarray,
    heat_capacity_j_per_k: float | np.ndarray,
    q_in_w: float | np.ndarray,
    q_conv_w: float | np.ndarray,
    q_rad_w: float | np.ndarray,
    dt_s: float,
) -> float | np.ndarray:
    """One explicit Euler step of ``m·c·dT/dt = Q_in − Q_conv − Q_rad``, in °C.

    ``T_{n+1} = T_n + dt·(Q_in − Q_conv − Q_rad) / C``. The heat flows are the
    signed values from :func:`convective_heat_flow_w` /
    :func:`radiative_heat_flow_w` (positive = loss) plus whatever positive
    input power :func:`tyre_slip_heat_w` / :func:`brake_heat_w` / engine or
    gearbox sources supply as ``q_in_w``. ``heat_capacity_j_per_k`` must be
    finite and positive (it is the divisor), ``dt_s`` finite and positive, and
    all flows finite. The result must stay finite and above absolute zero,
    otherwise the step raises instead of returning a state the radiation and
    pressure models could not consume. Stability is the caller's concern: an
    explicit Euler step with a small enough ``dt`` for the lightest node is
    stable, and this module does not pick one.
    """
    temperature = _celsius_to_kelvin("temp_c", temp_c) - _ABSOLUTE_ZERO_OFFSET
    capacity = _finite_nonnegative("heat_capacity_j_per_k", heat_capacity_j_per_k)
    if np.any(np.asarray(capacity) <= 0.0):
        raise ValueError(f"heat_capacity_j_per_k must be > 0, got {heat_capacity_j_per_k!r}")
    q_in = _finite("q_in_w", q_in_w)
    q_conv = _finite("q_conv_w", q_conv_w)
    q_rad = _finite("q_rad_w", q_rad_w)
    step = _finite("dt_s", dt_s)
    if np.any(np.asarray(step) <= 0.0):
        raise ValueError(f"dt_s must be > 0, got {dt_s!r}")
    new_temp = temperature + np.asarray(step) * (
        np.asarray(q_in) - np.asarray(q_conv) - np.asarray(q_rad)
    ) / np.asarray(capacity)
    if not np.all(np.isfinite(new_temp)):
        raise ArithmeticError("lumped_temperature_step_c integrated to a non-finite value")
    if np.any(np.asarray(new_temp) <= ABSOLUTE_ZERO_C):
        raise ValueError(
            "lumped_temperature_step_c would drive a node to or below absolute zero "
            f"({new_temp!r} °C); the caller's energy flows or step are not physical"
        )
    return _maybe_scalar(new_temp)


def tyre_pressure_pa(
    gas_mass_kg: float | np.ndarray,
    volume_m3: float | np.ndarray,
    temp_c: float | np.ndarray,
    gas_constant_j_per_kg_k: float | np.ndarray,
) -> float | np.ndarray:
    """Tyre absolute pressure from the ideal-gas law, ``p = m·R·T_K/V`` in Pa.

    Temperature enters in Kelvin via the explicit Celsius conversion; the mass
    is the declared gas-mass state (advanced by the caller, e.g. through
    :func:`tyre_gas_mass_step_kg`); ``volume_m3`` is a fixed/configured
    volume, so no volume change with temperature or deflection is modelled.
    ``gas_mass_kg`` must be finite and non-negative, ``volume_m3`` and
    ``gas_constant_j_per_kg_k`` finite and positive, and the temperature above
    absolute zero. A zero gas mass yields exactly zero pressure — an empty
    tyre, not an error. The gas constant is a synthetic placeholder for the
    tyre's air/nitrogen fill and is calibration data, not a universal value.
    """
    mass = _finite_nonnegative("gas_mass_kg", gas_mass_kg)
    volume = _finite_nonnegative("volume_m3", volume_m3)
    if np.any(np.asarray(volume) <= 0.0):
        raise ValueError(f"volume_m3 must be > 0, got {volume_m3!r}")
    gas_constant = _finite_nonnegative("gas_constant_j_per_kg_k", gas_constant_j_per_kg_k)
    if np.any(np.asarray(gas_constant) <= 0.0):
        raise ValueError(f"gas_constant_j_per_kg_k must be > 0, got {gas_constant_j_per_kg_k!r}")
    temp_k = _celsius_to_kelvin("temp_c", temp_c)
    return _maybe_scalar(np.asarray(mass) * np.asarray(gas_constant) * temp_k / np.asarray(volume))


def tyre_gas_mass_step_kg(
    gas_mass_kg: float | np.ndarray,
    leak_rate_kg_s: float | np.ndarray,
    dt_s: float,
) -> float | np.ndarray:
    """One step of a bounded leak on the tyre's declared gas mass, in kg.

    ``m_{n+1} = max(0, m_n − ṁ_leak·dt)``. The leak rate is non-negative and
    the step cannot push the mass below zero — an empty tyre stays empty — so
    pressure falls monotonically toward zero through :func:`tyre_pressure_pa`
    without any coupling to the thermal solver, the injector, or a repair.
    All inputs finite, ``leak_rate_kg_s`` non-negative, ``dt_s`` positive.
    A leak model with a pressure-dependent rate is deliberately out of scope:
    the rate here is whatever the caller declares.
    """
    mass = _finite_nonnegative("gas_mass_kg", gas_mass_kg)
    leak = _finite_nonnegative("leak_rate_kg_s", leak_rate_kg_s)
    step = _finite("dt_s", dt_s)
    if np.any(np.asarray(step) <= 0.0):
        raise ValueError(f"dt_s must be > 0, got {dt_s!r}")
    return _maybe_scalar(np.maximum(0.0, np.asarray(mass) - np.asarray(leak) * np.asarray(step)))


def _celsius_to_kelvin(label: str, value: object) -> np.ndarray:
    """Return the Celsius input as Kelvin, refusing non-finite or sub-absolute-zero."""
    array = _finite(label, value)
    if np.any(np.asarray(array) <= ABSOLUTE_ZERO_C):
        raise ValueError(
            f"{label} must be above absolute zero ({ABSOLUTE_ZERO_C} °C), got {value!r}"
        )
    return np.asarray(array) + _ABSOLUTE_ZERO_OFFSET


def _finite(label: str, value: object) -> np.ndarray:
    """Return a finite numeric input as a float array, or fail before arithmetic."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a real number, got {value!r}")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a real number, got {value!r}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return array


def _finite_nonnegative(label: str, value: object) -> np.ndarray:
    """Return a finite, non-negative numeric input as a float array, or fail."""
    array = _finite(label, value)
    if np.any(array < 0.0):
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    return array


def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
    """Collapse a 0-d result back to a Python float; keep arrays as arrays."""
    if np.ndim(value) == 0:
        return float(value)
    return value
