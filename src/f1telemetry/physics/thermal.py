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
cooling areas, emissivities, airflow coefficients, tyre volume, the gas
constant, the cold gauge pressure and the initial conditions live in the
``thermal`` section of ``car_spec.yaml`` as labelled placeholders;
:func:`validated_thermal_scalars` is the one boundary that reads them, and
:func:`simulate_thermal_trace` is its only in-module caller, so none of them
is written down here and calibrating them is a data edit. Nothing in this
module reads a clock, a seed, or a regulation.

**Validation shape.** Like the rest of :mod:`f1telemetry.physics`, every
entry point narrows its inputs and refuses non-finite values and non-positive
divisors *before* any arithmetic, so a NaN can never reach an integrator.
Functions accept Python scalars or numpy arrays; scalars come back as Python
floats, arrays as numpy arrays of the same shape. A lumped step that would
drive a node at or below absolute zero raises rather than clamping: that is a
physically declared boundary, and silently clamping it would hide an
insane energy imbalance in the caller's inputs.

**Two ways in, deliberately.** The primitives take flat scalars and
``float64`` arrays and are what a compiled kernel calls; they stay free of
the configuration object. :func:`validated_thermal_scalars` hands a kernel the
one validated form of every node value, to be read once outside its loop, and
:func:`simulate_thermal_trace` is the Python-facing composition that takes a
:class:`~f1telemetry.contracts.car_spec.KernelConfig` directly. The fixed-step
longitudinal kernel reads no thermal value, so nothing here changes a compiled
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from f1telemetry.contracts.car_spec import KernelConfig

__all__ = [
    "ABSOLUTE_ZERO_C",
    "CELSIUS_TO_KELVIN_OFFSET",
    "PA_PER_PSI",
    "STANDARD_ATMOSPHERE_PA",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "THERMAL_NODE_NAMES",
    "ThermalScalars",
    "ThermalTrace",
    "brake_heat_w",
    "convective_heat_flow_w",
    "lumped_temperature_step_c",
    "radiative_heat_flow_w",
    "simulate_thermal_trace",
    "tyre_gas_mass_step_kg",
    "tyre_pressure_pa",
    "tyre_slip_heat_w",
    "validated_thermal_scalars",
]

#: K added to a Celsius temperature to get Kelvin. The one conversion the module needs, and
#: named because the scenario boundary states its own ambient in the same units.
CELSIUS_TO_KELVIN_OFFSET: Final[float] = 273.15
_ABSOLUTE_ZERO_OFFSET: Final[float] = CELSIUS_TO_KELVIN_OFFSET

#: Celsius value of absolute zero; every temperature must stay above this.
ABSOLUTE_ZERO_C: Final[float] = -_ABSOLUTE_ZERO_OFFSET

#: Stefan–Boltzmann constant, W/(m²·K⁴). A physical constant, not a calibration.
STEFAN_BOLTZMANN_W_M2_K4: Final[float] = 5.670374419e-8

#: Standard atmospheric pressure, Pa. A physical standard rather than a calibration, kept
#: beside Stefan–Boltzmann because it plays the same role here: it is what a *gauge* pressure
#: is measured against, so the tyre pressure this module reports is only meaningful relative
#: to it, and the cold-pressure seed has to be lifted to absolute pressure by the same figure.
STANDARD_ATMOSPHERE_PA: Final[float] = 101_325.0

#: Pa per psi. A unit definition rather than a coefficient, for the same reason
#: ``STEFAN_BOLTZMANN_W_M2_K4`` is not one. The exact conversion is 6894.757293168361 Pa; the
#: rounded figure is kept because it is the one the committed traces were produced with, so
#: tightening it would move a published pressure for no physical gain.
PA_PER_PSI: Final[float] = 6_894.757

#: The four lumped thermal nodes, in the order every ``thermal_node_*`` configuration vector is
#: built: ``(tyre, brake, engine, gearbox)``. A node is a thermal mass with its own capacity,
#: cooling area, emissivity, airflow coefficient and initial temperature. The tyre and brake
#: records are shared by all four corners, which is the synthetic four-corner symmetry C11.1.2
#: asks for on the setup side; engine and gearbox are one node each.
#:
#: The order is duplicated from ``f1telemetry.contracts.car_spec.THERMAL_NODES`` rather than
#: imported. The physics core reads numbers out of the configuration and deliberately does not
#: reach back into the layer that produces it, so two definitions that must agree is the
#: honest shape here; ``tests/test_thermal.py`` pins them against each other and pins each node
#: index against the node its name claims, so a reordering fails a test rather than swapping
#: two thermal masses silently.
THERMAL_NODE_NAMES: Final[tuple[str, ...]] = ("tyre", "brake", "engine", "gearbox")
TYRE_NODE_INDEX: Final[int] = THERMAL_NODE_NAMES.index("tyre")
BRAKE_NODE_INDEX: Final[int] = THERMAL_NODE_NAMES.index("brake")
ENGINE_NODE_INDEX: Final[int] = THERMAL_NODE_NAMES.index("engine")
GEARBOX_NODE_INDEX: Final[int] = THERMAL_NODE_NAMES.index("gearbox")

# The corner count the per-corner tyre and brake nodes are replicated across, in the
# ``FL, FR, RL, RR`` order ``f1telemetry.physics.forces`` defines. Spelled out here rather
# than imported because this module reads no wheel state: it takes the per-corner work the
# caller already summed, and needs only how many corners there are.
_CORNERS_PER_NODE: Final[int] = 4


@dataclass(frozen=True, slots=True)
class ThermalScalars:
    """Every lumped-node value the thermal trace reads, narrowed and range-checked.

    Six vectors of one value per node in :data:`THERMAL_NODE_NAMES` order, the three heat
    shares that say how much of each heat source reaches each node, and the declared tyre gas
    state. Immutable, so a caller cannot adjust a node after the trace has read it and then
    report a temperature the model did not produce.
    """

    initial_temp_c: np.ndarray
    heat_capacity_j_per_k: np.ndarray
    cooling_area_m2: np.ndarray
    emissivity: np.ndarray
    airflow_base_w_m2_k: np.ndarray
    airflow_speed_gain_w_m2_k_per_m_s: np.ndarray
    brake_heat_fraction: float
    engine_waste_heat_share: float
    gearbox_loss_share: float
    tyre_volume_m3: float
    tyre_gas_constant_j_per_kg_k: float
    tyre_initial_pressure_psi_gauge: float


@dataclass(frozen=True, slots=True)
class ThermalTrace:
    """Computed ideal thermal channels, before the sensor pipeline.

    The temperatures are scenario outputs of the configured lumped model, not calibrated car
    data. The node capacities, cooling areas, emissivities, airflow coefficients and heat
    shares behind them are synthetic placeholders in ``car_spec.yaml``.
    """

    tyre_temp_c: np.ndarray
    brake_temp_c: np.ndarray
    tyre_pressure_psi: np.ndarray
    engine_temp_c: np.ndarray
    gearbox_temp_c: np.ndarray


def validated_thermal_scalars(config: KernelConfig, prefix: str) -> ThermalScalars:
    """Every lumped-node value the thermal model reads, narrowed and range-checked.

    **One validator for the model, shared by the trace and by the boundary that feeds it.**
    :func:`simulate_thermal_trace` reads the node values, and the scenario boundary reads the
    two drivetrain heat shares before it has anything for the trace to integrate; three copies
    of the same six names and the same sign rules would be three places for the rule to be
    wrong. ``prefix`` names the calling entry point in the message, because a shared boundary
    is a worse place for an ambiguous error than a named one.

    The rules are the ones the arithmetic cannot make for itself:

    * the six node vectors each a length-four, C-contiguous ``float64`` vector in
      :data:`THERMAL_NODE_NAMES` order. A wrong length or a non-contiguous view indexes the
      wrong thermal mass silently rather than raising, and the order is positional by design;
    * ``heat_capacity_j_per_k`` strictly positive at every node. It is the divisor of
      ``m c dT/dt``, so zero is a node with no thermal mass rather than a cold one;
    * ``cooling_area_m2`` and both airflow coefficients strictly positive. Zero would be a node
      that never cools by that path - a modelling decision the file can legitimately record -
      but a negative area or coefficient reverses that loss term and heats the node from the
      ambient, which is not a temperature the caller meant to ask for;
    * ``emissivity`` in ``(0, 1]``, for the same reason on the radiative path: a surface cannot
      emit more than a blackbody, and at or below zero the term vanishes;
    * ``initial_temp_c`` strictly above :data:`ABSOLUTE_ZERO_C`, because the radiation term
      raises its temperature to the fourth power and a negative temperature is not one;
    * the three heat shares in ``[0, 1]``. Zero is a legal declaration that a node absorbs none
      of the rejected energy; above one is a contradiction, since no node can take more heat
      than the boundary computed;
    * ``tyre_volume_m3`` and ``tyre_gas_constant_j_per_kg_k`` strictly positive, being the two
      divisors of the ideal gas law, and ``tyre_initial_pressure_psi_gauge`` non-negative: it is
      a *gauge* pressure, so zero is a tyre sitting at the standard atmosphere, while a
      negative one seeds a flat tyre rather than a cold one and is not a state this model
      starts from.

    It is deliberately *wider* than any one caller's read - the trace never reads the two
    drivetrain shares - because ``KernelConfig`` is a public frozen dataclass that
    ``dataclasses.replace`` can make inconsistent, and a boundary that checked only what this
    call happened to touch would let a bad value sit in the same object the next call reads.
    """
    names = (
        "initial_temp_c",
        "heat_capacity_j_per_k",
        "cooling_area_m2",
        "emissivity",
        "airflow_base_w_m2_k",
        "airflow_speed_gain_w_m2_k_per_m_s",
    )
    vectors = {
        name: _checked_node_vector(getattr(config, f"thermal_node_{name}"), name, prefix=prefix)
        for name in names
    }
    for index, node in enumerate(THERMAL_NODE_NAMES):
        for name in names[1:]:
            value = float(vectors[name][index])
            if value <= 0.0:
                raise ValueError(
                    f"{prefix}: config.thermal_node_{name}[{index}] ({node}) must be finite and "
                    f"> 0, got {value!r}. It is a divisor, a loss coefficient or a surface "
                    "property; at or below zero the node has no thermal mass, stops losing heat "
                    "through that path, or emits more than a blackbody"
                )
        emissivity = float(vectors["emissivity"][index])
        if emissivity > 1.0:
            raise ValueError(
                f"{prefix}: config.thermal_node_emissivity[{index}] ({node}) must be in (0, 1], "
                f"got {emissivity!r}. A surface cannot emit more than a blackbody"
            )
        initial = float(vectors["initial_temp_c"][index])
        if initial <= ABSOLUTE_ZERO_C:
            raise ValueError(
                f"{prefix}: config.thermal_node_initial_temp_c[{index}] ({node}) must be above "
                f"absolute zero ({ABSOLUTE_ZERO_C} °C), got {initial!r}. The radiation term "
                "raises its temperature to the fourth power"
            )
    shares: dict[str, float] = {}
    for name in ("brake_heat_fraction", "engine_waste_heat_share", "gearbox_loss_share"):
        value = _checked_float(
            f"config.thermal_{name}", getattr(config, f"thermal_{name}"), prefix=prefix
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{prefix}: config.thermal_{name} must be in [0, 1], got {value!r}. It is the "
                "share of the dissipated or rejected power that reaches this node, so zero is a "
                "legal declaration and more than one is a contradiction"
            )
        shares[name] = value
    divisors: dict[str, float] = {}
    for name in ("tyre_volume_m3", "tyre_gas_constant_j_per_kg_k"):
        value = _checked_float(
            f"config.thermal_{name}", getattr(config, f"thermal_{name}"), prefix=prefix
        )
        if value <= 0.0:
            raise ValueError(
                f"{prefix}: config.thermal_{name} must be finite and > 0, got {value!r}. Both "
                "divide the ideal gas law the tyre pressure comes from"
            )
        divisors[name] = value
    initial_pressure = _checked_float(
        "config.thermal_tyre_initial_pressure_psi_gauge",
        config.thermal_tyre_initial_pressure_psi_gauge,
        prefix=prefix,
    )
    if initial_pressure < 0.0:
        raise ValueError(
            f"{prefix}: config.thermal_tyre_initial_pressure_psi_gauge must be finite and >= 0, "
            f"got {initial_pressure!r}. It is a gauge pressure: zero is a tyre sitting at the "
            "standard atmosphere, and a negative one seeds a flat tyre rather than a cold one"
        )
    return ThermalScalars(
        initial_temp_c=vectors["initial_temp_c"],
        heat_capacity_j_per_k=vectors["heat_capacity_j_per_k"],
        cooling_area_m2=vectors["cooling_area_m2"],
        emissivity=vectors["emissivity"],
        airflow_base_w_m2_k=vectors["airflow_base_w_m2_k"],
        airflow_speed_gain_w_m2_k_per_m_s=vectors["airflow_speed_gain_w_m2_k_per_m_s"],
        brake_heat_fraction=shares["brake_heat_fraction"],
        engine_waste_heat_share=shares["engine_waste_heat_share"],
        gearbox_loss_share=shares["gearbox_loss_share"],
        tyre_volume_m3=divisors["tyre_volume_m3"],
        tyre_gas_constant_j_per_kg_k=divisors["tyre_gas_constant_j_per_kg_k"],
        tyre_initial_pressure_psi_gauge=initial_pressure,
    )


def simulate_thermal_trace(
    config: KernelConfig,
    *,
    dt_s: float,
    speed_m_s: np.ndarray,
    tyre_slip_work_j: np.ndarray,
    brake_work_j: np.ndarray,
    engine_heat_j: np.ndarray,
    gearbox_heat_j: np.ndarray,
    leak_rate_kg_s: float = 0.0,
    ambient_temp_c: float | None = None,
) -> ThermalTrace:
    """Advance the configured lumped nodes from interval energy inputs.

    The Python-facing composition, and the only place this module reads the configuration:
    the node capacities, cooling areas, emissivities, airflow coefficients, initial
    temperatures, the brake heat fraction and the declared tyre gas state all arrive through
    :func:`validated_thermal_scalars`, so none of them is a number written down here. A
    calibrated thermal model is therefore a ``car_spec.yaml`` edit rather than a refactor.

    Inputs carry energy accumulated over each record interval, so heating and airflow cooling
    are integrated at the scenario's 100 Hz boundary. The two drivetrain heat *shares* are the
    caller's to apply - they turn drivetrain work into node heat before this function is
    reached - so a caller that has not read them yet takes them from the same
    :func:`validated_thermal_scalars` rather than hardcoding them.

    ``ambient_temp_c`` defaults to the configured ``constants.air_temperature_k`` in Celsius,
    which is the one ambient the car file declares, and is otherwise a caller-overridable
    boundary: a scenario may place the car somewhere hotter or colder, and the default keeps
    the common case to an omitted argument without this module inventing a temperature.
    """
    values = validated_thermal_scalars(config, "simulate_thermal_trace")
    speed = _finite_nonnegative("speed_m_s", speed_m_s)
    slip = _finite_nonnegative("tyre_slip_work_j", tyre_slip_work_j)
    brake = _finite_nonnegative("brake_work_j", brake_work_j)
    engine = _finite_nonnegative("engine_heat_j", engine_heat_j)
    gearbox = _finite_nonnegative("gearbox_heat_j", gearbox_heat_j)
    if slip.ndim != 2 or slip.shape[1] != _CORNERS_PER_NODE or brake.shape != slip.shape:
        raise ValueError(
            f"tyre_slip_work_j and brake_work_j must have shape (steps, {_CORNERS_PER_NODE})"
        )
    count = slip.shape[0]
    if speed.shape != (count,) or engine.shape != (count,) or gearbox.shape != (count,):
        raise ValueError("speed_m_s, engine_heat_j, and gearbox_heat_j must match step count")
    dt = float(_finite("dt_s", dt_s))
    if dt <= 0.0:
        raise ValueError("dt_s must be > 0")
    leak = float(_finite_nonnegative("leak_rate_kg_s", leak_rate_kg_s))
    declared_ambient = (
        float(config.air_temperature_k) - CELSIUS_TO_KELVIN_OFFSET
        if ambient_temp_c is None
        else ambient_temp_c
    )

    ambient = float(_celsius_to_kelvin("ambient_temp_c", declared_ambient) - _ABSOLUTE_ZERO_OFFSET)
    tyre_temp = float(values.initial_temp_c[TYRE_NODE_INDEX])
    brake_temp = float(values.initial_temp_c[BRAKE_NODE_INDEX])
    engine_temp = float(values.initial_temp_c[ENGINE_NODE_INDEX])
    gearbox_temp = float(values.initial_temp_c[GEARBOX_NODE_INDEX])
    tyre_t = np.full(_CORNERS_PER_NODE, tyre_temp)
    brake_t = np.full(_CORNERS_PER_NODE, brake_temp)
    volume = values.tyre_volume_m3
    air_r = values.tyre_gas_constant_j_per_kg_k
    # Seed the gas mass to the configured cold gauge pressure, at the tyre node's own initial
    # temperature, so the declared pressure and the declared state agree by construction
    # rather than by two independently configured numbers happening to match.
    initial_gas_mass = (
        (STANDARD_ATMOSPHERE_PA + values.tyre_initial_pressure_psi_gauge * PA_PER_PSI)
        * volume
        / (air_r * (tyre_temp + CELSIUS_TO_KELVIN_OFFSET))
    )
    tyre_mass = np.full(_CORNERS_PER_NODE, initial_gas_mass)
    tyre_out = np.empty((count, _CORNERS_PER_NODE))
    brake_out = np.empty((count, _CORNERS_PER_NODE))
    pressure_out = np.empty((count, _CORNERS_PER_NODE))
    engine_out = np.empty(count)
    gearbox_out = np.empty(count)
    for i in range(count):
        speed_i = float(speed[i])
        tyre_q, brake_q = slip[i] / dt, brake[i] * values.brake_heat_fraction / dt
        tyre_t = _step_node(
            tyre_t,
            values,
            TYRE_NODE_INDEX,
            tyre_q,
            speed_i,
            ambient,
            dt,
        )
        brake_t = _step_node(
            brake_t,
            values,
            BRAKE_NODE_INDEX,
            brake_q,
            speed_i,
            ambient,
            dt,
        )
        engine_temp = float(
            _step_node(
                np.asarray(engine_temp),
                values,
                ENGINE_NODE_INDEX,
                engine[i] / dt,
                speed_i,
                ambient,
                dt,
            )
        )
        gearbox_temp = float(
            _step_node(
                np.asarray(gearbox_temp),
                values,
                GEARBOX_NODE_INDEX,
                gearbox[i] / dt,
                speed_i,
                ambient,
                dt,
            )
        )
        tyre_mass = tyre_gas_mass_step_kg(tyre_mass, leak, dt)
        tyre_out[i] = tyre_t
        brake_out[i] = brake_t
        pressure_out[i] = (
            np.asarray(tyre_pressure_pa(tyre_mass, volume, tyre_t, air_r)) - STANDARD_ATMOSPHERE_PA
        ) / PA_PER_PSI
        engine_out[i], gearbox_out[i] = engine_temp, gearbox_temp
    return ThermalTrace(tyre_out, brake_out, pressure_out, engine_out, gearbox_out)


def _step_node(
    temp_c: np.ndarray | float,
    values: ThermalScalars,
    node: int,
    q_in_w: np.ndarray,
    speed_m_s: float,
    ambient_temp_c: float,
    dt_s: float,
) -> np.ndarray | float:
    """One explicit Euler step of one configured lumped node, at one interval.

    The per-corner nodes pass a length-four temperature vector and the two powertrain nodes a
    scalar, so the node's own capacity, area, emissivity and airflow coefficient are taken from
    the validated configuration by position and the state keeps whichever shape the caller
    brought. The airflow coefficient is the configured ``base + speed_gain * v`` linear form,
    which is the whole of P3-T4's "cooling scaled by airflow, which should rise with speed".
    """
    coefficient = (
        float(values.airflow_base_w_m2_k[node])
        + float(values.airflow_speed_gain_w_m2_k_per_m_s[node]) * speed_m_s
    )
    area = float(values.cooling_area_m2[node])
    return lumped_temperature_step_c(
        temp_c,
        float(values.heat_capacity_j_per_k[node]),
        q_in_w,
        convective_heat_flow_w(coefficient, area, temp_c, ambient_temp_c),
        radiative_heat_flow_w(float(values.emissivity[node]), area, temp_c, ambient_temp_c),
        dt_s,
    )


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


def _checked_float(label: str, value: object, *, prefix: str) -> float:
    """A finite real scalar as ``float``, or a refusal before it reaches an integrator.

    Mirrors ``forces._checked_float`` and ``loads._checked_float`` rather than sharing one: each
    physics module validates a different field set for a different model, and a shared private
    helper between them would couple two models over six lines. ``bool`` is refused although it
    is an ``int``, because ``True`` as a heat fraction or a gas volume is a caller bug rather
    than a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {label} must be a real number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{prefix}: {label} must be finite, got {value!r}")
    return number


def _checked_node_vector(value: object, name: str, *, prefix: str) -> np.ndarray:
    """One configured node quantity, checked for the shape and dtype the node order relies on.

    The node order is positional, so a vector that is short, long, strided, ``float32`` or not
    an array at all does not raise later - it reads the wrong thermal mass and reports a
    plausible temperature. The length is fixed by :data:`THERMAL_NODE_NAMES` rather than by the
    array's own size, which is what makes the two able to disagree in a way this catches.
    """
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.ndim != 1
        or value.shape != (len(THERMAL_NODE_NAMES),)
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            f"{prefix}: config.thermal_node_{name} must be a C-contiguous float64 vector of "
            f"length {len(THERMAL_NODE_NAMES)} in THERMAL_NODE_NAMES order "
            f"{list(THERMAL_NODE_NAMES)}; got {type(value).__name__} of shape "
            f"{getattr(value, 'shape', None)} and dtype {getattr(value, 'dtype', None)}"
        )
    if not np.isfinite(value).all():
        raise ValueError(
            f"{prefix}: config.thermal_node_{name} must be finite at every node, got {list(value)}"
        )
    return value


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
