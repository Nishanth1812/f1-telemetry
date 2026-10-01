"""GENERATED FILE - DO NOT EDIT.
Regenerate with: uv run f1-codegen
Source of truth: channels.yaml (PLAN.md section 5.2)
CI runs `uv run f1-codegen --check` and fails on any diff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Self, TypeAlias

CONTRACT_VERSION: Final[str] = '1'
CONTRACT_DESCRIPTION: Final[str] = "THE contract. Every field's metadata - unit, rate, width, dtype, range, noise model, sigma, quantisation, fault eligibility - is declared exactly once, here. Python dataclasses, the Parquet schema and the TypeScript types are all code-generated from this file by `uv run f1-codegen`; none of them is hand-maintained. Channel inventory follows PLAN.md section 5.2 group by group; per-corner signals are declared once with a `corners` list and expanded by the generator, never written out four times. Sizing is computed from this file, not estimated: at the rates below the dictionary carries 7160 channel-samples/s (28.6 kB/s, 103 MB/h raw at float32), which supersedes the ~4600/s estimate in PLAN.md section 5.2."
CORNERS: Final[tuple[str, ...]] = ('FL', 'FR', 'RL', 'RR',)
FAULT_TYPES: Final[tuple[str, ...]] = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',)
NOISE_MODELS: Final[tuple[str, ...]] = ('gaussian', 'uniform', 'none',)
DTYPES: Final[tuple[str, ...]] = ('float32', 'float64', 'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'bool',)
GROUPS: Final[tuple[str, ...]] = ('chassis', 'imu', 'powertrain', 'aero', 'wheel', 'wheel_thermal', 'thermal', 'session', 'driver',)
RATES_HZ: Final[tuple[int, ...]] = (10, 20, 100, 200,)
SAMPLES_PER_SECOND: Final[float] = 7160.0
NoiseModel: TypeAlias = Literal['gaussian', 'uniform', 'none']
ChannelGroup: TypeAlias = Literal['chassis', 'imu', 'powertrain', 'aero', 'wheel', 'wheel_thermal', 'thermal', 'session', 'driver']
Corner: TypeAlias = Literal['FL', 'FR', 'RL', 'RR']
ChannelName: TypeAlias = Literal['speed', 'vx', 'vy', 'yaw_rate', 'accel_lateral', 'accel_longitudinal', 'roll', 'pitch', 'imu_accel_x', 'imu_accel_y', 'imu_accel_z', 'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z', 'ice_rpm', 'mgu_k_rpm', 'gear', 'throttle_pct', 'clutch_pct', 'ice_torque_nm', 'mgu_k_power_kw', 'boost_remaining_bar', 'eso_pct', 'fuel_flow_kg_h', 'aero_mode', 'fw_flap_deg', 'rw_flap_deg', 'zone_id', 'downforce_n', 'cl_a', 'cd_a', 'wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr', 'slip_ratio_fl', 'slip_ratio_fr', 'slip_ratio_rl', 'slip_ratio_rr', 'slip_angle_fl', 'slip_angle_fr', 'slip_angle_rl', 'slip_angle_rr', 'vertical_load_fl', 'vertical_load_fr', 'vertical_load_rl', 'vertical_load_rr', 'camber_fl', 'camber_fr', 'camber_rl', 'camber_rr', 'suspension_travel_fl', 'suspension_travel_fr', 'suspension_travel_rl', 'suspension_travel_rr', 'tyre_pressure_fl', 'tyre_pressure_fr', 'tyre_pressure_rl', 'tyre_pressure_rr', 'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr', 'brake_temp_fl', 'brake_temp_fr', 'brake_temp_rl', 'brake_temp_rr', 'engine_temp', 'gearbox_temp', 'coolant_temp', 'brake_duct_air_temp', 'lap_index', 'sector_index', 'lap_time', 'sector_time', 'delta', 'brake_pressure', 'steering_angle', 'pedal_travel']
CornerChannelName: TypeAlias = Literal['wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr', 'slip_ratio_fl', 'slip_ratio_fr', 'slip_ratio_rl', 'slip_ratio_rr', 'slip_angle_fl', 'slip_angle_fr', 'slip_angle_rl', 'slip_angle_rr', 'vertical_load_fl', 'vertical_load_fr', 'vertical_load_rl', 'vertical_load_rr', 'camber_fl', 'camber_fr', 'camber_rl', 'camber_rr', 'suspension_travel_fl', 'suspension_travel_fr', 'suspension_travel_rl', 'suspension_travel_rr', 'tyre_pressure_fl', 'tyre_pressure_fr', 'tyre_pressure_rl', 'tyre_pressure_rr', 'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr', 'brake_temp_fl', 'brake_temp_fr', 'brake_temp_rl', 'brake_temp_rr']
QuantisedBaseName: TypeAlias = Literal['speed', 'vx', 'vy', 'yaw_rate', 'accel_lateral', 'accel_longitudinal', 'roll', 'pitch', 'imu_accel_x', 'imu_accel_y', 'imu_accel_z', 'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z', 'ice_rpm', 'mgu_k_rpm', 'throttle_pct', 'clutch_pct', 'ice_torque_nm', 'mgu_k_power_kw', 'boost_remaining_bar', 'eso_pct', 'fuel_flow_kg_h', 'fw_flap_deg', 'rw_flap_deg', 'downforce_n', 'cl_a', 'cd_a', 'wheel_speed', 'slip_ratio', 'slip_angle', 'vertical_load', 'camber', 'suspension_travel', 'tyre_pressure', 'tyre_temp', 'brake_temp', 'engine_temp', 'gearbox_temp', 'coolant_temp', 'brake_duct_air_temp', 'brake_pressure', 'steering_angle', 'pedal_travel']

@dataclass(frozen=True, slots=True)
class Quantisation:
    """Analogue-to-digital resolution of the emulated sensor."""

    bits: int
    full_scale: float

    @property
    def step(self) -> float:
        return self.full_scale / float(1 << (self.bits - 1))

    def to_dict(self) -> dict[str, float]:
        bits = self.bits
        full_scale = self.full_scale
        return {'bits': bits, 'full_scale': full_scale, 'step': self.step}


@dataclass(frozen=True, slots=True)
class Channel:
    """One addressable channel. `name` is unique across the contract."""

    name: ChannelName
    base_name: str
    group: ChannelGroup
    unit: str
    dtype: str
    rate_hz: float | None
    event: bool
    range_min: float
    range_max: float
    noise_model: NoiseModel
    sigma: float
    quantisation: Quantisation | None
    fault_eligible: tuple[str, ...]
    corners: tuple[Corner, ...]
    corner: Corner | None
    description: str
    source: str

    @property
    def period_s(self) -> float | None:
        return None if self.rate_hz is None else 1.0 / self.rate_hz

    @property
    def is_corner_channel(self) -> bool:
        return self.corner is not None

    def to_dict(self) -> dict[str, object]:
        quant = None if self.quantisation is None else self.quantisation.to_dict()
        return {
            "name": self.name,
            "base_name": self.base_name,
            "group": self.group,
            "unit": self.unit,
            "dtype": self.dtype,
            "rate_hz": self.rate_hz,
            "event": self.event,
            "range": [self.range_min, self.range_max],
            "noise_model": self.noise_model,
            "sigma": self.sigma,
            "quantise": quant,
            "fault_eligible": list(self.fault_eligible),
            "corners": list(self.corners),
            "corner": self.corner,
            "description": self.description,
            "source": self.source,
        }


_CHANNELS: Final[tuple[Channel, ...]] = (
    Channel(name            = 'speed',
            base_name       = 'speed',
            group           = 'chassis',
            unit            = 'km/h',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 400.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=16, full_scale=400.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Ground-referenced vehicle speed, the driver-facing number.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'vx',
            base_name       = 'vx',
            group           = 'chassis',
            unit            = 'm/s',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -10.0,
            range_max       = 105.0,
            noise_model     = 'gaussian',
            sigma           = 0.02,
            quantisation    = Quantisation(bits=16, full_scale=110.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Body-frame longitudinal velocity, from the chassis estimator.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'vy',
            base_name       = 'vy',
            group           = 'chassis',
            unit            = 'm/s',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -30.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.02,
            quantisation    = Quantisation(bits=16, full_scale=32.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Body-frame lateral velocity, from the chassis estimator.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'yaw_rate',
            base_name       = 'yaw_rate',
            group           = 'chassis',
            unit            = 'deg/s',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -300.0,
            range_max       = 300.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=300.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Yaw rate about the vertical axis, chassis-filtered.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'accel_lateral',
            base_name       = 'accel_lateral',
            group           = 'chassis',
            unit            = 'm/s^2',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -60.0,
            range_max       = 60.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=60.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Filtered lateral acceleration, the "lateral g" the dashboard shows.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'accel_longitudinal',
            base_name       = 'accel_longitudinal',
            group           = 'chassis',
            unit            = 'm/s^2',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -70.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=70.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Filtered longitudinal acceleration, negative under braking.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'roll',
            base_name       = 'roll',
            group           = 'chassis',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -6.0,
            range_max       = 6.0,
            noise_model     = 'gaussian',
            sigma           = 0.02,
            quantisation    = Quantisation(bits=14, full_scale=6.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Body roll angle, positive rolling to the right.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'pitch',
            base_name       = 'pitch',
            group           = 'chassis',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -6.0,
            range_max       = 6.0,
            noise_model     = 'gaussian',
            sigma           = 0.02,
            quantisation    = Quantisation(bits=14, full_scale=6.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Body pitch angle, positive nose-up under braking.',
            source          = 'PLAN.md section 5.2 chassis group'),
    Channel(name            = 'imu_accel_x',
            base_name       = 'imu_accel_x',
            group           = 'imu',
            unit            = 'm/s^2',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -80.0,
            range_max       = 80.0,
            noise_model     = 'gaussian',
            sigma           = 0.08,
            quantisation    = Quantisation(bits=16, full_scale=80.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Raw IMU specific force, forward. Includes gravity on the other axes.',
            source          = 'PLAN.md section 5.2 imu group'),
    Channel(name            = 'imu_accel_y',
            base_name       = 'imu_accel_y',
            group           = 'imu',
            unit            = 'm/s^2',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -80.0,
            range_max       = 80.0,
            noise_model     = 'gaussian',
            sigma           = 0.08,
            quantisation    = Quantisation(bits=16, full_scale=80.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Raw IMU specific force, leftward positive.',
            source          = 'PLAN.md section 5.2 imu group'),
    Channel(name            = 'imu_accel_z',
            base_name       = 'imu_accel_z',
            group           = 'imu',
            unit            = 'm/s^2',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -80.0,
            range_max       = 80.0,
            noise_model     = 'gaussian',
            sigma           = 0.08,
            quantisation    = Quantisation(bits=16, full_scale=80.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Raw IMU specific force, upward positive, so the static value is +g and downforce reads as an increase.',
            source          = 'PLAN.md section 5.2 imu group'),
    Channel(name            = 'imu_gyro_x',
            base_name       = 'imu_gyro_x',
            group           = 'imu',
            unit            = 'deg/s',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -1000.0,
            range_max       = 1000.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=14, full_scale=1000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Raw IMU roll rate.',
            source          = 'PLAN.md section 5.2 imu group'),
    Channel(name            = 'imu_gyro_y',
            base_name       = 'imu_gyro_y',
            group           = 'imu',
            unit            = 'deg/s',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -1000.0,
            range_max       = 1000.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=14, full_scale=1000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Raw IMU pitch rate.',
            source          = 'PLAN.md section 5.2 imu group'),
    Channel(name            = 'imu_gyro_z',
            base_name       = 'imu_gyro_z',
            group           = 'imu',
            unit            = 'deg/s',
            dtype           = 'float32',
            rate_hz         = 200.0,
            event           = False,
            range_min       = -1000.0,
            range_max       = 1000.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=14, full_scale=1000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Raw IMU yaw rate.',
            source          = 'PLAN.md section 5.2 imu group'),
    Channel(name            = 'ice_rpm',
            base_name       = 'ice_rpm',
            group           = 'powertrain',
            unit            = 'rpm',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 13500.0,
            noise_model     = 'gaussian',
            sigma           = 8.0,
            quantisation    = Quantisation(bits=16, full_scale=13500.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Internal combustion engine shaft speed. A separate channel from mgu_k_rpm by contract - a single "rpm" is a 2022 concept and produces nonsense on a 2026 car. The range is illustrative: C5.14 gives a team per-car high-rev setting rather than a fixed public threshold, so there is no published overall limit to quote and 13500 rpm is our stand-in.',
            source          = 'PLAN.md section 5.2 powertrain group and naming correction'),
    Channel(name            = 'mgu_k_rpm',
            base_name       = 'mgu_k_rpm',
            group           = 'powertrain',
            unit            = 'rpm',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 35000.0,
            noise_model     = 'gaussian',
            sigma           = 25.0,
            quantisation    = Quantisation(bits=16, full_scale=35000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'MGU-K motor speed. Not on the ICE gear path, so it is free to spin far faster than the engine and the two must never be reported as one channel.',
            source          = 'PLAN.md section 5.2 powertrain group and naming correction'),
    Channel(name            = 'gear',
            base_name       = 'gear',
            group           = 'powertrain',
            unit            = 'enum',
            dtype           = 'int8',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -1.0,
            range_max       = 8.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'ICE gearbox state: -1 reverse, 0 neutral, 1..8 forward. FIA sets eight forward ratios and requires reverse; the integer encoding and neutral value are simulator conventions. Discrete state, so no gaussian noise or analogue quantisation.',
            source          = 'FIA 2026 Formula One Regulations Section C Issue 20 (2026-08-05), C9.6.1 (eight forward ratios) and C9.7 (reverse ratio), https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf'),
    Channel(name            = 'throttle_pct',
            base_name       = 'throttle_pct',
            group           = 'powertrain',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.2,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Driver throttle pedal travel as a percentage.',
            source          = 'PLAN.md section 5.2 powertrain group'),
    Channel(name            = 'clutch_pct',
            base_name       = 'clutch_pct',
            group           = 'powertrain',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.2,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Clutch engagement. Launch, trailing throttle and the upshift boost-cut all route through it, so a gearbox trace without it is not believable.',
            source          = 'PLAN.md section 6 clutch model'),
    Channel(name            = 'ice_torque_nm',
            base_name       = 'ice_torque_nm',
            group           = 'powertrain',
            unit            = 'Nm',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -200.0,
            range_max       = 400.0,
            noise_model     = 'gaussian',
            sigma           = 1.0,
            quantisation    = Quantisation(bits=14, full_scale=400.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'ICE shaft torque, including negative engine braking.',
            source          = 'PLAN.md section 6 powertrain'),
    Channel(name            = 'mgu_k_power_kw',
            base_name       = 'mgu_k_power_kw',
            group           = 'powertrain',
            unit            = 'kW',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -350.0,
            range_max       = 350.0,
            noise_model     = 'gaussian',
            sigma           = 1.0,
            quantisation    = Quantisation(bits=14, full_scale=350.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = "MGU-K electrical power at the motor, positive motoring, negative recuperating. The bounds are the rule's 350 kW in each direction - an operating limit, symmetric because recuperation is capped at the same figure as motoring, not a sensor span. C5.2.8 further constrains deployment by car speed and by mode; that is real regulation and this generator does not implement it, because Phase 0 is a noise-only signal generator with no energy model behind it.",
            source          = 'FIA 2026 Formula One Regulations Section C Issue 20 (2026-08-05), C5.2.7, https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf'),
    Channel(name            = 'boost_remaining_bar',
            base_name       = 'boost_remaining_bar',
            group           = 'powertrain',
            unit            = 'bar',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 8.5,
            noise_model     = 'gaussian',
            sigma           = 0.02,
            quantisation    = Quantisation(bits=14, full_scale=8.5),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Pneumatic overpressure available to the ICE for turbo boost.',
            source          = 'PLAN.md section 6 powertrain'),
    Channel(name            = 'eso_pct',
            base_name       = 'eso_pct',
            group           = 'powertrain',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = "Energy state of charge on a synthetic 0-100% display scale. The FIA's 4 MJ maximum state-of-charge swing is not the store capacity or an established mapping to this percentage, so the range and scale remain illustrative.",
            source          = 'PLAN.md section 6 powertrain'),
    Channel(name            = 'fuel_flow_kg_h',
            base_name       = 'fuel_flow_kg_h',
            group           = 'powertrain',
            unit            = 'kg/h',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 110.0,
            noise_model     = 'gaussian',
            sigma           = 0.15,
            quantisation    = Quantisation(bits=14, full_scale=110.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Fuel mass flow. One half of the energy-balance invariant; the other half is mgu_k_power_kw and the drag work. The range is illustrative because the regulation limits fuel *energy* flow, not mass flow. Converting that limit to kg/h requires the FIA-measured fuel energy density and lower heating value; this illustrative range is not that conversion.',
            source          = 'PLAN.md section 6 energy invariant'),
    Channel(name            = 'aero_mode',
            base_name       = 'aero_mode',
            group           = 'aero',
            unit            = 'enum',
            dtype           = 'int8',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 1.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Active aero state machine - 0 = Z-mode closed high downforce, 1 = X-mode open low drag.',
            source          = 'PLAN.md section 5.1 active aero'),
    Channel(name            = 'fw_flap_deg',
            base_name       = 'fw_flap_deg',
            group           = 'aero',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 25.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=25.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Front wing flap angle, moving through finite mechanical travel.',
            source          = 'PLAN.md section 5.1 active aero'),
    Channel(name            = 'rw_flap_deg',
            base_name       = 'rw_flap_deg',
            group           = 'aero',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -10.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=30.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Rear wing flap angle, moving through finite mechanical travel.',
            source          = 'PLAN.md section 5.1 active aero'),
    Channel(name            = 'zone_id',
            base_name       = 'zone_id',
            group           = 'aero',
            unit            = 'enum',
            dtype           = 'int16',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -1.0,
            range_max       = 20.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'freeze', 'spike', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'X-mode permission zone index, -1 where X-mode is not permitted. P4 supplies the track geometry that resolves it.',
            source          = 'PLAN.md section 5.1 active aero'),
    Channel(name            = 'downforce_n',
            base_name       = 'downforce_n',
            group           = 'aero',
            unit            = 'N',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 30000.0,
            noise_model     = 'gaussian',
            sigma           = 5.0,
            quantisation    = Quantisation(bits=16, full_scale=30000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Total vertical aerodynamic load, in newtons, so the dashboard shows a force rather than a dimensionless coefficient.',
            source          = 'PLAN.md section 5.1 derived channels'),
    Channel(name            = 'cl_a',
            base_name       = 'cl_a',
            group           = 'aero',
            unit            = 'm^2',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 5.2,
            noise_model     = 'gaussian',
            sigma           = 0.01,
            quantisation    = Quantisation(bits=14, full_scale=5.2),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Lift coefficient times reference area, the quantity downforce is built from.',
            source          = 'PLAN.md section 5.1 derived channels'),
    Channel(name            = 'cd_a',
            base_name       = 'cd_a',
            group           = 'aero',
            unit            = 'm^2',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 1.8,
            noise_model     = 'gaussian',
            sigma           = 0.005,
            quantisation    = Quantisation(bits=14, full_scale=1.8),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Drag coefficient times reference area, the quantity drag is built from.',
            source          = 'PLAN.md section 5.1 derived channels'),
    Channel(name            = 'wheel_speed_fl',
            base_name       = 'wheel_speed',
            group           = 'wheel',
            unit            = 'km/h',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 400.0,
            noise_model     = 'gaussian',
            sigma           = 0.6,
            quantisation    = Quantisation(bits=12, full_scale=400.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Per-corner wheel speed from the wheel-speed sensor.',
            source          = 'PLAN.md section 5.2 wheel group and worked example'),
    Channel(name            = 'wheel_speed_fr',
            base_name       = 'wheel_speed',
            group           = 'wheel',
            unit            = 'km/h',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 400.0,
            noise_model     = 'gaussian',
            sigma           = 0.6,
            quantisation    = Quantisation(bits=12, full_scale=400.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Per-corner wheel speed from the wheel-speed sensor.',
            source          = 'PLAN.md section 5.2 wheel group and worked example'),
    Channel(name            = 'wheel_speed_rl',
            base_name       = 'wheel_speed',
            group           = 'wheel',
            unit            = 'km/h',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 400.0,
            noise_model     = 'gaussian',
            sigma           = 0.6,
            quantisation    = Quantisation(bits=12, full_scale=400.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Per-corner wheel speed from the wheel-speed sensor.',
            source          = 'PLAN.md section 5.2 wheel group and worked example'),
    Channel(name            = 'wheel_speed_rr',
            base_name       = 'wheel_speed',
            group           = 'wheel',
            unit            = 'km/h',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 400.0,
            noise_model     = 'gaussian',
            sigma           = 0.6,
            quantisation    = Quantisation(bits=12, full_scale=400.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Per-corner wheel speed from the wheel-speed sensor.',
            source          = 'PLAN.md section 5.2 wheel group and worked example'),
    Channel(name            = 'slip_ratio_fl',
            base_name       = 'slip_ratio',
            group           = 'wheel',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -100.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Longitudinal slip ratio, kappa, as a percentage. Positive under drive, negative under braking.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_ratio_fr',
            base_name       = 'slip_ratio',
            group           = 'wheel',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -100.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Longitudinal slip ratio, kappa, as a percentage. Positive under drive, negative under braking.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_ratio_rl',
            base_name       = 'slip_ratio',
            group           = 'wheel',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -100.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Longitudinal slip ratio, kappa, as a percentage. Positive under drive, negative under braking.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_ratio_rr',
            base_name       = 'slip_ratio',
            group           = 'wheel',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -100.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Longitudinal slip ratio, kappa, as a percentage. Positive under drive, negative under braking.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_angle_fl',
            base_name       = 'slip_angle',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -30.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=30.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Slip angle at the contact patch, positive generating +y force.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_angle_fr',
            base_name       = 'slip_angle',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -30.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=30.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Slip angle at the contact patch, positive generating +y force.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_angle_rl',
            base_name       = 'slip_angle',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -30.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=30.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Slip angle at the contact patch, positive generating +y force.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'slip_angle_rr',
            base_name       = 'slip_angle',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -30.0,
            range_max       = 30.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=30.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Slip angle at the contact patch, positive generating +y force.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'vertical_load_fl',
            base_name       = 'vertical_load',
            group           = 'wheel',
            unit            = 'N',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 12000.0,
            noise_model     = 'gaussian',
            sigma           = 20.0,
            quantisation    = Quantisation(bits=16, full_scale=12000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Vertical load carried by the corner, static plus aerodynamic plus load transfer. The four of these must sum to the invariant-3 total.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'vertical_load_fr',
            base_name       = 'vertical_load',
            group           = 'wheel',
            unit            = 'N',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 12000.0,
            noise_model     = 'gaussian',
            sigma           = 20.0,
            quantisation    = Quantisation(bits=16, full_scale=12000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Vertical load carried by the corner, static plus aerodynamic plus load transfer. The four of these must sum to the invariant-3 total.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'vertical_load_rl',
            base_name       = 'vertical_load',
            group           = 'wheel',
            unit            = 'N',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 12000.0,
            noise_model     = 'gaussian',
            sigma           = 20.0,
            quantisation    = Quantisation(bits=16, full_scale=12000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Vertical load carried by the corner, static plus aerodynamic plus load transfer. The four of these must sum to the invariant-3 total.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'vertical_load_rr',
            base_name       = 'vertical_load',
            group           = 'wheel',
            unit            = 'N',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 12000.0,
            noise_model     = 'gaussian',
            sigma           = 20.0,
            quantisation    = Quantisation(bits=16, full_scale=12000.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Vertical load carried by the corner, static plus aerodynamic plus load transfer. The four of these must sum to the invariant-3 total.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'camber_fl',
            base_name       = 'camber',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 1.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=5.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Camber angle, negative at the top of the wheel.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'camber_fr',
            base_name       = 'camber',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 1.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=5.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Camber angle, negative at the top of the wheel.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'camber_rl',
            base_name       = 'camber',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 1.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=5.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Camber angle, negative at the top of the wheel.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'camber_rr',
            base_name       = 'camber',
            group           = 'wheel',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 1.0,
            noise_model     = 'gaussian',
            sigma           = 0.05,
            quantisation    = Quantisation(bits=14, full_scale=5.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Camber angle, negative at the top of the wheel.',
            source          = 'PLAN.md section 5.2 wheel group'),
    Channel(name            = 'suspension_travel_fl',
            base_name       = 'suspension_travel',
            group           = 'wheel_thermal',
            unit            = 'mm',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -25.0,
            range_max       = 25.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=25.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Per-corner suspension travel, quasi-static (PLAN.md section 2). Rate is 20 Hz because the ride-height sensor is a linear potentiometer, not because the underlying quantity is slow.',
            source          = 'PLAN.md section 5.2 wheel thermal group'),
    Channel(name            = 'suspension_travel_fr',
            base_name       = 'suspension_travel',
            group           = 'wheel_thermal',
            unit            = 'mm',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -25.0,
            range_max       = 25.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=25.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Per-corner suspension travel, quasi-static (PLAN.md section 2). Rate is 20 Hz because the ride-height sensor is a linear potentiometer, not because the underlying quantity is slow.',
            source          = 'PLAN.md section 5.2 wheel thermal group'),
    Channel(name            = 'suspension_travel_rl',
            base_name       = 'suspension_travel',
            group           = 'wheel_thermal',
            unit            = 'mm',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -25.0,
            range_max       = 25.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=25.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Per-corner suspension travel, quasi-static (PLAN.md section 2). Rate is 20 Hz because the ride-height sensor is a linear potentiometer, not because the underlying quantity is slow.',
            source          = 'PLAN.md section 5.2 wheel thermal group'),
    Channel(name            = 'suspension_travel_rr',
            base_name       = 'suspension_travel',
            group           = 'wheel_thermal',
            unit            = 'mm',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -25.0,
            range_max       = 25.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=12, full_scale=25.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Per-corner suspension travel, quasi-static (PLAN.md section 2). Rate is 20 Hz because the ride-height sensor is a linear potentiometer, not because the underlying quantity is slow.',
            source          = 'PLAN.md section 5.2 wheel thermal group'),
    Channel(name            = 'tyre_pressure_fl',
            base_name       = 'tyre_pressure',
            group           = 'wheel_thermal',
            unit            = 'psi',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 18.0,
            range_max       = 45.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=45.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Per-corner tyre pressure, driven by a gas-law node off carcass temperature, so it responds to both tyre_temp and a leak fault.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_pressure_fr',
            base_name       = 'tyre_pressure',
            group           = 'wheel_thermal',
            unit            = 'psi',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 18.0,
            range_max       = 45.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=45.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Per-corner tyre pressure, driven by a gas-law node off carcass temperature, so it responds to both tyre_temp and a leak fault.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_pressure_rl',
            base_name       = 'tyre_pressure',
            group           = 'wheel_thermal',
            unit            = 'psi',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 18.0,
            range_max       = 45.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=45.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Per-corner tyre pressure, driven by a gas-law node off carcass temperature, so it responds to both tyre_temp and a leak fault.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_pressure_rr',
            base_name       = 'tyre_pressure',
            group           = 'wheel_thermal',
            unit            = 'psi',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 18.0,
            range_max       = 45.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=45.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Per-corner tyre pressure, driven by a gas-law node off carcass temperature, so it responds to both tyre_temp and a leak fault.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_temp_fl',
            base_name       = 'tyre_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 150.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=12, full_scale=150.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Per-corner tyre carcass temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_temp_fr',
            base_name       = 'tyre_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 150.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=12, full_scale=150.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Per-corner tyre carcass temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_temp_rl',
            base_name       = 'tyre_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 150.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=12, full_scale=150.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Per-corner tyre carcass temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'tyre_temp_rr',
            base_name       = 'tyre_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = -5.0,
            range_max       = 150.0,
            noise_model     = 'gaussian',
            sigma           = 0.5,
            quantisation    = Quantisation(bits=12, full_scale=150.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Per-corner tyre carcass temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'brake_temp_fl',
            base_name       = 'brake_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 1100.0,
            noise_model     = 'gaussian',
            sigma           = 1.5,
            quantisation    = Quantisation(bits=14, full_scale=1100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FL',
            description     = 'Per-corner brake disc temperature, heated by brake torque times wheel speed.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'brake_temp_fr',
            base_name       = 'brake_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 1100.0,
            noise_model     = 'gaussian',
            sigma           = 1.5,
            quantisation    = Quantisation(bits=14, full_scale=1100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'FR',
            description     = 'Per-corner brake disc temperature, heated by brake torque times wheel speed.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'brake_temp_rl',
            base_name       = 'brake_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 1100.0,
            noise_model     = 'gaussian',
            sigma           = 1.5,
            quantisation    = Quantisation(bits=14, full_scale=1100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RL',
            description     = 'Per-corner brake disc temperature, heated by brake torque times wheel speed.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'brake_temp_rr',
            base_name       = 'brake_temp',
            group           = 'wheel_thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 20.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 1100.0,
            noise_model     = 'gaussian',
            sigma           = 1.5,
            quantisation    = Quantisation(bits=14, full_scale=1100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'swap', 'saturate', 'stale',),
            corners         = ('FL', 'FR', 'RL', 'RR',),
            corner          = 'RR',
            description     = 'Per-corner brake disc temperature, heated by brake torque times wheel speed.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'engine_temp',
            base_name       = 'engine_temp',
            group           = 'thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 10.0,
            event           = False,
            range_min       = -20.0,
            range_max       = 130.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=130.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'ICE coolant-block lumped node temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'gearbox_temp',
            base_name       = 'gearbox_temp',
            group           = 'thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 10.0,
            event           = False,
            range_min       = -20.0,
            range_max       = 130.0,
            noise_model     = 'gaussian',
            sigma           = 0.3,
            quantisation    = Quantisation(bits=12, full_scale=130.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Gearbox oil lumped node temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'coolant_temp',
            base_name       = 'coolant_temp',
            group           = 'thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 10.0,
            event           = False,
            range_min       = -20.0,
            range_max       = 120.0,
            noise_model     = 'gaussian',
            sigma           = 0.2,
            quantisation    = Quantisation(bits=12, full_scale=120.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Primary cooling loop lumped node temperature.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'brake_duct_air_temp',
            base_name       = 'brake_duct_air_temp',
            group           = 'thermal',
            unit            = 'degC',
            dtype           = 'float32',
            rate_hz         = 10.0,
            event           = False,
            range_min       = -30.0,
            range_max       = 90.0,
            noise_model     = 'gaussian',
            sigma           = 0.4,
            quantisation    = Quantisation(bits=12, full_scale=90.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Brake duct inlet air temperature, upstream of the disc.',
            source          = 'PLAN.md sections 5.2 and 7'),
    Channel(name            = 'lap_index',
            base_name       = 'lap_index',
            group           = 'session',
            unit            = 'count',
            dtype           = 'int16',
            rate_hz         = None,
            event           = True,
            range_min       = -1.0,
            range_max       = 5000.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Zero-based lap counter, emitted at each lap boundary.',
            source          = 'PLAN.md section 5.2 session group'),
    Channel(name            = 'sector_index',
            base_name       = 'sector_index',
            group           = 'session',
            unit            = 'count',
            dtype           = 'int8',
            rate_hz         = None,
            event           = True,
            range_min       = -1.0,
            range_max       = 3.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Sector index 0..2, emitted at each sector boundary.',
            source          = 'PLAN.md section 5.2 session group'),
    Channel(name            = 'lap_time',
            base_name       = 'lap_time',
            group           = 'session',
            unit            = 's',
            dtype           = 'float32',
            rate_hz         = None,
            event           = True,
            range_min       = 0.0,
            range_max       = 1200.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Completed lap time, emitted once per lap. sigma is 0 because the value is a timestamp difference, not a measurement.',
            source          = 'PLAN.md section 5.2 session group'),
    Channel(name            = 'sector_time',
            base_name       = 'sector_time',
            group           = 'session',
            unit            = 's',
            dtype           = 'float32',
            rate_hz         = None,
            event           = True,
            range_min       = 0.0,
            range_max       = 600.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Completed sector time, emitted once per sector.',
            source          = 'PLAN.md section 5.2 session group'),
    Channel(name            = 'delta',
            base_name       = 'delta',
            group           = 'session',
            unit            = 's',
            dtype           = 'float32',
            rate_hz         = None,
            event           = True,
            range_min       = -5.0,
            range_max       = 5.0,
            noise_model     = 'none',
            sigma           = 0.0,
            quantisation    = None,
            fault_eligible  = ('dropout', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Time delta against the reference lap at the current track position, emitted when it is recomputed rather than on a clock.',
            source          = 'PLAN.md sections 5.2 and 8.5'),
    Channel(name            = 'brake_pressure',
            base_name       = 'brake_pressure',
            group           = 'driver',
            unit            = 'bar',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 250.0,
            noise_model     = 'gaussian',
            sigma           = 0.4,
            quantisation    = Quantisation(bits=14, full_scale=250.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Master-cylinder brake pressure.',
            source          = 'PLAN.md section 5.2 driver group'),
    Channel(name            = 'steering_angle',
            base_name       = 'steering_angle',
            group           = 'driver',
            unit            = 'deg',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = -450.0,
            range_max       = 450.0,
            noise_model     = 'gaussian',
            sigma           = 0.1,
            quantisation    = Quantisation(bits=14, full_scale=450.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Steering wheel angle, positive left, before the steering limit.',
            source          = 'PLAN.md section 5.2 driver group'),
    Channel(name            = 'pedal_travel',
            base_name       = 'pedal_travel',
            group           = 'driver',
            unit            = '%',
            dtype           = 'float32',
            rate_hz         = 100.0,
            event           = False,
            range_min       = 0.0,
            range_max       = 100.0,
            noise_model     = 'gaussian',
            sigma           = 0.2,
            quantisation    = Quantisation(bits=12, full_scale=100.0),
            fault_eligible  = ('dropout', 'freeze', 'spike', 'step', 'gain', 'noise', 'quantise', 'saturate', 'stale',),
            corners         = (),
            corner          = None,
            description     = 'Combined driver pedal travel, whichever pedal is being worked.',
            source          = 'PLAN.md section 5.2 driver group'),
)

CHANNELS: Final[Mapping[str, Channel]] = MappingProxyType(
    {channel.name: channel for channel in _CHANNELS}
)

CHANNELS_BY_GROUP: Final[Mapping[str, tuple[Channel, ...]]] = MappingProxyType(
    {
        'chassis': (
            CHANNELS['speed'],
            CHANNELS['vx'],
            CHANNELS['vy'],
            CHANNELS['yaw_rate'],
            CHANNELS['accel_lateral'],
            CHANNELS['accel_longitudinal'],
            CHANNELS['roll'],
            CHANNELS['pitch'],
        ),
        'imu': (
            CHANNELS['imu_accel_x'],
            CHANNELS['imu_accel_y'],
            CHANNELS['imu_accel_z'],
            CHANNELS['imu_gyro_x'],
            CHANNELS['imu_gyro_y'],
            CHANNELS['imu_gyro_z'],
        ),
        'powertrain': (
            CHANNELS['ice_rpm'],
            CHANNELS['mgu_k_rpm'],
            CHANNELS['gear'],
            CHANNELS['throttle_pct'],
            CHANNELS['clutch_pct'],
            CHANNELS['ice_torque_nm'],
            CHANNELS['mgu_k_power_kw'],
            CHANNELS['boost_remaining_bar'],
            CHANNELS['eso_pct'],
            CHANNELS['fuel_flow_kg_h'],
        ),
        'aero': (
            CHANNELS['aero_mode'],
            CHANNELS['fw_flap_deg'],
            CHANNELS['rw_flap_deg'],
            CHANNELS['zone_id'],
            CHANNELS['downforce_n'],
            CHANNELS['cl_a'],
            CHANNELS['cd_a'],
        ),
        'wheel': (
            CHANNELS['wheel_speed_fl'],
            CHANNELS['wheel_speed_fr'],
            CHANNELS['wheel_speed_rl'],
            CHANNELS['wheel_speed_rr'],
            CHANNELS['slip_ratio_fl'],
            CHANNELS['slip_ratio_fr'],
            CHANNELS['slip_ratio_rl'],
            CHANNELS['slip_ratio_rr'],
            CHANNELS['slip_angle_fl'],
            CHANNELS['slip_angle_fr'],
            CHANNELS['slip_angle_rl'],
            CHANNELS['slip_angle_rr'],
            CHANNELS['vertical_load_fl'],
            CHANNELS['vertical_load_fr'],
            CHANNELS['vertical_load_rl'],
            CHANNELS['vertical_load_rr'],
            CHANNELS['camber_fl'],
            CHANNELS['camber_fr'],
            CHANNELS['camber_rl'],
            CHANNELS['camber_rr'],
        ),
        'wheel_thermal': (
            CHANNELS['suspension_travel_fl'],
            CHANNELS['suspension_travel_fr'],
            CHANNELS['suspension_travel_rl'],
            CHANNELS['suspension_travel_rr'],
            CHANNELS['tyre_pressure_fl'],
            CHANNELS['tyre_pressure_fr'],
            CHANNELS['tyre_pressure_rl'],
            CHANNELS['tyre_pressure_rr'],
            CHANNELS['tyre_temp_fl'],
            CHANNELS['tyre_temp_fr'],
            CHANNELS['tyre_temp_rl'],
            CHANNELS['tyre_temp_rr'],
            CHANNELS['brake_temp_fl'],
            CHANNELS['brake_temp_fr'],
            CHANNELS['brake_temp_rl'],
            CHANNELS['brake_temp_rr'],
        ),
        'thermal': (
            CHANNELS['engine_temp'],
            CHANNELS['gearbox_temp'],
            CHANNELS['coolant_temp'],
            CHANNELS['brake_duct_air_temp'],
        ),
        'session': (
            CHANNELS['lap_index'],
            CHANNELS['sector_index'],
            CHANNELS['lap_time'],
            CHANNELS['sector_time'],
            CHANNELS['delta'],
        ),
        'driver': (
            CHANNELS['brake_pressure'],
            CHANNELS['steering_angle'],
            CHANNELS['pedal_travel'],
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ChassisChannels:
    """One frame of every chassis channel. Field order matches the contract."""

    speed              : float
    vx                 : float
    vy                 : float
    yaw_rate           : float
    accel_lateral      : float
    accel_longitudinal : float
    roll               : float
    pitch              : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("speed", "vx", "vy", "yaw_rate", "accel_lateral", "accel_longitudinal", "roll", "pitch",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class ImuChannels:
    """One frame of every imu channel. Field order matches the contract."""

    imu_accel_x : float
    imu_accel_y : float
    imu_accel_z : float
    imu_gyro_x  : float
    imu_gyro_y  : float
    imu_gyro_z  : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("imu_accel_x", "imu_accel_y", "imu_accel_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class PowertrainChannels:
    """One frame of every powertrain channel. Field order matches the contract."""

    ice_rpm             : float
    mgu_k_rpm           : float
    gear                : float
    throttle_pct        : float
    clutch_pct          : float
    ice_torque_nm       : float
    mgu_k_power_kw      : float
    boost_remaining_bar : float
    eso_pct             : float
    fuel_flow_kg_h      : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("ice_rpm", "mgu_k_rpm", "gear", "throttle_pct", "clutch_pct", "ice_torque_nm", "mgu_k_power_kw", "boost_remaining_bar", "eso_pct", "fuel_flow_kg_h",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class AeroChannels:
    """One frame of every aero channel. Field order matches the contract."""

    aero_mode   : float
    fw_flap_deg : float
    rw_flap_deg : float
    zone_id     : float
    downforce_n : float
    cl_a        : float
    cd_a        : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("aero_mode", "fw_flap_deg", "rw_flap_deg", "zone_id", "downforce_n", "cl_a", "cd_a",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class WheelChannels:
    """One frame of every wheel channel. Field order matches the contract."""

    wheel_speed_fl   : float
    wheel_speed_fr   : float
    wheel_speed_rl   : float
    wheel_speed_rr   : float
    slip_ratio_fl    : float
    slip_ratio_fr    : float
    slip_ratio_rl    : float
    slip_ratio_rr    : float
    slip_angle_fl    : float
    slip_angle_fr    : float
    slip_angle_rl    : float
    slip_angle_rr    : float
    vertical_load_fl : float
    vertical_load_fr : float
    vertical_load_rl : float
    vertical_load_rr : float
    camber_fl        : float
    camber_fr        : float
    camber_rl        : float
    camber_rr        : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr", "slip_ratio_fl", "slip_ratio_fr", "slip_ratio_rl", "slip_ratio_rr", "slip_angle_fl", "slip_angle_fr", "slip_angle_rl", "slip_angle_rr", "vertical_load_fl", "vertical_load_fr", "vertical_load_rl", "vertical_load_rr", "camber_fl", "camber_fr", "camber_rl", "camber_rr",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class WheelThermalChannels:
    """One frame of every wheel thermal channel. Field order matches the contract."""

    suspension_travel_fl : float
    suspension_travel_fr : float
    suspension_travel_rl : float
    suspension_travel_rr : float
    tyre_pressure_fl     : float
    tyre_pressure_fr     : float
    tyre_pressure_rl     : float
    tyre_pressure_rr     : float
    tyre_temp_fl         : float
    tyre_temp_fr         : float
    tyre_temp_rl         : float
    tyre_temp_rr         : float
    brake_temp_fl        : float
    brake_temp_fr        : float
    brake_temp_rl        : float
    brake_temp_rr        : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("suspension_travel_fl", "suspension_travel_fr", "suspension_travel_rl", "suspension_travel_rr", "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr", "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr", "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class ThermalChannels:
    """One frame of every thermal channel. Field order matches the contract."""

    engine_temp         : float
    gearbox_temp        : float
    coolant_temp        : float
    brake_duct_air_temp : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("engine_temp", "gearbox_temp", "coolant_temp", "brake_duct_air_temp",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class SessionChannels:
    """One frame of every session channel. Field order matches the contract."""

    lap_index    : float
    sector_index : float
    lap_time     : float
    sector_time  : float
    delta        : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("lap_index", "sector_index", "lap_time", "sector_time", "delta",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}


@dataclass(frozen=True, slots=True)
class DriverChannels:
    """One frame of every driver channel. Field order matches the contract."""

    brake_pressure : float
    steering_angle : float
    pedal_travel   : float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """Build from a channel-name mapping; a missing channel is an error."""
        return cls(**{name: values[name] for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return ("brake_pressure", "steering_angle", "pedal_travel",)

    def as_mapping(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.field_names()}



@dataclass(frozen=True, slots=True)
class Contract:
    """Runtime handle on the contract, for callers that need to re-read the YAML."""

    version: str
    channels: Mapping[str, Channel]

    def __len__(self) -> int:
        return len(self.channels)

    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.channels)

    def by_name(self, name: str) -> Channel:
        return self.channels[name]

    def group(self, group: str) -> tuple[Channel, ...]:
        return CHANNELS_BY_GROUP[group]

    def base_names(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for channel in self.channels.values():
            seen.setdefault(channel.base_name, None)
        return tuple(seen)

    def corner_channels(self, base_name: str) -> tuple[Channel, ...]:
        matched = (c for c in self.channels.values() if c.base_name == base_name)
        return tuple(matched)

    def corner_map(self, base_name: str) -> dict[Corner, Channel]:
        found: dict[Corner, Channel] = {}
        for channel in self.corner_channels(base_name):
            if channel.corner is not None:
                found[channel.corner] = channel
        return found

    def corner_values(
        self, values: Mapping[str, float], base_name: str
    ) -> dict[str, float]:
        channels = self.corner_channels(base_name)
        return {str(c.corner).lower(): values[c.name] for c in channels}

    def samples_per_second(self) -> float:
        return SAMPLES_PER_SECOND

    def event_channels(self) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels.values() if c.event)


def load_contract(channels_yaml: Path | None = None) -> Contract:
    """Re-read `channels.yaml` and assert it still matches this generated file.

    Raises if the contract has drifted, which means codegen was not re-run.
    """
    from f1telemetry.contracts.channels import (
        ContractError,
        load_channel_contract,
    )

    loaded = load_channel_contract(channels_yaml)
    if loaded.version != CONTRACT_VERSION:
        msg = f'contract version {loaded.version} != '
        msg += f'generated {CONTRACT_VERSION}'
        raise ContractError(msg)
    if loaded.names != tuple(CHANNELS):
        msg = 'channels.yaml no longer matches the generated registry; '
        msg += 'run uv run f1-codegen'
        raise ContractError(msg)
    return Contract(version=loaded.version, channels=CHANNELS)
