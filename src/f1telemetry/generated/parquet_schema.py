"""GENERATED FILE - DO NOT EDIT.
Regenerate with: uv run f1-codegen
Source of truth: channels.yaml (PLAN.md section 5.2)
CI runs `uv run f1-codegen --check` and fails on any diff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pyarrow as pa

TIME_COLUMN: Final[str] = 't_s'

@dataclass(frozen=True, slots=True)
class ParquetColumn:
    """One Parquet column and the contract metadata a reader needs."""

    name: str
    arrow_type: str
    unit: str
    rate_hz: float | None
    event: bool
    dtype: str


@dataclass(frozen=True, slots=True)
class ParquetGroupSchema:
    """Schema of one group file: the time column plus that group's channels."""

    group: str
    rate_hz: float | None
    event: bool
    columns: tuple[ParquetColumn, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def arrow_schema(self) -> pa.Schema:
        """Arrow schema for the group, time column first."""
        fields = [pa.field(TIME_COLUMN, pa.float64(), nullable=False)]
        for column in self.columns:
            fields.append(pa.field(column.name, column.arrow_type, nullable=False))
        return pa.schema(fields)

    def arrow_type_map(self) -> dict[str, pa.DataType]:
        return {name: _ARROW[name] for name in self.names}

_ARROW: Final[Mapping[str, pa.DataType]] = MappingProxyType(
{
        'float32': pa.float32(),
        'float64': pa.float64(),
        'int8': pa.int8(),
        'int16': pa.int16(),
        'int32': pa.int32(),
        'uint8': pa.uint8(),
        'uint16': pa.uint16(),
        'uint32': pa.uint32(),
        'bool': pa.bool_(),
})


CHASSIS: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='chassis',
    rate_hz=200.0,
    event=False,
    columns=(
        ParquetColumn(
            name='speed',
            arrow_type=pa.float32(),
            unit='km/h',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='vx',
            arrow_type=pa.float32(),
            unit='m/s',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='vy',
            arrow_type=pa.float32(),
            unit='m/s',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='yaw_rate',
            arrow_type=pa.float32(),
            unit='deg/s',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='accel_lateral',
            arrow_type=pa.float32(),
            unit='m/s^2',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='accel_longitudinal',
            arrow_type=pa.float32(),
            unit='m/s^2',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='roll',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='pitch',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
    ),
)

IMU: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='imu',
    rate_hz=200.0,
    event=False,
    columns=(
        ParquetColumn(
            name='imu_accel_x',
            arrow_type=pa.float32(),
            unit='m/s^2',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='imu_accel_y',
            arrow_type=pa.float32(),
            unit='m/s^2',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='imu_accel_z',
            arrow_type=pa.float32(),
            unit='m/s^2',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='imu_gyro_x',
            arrow_type=pa.float32(),
            unit='deg/s',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='imu_gyro_y',
            arrow_type=pa.float32(),
            unit='deg/s',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='imu_gyro_z',
            arrow_type=pa.float32(),
            unit='deg/s',
            rate_hz=200.0,
            event=False,
            dtype='float32',
        ),
    ),
)

POWERTRAIN: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='powertrain',
    rate_hz=100.0,
    event=False,
    columns=(
        ParquetColumn(
            name='ice_rpm',
            arrow_type=pa.float32(),
            unit='rpm',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='mgu_k_rpm',
            arrow_type=pa.float32(),
            unit='rpm',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='gear',
            arrow_type=pa.int8(),
            unit='enum',
            rate_hz=100.0,
            event=False,
            dtype='int8',
        ),
        ParquetColumn(
            name='throttle_pct',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='clutch_pct',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='ice_torque_nm',
            arrow_type=pa.float32(),
            unit='Nm',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='mgu_k_power_kw',
            arrow_type=pa.float32(),
            unit='kW',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='boost_remaining_bar',
            arrow_type=pa.float32(),
            unit='bar',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='eso_pct',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='fuel_flow_kg_h',
            arrow_type=pa.float32(),
            unit='kg/h',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
    ),
)

AERO: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='aero',
    rate_hz=100.0,
    event=False,
    columns=(
        ParquetColumn(
            name='aero_mode',
            arrow_type=pa.int8(),
            unit='enum',
            rate_hz=100.0,
            event=False,
            dtype='int8',
        ),
        ParquetColumn(
            name='fw_flap_deg',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='rw_flap_deg',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='zone_id',
            arrow_type=pa.int16(),
            unit='enum',
            rate_hz=100.0,
            event=False,
            dtype='int16',
        ),
        ParquetColumn(
            name='downforce_n',
            arrow_type=pa.float32(),
            unit='N',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='cl_a',
            arrow_type=pa.float32(),
            unit='m^2',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='cd_a',
            arrow_type=pa.float32(),
            unit='m^2',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
    ),
)

WHEEL: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='wheel',
    rate_hz=100.0,
    event=False,
    columns=(
        ParquetColumn(
            name='wheel_speed_fl',
            arrow_type=pa.float32(),
            unit='km/h',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='wheel_speed_fr',
            arrow_type=pa.float32(),
            unit='km/h',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='wheel_speed_rl',
            arrow_type=pa.float32(),
            unit='km/h',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='wheel_speed_rr',
            arrow_type=pa.float32(),
            unit='km/h',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_ratio_fl',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_ratio_fr',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_ratio_rl',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_ratio_rr',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_angle_fl',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_angle_fr',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_angle_rl',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='slip_angle_rr',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='vertical_load_fl',
            arrow_type=pa.float32(),
            unit='N',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='vertical_load_fr',
            arrow_type=pa.float32(),
            unit='N',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='vertical_load_rl',
            arrow_type=pa.float32(),
            unit='N',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='vertical_load_rr',
            arrow_type=pa.float32(),
            unit='N',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='camber_fl',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='camber_fr',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='camber_rl',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='camber_rr',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
    ),
)

WHEEL_THERMAL: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='wheel_thermal',
    rate_hz=20.0,
    event=False,
    columns=(
        ParquetColumn(
            name='suspension_travel_fl',
            arrow_type=pa.float32(),
            unit='mm',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='suspension_travel_fr',
            arrow_type=pa.float32(),
            unit='mm',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='suspension_travel_rl',
            arrow_type=pa.float32(),
            unit='mm',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='suspension_travel_rr',
            arrow_type=pa.float32(),
            unit='mm',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_pressure_fl',
            arrow_type=pa.float32(),
            unit='psi',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_pressure_fr',
            arrow_type=pa.float32(),
            unit='psi',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_pressure_rl',
            arrow_type=pa.float32(),
            unit='psi',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_pressure_rr',
            arrow_type=pa.float32(),
            unit='psi',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_temp_fl',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_temp_fr',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_temp_rl',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='tyre_temp_rr',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='brake_temp_fl',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='brake_temp_fr',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='brake_temp_rl',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='brake_temp_rr',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=20.0,
            event=False,
            dtype='float32',
        ),
    ),
)

THERMAL: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='thermal',
    rate_hz=10.0,
    event=False,
    columns=(
        ParquetColumn(
            name='engine_temp',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=10.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='gearbox_temp',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=10.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='coolant_temp',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=10.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='brake_duct_air_temp',
            arrow_type=pa.float32(),
            unit='degC',
            rate_hz=10.0,
            event=False,
            dtype='float32',
        ),
    ),
)

SESSION: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='session',
    rate_hz=None,
    event=True,
    columns=(
        ParquetColumn(
            name='lap_index',
            arrow_type=pa.int16(),
            unit='count',
            rate_hz=None,
            event=True,
            dtype='int16',
        ),
        ParquetColumn(
            name='sector_index',
            arrow_type=pa.int8(),
            unit='count',
            rate_hz=None,
            event=True,
            dtype='int8',
        ),
        ParquetColumn(
            name='lap_time',
            arrow_type=pa.float32(),
            unit='s',
            rate_hz=None,
            event=True,
            dtype='float32',
        ),
        ParquetColumn(
            name='sector_time',
            arrow_type=pa.float32(),
            unit='s',
            rate_hz=None,
            event=True,
            dtype='float32',
        ),
        ParquetColumn(
            name='delta',
            arrow_type=pa.float32(),
            unit='s',
            rate_hz=None,
            event=True,
            dtype='float32',
        ),
    ),
)

DRIVER: Final[ParquetGroupSchema] = ParquetGroupSchema(
    group='driver',
    rate_hz=100.0,
    event=False,
    columns=(
        ParquetColumn(
            name='brake_pressure',
            arrow_type=pa.float32(),
            unit='bar',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='steering_angle',
            arrow_type=pa.float32(),
            unit='deg',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
        ParquetColumn(
            name='pedal_travel',
            arrow_type=pa.float32(),
            unit='%',
            rate_hz=100.0,
            event=False,
            dtype='float32',
        ),
    ),
)


PARQUET_SCHEMAS: Final[Mapping[str, ParquetGroupSchema]] = MappingProxyType(
{
    'chassis': CHASSIS,
    'imu': IMU,
    'powertrain': POWERTRAIN,
    'aero': AERO,
    'wheel': WHEEL,
    'wheel_thermal': WHEEL_THERMAL,
    'thermal': THERMAL,
    'session': SESSION,
    'driver': DRIVER,
})



def chassis_schema() -> ParquetGroupSchema:
    return CHASSIS


def chassis_arrow_schema() -> pa.Schema:
    return CHASSIS.arrow_schema()


def imu_schema() -> ParquetGroupSchema:
    return IMU


def imu_arrow_schema() -> pa.Schema:
    return IMU.arrow_schema()


def powertrain_schema() -> ParquetGroupSchema:
    return POWERTRAIN


def powertrain_arrow_schema() -> pa.Schema:
    return POWERTRAIN.arrow_schema()


def aero_schema() -> ParquetGroupSchema:
    return AERO


def aero_arrow_schema() -> pa.Schema:
    return AERO.arrow_schema()


def wheel_schema() -> ParquetGroupSchema:
    return WHEEL


def wheel_arrow_schema() -> pa.Schema:
    return WHEEL.arrow_schema()


def wheel_thermal_schema() -> ParquetGroupSchema:
    return WHEEL_THERMAL


def wheel_thermal_arrow_schema() -> pa.Schema:
    return WHEEL_THERMAL.arrow_schema()


def thermal_schema() -> ParquetGroupSchema:
    return THERMAL


def thermal_arrow_schema() -> pa.Schema:
    return THERMAL.arrow_schema()


def session_schema() -> ParquetGroupSchema:
    return SESSION


def session_arrow_schema() -> pa.Schema:
    return SESSION.arrow_schema()


def driver_schema() -> ParquetGroupSchema:
    return DRIVER


def driver_arrow_schema() -> pa.Schema:
    return DRIVER.arrow_schema()


def parquet_group_schemas() -> Mapping[str, ParquetGroupSchema]:
    return PARQUET_SCHEMAS


def parquet_arrow_schema(group: str) -> pa.Schema:
    return PARQUET_SCHEMAS[group].arrow_schema()
