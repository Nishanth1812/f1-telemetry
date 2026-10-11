/*
 * Rule-based car health checks. Pure functions only: no store, no React.
 *
 * Every threshold below is an uncalibrated placeholder chosen for readable
 * demo output. None of them are FIA or team limits.
 */

// Placeholder: wheel slip ratio (percent) at which spin or lock is worth flagging.
export const SLIP_WARN_PCT = 10;
// Placeholder: wheel slip ratio (percent) above which the warning is strong.
export const SLIP_STRONG_PCT = 20;
// Placeholder: tyre temperature (°C) at which the tyre is near its working limit.
export const TYRE_WARN_C = 100;
// Placeholder: tyre temperature (°C) above which the warning is strong.
export const TYRE_STRONG_C = 115;
// Placeholder: brake disc temperature (°C) at which the disc is very hot.
export const BRAKE_WARN_C = 700;
// Placeholder: brake disc temperature (°C) above which the warning is strong.
export const BRAKE_STRONG_C = 900;
// Placeholder: yaw rate (deg/s) that is unusual without steering input.
export const YAW_WARN_DEG_S = 8;
// Placeholder: yaw rate (deg/s) that is a strong sign of a slide.
export const YAW_STRONG_DEG_S = 25;
// Placeholder: steering angle (deg) treated as "not steering".
export const STRAIGHT_STEER_DEG = 2;
// Placeholder: speed (km/h) below which the stability check is not applied.
export const STABILITY_MIN_SPEED_KMH = 30;
// Placeholder: engine speed (RPM) at which the engine is at its rev limit.
export const REV_LIMIT_RPM = 11_800;
// Placeholder: engine temperature (°C) at which the engine is running hot.
export const ENGINE_HOT_C = 120;

export type HealthStatus = 'ok' | 'warn' | 'nodata';

export interface HealthCheck {
  id: string;
  title: string;
  status: HealthStatus;
  message: string;
  detail: string;
  /** 0 = ok or no data, 1 = warning, 2 = strong warning. */
  severity: number;
}

interface Corner {
  key: 'fl' | 'fr' | 'rl' | 'rr';
  label: string;
}

const CORNERS: readonly Corner[] = [
  { key: 'fl', label: 'front left' },
  { key: 'fr', label: 'front right' },
  { key: 'rl', label: 'rear left' },
  { key: 'rr', label: 'rear right' },
];

type Channels = Readonly<Record<string, number>>;

function readFinite(channels: Channels, name: string): number | undefined {
  const value = channels[name];
  return value !== undefined && Number.isFinite(value) ? value : undefined;
}

/** Readings for the corners that have a finite value, in fl, fr, rl, rr order. */
function cornerReadings(channels: Channels, prefix: string): Reading[] {
  const readings: Reading[] = [];
  for (const corner of CORNERS) {
    const value = readFinite(channels, `${prefix}_${corner.key}`);
    if (value !== undefined) {
      readings.push({ corner, value });
    }
  }
  return readings;
}

type Reading = { corner: Corner; value: number };

/** The reading with the highest score, ties going to the earlier corner. */
function pickWorst(readings: Reading[], score: (reading: Reading) => number): Reading | undefined {
  let worst: Reading | undefined;
  for (const reading of readings) {
    if (worst === undefined || score(reading) > score(worst)) {
      worst = reading;
    }
  }
  return worst;
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function oneDecimal(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Object.is(rounded, -0) ? '0.0' : rounded.toFixed(1);
}

function wholeNumber(value: number): string {
  const rounded = Math.round(value);
  return Object.is(rounded, -0) ? '0' : String(rounded);
}

function nodata(id: string, title: string, missing: string): HealthCheck {
  return {
    id,
    title,
    status: 'nodata',
    message: 'No data for this check yet',
    detail: `Waiting for ${missing}`,
    severity: 0,
  };
}

function traction(channels: Channels): HealthCheck {
  const id = 'traction';
  const title = 'Traction';
  const readings = cornerReadings(channels, 'slip_ratio');
  const worst = pickWorst(readings, (reading) => Math.abs(reading.value));
  if (worst === undefined) {
    return nodata(id, title, 'wheel slip readings');
  }
  const magnitude = Math.abs(worst.value);
  const detail = `worst: ${worst.corner.label} ${oneDecimal(magnitude)} %`;
  if (magnitude >= SLIP_WARN_PCT) {
    return {
      id,
      title,
      status: 'warn',
      message: `Wheels are spinning or locking: ${worst.corner.label} slip ${oneDecimal(magnitude)}%`,
      detail,
      severity: magnitude > SLIP_STRONG_PCT ? 2 : 1,
    };
  }
  return {
    id,
    title,
    status: 'ok',
    message: 'Tyres are gripping normally',
    detail,
    severity: 0,
  };
}

function tyreTemperature(channels: Channels): HealthCheck {
  const id = 'tyre-temp';
  const title = 'Tyre temperature';
  const readings = cornerReadings(channels, 'tyre_temp');
  const hottest = pickWorst(readings, (reading) => reading.value);
  if (hottest === undefined) {
    return nodata(id, title, 'tyre temperatures');
  }
  const detail = `hottest: ${hottest.corner.label} ${wholeNumber(hottest.value)} °C`;
  if (hottest.value >= TYRE_WARN_C) {
    return {
      id,
      title,
      status: 'warn',
      message: `${capitalise(hottest.corner.label)} tyre is approaching its thermal limit (${wholeNumber(hottest.value)} °C)`,
      detail,
      severity: hottest.value >= TYRE_STRONG_C ? 2 : 1,
    };
  }
  return {
    id,
    title,
    status: 'ok',
    message: 'Tyres are in a safe temperature range',
    detail,
    severity: 0,
  };
}

function brakeTemperature(channels: Channels): HealthCheck {
  const id = 'brake-temp';
  const title = 'Brake temperature';
  const readings = cornerReadings(channels, 'brake_temp');
  const hottest = pickWorst(readings, (reading) => reading.value);
  if (hottest === undefined) {
    return nodata(id, title, 'brake temperatures');
  }
  const detail = `hottest: ${hottest.corner.label} ${wholeNumber(hottest.value)} °C`;
  if (hottest.value >= BRAKE_WARN_C) {
    return {
      id,
      title,
      status: 'warn',
      message: `${capitalise(hottest.corner.label)} brake disc is very hot`,
      detail,
      severity: hottest.value >= BRAKE_STRONG_C ? 2 : 1,
    };
  }
  return {
    id,
    title,
    status: 'ok',
    message: 'Brake discs are in a normal temperature range',
    detail,
    severity: 0,
  };
}

function stability(channels: Channels): HealthCheck {
  const id = 'stability';
  const title = 'Stability';
  const yaw = readFinite(channels, 'yaw_rate');
  const steer = readFinite(channels, 'steering_angle');
  const speed = readFinite(channels, 'speed');
  if (yaw === undefined || steer === undefined || speed === undefined) {
    return nodata(id, title, 'yaw rate, steering angle and speed');
  }
  const detail = `yaw rate ${oneDecimal(yaw)} deg/s, steering ${oneDecimal(steer)} deg, speed ${wholeNumber(speed)} km/h`;
  const sliding =
    Math.abs(yaw) > YAW_WARN_DEG_S &&
    Math.abs(steer) < STRAIGHT_STEER_DEG &&
    speed > STABILITY_MIN_SPEED_KMH;
  if (sliding) {
    return {
      id,
      title,
      status: 'warn',
      message: 'The car is rotating without steering input - it is sliding',
      detail,
      severity: Math.abs(yaw) > YAW_STRONG_DEG_S ? 2 : 1,
    };
  }
  return {
    id,
    title,
    status: 'ok',
    message: 'The car is tracking the way it is steered',
    detail,
    severity: 0,
  };
}

function powertrain(channels: Channels): HealthCheck {
  const id = 'powertrain';
  const title = 'Powertrain';
  const rpm = readFinite(channels, 'ice_rpm');
  const engineTemp = readFinite(channels, 'engine_temp');
  if (rpm === undefined && engineTemp === undefined) {
    return nodata(id, title, 'engine speed and engine temperature');
  }
  const parts: string[] = [];
  if (rpm !== undefined) {
    parts.push(`engine ${wholeNumber(rpm)} RPM`);
  }
  if (engineTemp !== undefined) {
    parts.push(`engine temperature ${wholeNumber(engineTemp)} °C`);
  }
  const detail = parts.join(', ');
  if (rpm !== undefined && rpm > REV_LIMIT_RPM) {
    return { id, title, status: 'warn', message: 'Engine is at the rev limit', detail, severity: 1 };
  }
  if (engineTemp !== undefined && engineTemp > ENGINE_HOT_C) {
    return { id, title, status: 'warn', message: 'Engine is running hot', detail, severity: 1 };
  }
  if (rpm === undefined || engineTemp === undefined) {
    // One of the two inputs is missing, so "normal" cannot be claimed.
    return nodata(id, title, rpm === undefined ? 'engine speed' : 'engine temperature');
  }
  return { id, title, status: 'ok', message: 'Powertrain behaviour normal', detail, severity: 0 };
}

/**
 * Runs every rule against the latest channel values. A missing channel never
 * produces a warning: the affected check reports "no data" instead.
 */
export function evaluateHealth(channels: Record<string, number>): HealthCheck[] {
  return [
    traction(channels),
    tyreTemperature(channels),
    brakeTemperature(channels),
    stability(channels),
    powertrain(channels),
  ];
}
