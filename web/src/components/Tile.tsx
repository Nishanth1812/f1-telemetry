import { memo } from 'react';
import { CHANNELS, type ChannelSpec } from '../generated/channels';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

/**
 * Plain-English names for the contract's channels. Only the channels whose raw
 * names are abbreviations or otherwise hard to read are listed; everything else
 * falls back to `humanise`, which turns `tyre_temp_fl` into "Tyre temp front left".
 * The raw name always stays in the `title` tooltip.
 */
export const CHANNEL_LABELS: Readonly<Record<string, string>> = {
  speed: 'Speed',
  vx: 'Forward speed',
  vy: 'Sideways speed',
  yaw_rate: 'Yaw rate (turning speed)',
  accel_lateral: 'Sideways acceleration',
  accel_longitudinal: 'Forward acceleration',
  ice_rpm: 'Engine speed (RPM)',
  ice_torque_nm: 'Engine torque (Nm)',
  mgu_k_rpm: 'Hybrid motor speed (RPM)',
  mgu_k_power_kw: 'Hybrid motor power (kW)',
  throttle_pct: 'Throttle (%)',
  clutch_pct: 'Clutch (%)',
  boost_remaining_bar: 'Boost left (bar)',
  eso_pct: 'Energy store charge (%)',
  fuel_flow_kg_h: 'Fuel flow (kg/h)',
  aero_mode: 'Aero mode',
  fw_flap_deg: 'Front wing flap angle',
  rw_flap_deg: 'Rear wing flap angle',
  zone_id: 'Track zone',
  downforce_n: 'Downforce (N)',
  cl_a: 'Lift coefficient',
  cd_a: 'Drag coefficient',
  lap_index: 'Lap number',
  sector_index: 'Current sector',
  lap_time: 'Lap time',
  sector_time: 'Sector time',
  delta: 'Gap to reference lap',
  brake_pressure: 'Brake pressure',
  steering_angle: 'Steering angle',
  pedal_travel: 'Pedal travel',
};

const TOKEN_WORDS: Readonly<Record<string, string>> = {
  fl: 'front left',
  fr: 'front right',
  rl: 'rear left',
  rr: 'rear right',
  pct: '%',
  imu: 'IMU',
  rpm: 'RPM',
  kw: 'kW',
  nm: 'Nm',
  ice: 'engine',
  mgu: 'MGU',
  eso: 'ESO',
};

function humanise(name: string): string {
  const words = name.split('_').map((token) => TOKEN_WORDS[token] ?? token);
  const text = words.join(' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Plain-English name for a channel, e.g. `wheel_speed_fl` → "Wheel speed front left". */
export function channelLabel(name: string): string {
  return CHANNEL_LABELS[name] ?? humanise(name);
}

/** Plain-English name for a fault type code, e.g. `dropout` → "Signal dropout". */
export function faultLabel(faultType: string): string {
  return FAULT_LABELS[faultType] ?? humanise(faultType);
}

const FAULT_LABELS: Readonly<Record<string, string>> = {
  dropout: 'Signal dropout',
  freeze: 'Frozen reading',
  spike: 'Spike',
  step: 'Sudden shift',
  gain: 'Gain error',
  noise: 'Extra noise',
  quantise: 'Coarse rounding',
  swap: 'Swapped with another sensor',
  saturate: 'Clipped at a limit',
  stale: 'Stale (old value repeated)',
};

export const Tile = memo(function Tile({ name, hint }: { name: string; hint?: string }) {
  const value = useTelemetryStore((state) =>
    state.frame === null ? undefined : state.frame.channels[name],
  );
  const spec = (CHANNELS as Record<string, ChannelSpec | undefined>)[name];
  const sampleRate = spec?.rateHz;
  return (
    <li
      className="tile"
      data-channel={name}
      data-declared-hz={sampleRate ?? ''}
    >
      <span className="tile__name" title={name}>
        {channelLabel(name)}
      </span>
      <span
        className="tile__value"
        title={value === undefined ? 'No reading received yet' : undefined}
      >
        {value === undefined ? '—' : formatValue(value)}
      </span>
      <span className="tile__meta">
        {sampleRate === null || sampleRate === undefined
          ? 'Sent only when an event happens'
          : `Updates ${sampleRate} times a second`}
      </span>
      {hint !== undefined ? (
        <span className="tile__hint-caption" title={hint} aria-label={hint}>
          {hint}
        </span>
      ) : null}
    </li>
  );
});
