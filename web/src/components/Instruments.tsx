import '../styles/instruments.css';
import { memo, useId } from 'react';
import { CHANNELS, type ChannelSpec } from '../generated/channels';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

const STANDARD_GRAVITY_MS2 = 9.80665;

const SPECS = CHANNELS as Record<string, ChannelSpec | undefined>;

function unitOf(name: string, fallback: string): string {
  return SPECS[name]?.unit ?? fallback;
}

/** Upper end of a channel's contract range, used to scale the thin bars. */
function rangeMax(name: string): number {
  return SPECS[name]?.range[1] ?? 1;
}

/** Each cell reads only its own channel, so a change elsewhere does not re-render it. */
function useChannel(name: string): number | undefined {
  return useTelemetryStore((state) =>
    state.frame === null ? undefined : state.frame.channels[name],
  );
}

/** Signed number with the sign shown only when it carries information. */
function signed(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude < 0.01) {
    return '0';
  }
  return `${value > 0 ? '+' : '−'}${formatValue(magnitude)}`;
}

/** Acceleration in g, two decimals, signed. */
function signedG(accelMs2: number): string {
  const g = Math.round((accelMs2 / STANDARD_GRAVITY_MS2) * 100) / 100;
  if (g === 0) {
    return '0.00';
  }
  return `${g > 0 ? '+' : '−'}${Math.abs(g).toFixed(2)}`;
}

function gearText(value: number): string {
  const gear = Math.round(value);
  if (gear === 0) {
    return 'N';
  }
  if (gear === -1) {
    return 'R';
  }
  return String(gear);
}

function fractionOf(value: number | undefined, max: number): number | undefined {
  if (value === undefined || !Number.isFinite(value) || max <= 0) {
    return undefined;
  }
  return Math.min(1, Math.max(0, value / max));
}

interface CellProps {
  channel: string;
  label: string;
  text: string;
  missing: boolean;
  unit?: string;
  fraction?: number;
  caption?: string;
  modifier?: string;
}

const Cell = memo(function Cell({
  channel,
  label,
  text,
  missing,
  unit,
  fraction,
  caption,
  modifier,
}: CellProps) {
  const barWidth = fraction === undefined || missing ? 0 : fraction * 100;
  return (
    <li
      className={modifier === undefined ? 'instrument' : `instrument ${modifier}`}
      data-channel={channel}
      title={channel}
    >
      <span className="instrument__label">{label}</span>
      <span className="instrument__readout">
        <span className={missing ? 'instrument__value instrument__value--missing' : 'instrument__value'}>
          {missing ? '—' : text}
        </span>
        {unit !== undefined && !missing ? <span className="instrument__unit">{unit}</span> : null}
      </span>
      {fraction !== undefined ? (
        <span className="instrument__bar" aria-hidden="true">
          <span className="instrument__bar-fill" style={{ width: `${barWidth}%` }} />
        </span>
      ) : null}
      {caption !== undefined ? <span className="instrument__caption">{caption}</span> : null}
    </li>
  );
});

const SpeedCell = memo(function SpeedCell() {
  const value = useChannel('speed');
  return (
    <Cell
      channel="speed"
      label="Speed"
      text={value === undefined ? '—' : formatValue(value)}
      missing={value === undefined}
      unit={unitOf('speed', 'km/h')}
    />
  );
});

const RpmCell = memo(function RpmCell() {
  const value = useChannel('ice_rpm');
  const max = rangeMax('ice_rpm');
  return (
    <Cell
      channel="ice_rpm"
      label="Engine RPM"
      text={value === undefined ? '—' : formatValue(value)}
      missing={value === undefined}
      unit={unitOf('ice_rpm', 'rpm')}
      fraction={fractionOf(value, max)}
    />
  );
});

const GearCell = memo(function GearCell() {
  const value = useChannel('gear');
  return (
    <Cell
      channel="gear"
      label="Gear"
      text={value === undefined ? '—' : gearText(value)}
      missing={value === undefined}
      modifier="instrument--gear"
    />
  );
});

const ThrottleCell = memo(function ThrottleCell() {
  const value = useChannel('throttle_pct');
  const max = rangeMax('throttle_pct');
  return (
    <Cell
      channel="throttle_pct"
      label="Throttle"
      text={value === undefined ? '—' : formatValue(value)}
      missing={value === undefined}
      unit={unitOf('throttle_pct', '%')}
      fraction={fractionOf(value, max)}
    />
  );
});

const SteeringCell = memo(function SteeringCell() {
  const value = useChannel('steering_angle');
  return (
    <Cell
      channel="steering_angle"
      label="Steering"
      text={value === undefined ? '—' : signed(value)}
      missing={value === undefined}
      unit={unitOf('steering_angle', 'deg')}
      caption="Positive is left"
    />
  );
});

const LateralGCell = memo(function LateralGCell() {
  const value = useChannel('accel_lateral');
  return (
    <Cell
      channel="accel_lateral"
      label="Lateral G"
      text={value === undefined ? '—' : signedG(value)}
      missing={value === undefined}
      unit="g"
      caption="sideways"
    />
  );
});

const LongitudinalGCell = memo(function LongitudinalGCell() {
  const value = useChannel('accel_longitudinal');
  return (
    <Cell
      channel="accel_longitudinal"
      label="Longitudinal G"
      text={value === undefined ? '—' : signedG(value)}
      missing={value === undefined}
      unit="g"
      caption="forward (+) or braking (−)"
    />
  );
});

const YawCell = memo(function YawCell() {
  const value = useChannel('yaw_rate');
  return (
    <Cell
      channel="yaw_rate"
      label="Yaw rate"
      text={value === undefined ? '—' : signed(value)}
      missing={value === undefined}
      unit={unitOf('yaw_rate', 'deg/s')}
    />
  );
});

export function Instruments() {
  const headingId = useId();
  return (
    <section className="instruments" aria-labelledby={headingId}>
      <header className="instruments__header">
        <h2 id={headingId} className="instruments__title">
          Live car data
        </h2>
        <p className="instruments__caption">
          Updates about 30 times a second from the simulated car. Start a virtual test to watch it
          drive.
        </p>
      </header>
      <ul className="instruments__strip" aria-live="off">
        <SpeedCell />
        <RpmCell />
        <GearCell />
        <ThrottleCell />
        <SteeringCell />
        <LateralGCell />
        <LongitudinalGCell />
        <YawCell />
      </ul>
    </section>
  );
}
