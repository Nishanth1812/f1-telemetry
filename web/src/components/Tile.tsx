import { memo } from 'react';
import { CHANNELS, type ChannelSpec } from '../generated/channels';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

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
        {name}
      </span>
      <span className="tile__value">{value === undefined ? '-' : formatValue(value)}</span>
      <span className="tile__meta">
        {sampleRate === null || sampleRate === undefined
          ? 'event-based channel'
          : `${sampleRate} Hz sensor`}
      </span>
      {hint !== undefined ? (
        <span className="tile__hint-caption" title={hint} aria-label={hint}>
          {hint}
        </span>
      ) : null}
    </li>
  );
});
