import { memo } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

export const Tile = memo(function Tile({ name }: { name: string }) {
  const value = useTelemetryStore((state) =>
    state.frame === null ? undefined : state.frame.channels[name],
  );
  const diagnostics = useTelemetryStore((state) => state.diagnostics);
  const channelDiagnostics = diagnostics === null ? undefined : diagnostics.channels[name];
  return (
    <li
      className="tile"
      data-channel={name}
      data-sample-count={channelDiagnostics === undefined ? 0 : channelDiagnostics.count}
      data-declared-hz={channelDiagnostics?.declaredHz ?? ''}
    >
      <span className="tile__name" title={name}>
        {name}
      </span>
      <span className="tile__value">{value === undefined ? '-' : formatValue(value)}</span>
      <span className="tile__meta">
        {channelDiagnostics === undefined
          ? 'no samples yet'
          : `${channelDiagnostics.declaredHz === null ? 'auto' : `${channelDiagnostics.declaredHz} Hz`} declared · observed ${channelDiagnostics.observedHz.toFixed(1)} Hz (sample timestamps) · ${channelDiagnostics.count} samples`}
      </span>
    </li>
  );
});
