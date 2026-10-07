import { memo } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import { formatValue } from '../telemetry/format';

export const Tile = memo(function Tile({ name }: { name: string }) {
  const value = useTelemetryStore((state) =>
    state.frame === null ? undefined : state.frame.channels[name],
  );
  return (
    <li className="tile">
      <span className="tile__name" title={name}>
        {name}
      </span>
      <span className="tile__value">{value === undefined ? '—' : formatValue(value)}</span>
    </li>
  );
});
