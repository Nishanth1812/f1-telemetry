const THOUSANDS = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

export function formatValue(value: number): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const magnitude = Math.abs(value);
  if (magnitude === 0) {
    return '0';
  }
  if (magnitude >= 10_000) {
    return THOUSANDS.format(value);
  }
  if (magnitude >= 100) {
    return value.toFixed(1);
  }
  if (magnitude >= 1) {
    return value.toFixed(2);
  }
  if (magnitude >= 0.01) {
    return value.toFixed(3);
  }
  return value.toExponential(1);
}

export function formatDurationMs(milliseconds: number): string {
  if (milliseconds < 1_000) {
    return `${Math.max(0, Math.round(milliseconds))} ms`;
  }
  return `${(milliseconds / 1_000).toFixed(1)} s`;
}
