import '../styles/health.css';
import { useEffect, useRef, useState } from 'react';
import { useLabStore } from '../store/labStore';
import { useTelemetryStore } from '../store/telemetryStore';
import { evaluateHealth, type HealthCheck } from '../telemetry/health';

/** Minimum gap between two evaluations, so the panel renders at most ~10 times a second. */
const REFRESH_MS = 100;

type Tone = 'ok' | 'warn' | 'strong' | 'nodata';

const ICONS: Readonly<Record<Tone, string>> = {
  ok: '✓',
  warn: '⚠',
  strong: '⚠',
  nodata: '–',
};

const SCREEN_READER_PREFIX: Readonly<Record<Tone, string>> = {
  ok: 'OK: ',
  warn: 'Warning: ',
  strong: 'Strong warning: ',
  nodata: 'No data: ',
};

interface PanelView {
  checks: HealthCheck[];
  /** Highest-severity result per check id since the current test started. */
  worst: ReadonlyMap<string, HealthCheck>;
}

function toneOf(check: HealthCheck): Tone {
  if (check.status === 'nodata') {
    return 'nodata';
  }
  if (check.severity >= 2) {
    return 'strong';
  }
  return check.severity === 1 ? 'warn' : 'ok';
}

/** Keeps a check only if it is at least as bad as the one already remembered. */
function recordWorst(memory: Map<string, HealthCheck>, checks: HealthCheck[]): void {
  for (const check of checks) {
    if (check.status === 'nodata') {
      continue;
    }
    const previous = memory.get(check.id);
    if (previous === undefined || check.severity > previous.severity) {
      memory.set(check.id, check);
    }
  }
}

/**
 * Evaluates the rules from the live frame at most once per REFRESH_MS, and only
 * while the tab is visible. The store is read outside render, so frames that
 * arrive faster than the refresh rate do not re-render the panel.
 */
function usePanelView(): PanelView {
  const startedAtMs = useLabStore((state) => state.startedAtMs);
  const [view, setView] = useState<PanelView>(() => ({
    checks: evaluateHealth(useTelemetryStore.getState().frame?.channels ?? {}),
    worst: new Map(),
  }));
  const memoryRef = useRef<Map<string, HealthCheck>>(new Map());

  // A new test starts: forget the worst results from the previous one.
  useEffect(() => {
    memoryRef.current = new Map();
    setView((current) => (current.worst.size === 0 ? current : { ...current, worst: new Map() }));
  }, [startedAtMs]);

  useEffect(() => {
    let latest = useTelemetryStore.getState().frame?.channels;
    let dirty = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const isVisible = () => document.visibilityState !== 'hidden';

    const schedule = () => {
      if (timer === undefined && dirty && isVisible()) {
        timer = setTimeout(publish, REFRESH_MS);
      }
    };

    const publish = () => {
      timer = undefined;
      if (!isVisible()) {
        // Left unsent; the visibilitychange listener schedules it again.
        return;
      }
      dirty = false;
      const checks = evaluateHealth(latest ?? {});
      recordWorst(memoryRef.current, checks);
      setView({ checks, worst: new Map(memoryRef.current) });
    };

    const unsubscribe = useTelemetryStore.subscribe((state) => {
      const channels = state.frame?.channels;
      if (channels !== latest) {
        latest = channels;
        dirty = true;
        schedule();
      }
    });

    document.addEventListener('visibilitychange', schedule);
    schedule();

    return () => {
      unsubscribe();
      document.removeEventListener('visibilitychange', schedule);
      if (timer !== undefined) {
        clearTimeout(timer);
        timer = undefined;
      }
    };
  }, []);

  return view;
}

function HealthRow({ check, worst }: { check: HealthCheck; worst: HealthCheck | undefined }) {
  const tone = toneOf(check);
  const showWorst =
    worst !== undefined && worst.severity > 0 && worst.message !== check.message;
  return (
    <li
      className={`health__row health__row--${tone}`}
      data-check={check.id}
      data-status={check.status}
    >
      <span className="health__icon" aria-hidden="true">
        {ICONS[tone]}
      </span>
      <span className="health__sr">{SCREEN_READER_PREFIX[tone]}</span>
      <span className="health__body">
        <span className="health__title">{check.title}</span>
        <span className="health__message">{check.message}</span>
        <span className="health__detail">{check.detail}</span>
        {showWorst ? (
          <span className="health__worst">Worst this test: {worst.message}</span>
        ) : null}
      </span>
    </li>
  );
}

export function HealthPanel() {
  const view = usePanelView();
  return (
    <div className="health">
      <p className="health__intro">
        Simple rule checks on the live data. Machine-learning models are planned; thresholds here
        are placeholders.
      </p>
      <ul className="health__list" aria-live="off">
        {view.checks.map((check) => (
          <HealthRow key={check.id} check={check} worst={view.worst.get(check.id)} />
        ))}
      </ul>
    </div>
  );
}
