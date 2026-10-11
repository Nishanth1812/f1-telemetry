import { create } from 'zustand';
import { requestScenarioRun, type LabScenarioId, type ScenarioResult } from '../telemetry/socket';
import { useTelemetryStore } from './telemetryStore';
import { DEFAULT_SETTING } from './labScenarios';

export interface LabState {
  scenarioId: LabScenarioId;
  setting: number;
  /** True from the click until the server replies with the run summary. */
  running: boolean;
  /** Summary of the latest run, shown in "Last test result". */
  result: ScenarioResult | null;
  /** The previous run of the same scenario, used for the "What changed" line. */
  prevResult: ScenarioResult | null;
  lastRuns: Partial<Record<LabScenarioId, ScenarioResult>>;
  error: string | null;
  /** performance.now() when the run reply arrived; the live car starts driving then. */
  startedAtMs: number | null;
  chooseScenario: (next: LabScenarioId) => void;
  setSetting: (value: number) => void;
  run: () => Promise<void>;
}

export const useLabStore = create<LabState>((set, get) => ({
  scenarioId: 'launch',
  setting: DEFAULT_SETTING.launch,
  running: false,
  result: null,
  prevResult: null,
  lastRuns: {},
  error: null,
  startedAtMs: null,
  chooseScenario: (next) => {
    set({
      scenarioId: next,
      setting: DEFAULT_SETTING[next],
      result: null,
      error: null,
    });
  },
  setSetting: (value) => {
    // Changing the setting keeps the shown result until the next run replaces it.
    set({ setting: value });
  },
  run: async () => {
    if (get().running) {
      return;
    }
    const { scenarioId, setting } = get();
    set({ running: true, error: null, startedAtMs: null });
    try {
      const newResult = await requestScenarioRun(scenarioId, setting);
      // The server queues the run before it replies, so the car's first frames follow this
      // reply. Clearing now drops the held values from before the test, not the new run.
      useTelemetryStore.getState().resetForRun();
      set((state) => ({
        prevResult: state.lastRuns[scenarioId] ?? null,
        lastRuns: { ...state.lastRuns, [scenarioId]: newResult },
        result: newResult,
        startedAtMs: performance.now(),
      }));
    } catch (reason) {
      set({ error: reason instanceof Error ? reason.message : 'The run could not finish.' });
    } finally {
      set({ running: false });
    }
  },
}));
