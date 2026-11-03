import { create } from 'zustand';
import {
  requestScenarioRun, requestSavedRuns, requestSavedReplay,
  type LabScenarioId, type ScenarioResult,
} from '../telemetry/socket';
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
  savedRuns: ScenarioResult[];
  loadingSaved: boolean;
  refreshSaved: () => Promise<void>;
  replaySaved: (runId: string) => Promise<void>;
  compareSaved: (runId: string, baselineId: string) => void;
  receiveResult: (result: ScenarioResult) => void;
  readOnly: boolean;
}

async function playRun(request: () => Promise<ScenarioResult>): Promise<void> {
  if (useLabStore.getState().running) return;
  useLabStore.setState({ running: true, error: null, startedAtMs: null });
  try {
    const result = await request();
    useLabStore.getState().receiveResult(result);
  } catch (reason) {
    useLabStore.setState({ error: reason instanceof Error ? reason.message : 'The run could not finish.' });
  } finally {
    useLabStore.setState({ running: false });
  }
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
  savedRuns: [],
  loadingSaved: false,
  readOnly: false,
  receiveResult: (result) => {
    if (get().result?.request_id === result.request_id && get().result?.run_id === result.run_id) return;
    useTelemetryStore.getState().resetForRun();
    set((state) => ({
      prevResult: state.lastRuns[result.scenario_id]?.run_id === result.run_id
        ? state.prevResult : state.lastRuns[result.scenario_id] ?? null,
      lastRuns: { ...state.lastRuns, [result.scenario_id]: result },
      result, scenarioId: result.scenario_id, setting: result.setting, error: null,
      startedAtMs: performance.now() - (result.playback_elapsed_s ?? 0) * 1000,
      savedRuns: result.run_id === undefined || state.savedRuns.some((run) => run.run_id === result.run_id)
        ? state.savedRuns : [result, ...state.savedRuns],
    }));
  },
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
  run: () => {
    const { scenarioId, setting } = get();
    return playRun(() => requestScenarioRun(scenarioId, setting));
  },
  replaySaved: (runId) => playRun(() => requestSavedReplay(runId)),
  refreshSaved: async () => {
    if (get().loadingSaved) return;
    set({ loadingSaved: true, error: null });
    try {
      set({ savedRuns: await requestSavedRuns() });
    } catch (reason) {
      set({ error: reason instanceof Error ? reason.message : 'Saved tests could not be loaded.' });
    } finally {
      set({ loadingSaved: false });
    }
  },
  compareSaved: (runId, baselineId) => {
    const result = get().savedRuns.find((run) => run.run_id === runId);
    const baseline = get().savedRuns.find((run) => run.run_id === baselineId);
    if (!result || !baseline || result.scenario_id !== baseline.scenario_id || runId === baselineId) {
      set({ error: 'Choose two different saved runs of the same test.' });
      return;
    }
    set({ result, prevResult: baseline, scenarioId: result.scenario_id, setting: result.setting, error: null });
  },
}));
