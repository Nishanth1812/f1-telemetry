import { decodeFrame, type TelemetryFrame } from './types';

const BASE_DELAY_MS = 500;
const MAX_DELAY_MS = 10_000;
const BACKOFF_FACTOR = 2;
const JITTER = 0.25;

export interface TelemetrySocketCallbacks {
  onConnecting: () => void;
  onOpen: () => void;
  onFrame: (frame: TelemetryFrame) => void;
  onMalformed: (count: number) => void;
  onRetry: (attempt: number, delayMs: number, reason: string | null) => void;
  onResult?: (result: ScenarioResult) => void;
  onReadOnly?: (readOnly: boolean) => void;
}

export type LabScenarioId = 'launch' | 'braking' | 'cornering';

export interface ScenarioResult {
  type: 'scenario_result';
  run_id?: string;
  playback_elapsed_s?: number;
  request_id: string;
  scenario_id: LabScenarioId;
  title: string;
  description: string;
  setting_name: string;
  setting: number;
  duration_s: number;
  peak_speed_kmh: number;
  peak_longitudinal_g: number;
  peak_lateral_g: number;
  simulation_label: string;
  points: Array<{
    time_s: number;
    speed_kmh: number;
    accel_longitudinal_m_s2: number;
    accel_lateral_m_s2: number;
  }>;
}

interface ScenarioErrorResponse {
  type: 'scenario_error';
  request_id: string | null;
  error?: string;
}

interface SavedRunsResponse {
  type: 'runs_list_result';
  request_id: string;
  runs: ScenarioResult[];
}

type CommandSuccess = ScenarioResult | SavedRunsResponse;
type ScenarioResponse = CommandSuccess | ScenarioErrorResponse;

interface PendingScenario {
  resolve: (result: CommandSuccess) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const SCENARIO_TIMEOUT_MS = 120_000;
let activeSocket: TelemetrySocket | null = null;
let requestSequence = 0;

export function requestScenarioRun(
  scenarioId: LabScenarioId,
  setting: number,
): Promise<ScenarioResult> {
  if (activeSocket === null) {
    return Promise.reject(new Error('Connect to the telemetry server before running a scenario.'));
  }
  return activeSocket.runScenario(scenarioId, setting);
}

export function setDetailedTelemetry(enabled: boolean): void {
  activeSocket?.setDetailedTelemetry(enabled);
}

export function requestSavedRuns(): Promise<ScenarioResult[]> {
  if (activeSocket === null) {
    return Promise.reject(new Error('Connect to the telemetry server to load saved tests.'));
  }
  return activeSocket.listSavedRuns();
}

export function requestSavedReplay(runId: string): Promise<ScenarioResult> {
  if (activeSocket === null) {
    return Promise.reject(new Error('Connect to the telemetry server to replay a test.'));
  }
  return activeSocket.replaySavedRun(runId);
}

export function backoffDelayMs(attempt: number, random: () => number = Math.random): number {
  const exponential = BASE_DELAY_MS * BACKOFF_FACTOR ** Math.max(0, attempt - 1);
  const jittered = exponential * (1 - JITTER + random() * 2 * JITTER);
  return Math.min(MAX_DELAY_MS, Math.round(jittered));
}

export class TelemetrySocket {
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private attempt = 0;
  private readonly pendingScenarios = new Map<string, PendingScenario>();
  private detailedTelemetry = false;

  constructor(
    private readonly url: string,
    private readonly callbacks: TelemetrySocketCallbacks,
  ) {}

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    this.attempt = 0;
    activeSocket = this;
    this.connect();
  }

  stop(): void {
    this.running = false;
    this.attempt = 0;
    if (activeSocket === this) {
      activeSocket = null;
    }
    for (const [requestId, pending] of this.pendingScenarios) {
      clearTimeout(pending.timer);
      pending.reject(new Error('Connection closed before the scenario finished.'));
      this.pendingScenarios.delete(requestId);
    }
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const socket = this.socket;
    this.socket = null;
    if (socket === null) {
      return;
    }
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close();
    }
  }

  runScenario(scenarioId: LabScenarioId, setting: number): Promise<ScenarioResult> {
    return this.request<ScenarioResult>(
      { type: 'scenario_run', scenario_id: scenarioId, setting }, 'scenario_result',
    );
  }

  listSavedRuns(): Promise<ScenarioResult[]> {
    return this.request<SavedRunsResponse>({ type: 'runs_list' }, 'runs_list_result')
      .then((response) => response.runs);
  }

  replaySavedRun(runId: string): Promise<ScenarioResult> {
    return this.request<ScenarioResult>(
      { type: 'run_replay', run_id: runId }, 'scenario_result',
    );
  }

  private request<T extends CommandSuccess>(
    body: Record<string, unknown>, expectedType: T['type'],
  ): Promise<T> {
    const socket = this.socket;
    if (socket === null || socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('Connect to the telemetry server before running a scenario.'));
    }
    const requestId = `lab-${Date.now()}-${requestSequence++}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingScenarios.delete(requestId);
        reject(new Error('The run took too long. Check that the server is still running.'));
      }, SCENARIO_TIMEOUT_MS);
      this.pendingScenarios.set(requestId, {
        resolve: (response) => {
          if (response.type !== expectedType) {
            reject(new Error('The server returned an unexpected response.'));
          } else {
            resolve(response as T);
          }
        },
        reject,
        timer,
      });
      try {
        socket.send(
          JSON.stringify({
            ...body,
            request_id: requestId,
          }),
        );
      } catch (error) {
        clearTimeout(timer);
        this.pendingScenarios.delete(requestId);
        reject(error instanceof Error ? error : new Error('Could not send the run request.'));
      }
    });
  }

  setDetailedTelemetry(enabled: boolean): void {
    this.detailedTelemetry = enabled;
    const socket = this.socket;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'telemetry_detail', enabled }));
    }
  }

  private connect(): void {
    if (!this.running) {
      return;
    }
    if (this.socket !== null) {
      const state = this.socket.readyState;
      if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) {
        return;
      }
      this.socket = null;
    }
    this.callbacks.onConnecting();
    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url);
    } catch (error) {
      this.scheduleRetry(error instanceof Error ? error.message : 'socket could not be created');
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (!this.running || this.socket !== socket) {
        return;
      }
      this.attempt = 0;
      this.callbacks.onOpen();
      socket.send(JSON.stringify({ type: 'telemetry_detail', enabled: this.detailedTelemetry }));
    };
    socket.onmessage = (event: MessageEvent<unknown>) => {
      if (!this.running || this.socket !== socket) {
        return;
      }
      const payload = parsePayload(event.data);
      if (payload !== null && typeof payload === 'object' && 'type' in payload && payload.type === 'server_info') {
        const info = payload as Record<string, unknown>;
        if (typeof info.read_only === 'boolean' &&
            (info.result === null || (isScenarioResponse(info.result) && info.result.type === 'scenario_result'))) {
          this.callbacks.onReadOnly?.(info.read_only);
          if (info.result !== null) this.callbacks.onResult?.(info.result as ScenarioResult);
        } else {
          this.callbacks.onMalformed(1);
        }
        return;
      }
      if (isScenarioResponse(payload)) {
        this.resolveScenario(payload);
        if (payload.type === 'scenario_result') this.callbacks.onResult?.(payload);
        return;
      }
      if (payload !== null && typeof payload === 'object') {
        const badReply = payload as Record<string, unknown>;
        if (typeof badReply.request_id === 'string' && this.pendingScenarios.has(badReply.request_id)) {
          const pending = this.pendingScenarios.get(badReply.request_id)!;
          clearTimeout(pending.timer);
          this.pendingScenarios.delete(badReply.request_id);
          pending.reject(new Error('The server returned an invalid response.'));
          return;
        }
      }
      const frame = decodeFrame(payload);
      if (frame === null) {
        this.callbacks.onMalformed(1);
        return;
      }
      this.callbacks.onFrame(frame);
    };
    socket.onerror = () => {
      if (!this.running || this.socket !== socket) {
        return;
      }
      socket.close();
    };
    socket.onclose = (event: CloseEvent) => {
      if (!this.running || this.socket !== socket) {
        return;
      }
      this.socket = null;
      this.rejectPendingScenarios(new Error('Connection lost before the scenario finished.'));
      this.scheduleRetry(event.reason === '' ? null : event.reason);
    };
  }

  private resolveScenario(response: ScenarioResponse): void {
    if (response.request_id === null) {
      return;
    }
    const pending = this.pendingScenarios.get(response.request_id);
    if (pending === undefined) {
      return;
    }
    clearTimeout(pending.timer);
    this.pendingScenarios.delete(response.request_id);
    if (response.type === 'scenario_error') {
      pending.reject(
        new Error(typeof response.error === 'string' ? response.error : 'The run could not finish.'),
      );
    } else {
      pending.resolve(response);
    }
  }

  private rejectPendingScenarios(error: Error): void {
    for (const [requestId, pending] of this.pendingScenarios) {
      clearTimeout(pending.timer);
      pending.reject(error);
      this.pendingScenarios.delete(requestId);
    }
  }

  private scheduleRetry(reason: string | null): void {
    this.attempt += 1;
    const delayMs = backoffDelayMs(this.attempt);
    this.callbacks.onRetry(this.attempt, delayMs, reason);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.connect();
    }, delayMs);
  }
}

function isScenarioResponse(value: unknown): value is ScenarioResponse {
  if (value === null || typeof value !== 'object') {
    return false;
  }
  const response = value as Record<string, unknown>;
  if (response.type === 'runs_list_result') {
    return typeof response.request_id === 'string' && Array.isArray(response.runs) &&
      response.runs.every((run: unknown) => {
        if (!isScenarioResponse(run) || run.type !== 'scenario_result') return false;
        return typeof run.run_id === 'string' && /^[a-f0-9]{32}$/.test(run.run_id);
      });
  }
  if (response.type === 'scenario_error') {
    return (
      (typeof response.request_id === 'string' || response.request_id === null) &&
      (response.error === undefined || typeof response.error === 'string')
    );
  }
  if (
    response.type !== 'scenario_result' ||
    (response.playback_elapsed_s !== undefined &&
      (!isFiniteNumber(response.playback_elapsed_s) || response.playback_elapsed_s < 0)) ||
    (response.run_id !== undefined &&
      (typeof response.run_id !== 'string' || !/^[a-f0-9]{32}$/.test(response.run_id))) ||
    typeof response.request_id !== 'string' ||
    !['launch', 'braking', 'cornering'].includes(String(response.scenario_id)) ||
    typeof response.title !== 'string' ||
    typeof response.description !== 'string' ||
    typeof response.setting_name !== 'string' ||
    typeof response.simulation_label !== 'string' ||
    !isFiniteNumber(response.setting) ||
    !isFiniteNumber(response.duration_s) ||
    !isFiniteNumber(response.peak_speed_kmh) ||
    !isFiniteNumber(response.peak_longitudinal_g) ||
    !isFiniteNumber(response.peak_lateral_g) ||
    !Array.isArray(response.points) ||
    response.points.length === 0 ||
    response.points.length > 120
  ) {
    return false;
  }
  return response.points.every((point: unknown) => {
    if (point === null || typeof point !== 'object') {
      return false;
    }
    const sample = point as Record<string, unknown>;
    return (
      isFiniteNumber(sample.time_s) &&
      isFiniteNumber(sample.speed_kmh) &&
      isFiniteNumber(sample.accel_longitudinal_m_s2) &&
      isFiniteNumber(sample.accel_lateral_m_s2)
    );
  });
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function parsePayload(data: unknown): unknown {
  if (typeof data !== 'string') {
    return null;
  }
  try {
    return JSON.parse(data) as unknown;
  } catch {
    return null;
  }
}
