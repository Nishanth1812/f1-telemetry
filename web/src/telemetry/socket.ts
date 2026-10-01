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
    this.connect();
  }

  stop(): void {
    this.running = false;
    this.attempt = 0;
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
    };
    socket.onmessage = (event: MessageEvent<unknown>) => {
      if (!this.running || this.socket !== socket) {
        return;
      }
      const frame = decodeFrame(parsePayload(event.data));
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
      this.scheduleRetry(event.reason === '' ? null : event.reason);
    };
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
