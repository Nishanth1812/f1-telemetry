import type { LabScenarioId } from '../telemetry/socket';

export interface LabScenario {
  id: LabScenarioId;
  title: string;
  detail: string;
  story: string;
  setting: string;
  min: number;
  max: number;
  unit: string;
}

export const SCENARIOS: LabScenario[] = [
  {
    id: 'launch',
    title: 'Standing launch',
    detail: 'See how throttle changes the first seconds from a standstill.',
    story: 'The car starts from a standstill and accelerates for 7 seconds. More throttle means harder acceleration.',
    setting: 'Throttle',
    min: 50,
    max: 100,
    unit: '%',
  },
  {
    id: 'braking',
    title: 'Braking run',
    detail: 'Compare the speed trace with a lighter or stronger brake request.',
    story: 'The car is rolling at about 90 km/h, then the brakes come on. A stronger brake request stops it faster.',
    setting: 'Brake',
    min: 25,
    max: 100,
    unit: '%',
  },
  {
    id: 'cornering',
    title: 'Constant-radius corner',
    detail: 'Change steering around a fixed 50 m radius scenario.',
    story: 'The car drives around a 50 m radius circle at about 72 km/h. More steering tightens the turn, so the car is pushed harder sideways.',
    setting: 'Steering',
    min: 50,
    max: 150,
    unit: '% of baseline',
  },
];

export const SCENARIO_BY_ID = Object.fromEntries(
  SCENARIOS.map((item) => [item.id, item]),
) as Record<LabScenarioId, LabScenario>;

export const DEFAULT_SETTING: Record<LabScenarioId, number> = {
  launch: 100,
  braking: 100,
  cornering: 100,
};

export function formatSetting(scenario: LabScenario, value: number): string {
  return scenario.unit === '% of baseline' ? `${value}% of baseline` : `${value}%`;
}

export const STANDARD_GRAVITY = 9.80665;

/** Above this sideways load the simulated car is turning or sliding. */
export const SLIDE_WARNING_G = 0.3;

/**
 * Approximate drive length in seconds, used only to time the progress chip.
 * The exact duration arrives with the result and is shown from there on.
 */
export const APPROX_DURATION_S: Record<LabScenarioId, number> = {
  launch: 7.0,
  braking: 1.6,
  cornering: 1.0,
};
