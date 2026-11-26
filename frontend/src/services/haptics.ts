import type { GuidancePayload, SteeringAction } from '../types/telemetry';

export interface HapticPulseEvent {
  patternLabel: string;
  pattern: number[];
  nativeSupported: boolean;
  timestamp: number;
}

export type ForwardBlockKind = 'clear' | 'edge' | 'close' | 'contact' | 'curb';

export interface ForwardBlockState {
  blocked: boolean;
  distanceM: number;
  kind: ForwardBlockKind;
  reason: string;
}

export const DEFAULT_METAL_DETECTOR_STOP_M = 1.15;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Strict "Metal Detector" forward-cone blocking gate.
 *
 * Returns `blocked = true` ONLY when walking forward in the direction the phone
 * currently points is unsafe (`STOP`, `CURB_DROP`, covered lens, glass barrier,
 * overhead hazard, or a close obstacle directly inside the center walking beam).
 *
 * Peripheral-only obstacles (`VEER_LEFT` / `VEER_RIGHT` while `center` is clear)
 * and distant caution objects (`> stopRadiusM`) return `blocked = false` (100% silence)
 * so the user can sweep the phone radially around their body to find an open path.
 */
export function evaluateForwardBlock(
  guidance: GuidancePayload,
  stopRadiusM = DEFAULT_METAL_DETECTOR_STOP_M
): ForwardBlockState {
  const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
  if (isCurb) {
    return {
      blocked: true,
      distanceM: Math.min(guidance.center.min_distance_m, 0.8),
      kind: 'curb',
      reason: guidance.curb.stairs_likely ? 'STAIRS_DROP' : 'CURB_DROP',
    };
  }

  if (guidance.lens_blocked) {
    return {
      blocked: true,
      distanceM: 0.35,
      kind: 'contact',
      reason: 'LENS_BLOCKED',
    };
  }

  if (guidance.glass_barrier) {
    const d = guidance.ultrasonic_dist_m ?? guidance.center.min_distance_m;
    return {
      blocked: true,
      distanceM: d,
      kind: d <= 0.6 ? 'contact' : 'close',
      reason: 'GLASS_BARRIER',
    };
  }

  if (guidance.overhead_hazard && guidance.overall_severity === 'danger') {
    return {
      blocked: true,
      distanceM: Math.min(guidance.center.min_distance_m, 0.8),
      kind: 'close',
      reason: 'OVERHEAD_HAZARD',
    };
  }

  const isMetricModel = guidance.model_name.toLowerCase().includes('metric');
  const minAllDist = Math.min(
    guidance.left.min_distance_m,
    guidance.center.min_distance_m,
    guidance.right.min_distance_m
  );
  const centerDist = guidance.center.min_distance_m;

  const hardStop =
    guidance.wall_override === true ||
    (guidance.steering === 'STOP' &&
      (!isMetricModel || minAllDist <= stopRadiusM + 0.25));

  const centerBlocked = isMetricModel
    ? (guidance.center.severity === 'danger' && centerDist <= stopRadiusM + 0.2) ||
      (guidance.center.severity === 'caution' && centerDist <= stopRadiusM)
    : guidance.center.severity === 'danger' ||
      (guidance.center.severity === 'caution' && centerDist <= stopRadiusM);

  if (!hardStop && !centerBlocked) {
    return {
      blocked: false,
      distanceM: centerDist,
      kind: 'clear',
      reason: 'OPEN_PATH',
    };
  }

  const effectiveDist = centerBlocked
    ? centerDist
    : Math.min(centerDist, minAllDist);

  if (
    guidance.wall_override ||
    effectiveDist <= 0.55 ||
    (!isMetricModel && guidance.center.obstacle_score >= 0.78)
  ) {
    return {
      blocked: true,
      distanceM: effectiveDist,
      kind: 'contact',
      reason: 'STOP_CONTACT',
    };
  }
  if (
    effectiveDist <= 0.85 ||
    (!isMetricModel && guidance.center.obstacle_score >= 0.58)
  ) {
    return {
      blocked: true,
      distanceM: effectiveDist,
      kind: 'close',
      reason: 'OBSTACLE_CLOSE',
    };
  }
  return {
    blocked: true,
    distanceM: effectiveDist,
    kind: 'edge',
    reason: 'OBSTACLE_AHEAD',
  };
}

/**
 * Tactile scheduler supporting two distinct operating modes:
 * 1. Metal Detector Mode (default for blindfolded radial sweep):
 *    - 100% silent whenever the forward walking cone is open (zero vibration on CLEAR,
 *      distant caution, or peripheral-only side objects).
 *    - Calls `navigator.vibrate(0)` immediately on transition to clear for razor-sharp
 *      angular edges while sweeping around the body.
 *    - Continuous proximity-driven Geiger cadence (no stationary backoff) while pointing
 *      at a close blocking obstacle.
 * 2. Smart Calm / Directional Mode (legacy corridor guidance with asymmetric L/R nudges).
 */
export class HapticsEngine {
  private lastFiredAt = 0;
  private lastCooldownMs = 0;
  private lastRank = 0;
  private lastSteering: SteeringAction = 'CLEAR';
  private sameStateBuzzCount = 0;
  private minDistAtLastBuzz = 99;
  private calmBackoff = false;
  private metalDetectorMode = true;
  private stopRadiusM = DEFAULT_METAL_DETECTOR_STOP_M;
  private isCurrentlyBlocked = false;
  private enabled = true;
  private iosSwitchLabel: HTMLLabelElement | null = null;
  private onPulse?: (evt: HapticPulseEvent) => void;
  private onIosTactileFallback?: (rank: number) => void;

  constructor(
    onPulse?: (evt: HapticPulseEvent) => void,
    onIosTactileFallback?: (rank: number) => void
  ) {
    this.onPulse = onPulse;
    this.onIosTactileFallback = onIosTactileFallback;
  }

  public setCalmBackoff(calm: boolean): void {
    this.calmBackoff = calm;
  }

  public setMetalDetectorMode(enabled: boolean): void {
    this.metalDetectorMode = enabled;
    this.resetSessionState();
  }

  public setStopRadiusM(meters: number): void {
    this.stopRadiusM = clamp(meters, 0.5, 2.5);
  }

  public cancelVibration(): void {
    if (this.isNativeVibrationSupported()) {
      try {
        navigator.vibrate(0);
      } catch {
        // Ignore before first user activation
      }
    }
  }

  public resetSessionState(): void {
    if (this.isCurrentlyBlocked || this.lastRank > 0) {
      this.cancelVibration();
    }
    this.isCurrentlyBlocked = false;
    this.lastRank = 0;
    this.lastSteering = 'CLEAR';
    this.sameStateBuzzCount = 0;
    this.minDistAtLastBuzz = 99;
  }

  public isNativeVibrationSupported(): boolean {
    return (
      typeof navigator !== 'undefined' &&
      typeof navigator.vibrate === 'function'
    );
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.cancelVibration();
    }
  }

  /** Used for direct physical UI button taps on iOS Safari 17.4+ */
  public attachIosSwitchFallback(label: HTMLLabelElement | null): void {
    this.iosSwitchLabel = label;
  }

  public triggerTapFeedback(): void {
    if (!this.enabled) return;
    if (this.isNativeVibrationSupported()) {
      try {
        navigator.vibrate(25);
      } catch {
        // Ignore
      }
    } else if (this.iosSwitchLabel) {
      try {
        this.iosSwitchLabel.click();
      } catch {
        // Ignore
      }
    }
  }

  public triggerTestBuzz(): HapticPulseEvent {
    return this.firePattern('TEST_PULSE', [110, 55, 110], 3, 350, true);
  }

  public triggerFailsafeDisconnect(): void {
    if (!this.enabled) return;
    const now = performance.now();
    if (this.lastRank < 5 || now - this.lastFiredAt >= 3200) {
      this.firePattern(
        'FAILSAFE_SIGNAL_LOST',
        [200, 70, 90, 70, 200],
        5,
        3200,
        this.lastRank < 5
      );
    }
  }

  public handleGuidance(guidance: GuidancePayload): void {
    if (!this.enabled) return;

    if (this.metalDetectorMode) {
      this.handleMetalDetectorGuidance(guidance);
      return;
    }

    const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
    if (guidance.overall_severity === 'clear' && !isCurb) {
      if (this.lastRank > 0) {
        this.cancelVibration();
      }
      this.lastRank = 0;
      this.lastSteering = 'CLEAR';
      this.sameStateBuzzCount = 0;
      this.minDistAtLastBuzz = 99;
      return;
    }

    const minDist = Math.min(
      guidance.left.min_distance_m,
      guidance.center.min_distance_m,
      guidance.right.min_distance_m
    );
    const distCadenceMs = Math.round(
      clamp(190 + 250 * Math.pow(clamp(minDist, 0.35, 3.5), 1.28), 240, 1050)
    );

    let rank = 1;
    let label = 'CAUTION_TICK';
    let pattern: number[] = [45];

    if (isCurb) {
      rank = 4;
      label = guidance.curb.stairs_likely ? 'STAIRS_DROP_ALERT' : 'CURB_TRIPLE_DROP';
      pattern = [55, 40, 55, 40, 55, 50, 165]; // 460ms
    } else if (guidance.steering === 'STOP') {
      rank = 3;
      label = 'STOP_DOUBLE_SLAM';
      pattern = [180, 55, 180]; // 415ms
    } else if (guidance.overall_severity === 'danger') {
      rank = 2;
      if (guidance.steering === 'VEER_LEFT') {
        label = 'STEER_LEFT_SHORT_LONG';
        pattern = [45, 40, 45, 40, 145]; // 315ms
      } else {
        label = 'STEER_RIGHT_LONG_SHORT';
        pattern = [145, 40, 45, 40, 45]; // 315ms
      }
    } else {
      rank = 1;
      if (guidance.steering === 'VEER_LEFT') {
        label = 'CAUTION_NUDGE_LEFT';
        pattern = [38, 35, 75];
      } else if (guidance.steering === 'VEER_RIGHT') {
        label = 'CAUTION_NUDGE_RIGHT';
        pattern = [75, 35, 38];
      } else {
        label = 'CAUTION_TICK';
        pattern = [45];
      }
    }

    const stateChanged =
      rank !== this.lastRank || guidance.steering !== this.lastSteering;
    const movedCloser = this.minDistAtLastBuzz - minDist >= 0.25;

    if (stateChanged) {
      this.sameStateBuzzCount = 0;
    }

    const patternDurationMs = pattern.reduce((acc, v) => acc + v, 0);
    const backoffFactor =
      this.calmBackoff && !movedCloser
        ? Math.pow(1.85, Math.min(4, this.sameStateBuzzCount))
        : 1.0;
    const requiredCooldownMs = Math.min(
      6000,
      Math.round(Math.max(distCadenceMs * backoffFactor, patternDurationMs + 95))
    );

    const now = performance.now();
    const urgentPreempt =
      rank > this.lastRank ||
      (rank >= 2 && guidance.steering !== this.lastSteering);

    if (
      urgentPreempt ||
      (movedCloser && now - this.lastFiredAt >= patternDurationMs + 90) ||
      now - this.lastFiredAt >= this.lastCooldownMs
    ) {
      if (
        this.calmBackoff &&
        rank === 1 &&
        !stateChanged &&
        !movedCloser &&
        this.sameStateBuzzCount >= 2
      ) {
        return;
      }
      this.lastSteering = guidance.steering;
      this.minDistAtLastBuzz = minDist;
      this.sameStateBuzzCount += 1;
      this.firePattern(label, pattern, rank, requiredCooldownMs, urgentPreempt);
    }
  }

  private handleMetalDetectorGuidance(guidance: GuidancePayload): void {
    const block = evaluateForwardBlock(guidance, this.stopRadiusM);
    if (!block.blocked) {
      // Razor-sharp cutoff: stop active motor vibration the exact frame the path opens!
      if (this.isCurrentlyBlocked || this.lastRank > 0) {
        this.cancelVibration();
      }
      this.isCurrentlyBlocked = false;
      this.lastRank = 0;
      this.lastSteering = 'CLEAR';
      this.sameStateBuzzCount = 0;
      this.minDistAtLastBuzz = 99;
      return;
    }

    let rank = 1;
    let label = 'OBSTACLE_AHEAD_PULSE';
    let pattern: number[] = [60];
    let cooldownMs = 320;

    if (block.kind === 'curb') {
      rank = 4;
      label = guidance.curb.stairs_likely ? 'STOP_STAIRS_DROP' : 'STOP_CURB_DROP';
      pattern = [50, 35, 50, 35, 145]; // 315ms
      cooldownMs = 400;
    } else if (block.kind === 'contact') {
      rank = 3;
      label = 'STOP_CONTACT_SLAM';
      pattern = [130, 35, 130]; // 295ms
      cooldownMs = 345;
    } else if (block.kind === 'close') {
      rank = 2;
      label = 'OBSTACLE_CLOSE_PULSE';
      pattern = [90];
      cooldownMs = 195;
    }

    const now = performance.now();
    const justEnteredBlock = !this.isCurrentlyBlocked;
    const escalated = rank > this.lastRank;

    // Zero stationary backoff: keeps pulsing steadily as long as phone points at the obstacle!
    if (
      justEnteredBlock ||
      escalated ||
      now - this.lastFiredAt >= this.lastCooldownMs
    ) {
      this.isCurrentlyBlocked = true;
      this.lastSteering = guidance.steering;
      this.minDistAtLastBuzz = block.distanceM;
      this.sameStateBuzzCount += 1;
      this.firePattern(label, pattern, rank, cooldownMs, justEnteredBlock || escalated);
    }
  }

  private firePattern(
    patternLabel: string,
    pattern: number[],
    rank: number,
    cooldownMs: number,
    preempt: boolean
  ): HapticPulseEvent {
    const now = performance.now();
    this.lastFiredAt = now;
    this.lastCooldownMs = cooldownMs;
    this.lastRank = rank;

    const nativeSupported = this.isNativeVibrationSupported();
    if (nativeSupported) {
      try {
        if (preempt) {
          navigator.vibrate(0);
        }
        navigator.vibrate(pattern);
      } catch {
        // Ignore if browser blocks vibration prior to user gesture
      }
    } else {
      // Async WebSocket/Worker callbacks lack transient activation for iOS `<input switch>`,
      // so delegate a crisp 125Hz sub-bass tactile click to WebAudio!
      this.onIosTactileFallback?.(rank);
    }

    const evt: HapticPulseEvent = {
      patternLabel,
      pattern,
      nativeSupported,
      timestamp: Date.now(),
    };
    this.onPulse?.(evt);
    return evt;
  }
}
