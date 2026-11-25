import type { GuidancePayload, SteeringAction } from '../types/telemetry';

export interface HapticPulseEvent {
  patternLabel: string;
  pattern: number[];
  nativeSupported: boolean;
  timestamp: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Priority-preempting tactile scheduler for Android LRA/ERM motors + iOS acoustic-tactile bridge.
 *
 * Upgrades over v1 (from NavCog + sonic-vision + web-haptics audit):
 * 1. Urgency rank preemption: higher-rank hazards (`STOP`, `CURB_DROP`, `FAILSAFE`) immediately
 *    preempt active lower-rank (`CAUTION`) cooldowns with 0ms starvation delay.
 * 2. Non-clipping dynamic distance cadence: cooldown is guaranteed >= patternDuration + 75ms
 *    so Android `navigator.vibrate()` never clips its own final pulse.
 * 3. Asymmetric directional tactile vocabulary:
 *    - `VEER_LEFT`:  short-short-long `[45, 40, 45, 40, 145]` (guides walker left)
 *    - `VEER_RIGHT`: long-short-short `[145, 40, 45, 40, 45]` (guides walker right)
 *    - `STOP`:       heavy double slam `[180, 55, 180]`
 *    - `CURB_DROP`:  staccato triple-tap + drop thud `[55, 40, 55, 40, 55, 50, 165]`
 *    - `FAILSAFE`:   SOS disconnect alert `[240, 75, 90, 75, 240]`
 */
export class HapticsEngine {
  private lastFiredAt = 0;
  private lastCooldownMs = 0;
  private lastRank = 0;
  private lastSteering: SteeringAction = 'CLEAR';
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

  public isNativeVibrationSupported(): boolean {
    return (
      typeof navigator !== 'undefined' &&
      typeof navigator.vibrate === 'function'
    );
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled && this.isNativeVibrationSupported()) {
      try {
        navigator.vibrate(0);
      } catch {
        // Ignore before first user activation
      }
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
    if (this.lastRank < 5 || now - this.lastFiredAt >= 750) {
      this.firePattern(
        'FAILSAFE_SIGNAL_LOST',
        [240, 75, 90, 75, 240],
        5,
        820,
        this.lastRank < 5
      );
    }
  }

  public handleGuidance(guidance: GuidancePayload): void {
    if (!this.enabled) return;

    const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
    if (guidance.overall_severity === 'clear' && !isCurb) {
      this.lastRank = 0;
      this.lastSteering = 'CLEAR';
      return;
    }

    const minDist = Math.min(
      guidance.left.min_distance_m,
      guidance.center.min_distance_m,
      guidance.right.min_distance_m
    );
    // Weber-Fechner inverse-exponential distance cadence (`sonic-vision` formula)
    const distCadenceMs = Math.round(
      clamp(155 + 225 * Math.pow(clamp(minDist, 0.35, 3.5), 1.28), 185, 820)
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

    const patternDurationMs = pattern.reduce((acc, v) => acc + v, 0);
    const requiredCooldownMs = Math.max(distCadenceMs, patternDurationMs + 75);

    const now = performance.now();
    const urgentPreempt =
      rank > this.lastRank ||
      (rank >= 2 && guidance.steering !== this.lastSteering);

    if (urgentPreempt || now - this.lastFiredAt >= this.lastCooldownMs) {
      this.lastSteering = guidance.steering;
      this.firePattern(label, pattern, rank, requiredCooldownMs, urgentPreempt);
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
