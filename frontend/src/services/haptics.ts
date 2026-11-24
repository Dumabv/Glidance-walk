import type { GuidancePayload } from '../types/telemetry';

export interface HapticPulseEvent {
  patternLabel: string;
  pattern: number[];
  nativeSupported: boolean;
  timestamp: number;
}

/**
 * Client-side haptic scheduler with cooldown rate-limiting.
 * Runs locally on the phone so 15 FPS guidance frames never cancel/stutter the vibration motor.
 */
export class HapticsEngine {
  private lastFiredAt = 0;
  private enabled = true;
  private iosSwitchLabel: HTMLLabelElement | null = null;
  private onPulse?: (evt: HapticPulseEvent) => void;

  constructor(onPulse?: (evt: HapticPulseEvent) => void) {
    this.onPulse = onPulse;
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
      navigator.vibrate(0);
    }
  }

  /** Attaches an off-screen iOS 17.4+ switch label for subtle Safari Taptic pulses. */
  public attachIosSwitchFallback(label: HTMLLabelElement | null): void {
    this.iosSwitchLabel = label;
  }

  public triggerTestBuzz(): HapticPulseEvent {
    return this.firePattern('TEST_PULSE', [120, 60, 120], true);
  }

  public handleGuidance(guidance: GuidancePayload): void {
    if (!this.enabled) return;

    if (guidance.overall_severity === 'clear' && !guidance.curb.detected) {
      return;
    }

    const now = performance.now();

    if (guidance.curb.detected || guidance.steering === 'CURB_DROP') {
      if (now - this.lastFiredAt >= 430) {
        this.firePattern('CURB_TRIPLE_TAP', [95, 55, 95, 55, 160], false);
      }
      return;
    }

    if (guidance.steering === 'STOP') {
      if (now - this.lastFiredAt >= 420) {
        this.firePattern('STOP_DOUBLE_BEAT', [170, 65, 170], false);
      }
      return;
    }

    if (guidance.overall_severity === 'danger') {
      if (now - this.lastFiredAt >= 460) {
        const label =
          guidance.steering === 'VEER_LEFT' ? 'DANGER_RIGHT_SIDE' : 'DANGER_LEFT_SIDE';
        this.firePattern(label, [140, 55, 70], false);
      }
      return;
    }

    // Caution cadence: gentle reminder every 700 ms
    if (guidance.overall_severity === 'caution' && now - this.lastFiredAt >= 700) {
      this.firePattern('CAUTION_PING', [70], false);
    }
  }

  private firePattern(
    patternLabel: string,
    pattern: number[],
    force: boolean
  ): HapticPulseEvent {
    const now = performance.now();
    if (!force) {
      this.lastFiredAt = now;
    }

    const nativeSupported = this.isNativeVibrationSupported();
    if (nativeSupported) {
      try {
        navigator.vibrate(pattern);
      } catch {
        // Ignore if browser blocks vibration before first user tap
      }
    } else if (this.iosSwitchLabel) {
      // Fallback for iOS Safari 17.4+ native switch Taptic feedback
      try {
        this.iosSwitchLabel.click();
      } catch {
        // Ignore gracefully on unsupported browsers
      }
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
