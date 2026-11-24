import type { GuidancePayload } from '../types/telemetry';

/**
 * 3D stereo sonar pings via Web Audio API StereoPannerNode + debounced speech cues.
 * Works identically on Android Chrome and iPhone Safari (crucial for iOS earbuds testing).
 */
export class SpatialAudioEngine {
  private ctx: AudioContext | null = null;
  private sonarEnabled = true;
  private voiceEnabled = false;
  private lastPingAt = 0;
  private lastSpokenAt = 0;
  private lastSpokenCue = '';

  public unlock(): void {
    if (typeof window === 'undefined') return;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
  }

  public setSonarEnabled(enabled: boolean): void {
    this.sonarEnabled = enabled;
  }

  public setVoiceEnabled(enabled: boolean): void {
    this.voiceEnabled = enabled;
    if (!enabled && typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }

  public playTestSonar(): void {
    this.unlock();
    this.playTone(540, 0.12, -0.8, 0.14);
    setTimeout(() => this.playTone(720, 0.12, 0.8, 0.14), 160);
  }

  public handleGuidance(guidance: GuidancePayload): void {
    const now = performance.now();

    if (this.sonarEnabled && guidance.overall_severity !== 'clear') {
      const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
      const minInterval = isCurb
        ? 450
        : guidance.overall_severity === 'danger'
          ? 420
          : 780;

      if (now - this.lastPingAt >= minInterval) {
        this.lastPingAt = now;
        this.unlock();

        if (isCurb) {
          // Descending dual tone = step DOWN ahead
          this.playTone(660, 0.09, 0.0, 0.18);
          setTimeout(() => this.playTone(310, 0.15, 0.0, 0.2), 100);
        } else {
          // Pan toward the obstacle so user senses direction (left=-0.85, right=+0.85)
          const pan = this.computeObstaclePan(guidance);
          const freq = guidance.overall_severity === 'danger' ? 880 : 490;
          const dur = guidance.overall_severity === 'danger' ? 0.1 : 0.08;
          this.playTone(freq, dur, pan, 0.16);
        }
      }
    }

    if (
      this.voiceEnabled &&
      guidance.spoken_cue &&
      typeof window !== 'undefined' &&
      'speechSynthesis' in window
    ) {
      const cueChanged = guidance.spoken_cue !== this.lastSpokenCue;
      const cooldown = guidance.overall_severity === 'danger' ? 2000 : 3500;
      if (
        (cueChanged && now - this.lastSpokenAt >= 1200) ||
        (guidance.overall_severity === 'danger' && now - this.lastSpokenAt >= cooldown)
      ) {
        this.lastSpokenAt = now;
        this.lastSpokenCue = guidance.spoken_cue;
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(guidance.spoken_cue);
        utter.rate = 1.15;
        utter.lang = 'en-US';
        window.speechSynthesis.speak(utter);
      }
    }
  }

  private computeObstaclePan(guidance: GuidancePayload): number {
    if (guidance.steering === 'STOP' || guidance.center.severity === 'danger') {
      return 0.0;
    }
    const diff = guidance.right.obstacle_score - guidance.left.obstacle_score;
    return Math.max(-0.9, Math.min(0.9, diff * 2.0));
  }

  private playTone(
    freq: number,
    durationSec: number,
    pan: number,
    volume: number
  ): void {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, now);

    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(volume, now + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.001, now + durationSec);

    if (typeof this.ctx.createStereoPanner === 'function') {
      const panner = this.ctx.createStereoPanner();
      panner.pan.setValueAtTime(pan, now);
      osc.connect(gain);
      gain.connect(panner);
      panner.connect(this.ctx.destination);
    } else {
      osc.connect(gain);
      gain.connect(this.ctx.destination);
    }

    osc.start(now);
    osc.stop(now + durationSec + 0.02);
  }
}
