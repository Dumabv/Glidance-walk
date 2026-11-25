import type { GuidancePayload } from '../types/telemetry';

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * 3D Binaural HRTF Spatial Sonar + Prioritized Speech Engine.
 *
 * Upgrades over v1 (from Microsoft Soundscape + Google Project Guideline + vOICe_depth):
 * 1. True 3D `PannerNode` (`panningModel = 'HRTF'`, `(x, y, z)`) routed into a persistent
 *    `DynamicsCompressorNode` master limiter:
 *    - Curbs physically localize BELOW ear level (`posY = -0.85m`) with a hardware-clocked
 *      descending frequency sweep (`680Hz -> 240Hz`, zero `setTimeout` jitter!).
 *    - Obstacles localize laterally (`posX` in `[-1.2m, +1.2m]`) and in depth (`posZ = -dist`),
 *      using filtered `triangle` harmonics for crisp pinna/ITD localization without sine fatigue.
 * 2. Zero WebAudio node leaks: every ephemeral oscillator/filter/gain/panner disconnects cleanly
 *    inside `osc.onended`.
 * 3. Subtle `-34dB` `2.5s` active-assurance wooden tick on `CLEAR` so an eyes-closed walker
 *    always knows camera guidance is alive.
 * 4. Chrome/Android `speechSynthesis.cancel()` race fix + priority preemption + `aria-live`
 *    double-talk prevention.
 */
export class SpatialAudioEngine {
  private ctx: AudioContext | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private sonarEnabled = true;
  private voiceEnabled = false;
  private lastPingAt = 0;
  private lastHeartbeatAt = 0;
  private lastRank = 0;
  private lastSpokenAt = 0;
  private lastSpokenCue = '';
  private speakTimer: number | null = null;

  public unlock(): void {
    if (typeof window === 'undefined') return;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
        const comp = this.ctx.createDynamicsCompressor();
        comp.threshold.setValueAtTime(-6, this.ctx.currentTime);
        comp.knee.setValueAtTime(12, this.ctx.currentTime);
        comp.ratio.setValueAtTime(12, this.ctx.currentTime);
        comp.attack.setValueAtTime(0.003, this.ctx.currentTime);
        comp.release.setValueAtTime(0.15, this.ctx.currentTime);
        comp.connect(this.ctx.destination);
        this.compressor = comp;
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
  }

  public dispose(): void {
    if (this.speakTimer !== null && typeof window !== 'undefined') {
      window.clearTimeout(this.speakTimer);
      this.speakTimer = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    if (this.ctx) {
      void this.ctx.close().catch(() => {});
      this.ctx = null;
      this.compressor = null;
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

  public isVoiceEnabled(): boolean {
    return this.voiceEnabled;
  }

  /** Crisp 125Hz sub-bass tactile click fallback for iOS devices lacking `navigator.vibrate` */
  public playIosTactileClick(rank: number): void {
    if (!this.sonarEnabled) return;
    this.unlock();
    if (!this.ctx || !this.compressor) return;
    const t0 = this.ctx.currentTime;
    const gainVal = rank >= 3 ? 0.24 : 0.14;
    this.schedule3DPing({
      startTime: t0,
      freqStart: 135,
      freqEnd: 95,
      durationSec: 0.032,
      posX: 0,
      posY: -0.2,
      posZ: -0.7,
      volume: gainVal,
      cutoffHz: 420,
    });
  }

  public playTestSonar(): void {
    this.unlock();
    if (!this.ctx) return;
    const t0 = this.ctx.currentTime;
    this.schedule3DPing({
      startTime: t0,
      freqStart: 520,
      freqEnd: 520,
      durationSec: 0.11,
      posX: -1.0,
      posY: 0,
      posZ: -1.1,
      volume: 0.16,
      cutoffHz: 2200,
    });
    this.schedule3DPing({
      startTime: t0 + 0.15,
      freqStart: 740,
      freqEnd: 740,
      durationSec: 0.11,
      posX: 1.0,
      posY: 0,
      posZ: -1.1,
      volume: 0.16,
      cutoffHz: 2600,
    });
  }

  public triggerDisconnectAlarm(reason = 'Signal lost. Stop.'): void {
    const now = performance.now();
    if (this.sonarEnabled && (this.lastRank < 5 || now - this.lastPingAt >= 750)) {
      this.lastPingAt = now;
      this.lastRank = 5;
      this.unlock();
      if (this.ctx) {
        const t0 = this.ctx.currentTime;
        this.schedule3DPing({
          startTime: t0,
          freqStart: 540,
          freqEnd: 540,
          durationSec: 0.11,
          posX: 0,
          posY: 0,
          posZ: -0.6,
          volume: 0.22,
          cutoffHz: 2400,
        });
        this.schedule3DPing({
          startTime: t0 + 0.13,
          freqStart: 360,
          freqEnd: 360,
          durationSec: 0.16,
          posX: 0,
          posY: -0.3,
          posZ: -0.6,
          volume: 0.24,
          cutoffHz: 2000,
        });
      }
    }
    this.speakPrioritized(reason, true);
  }

  public speakStatusImmediate(text: string): void {
    this.speakPrioritized(text, true, true);
  }

  public handleGuidance(guidance: GuidancePayload): void {
    const now = performance.now();
    const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';

    if (this.sonarEnabled) {
      if (guidance.overall_severity === 'clear' && !isCurb) {
        this.lastRank = 0;
        // Subtle 2.5s active-assurance wooden tick (-34dB) so eyes-closed walker knows system is live
        if (now - this.lastHeartbeatAt >= 2500) {
          this.lastHeartbeatAt = now;
          this.unlock();
          if (this.ctx) {
            this.schedule3DPing({
              startTime: this.ctx.currentTime,
              freqStart: 330,
              freqEnd: 290,
              durationSec: 0.018,
              posX: 0,
              posY: 0.1,
              posZ: -1.8,
              volume: 0.022,
              cutoffHz: 950,
            });
          }
        }
      } else {
        const minDist = Math.min(
          guidance.left.min_distance_m,
          guidance.center.min_distance_m,
          guidance.right.min_distance_m
        );
        const rank = isCurb
          ? 4
          : guidance.steering === 'STOP'
            ? 3
            : guidance.overall_severity === 'danger'
              ? 2
              : 1;

        const minInterval = isCurb
          ? 440
          : Math.round(clamp(165 + 220 * Math.pow(clamp(minDist, 0.35, 3.5), 1.25), 185, 800));

        if (rank > this.lastRank || now - this.lastPingAt >= minInterval) {
          this.lastPingAt = now;
          this.lastHeartbeatAt = now;
          this.lastRank = rank;
          this.unlock();

          if (this.ctx) {
            const t0 = this.ctx.currentTime;
            if (isCurb) {
              // Continuous downward pitch glide localized below ear height (`posY = -0.85m`)
              this.schedule3DPing({
                startTime: t0,
                freqStart: 680,
                freqEnd: 240,
                durationSec: 0.2,
                posX: 0,
                posY: -0.85,
                posZ: -0.9,
                volume: 0.22,
                cutoffHz: 2400,
              });
            } else {
              const posX = this.computeObstacleX(guidance);
              const posY = guidance.overhead_hazard ? 0.75 : 0.0;
              const posZ = -clamp(minDist, 0.4, 3.6);
              const freq = Math.round(940 - 160 * clamp(minDist, 0.4, 3.2));
              const dur = guidance.overall_severity === 'danger' ? 0.105 : 0.075;
              const cutoff = guidance.overall_severity === 'danger' ? 3000 : 1800;

              this.schedule3DPing({
                startTime: t0,
                freqStart: freq,
                freqEnd: freq,
                durationSec: dur,
                posX,
                posY,
                posZ,
                volume: guidance.overall_severity === 'danger' ? 0.19 : 0.13,
                cutoffHz: cutoff,
              });

              // Optional soft open-corridor guide chime in opposite ear on VEER_LEFT / VEER_RIGHT
              if (
                guidance.steering === 'VEER_LEFT' ||
                guidance.steering === 'VEER_RIGHT'
              ) {
                const safeX = guidance.steering === 'VEER_LEFT' ? -0.85 : 0.85;
                this.schedule3DPing({
                  startTime: t0 + 0.095,
                  freqStart: 528,
                  freqEnd: 528,
                  durationSec: 0.065,
                  posX: safeX,
                  posY: 0.05,
                  posZ: -1.4,
                  volume: 0.065,
                  cutoffHz: 1500,
                });
              }
            }
          }
        }
      }
    }

    if (this.voiceEnabled && guidance.spoken_cue) {
      const urgent =
        guidance.overall_severity === 'danger' ||
        guidance.steering === 'STOP' ||
        isCurb;
      const cueChanged = guidance.spoken_cue !== this.lastSpokenCue;
      const cooldown = urgent ? 1900 : 3400;
      if (
        (cueChanged && (urgent || now - this.lastSpokenAt >= 1100)) ||
        (urgent && now - this.lastSpokenAt >= cooldown)
      ) {
        this.speakPrioritized(guidance.spoken_cue, urgent);
      }
    }
  }

  private speakPrioritized(
    text: string,
    urgent: boolean,
    forceEvenIfMuted = false
  ): void {
    if (
      (!this.voiceEnabled && !forceEvenIfMuted) ||
      typeof window === 'undefined' ||
      !('speechSynthesis' in window)
    ) {
      return;
    }
    const now = performance.now();
    this.lastSpokenAt = now;
    this.lastSpokenCue = text;

    const synth = window.speechSynthesis;
    if (urgent || forceEvenIfMuted) {
      synth.cancel();
      if (this.speakTimer !== null) {
        window.clearTimeout(this.speakTimer);
      }
      // 45ms delay after `cancel()` avoids Chrome/Android WebSpeech utterance drop bug!
      this.speakTimer = window.setTimeout(() => {
        this.speakTimer = null;
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = urgent ? 1.2 : 1.12;
        utter.lang = 'en-US';
        synth.speak(utter);
      }, 45);
    } else if (!synth.speaking) {
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.12;
      utter.lang = 'en-US';
      synth.speak(utter);
    }
  }

  private computeObstacleX(guidance: GuidancePayload): number {
    if (guidance.steering === 'STOP' || guidance.center.severity === 'danger') {
      return 0.0;
    }
    const diff = guidance.right.obstacle_score - guidance.left.obstacle_score;
    return clamp(diff * 2.2, -1.15, 1.15);
  }

  private schedule3DPing(opts: {
    startTime: number;
    freqStart: number;
    freqEnd: number;
    durationSec: number;
    posX: number;
    posY: number;
    posZ: number;
    volume: number;
    cutoffHz: number;
  }): void {
    if (!this.ctx || !this.compressor) return;
    const {
      startTime,
      freqStart,
      freqEnd,
      durationSec,
      posX,
      posY,
      posZ,
      volume,
      cutoffHz,
    } = opts;

    const osc = this.ctx.createOscillator();
    const filter = this.ctx.createBiquadFilter();
    const gain = this.ctx.createGain();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(Math.max(40, freqStart), startTime);
    if (Math.abs(freqEnd - freqStart) > 1) {
      osc.frequency.exponentialRampToValueAtTime(
        Math.max(40, freqEnd),
        startTime + durationSec
      );
    }

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(cutoffHz, startTime);

    gain.gain.setValueAtTime(0.0008, startTime);
    gain.gain.exponentialRampToValueAtTime(
      Math.max(0.002, volume),
      startTime + Math.min(0.012, durationSec * 0.35)
    );
    gain.gain.exponentialRampToValueAtTime(0.0008, startTime + durationSec);

    osc.connect(filter);
    filter.connect(gain);

    let spatialNode: AudioNode | null = null;
    if (typeof this.ctx.createPanner === 'function') {
      const panner = this.ctx.createPanner();
      panner.panningModel = 'HRTF';
      panner.distanceModel = 'inverse';
      panner.refDistance = 0.85;
      panner.rolloffFactor = 1.1;
      if (panner.positionX) {
        panner.positionX.setValueAtTime(posX, startTime);
        panner.positionY.setValueAtTime(posY, startTime);
        panner.positionZ.setValueAtTime(posZ, startTime);
      } else {
        panner.setPosition(posX, posY, posZ);
      }
      gain.connect(panner);
      panner.connect(this.compressor);
      spatialNode = panner;
    } else if (typeof this.ctx.createStereoPanner === 'function') {
      const panner = this.ctx.createStereoPanner();
      panner.pan.setValueAtTime(clamp(posX, -1, 1), startTime);
      gain.connect(panner);
      panner.connect(this.compressor);
      spatialNode = panner;
    } else {
      gain.connect(this.compressor);
    }

    osc.onended = () => {
      osc.disconnect();
      filter.disconnect();
      gain.disconnect();
      spatialNode?.disconnect();
    };

    osc.start(startTime);
    osc.stop(startTime + durationSec + 0.015);
  }
}
