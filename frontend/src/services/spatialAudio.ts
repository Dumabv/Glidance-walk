import type { GuidancePayload } from '../types/telemetry';
import { evaluateForwardBlock } from './haptics.ts';

export type GuidanceProfile =
  | 'metal_detector'
  | 'headphones_3d'
  | 'smart_calm'
  | 'sonar_3d'
  | 'tactile_only';
export type VoiceLang = 'ru' | 'en';

export interface HeadphonesCuePlan {
  kind: 'silence' | 'left_obstacle' | 'right_obstacle' | 'wall_stop' | 'stairs';
  /** Stereo pan: -0.9 (left ear) .. 0.0 (both ears) .. +0.9 (right ear) */
  pan: number;
  freqStart: number;
  freqEnd: number;
  intervalMs: number;
  volume: number;
  doublePulse: boolean;
}

/**
 * Pure, cross-platform (iOS Safari + Android Chrome) binaural headphone cue planner.
 * - Obstacle on LEFT (VEER_RIGHT)  -> Left ear  (pan = -0.88)
 * - Obstacle on RIGHT (VEER_LEFT)  -> Right ear (pan = +0.88)
 * - Wall / STOP ahead              -> Both ears (pan = 0.0, low double bass)
 * - Stairs / Curb drop             -> Both ears (pan = 0.0, descending 680->420Hz chime)
 * - Clear path                     -> Complete silence
 */
export function computeHeadphonesCue(guidance: GuidancePayload): HeadphonesCuePlan {
  const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
  const minDist = Math.min(
    guidance.left.min_distance_m,
    guidance.center.min_distance_m,
    guidance.right.min_distance_m
  );
  const closeness = clamp(1.0 - (minDist - 0.35) / 2.25, 0.0, 1.0);

  if (guidance.overall_severity === 'clear' && !isCurb) {
    return {
      kind: 'silence',
      pan: 0,
      freqStart: 0,
      freqEnd: 0,
      intervalMs: 1000,
      volume: 0,
      doublePulse: false,
    };
  }

  if (isCurb) {
    return {
      kind: 'stairs',
      pan: 0,
      freqStart: 680,
      freqEnd: 420,
      intervalMs: Math.round(520 - closeness * 220),
      volume: 0.22,
      doublePulse: true,
    };
  }

  if (guidance.steering === 'STOP' || guidance.wall_override || guidance.lens_blocked) {
    return {
      kind: 'wall_stop',
      pan: 0,
      freqStart: 270,
      freqEnd: 210,
      intervalMs: Math.round(440 - closeness * 210),
      volume: 0.24,
      doublePulse: true,
    };
  }

  if (guidance.steering === 'VEER_LEFT') {
    // Obstacle is on the RIGHT -> sound in RIGHT ear (+0.88)
    const freq = Math.round(440 + closeness * 380);
    return {
      kind: 'right_obstacle',
      pan: 0.88,
      freqStart: freq,
      freqEnd: freq,
      intervalMs: Math.round(620 - closeness * 360),
      volume: clamp(0.13 + closeness * 0.11, 0.13, 0.24),
      doublePulse: guidance.overall_severity === 'danger',
    };
  }

  if (guidance.steering === 'VEER_RIGHT') {
    // Obstacle is on the LEFT -> sound in LEFT ear (-0.88)
    const freq = Math.round(440 + closeness * 380);
    return {
      kind: 'left_obstacle',
      pan: -0.88,
      freqStart: freq,
      freqEnd: freq,
      intervalMs: Math.round(620 - closeness * 360),
      volume: clamp(0.13 + closeness * 0.11, 0.13, 0.24),
      doublePulse: guidance.overall_severity === 'danger',
    };
  }

  const diff = guidance.right.obstacle_score - guidance.left.obstacle_score;
  const pan = clamp(diff * 1.4, -0.88, 0.88);
  const freq = Math.round(420 + closeness * 340);
  return {
    kind: pan < 0 ? 'left_obstacle' : 'right_obstacle',
    pan: Math.round(pan * 100) / 100,
    freqStart: freq,
    freqEnd: freq,
    intervalMs: Math.round(640 - closeness * 340),
    volume: 0.16,
    doublePulse: false,
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

const RU_OBJECT_NAMES: Record<string, string> = {
  person: 'Человек',
  stairs: 'Лестница',
  step: 'Ступенька',
  curb: 'Бордюр',
  door: 'Дверь',
  'glass door': 'Стеклянная дверь',
  wall: 'Стена',
  pillar: 'Колонна',
  railing: 'Перила',
  chair: 'Стул',
  table: 'Стол',
  'dining table': 'Стол',
  desk: 'Стол',
  bench: 'Скамейка',
  couch: 'Диван',
  box: 'Коробка',
  backpack: 'Рюкзак',
  suitcase: 'Чемодан',
  'trash can': 'Урна',
  manhole: 'Люк',
  bicycle: 'Велосипед',
  car: 'Машина',
  dog: 'Собака',
  pole: 'Столб',
};

function getNearestNamedObject(guidance: GuidancePayload): string | null {
  const objs = guidance.objects;
  if (!objs || objs.length === 0) return null;
  const active = objs.filter((o) => o.severity !== 'clear' && o.confidence >= 0.3);
  if (active.length === 0) return null;
  const centerObj = active.find((o) => o.corridor === 'center');
  return (centerObj ?? active[0]).label.toLowerCase();
}

function formatObjName(label: string, lang: VoiceLang): string {
  if (lang === 'ru') {
    return RU_OBJECT_NAMES[label] ?? label;
  }
  return label.charAt(0).toUpperCase() + label.slice(1);
}

/**
 * Action-only 1–2 word aviation/rally callouts with specific obstacle naming (stairs/door/chair/person/wall).
 * NEVER mixes two directions in one phrase (eliminates "Obstacle on RIGHT, step LEFT" confusion!).
 */
function getCrispCallout(
  guidance: GuidancePayload,
  lang: VoiceLang
): { key: string; text: string; urgent: boolean } | null {
  const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
  const objLabel = getNearestNamedObject(guidance);

  if (guidance.glass_barrier || objLabel === 'glass door') {
    return {
      key: 'GLASS_BARRIER',
      text: lang === 'ru' ? 'Стекло впереди' : 'Glass ahead',
      urgent: true,
    };
  }
  if (guidance.lens_blocked) {
    return {
      key: 'LENS_BLOCKED',
      text: lang === 'ru' ? 'Камера закрыта' : 'Camera blocked',
      urgent: true,
    };
  }
  if (isCurb || objLabel === 'stairs' || objLabel === 'step' || objLabel === 'curb') {
    if (guidance.curb.stairs_likely || objLabel === 'stairs') {
      return {
        key: 'STAIRS',
        text: lang === 'ru' ? 'Лестница' : 'Stairs ahead',
        urgent: true,
      };
    }
    return {
      key: 'CURB',
      text: lang === 'ru' ? 'Ступенька' : 'Step ahead',
      urgent: true,
    };
  }
  if (guidance.overhead_hazard && guidance.overall_severity === 'danger') {
    return {
      key: 'OVERHEAD',
      text: lang === 'ru' ? 'Препятствие сверху' : 'Watch head',
      urgent: true,
    };
  }
  if (guidance.steering === 'STOP') {
    const noun = objLabel
      ? formatObjName(objLabel, lang)
      : guidance.wall_override
        ? lang === 'ru'
          ? 'Стена'
          : 'Wall'
        : null;
    if (typeof guidance.best_escape_deg === 'number' && Math.abs(guidance.best_escape_deg) >= 25) {
      const goLeft = guidance.best_escape_deg < 0;
      const prefix = noun ? `${noun}, ` : '';
      return {
        key: `${noun ?? 'STOP'}_TURN_${goLeft ? 'L' : 'R'}`,
        text:
          lang === 'ru'
            ? `${prefix}${goLeft ? 'стоп, влево' : 'стоп, вправо'}`
            : `${prefix}${goLeft ? 'stop, turn left' : 'stop, turn right'}`,
        urgent: true,
      };
    }
    return {
      key: noun ? `STOP_${objLabel ?? 'WALL'}` : 'STOP',
      text: noun
        ? lang === 'ru'
          ? `${noun}, стоп`
          : `${noun}, stop`
        : lang === 'ru'
          ? 'Стоп'
          : 'Stop',
      urgent: true,
    };
  }
  if (guidance.steering === 'VEER_LEFT') {
    const isDanger = guidance.overall_severity === 'danger';
    const noun = objLabel ? formatObjName(objLabel, lang) : null;
    return {
      key: `${isDanger ? 'DANGER' : 'CAUTION'}_LEFT_${objLabel ?? 'GEN'}`,
      text: noun
        ? lang === 'ru'
          ? `${noun}, левее`
          : `${noun}, left`
        : lang === 'ru'
          ? 'Левее'
          : 'Left',
      urgent: isDanger,
    };
  }
  if (guidance.steering === 'VEER_RIGHT') {
    const isDanger = guidance.overall_severity === 'danger';
    const noun = objLabel ? formatObjName(objLabel, lang) : null;
    return {
      key: `${isDanger ? 'DANGER' : 'CAUTION'}_RIGHT_${objLabel ?? 'GEN'}`,
      text: noun
        ? lang === 'ru'
          ? `${noun}, правее`
          : `${noun}, right`
        : lang === 'ru'
          ? 'Правее'
          : 'Right',
      urgent: isDanger,
    };
  }
  return null;
}

/**
 * Calm Event-Driven 3D Binaural HRTF Sonar + Zero-Nag Rally Callout Engine.
 *
 * Designed around real blind-navigation HCI principles (Soundscape / TCAS / Rally Co-Driver):
 * 1. Silence-by-default on clear paths (zero background chatter, preserves natural street hearing).
 * 2. Edge-Triggered + Closing-Velocity Gate ("Zero Stationary Nag"):
 *    - Fires immediately (0ms) when a new hazard appears or direction changes.
 *    - If you stand still or hold the phone steady to inspect a scene, it DOES NOT repeat!
 *    - Only repeats if you physically move closer (`minDist` drops by >= 0.28m) or after a long
 *      safety reminder (`6.5s+`) if still walking into `STOP`.
 * 3. 1-word action callouts (`"Левее"`, `"Правее"`, `"Стоп"`, `"Ступенька"`) in < 350ms, never
 *    mixing conflicting words like "Obstacle on right, step left".
 * 4. Stability filter: mild `caution` requires 2 consecutive frames before speaking, and `"Свободно"`
 *    only plays once after clearing a sustained `DANGER`/`STOP`/`CURB` hazard.
 */
export class SpatialAudioEngine {
  private ctx: AudioContext | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private profile: GuidanceProfile = 'metal_detector';
  private sonarEnabled = true;
  private voiceEnabled = false;
  private voiceLang: VoiceLang = 'en';

  private lastPingAt = 0;
  private lastRank = 0;

  // Anti-nag state machine
  private activeHazardKey = 'CLEAR';
  private pendingKey = 'CLEAR';
  private pendingStreak = 0;
  private hazardStreak = 0;
  private clearStreak = 0;
  private hadSustainedDanger = false;

  private lastSpokenKey = 'CLEAR';
  private lastSpokenAt = 0;
  private minDistAtLastAlert = 99;
  private sameStatePingCount = 0;
  private lastDisconnectSpokenAt = 0;
  private lastDisconnectPingAt = 0;
  private speakTimer: number | null = null;

  public unlock(): void {
    if (typeof window === 'undefined') return;
    // iOS 17+ Safari: switch audio session to 'playback' so headphones/speaker work even if mute switch is on
    try {
      const nav = navigator as unknown as { audioSession?: { type?: string } };
      if (nav.audioSession && nav.audioSession.type !== 'playback') {
        nav.audioSession.type = 'playback';
      }
    } catch {
      // Ignore on browsers without navigator.audioSession
    }
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
    if (this.ctx) {
      if (this.ctx.state === 'suspended') {
        void this.ctx.resume();
      }
      // iOS Safari gesture unlock: trigger a 1-sample silent buffer so Bluetooth/AirPods wake immediately
      try {
        const silentBuf = this.ctx.createBuffer(1, 1, 22050);
        const src = this.ctx.createBufferSource();
        src.buffer = silentBuf;
        src.connect(this.ctx.destination);
        src.start(0);
      } catch {
        // Ignore if context is closed or mocked
      }
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

  public resetSessionState(): void {
    this.activeHazardKey = 'CLEAR';
    this.pendingKey = 'CLEAR';
    this.pendingStreak = 0;
    this.hazardStreak = 0;
    this.clearStreak = 0;
    this.hadSustainedDanger = false;
    this.lastSpokenKey = 'CLEAR';
    this.minDistAtLastAlert = 99;
    this.sameStatePingCount = 0;
  }

  public setProfile(profile: GuidanceProfile): void {
    this.profile = profile;
    if (profile === 'tactile_only' && typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
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

  public setVoiceLang(lang: VoiceLang): void {
    this.voiceLang = lang;
  }

  public isVoiceEnabled(): boolean {
    return this.voiceEnabled && this.profile !== 'tactile_only';
  }

  /** Crisp 125Hz sub-bass tactile click fallback for iOS devices lacking `navigator.vibrate` */
  public playIosTactileClick(rank: number): void {
    if (!this.sonarEnabled || this.profile === 'tactile_only') return;
    this.unlock();
    if (!this.ctx || !this.compressor) return;
    const t0 = this.ctx.currentTime;
    const gainVal = rank >= 3 ? 0.22 : 0.12;
    this.schedule3DPing({
      startTime: t0,
      freqStart: 135,
      freqEnd: 95,
      durationSec: 0.03,
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
      durationSec: 0.1,
      posX: -1.0,
      posY: 0,
      posZ: -1.1,
      volume: 0.15,
      cutoffHz: 2200,
    });
    this.schedule3DPing({
      startTime: t0 + 0.14,
      freqStart: 740,
      freqEnd: 740,
      durationSec: 0.1,
      posX: 1.0,
      posY: 0,
      posZ: -1.1,
      volume: 0.15,
      cutoffHz: 2600,
    });
  }

  public triggerDisconnectAlarm(kind: 'camera_dead' | 'signal_delayed'): void {
    if (this.profile === 'tactile_only') return;
    const now = performance.now();

    // Cooldown prevents 250ms watchdog loop from machine-gunning alarms!
    if (this.sonarEnabled && now - this.lastDisconnectPingAt >= 3200) {
      this.lastDisconnectPingAt = now;
      this.lastRank = 5;
      this.unlock();
      if (this.ctx) {
        const t0 = this.ctx.currentTime;
        this.schedule3DPing({
          startTime: t0,
          freqStart: 520,
          freqEnd: 360,
          durationSec: 0.16,
          posX: 0,
          posY: -0.2,
          posZ: -0.7,
          volume: 0.18,
          cutoffHz: 2000,
        });
      }
    }

    if (this.voiceEnabled && now - this.lastDisconnectSpokenAt >= 7500) {
      this.lastDisconnectSpokenAt = now;
      const msg =
        kind === 'camera_dead'
          ? this.voiceLang === 'ru'
            ? 'Камера отключена'
            : 'Camera lost'
          : this.voiceLang === 'ru'
            ? 'Слабая связь'
            : 'Weak signal';
      this.speakNow(msg, true, false);
    }
  }

  public speakStatusImmediate(guidance: GuidancePayload | null, signalLost: boolean): void {
    let text: string;
    if (signalLost) {
      text = this.voiceLang === 'ru' ? 'Связь потеряна. Подождите.' : 'Signal lost. Wait.';
    } else if (!guidance) {
      text = this.voiceLang === 'ru' ? 'Готов к работе' : 'Ready to walk';
    } else {
      const minDist = Math.min(
        guidance.left.min_distance_m,
        guidance.center.min_distance_m,
        guidance.right.min_distance_m
      ).toFixed(1);
      const callout = getCrispCallout(guidance, this.voiceLang);
      text = callout
        ? this.voiceLang === 'ru'
          ? `${callout.text}. ${minDist} метра`
          : `${callout.text}. ${minDist} meters`
        : this.voiceLang === 'ru'
          ? `Свободно. ${minDist} метра`
          : `Clear. ${minDist} meters`;
    }
    this.speakNow(text, true, true);
  }

  /** Cross-platform L -> Center -> R stereo sweep to verify AirPods / Android headphones */
  public playHeadphonesLRTest(): void {
    this.unlock();
    if (!this.ctx || !this.compressor) return;
    const t0 = this.ctx.currentTime;
    this.scheduleStereoTone({ startTime: t0, freqStart: 480, freqEnd: 480, durationSec: 0.14, pan: -0.9, volume: 0.22 });
    this.scheduleStereoTone({ startTime: t0 + 0.22, freqStart: 600, freqEnd: 600, durationSec: 0.14, pan: 0.0, volume: 0.22 });
    this.scheduleStereoTone({ startTime: t0 + 0.44, freqStart: 740, freqEnd: 740, durationSec: 0.14, pan: 0.9, volume: 0.22 });
  }

  private handleHeadphones3D(guidance: GuidancePayload, now: number): void {
    const plan = computeHeadphonesCue(guidance);
    if (plan.kind === 'silence') {
      this.lastRank = 0;
      return;
    }

    if (this.sonarEnabled && now - this.lastPingAt >= plan.intervalMs) {
      this.lastPingAt = now;
      this.unlock();
      if (this.ctx && this.compressor) {
        const t0 = this.ctx.currentTime;
        this.scheduleStereoTone({
          startTime: t0,
          freqStart: plan.freqStart,
          freqEnd: plan.freqEnd,
          durationSec: plan.kind === 'stairs' ? 0.12 : 0.085,
          pan: plan.pan,
          volume: plan.volume,
        });
        if (plan.doublePulse) {
          this.scheduleStereoTone({
            startTime: t0 + 0.11,
            freqStart: plan.kind === 'stairs' ? plan.freqEnd : plan.freqStart,
            freqEnd: plan.kind === 'stairs' ? Math.max(260, plan.freqEnd - 120) : plan.freqEnd,
            durationSec: 0.09,
            pan: plan.pan,
            volume: plan.volume,
          });
        }
      }
    }

    // Optional crisp voice callout on new urgent hazard while wearing headphones
    if (this.voiceEnabled) {
      const callout = getCrispCallout(guidance, this.voiceLang);
      if (callout && (callout.key !== this.lastSpokenKey || now - this.lastSpokenAt >= 4500)) {
        this.lastSpokenKey = callout.key;
        this.lastSpokenAt = now;
        this.speakNow(callout.text, callout.urgent, false);
      }
    }
  }

  public handleGuidance(guidance: GuidancePayload): void {
    if (this.profile === 'tactile_only') return;

    const now = performance.now();
    if (this.profile === 'metal_detector') {
      this.handleMetalDetectorAudio(guidance, now);
      return;
    }
    if (this.profile === 'headphones_3d') {
      this.handleHeadphones3D(guidance, now);
      return;
    }

    const isCurb = guidance.curb.detected || guidance.steering === 'CURB_DROP';
    const minDist = Math.min(
      guidance.left.min_distance_m,
      guidance.center.min_distance_m,
      guidance.right.min_distance_m
    );

    // 1. CLEAR PATH: 100% calm silence (only say a soft "Clear" once after leaving sustained danger!)
    if (guidance.overall_severity === 'clear' && !isCurb) {
      this.lastRank = 0;
      this.clearStreak += 1;

      if (this.clearStreak >= 3) {
        if (
          this.hadSustainedDanger &&
          this.voiceEnabled &&
          now - this.lastSpokenAt >= 1200
        ) {
          this.hadSustainedDanger = false;
          // Gentle ascending two-note "all clear" chime instead of verbose chatter, or 1 soft word
          if (this.sonarEnabled) {
            this.playAllClearChime();
          } else {
            this.speakNow(this.voiceLang === 'ru' ? 'Свободно' : 'Clear', false, false);
          }
          this.lastSpokenKey = 'CLEAR';
          this.lastSpokenAt = now;
        }
        this.activeHazardKey = 'CLEAR';
        this.hazardStreak = 0;
        this.sameStatePingCount = 0;
        this.minDistAtLastAlert = 99;
      }
      return;
    }

    this.clearStreak = 0;
    const callout = getCrispCallout(guidance, this.voiceLang);
    const nextKey = callout?.key ?? `${guidance.overall_severity}:${guidance.steering}`;
    const isUrgent =
      isCurb || guidance.steering === 'STOP' || guidance.overall_severity === 'danger';

    // 2. Stability debounce: urgent hazards trigger immediately; mild caution needs 2 frames
    if (nextKey === this.pendingKey) {
      this.pendingStreak += 1;
    } else {
      this.pendingKey = nextKey;
      this.pendingStreak = 1;
    }
    if (!isUrgent && this.pendingStreak < 2) {
      return;
    }

    const stateChanged = nextKey !== this.activeHazardKey;
    if (stateChanged) {
      this.activeHazardKey = nextKey;
      this.sameStatePingCount = 0;
      this.minDistAtLastAlert = minDist;
    }

    if (isUrgent) {
      this.hazardStreak += 1;
      if (this.hazardStreak >= 3) {
        this.hadSustainedDanger = true;
      }
    }

    // Did the user physically walk closer by >= 28cm since the last alert?
    const movedCloser = this.minDistAtLastAlert - minDist >= 0.28;

    // 3. 3D HRTF Sonar:
    //    - In `smart_calm`: completely silent on mild `caution` (haptics handles it!),
    //      fires once on new urgent state or closing distance, then backs off cleanly when stationary!
    //    - In `sonar_3d`: continuous distance radar with gentle stationary backoff.
    if (this.sonarEnabled) {
      const skipMildCautionInCalm =
        this.profile === 'smart_calm' && !isUrgent && guidance.overall_severity === 'caution';

      if (!skipMildCautionInCalm) {
        const rank = isCurb ? 4 : guidance.steering === 'STOP' ? 3 : isUrgent ? 2 : 1;
        const baseInterval = isCurb
          ? 650
          : Math.round(clamp(220 + 260 * Math.pow(clamp(minDist, 0.35, 3.5), 1.25), 260, 1100));

        // Exponential anti-nag backoff when standing still looking at the same object!
        const stationaryBackoff =
          this.profile === 'smart_calm'
            ? Math.min(8000, baseInterval * Math.pow(2.1, Math.min(4, this.sameStatePingCount)))
            : Math.min(2600, baseInterval * (this.sameStatePingCount >= 4 ? 1.8 : 1.0));

        const shouldPing =
          stateChanged ||
          rank > this.lastRank ||
          (movedCloser && now - this.lastPingAt >= 350) ||
          (this.profile === 'sonar_3d' && now - this.lastPingAt >= stationaryBackoff) ||
          (this.profile === 'smart_calm' &&
            this.sameStatePingCount < 2 &&
            now - this.lastPingAt >= stationaryBackoff);

        if (shouldPing) {
          this.lastPingAt = now;
          this.lastRank = rank;
          this.sameStatePingCount += 1;
          if (movedCloser) {
            this.minDistAtLastAlert = minDist;
          }
          this.emitSpatialEarcon(guidance, isCurb, minDist);
        }
      }
    }

    // 4. Voice Callouts (Zero-Nag Rally Co-Driver):
    //    - In `smart_calm`: ONLY speaks on URGENT hazards (`DANGER`, `STOP`, `CURB`, `LENS_BLOCKED`),
    //      NEVER nags on mild `caution` walls!
    //    - Speaks ONCE when entering a new hazard state, and NEVER repeats while standing still
    //      unless you walk >= 32cm closer OR stay in `STOP` after 7.5 seconds!
    if (this.voiceEnabled && callout) {
      const allowVoiceForSeverity =
        this.profile === 'sonar_3d' ? true : callout.urgent;

      if (allowVoiceForSeverity) {
        const isNewSpokenState = callout.key !== this.lastSpokenKey;
        const minGapMs = callout.urgent ? 1350 : 2800;
        const approachedSignificantly =
          this.minDistAtLastAlert - minDist >= 0.32 && now - this.lastSpokenAt >= 1800;
        const rareSafetyReminder =
          (guidance.steering === 'STOP' || isCurb) &&
          minDist <= 1.0 &&
          now - this.lastSpokenAt >= 7500;

        if (
          (isNewSpokenState && now - this.lastSpokenAt >= minGapMs) ||
          (isNewSpokenState && callout.urgent && this.lastSpokenKey === 'CLEAR') ||
          approachedSignificantly ||
          rareSafetyReminder
        ) {
          this.lastSpokenKey = callout.key;
          this.minDistAtLastAlert = minDist;
          this.speakNow(callout.text, callout.urgent, false);
        }
      }
    }
  }

  private handleMetalDetectorAudio(guidance: GuidancePayload, now: number): void {
    const block = evaluateForwardBlock(guidance);
    if (!block.blocked) {
      this.lastRank = 0;
      this.activeHazardKey = 'CLEAR';
      this.lastSpokenKey = 'CLEAR';
      return;
    }

    const rank =
      block.kind === 'curb'
        ? 4
        : block.kind === 'contact'
          ? 3
          : block.kind === 'close'
            ? 2
            : 1;

    if (this.sonarEnabled) {
      const intervalMs =
        block.kind === 'curb'
          ? 400
          : block.kind === 'contact'
            ? 175
            : block.kind === 'close'
              ? 230
              : 340;
      const justEntered = this.activeHazardKey === 'CLEAR';
      const escalated = rank > this.lastRank;

      if (justEntered || escalated || now - this.lastPingAt >= intervalMs) {
        this.lastPingAt = now;
        this.lastRank = rank;
        this.activeHazardKey = block.reason;
        this.unlock();
        if (this.ctx) {
          const t0 = this.ctx.currentTime;
          if (block.kind === 'curb') {
            this.schedule3DPing({
              startTime: t0,
              freqStart: 680,
              freqEnd: 260,
              durationSec: 0.14,
              posX: 0,
              posY: -0.75,
              posZ: -0.8,
              volume: 0.2,
              cutoffHz: 2200,
            });
          } else {
            const freq =
              block.kind === 'contact' ? 940 : block.kind === 'close' ? 740 : 540;
            this.schedule3DPing({
              startTime: t0,
              freqStart: freq,
              freqEnd: freq,
              durationSec: block.kind === 'contact' ? 0.08 : 0.065,
              posX: 0,
              posY: 0,
              posZ: -clamp(block.distanceM, 0.35, 1.5),
              volume: block.kind === 'contact' ? 0.18 : 0.13,
              cutoffHz: 2600,
            });
          }
        }
      }
    } else {
      this.activeHazardKey = block.reason;
      this.lastRank = rank;
    }

    if (
      this.voiceEnabled &&
      (block.kind === 'contact' || block.kind === 'curb') &&
      (this.lastSpokenKey !== block.kind || now - this.lastSpokenAt >= 3200)
    ) {
      this.lastSpokenKey = block.kind;
      const word =
        block.kind === 'curb'
          ? this.voiceLang === 'ru'
            ? 'Ступенька'
            : 'Step down'
          : this.voiceLang === 'ru'
            ? 'Стоп'
            : 'Stop';
      this.speakNow(word, true, false);
    }
  }

  private playAllClearChime(): void {
    this.unlock();
    if (!this.ctx) return;
    const t0 = this.ctx.currentTime;
    this.schedule3DPing({
      startTime: t0,
      freqStart: 440,
      freqEnd: 587,
      durationSec: 0.085,
      posX: 0,
      posY: 0.1,
      posZ: -1.4,
      volume: 0.08,
      cutoffHz: 1600,
    });
  }

  private emitSpatialEarcon(
    guidance: GuidancePayload,
    isCurb: boolean,
    minDist: number
  ): void {
    this.unlock();
    if (!this.ctx) return;
    const t0 = this.ctx.currentTime;

    if (isCurb) {
      this.schedule3DPing({
        startTime: t0,
        freqStart: 660,
        freqEnd: 240,
        durationSec: 0.19,
        posX: 0,
        posY: -0.85,
        posZ: -0.9,
        volume: 0.2,
        cutoffHz: 2200,
      });
      return;
    }

    const posX = this.computeObstacleX(guidance);
    const posY = guidance.overhead_hazard ? 0.75 : 0.0;
    const posZ = -clamp(minDist, 0.4, 3.6);
    const freq = Math.round(880 - 150 * clamp(minDist, 0.4, 3.2));
    const isDanger = guidance.overall_severity === 'danger';

    this.schedule3DPing({
      startTime: t0,
      freqStart: freq,
      freqEnd: freq,
      durationSec: isDanger ? 0.095 : 0.065,
      posX,
      posY,
      posZ,
      volume: isDanger ? 0.16 : 0.09,
      cutoffHz: isDanger ? 2600 : 1600,
    });

    if (guidance.steering === 'VEER_LEFT' || guidance.steering === 'VEER_RIGHT') {
      const safeX = guidance.steering === 'VEER_LEFT' ? -0.85 : 0.85;
      this.schedule3DPing({
        startTime: t0 + 0.09,
        freqStart: 528,
        freqEnd: 528,
        durationSec: 0.06,
        posX: safeX,
        posY: 0.05,
        posZ: -1.4,
        volume: 0.06,
        cutoffHz: 1400,
      });
    }
  }

  private speakNow(
    text: string,
    urgent: boolean,
    forceEvenIfMuted: boolean
  ): void {
    if (
      (!this.voiceEnabled && !forceEvenIfMuted) ||
      typeof window === 'undefined' ||
      !('speechSynthesis' in window)
    ) {
      return;
    }
    this.lastSpokenAt = performance.now();

    const synth = window.speechSynthesis;
    const langTag = this.voiceLang === 'ru' ? 'ru-RU' : 'en-US';

    if (urgent || forceEvenIfMuted) {
      synth.cancel();
      if (this.speakTimer !== null) {
        window.clearTimeout(this.speakTimer);
      }
      this.speakTimer = window.setTimeout(() => {
        this.speakTimer = null;
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = langTag;
        utter.rate = this.voiceLang === 'ru' ? 1.18 : 1.2;
        synth.speak(utter);
      }, 45);
    } else if (!synth.speaking) {
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = langTag;
      utter.rate = 1.12;
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

  /**
   * Cross-platform stereo panning: uses StereoPannerNode (Android Chrome & iOS 14.5+)
   * with automatic PannerNode equalpower fallback for older iOS WebKit builds.
   */
  private scheduleStereoTone(opts: {
    startTime: number;
    freqStart: number;
    freqEnd: number;
    durationSec: number;
    pan: number;
    volume: number;
  }): void {
    if (!this.ctx || !this.compressor) return;
    const { startTime, freqStart, freqEnd, durationSec, pan, volume } = opts;
    const clampedPan = clamp(pan, -1, 1);

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = freqStart < 320 ? 'sine' : 'triangle';
    osc.frequency.setValueAtTime(Math.max(60, freqStart), startTime);
    if (Math.abs(freqEnd - freqStart) > 1) {
      osc.frequency.exponentialRampToValueAtTime(Math.max(60, freqEnd), startTime + durationSec);
    }

    gain.gain.setValueAtTime(0.0008, startTime);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.002, volume), startTime + Math.min(0.014, durationSec * 0.3));
    gain.gain.exponentialRampToValueAtTime(0.0008, startTime + durationSec);
    osc.connect(gain);

    let panNode: AudioNode | null = null;
    if (typeof this.ctx.createStereoPanner === 'function') {
      const sp = this.ctx.createStereoPanner();
      sp.pan.setValueAtTime(clampedPan, startTime);
      gain.connect(sp);
      sp.connect(this.compressor);
      panNode = sp;
    } else if (typeof this.ctx.createPanner === 'function') {
      const p = this.ctx.createPanner();
      p.panningModel = 'equalpower';
      const z = -Math.sqrt(Math.max(0.05, 1 - clampedPan * clampedPan));
      if (p.positionX) {
        p.positionX.setValueAtTime(clampedPan, startTime);
        p.positionY.setValueAtTime(0, startTime);
        p.positionZ.setValueAtTime(z, startTime);
      } else {
        p.setPosition(clampedPan, 0, z);
      }
      gain.connect(p);
      p.connect(this.compressor);
      panNode = p;
    } else {
      gain.connect(this.compressor);
    }

    osc.onended = () => {
      try {
        osc.disconnect();
        gain.disconnect();
        panNode?.disconnect();
      } catch {
        // Ignore already-disconnected nodes
      }
    };
    osc.start(startTime);
    osc.stop(startTime + durationSec + 0.015);
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
      panner.refDistance = 0.8;
      panner.maxDistance = 8.0;
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
    } else {
      gain.connect(this.compressor);
    }

    osc.onended = () => {
      try {
        osc.disconnect();
        filter.disconnect();
        gain.disconnect();
        spatialNode?.disconnect();
      } catch {
        // Ignore already-disconnected nodes
      }
    };

    osc.start(startTime);
    osc.stop(startTime + durationSec + 0.015);
  }
}
