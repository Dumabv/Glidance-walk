import assert from 'node:assert/strict';
import test from 'node:test';
import { computeHeadphonesCue, SpatialAudioEngine } from './spatialAudio.ts';
import type { GuidancePayload } from '../types/telemetry.ts';

function makeGuidance(overrides: Partial<GuidancePayload> = {}): GuidancePayload {
  return {
    frame_id: 1,
    client_ts: 0,
    server_ts: 0,
    overall_severity: 'clear',
    steering: 'CLEAR',
    spoken_cue: 'Path clear.',
    left: { severity: 'clear', obstacle_score: 0.05, min_distance_m: 4.0, coverage_ratio: 0.0 },
    center: { severity: 'clear', obstacle_score: 0.05, min_distance_m: 4.0, coverage_ratio: 0.0 },
    right: { severity: 'clear', obstacle_score: 0.05, min_distance_m: 4.0, coverage_ratio: 0.0 },
    curb: { detected: false, confidence: 0.0, drop_row_ratio: 0.7, gradient_peak: 0.0, stairs_likely: false },
    objects: [],
    latency: { decode_ms: 1, inference_ms: 25, yolo_ms: 20, hazard_ms: 0.2, server_total_ms: 34 },
    device: 'cuda-rtx',
    model_name: 'Small-FP16',
    kernel_backend: 'rust-simd',
    ...overrides,
  };
}

test('Headphones 3D: CLEAR path stays completely silent', () => {
  const plan = computeHeadphonesCue(makeGuidance());
  assert.equal(plan.kind, 'silence');
  assert.equal(plan.volume, 0);
});

test('Headphones 3D: obstacle on LEFT (VEER_RIGHT) pans to LEFT ear (-0.88)', () => {
  const plan = computeHeadphonesCue(
    makeGuidance({
      overall_severity: 'danger',
      steering: 'VEER_RIGHT',
      left: { severity: 'danger', obstacle_score: 0.85, min_distance_m: 0.9, coverage_ratio: 0.5 },
    })
  );
  assert.equal(plan.kind, 'left_obstacle');
  assert.equal(plan.pan, -0.88);
  assert.equal(plan.doublePulse, true);
});

test('Headphones 3D: obstacle on RIGHT (VEER_LEFT) pans to RIGHT ear (+0.88)', () => {
  const plan = computeHeadphonesCue(
    makeGuidance({
      overall_severity: 'caution',
      steering: 'VEER_LEFT',
      right: { severity: 'caution', obstacle_score: 0.6, min_distance_m: 1.7, coverage_ratio: 0.3 },
    })
  );
  assert.equal(plan.kind, 'right_obstacle');
  assert.equal(plan.pan, 0.88);
  assert.equal(plan.doublePulse, false);
});

test('Headphones 3D: STOP / wall_override plays low double bass in BOTH ears (pan = 0)', () => {
  const plan = computeHeadphonesCue(
    makeGuidance({
      overall_severity: 'danger',
      steering: 'STOP',
      wall_override: true,
      center: { severity: 'danger', obstacle_score: 0.92, min_distance_m: 0.5, coverage_ratio: 0.85 },
    })
  );
  assert.equal(plan.kind, 'wall_stop');
  assert.equal(plan.pan, 0);
  assert.equal(plan.doublePulse, true);
  assert.ok(plan.freqStart < 320);
});

test('Headphones 3D: CURB_DROP / stairs plays descending chime in BOTH ears (pan = 0)', () => {
  const plan = computeHeadphonesCue(
    makeGuidance({
      overall_severity: 'danger',
      steering: 'CURB_DROP',
      curb: { detected: true, confidence: 0.8, drop_row_ratio: 0.78, gradient_peak: 0.14, stairs_likely: true },
    })
  );
  assert.equal(plan.kind, 'stairs');
  assert.equal(plan.pan, 0);
  assert.ok(plan.freqStart > plan.freqEnd);
  assert.equal(plan.doublePulse, true);
});

test('Headphones 3D: closer obstacle produces faster pulse & higher pitch', () => {
  const far = computeHeadphonesCue(
    makeGuidance({
      overall_severity: 'caution',
      steering: 'VEER_LEFT',
      right: { severity: 'caution', obstacle_score: 0.5, min_distance_m: 2.1, coverage_ratio: 0.2 },
    })
  );
  const near = computeHeadphonesCue(
    makeGuidance({
      overall_severity: 'danger',
      steering: 'VEER_LEFT',
      right: { severity: 'danger', obstacle_score: 0.9, min_distance_m: 0.55, coverage_ratio: 0.6 },
    })
  );
  assert.ok(near.intervalMs < far.intervalMs);
  assert.ok(near.freqStart > far.freqStart);
});

test('Android Chrome AudioContext: StereoPannerNode emits L (-0.9), C (0), R (+0.9) on test sweep', () => {
  const pans: number[] = [];
  const makeParam = () => ({ setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {} });
  class MockAndroidAudioContext {
    currentTime = 0;
    state = 'running';
    destination = {};
    resume() { return Promise.resolve(); }
    close() { return Promise.resolve(); }
    createDynamicsCompressor() {
      return { threshold: makeParam(), knee: makeParam(), ratio: makeParam(), attack: makeParam(), release: makeParam(), connect: () => {} };
    }
    createBuffer() { return {}; }
    createBufferSource() { return { buffer: null, connect: () => {}, start: () => {} }; }
    createOscillator() { return { type: 'sine', frequency: makeParam(), connect: () => {}, start: () => {}, stop: () => {}, disconnect: () => {} }; }
    createGain() { return { gain: makeParam(), connect: () => {}, disconnect: () => {} }; }
    createStereoPanner() {
      return {
        pan: { setValueAtTime: (v: number) => pans.push(v) },
        connect: () => {},
        disconnect: () => {},
      };
    }
  }

  (globalThis as unknown as { window: unknown }).window = { AudioContext: MockAndroidAudioContext };
  const eng = new SpatialAudioEngine();
  eng.setProfile('headphones_3d');
  eng.playHeadphonesLRTest();
  assert.deepEqual(pans, [-0.9, 0, 0.9]);
  eng.dispose();
});

test('iOS Safari fallback: unlocks audioSession playback + silent buffer & falls back to PannerNode equalpower', () => {
  const posXs: number[] = [];
  let silentBufferStarted = false;
  const audioSession = { type: 'auto' };
  const makeParam = () => ({ setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {} });

  class MockIosWebkitAudioContext {
    currentTime = 0;
    state = 'suspended';
    destination = {};
    resume() { this.state = 'running'; return Promise.resolve(); }
    close() { return Promise.resolve(); }
    createDynamicsCompressor() {
      return { threshold: makeParam(), knee: makeParam(), ratio: makeParam(), attack: makeParam(), release: makeParam(), connect: () => {} };
    }
    createBuffer() { return {}; }
    createBufferSource() { return { buffer: null, connect: () => {}, start: () => { silentBufferStarted = true; } }; }
    createOscillator() { return { type: 'sine', frequency: makeParam(), connect: () => {}, start: () => {}, stop: () => {}, disconnect: () => {} }; }
    createGain() { return { gain: makeParam(), connect: () => {}, disconnect: () => {} }; }
    // Note: NO createStereoPanner on older iOS Safari -> must use PannerNode!
    createPanner() {
      return {
        panningModel: '',
        positionX: { setValueAtTime: (v: number) => posXs.push(v) },
        positionY: { setValueAtTime: () => {} },
        positionZ: { setValueAtTime: () => {} },
        connect: () => {},
        disconnect: () => {},
      };
    }
  }

  (globalThis as unknown as { window: unknown }).window = { webkitAudioContext: MockIosWebkitAudioContext };
  Object.defineProperty(globalThis, 'navigator', { value: { audioSession }, configurable: true });
  const eng = new SpatialAudioEngine();
  eng.setProfile('headphones_3d');
  eng.playHeadphonesLRTest();

  assert.equal(audioSession.type, 'playback');
  assert.equal(silentBufferStarted, true);
  assert.deepEqual(posXs, [-0.9, 0, 0.9]);
  eng.dispose();
});
