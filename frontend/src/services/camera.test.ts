import assert from 'node:assert/strict';
import test from 'node:test';
import {
  acquireCameraStream,
  classifyLabelFacingMode,
  switchCameraStream,
} from './camera.ts';

interface FakeTrack {
  label: string;
  stopped: boolean;
  stop: () => void;
  getSettings: () => { deviceId?: string; facingMode?: string };
}

function makeFakeStream(opts: {
  deviceId: string;
  label: string;
  facingMode?: string;
}): MediaStream & { _track: FakeTrack } {
  const track: FakeTrack = {
    label: opts.label,
    stopped: false,
    stop() {
      this.stopped = true;
    },
    getSettings() {
      return {
        deviceId: opts.deviceId,
        ...(opts.facingMode ? { facingMode: opts.facingMode } : {}),
      };
    },
  };
  return {
    _track: track,
    getTracks: () => [track],
    getVideoTracks: () => [track],
  } as unknown as MediaStream & { _track: FakeTrack };
}

function setNavigatorMediaDevices(mediaDevices: Partial<MediaDevices>): () => void {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { mediaDevices },
  });
  return () => {
    if (original) {
      Object.defineProperty(globalThis, 'navigator', original);
    }
  };
}

test('classifyLabelFacingMode supports English and Russian iOS camera labels', () => {
  assert.equal(classifyLabelFacingMode('Back Camera'), 'environment');
  assert.equal(classifyLabelFacingMode('Back Dual Wide Camera'), 'environment');
  assert.equal(classifyLabelFacingMode('Задняя камера'), 'environment');
  assert.equal(classifyLabelFacingMode('Front Camera'), 'user');
  assert.equal(classifyLabelFacingMode('Фронтальная камера'), 'user');
  assert.equal(classifyLabelFacingMode('FaceTime HD Camera'), 'user');
});

test('iPhone Safari initial acquire: auto-corrects to main Back Camera when first call opens Front', async () => {
  const calls: MediaStreamConstraints[] = [];
  let permissionGranted = false;

  const frontStream = makeFakeStream({
    deviceId: 'ios-front',
    label: 'Фронтальная камера',
    facingMode: 'user',
  });
  const backMainStream = makeFakeStream({
    deviceId: 'ios-back-main',
    label: 'Задняя камера',
    facingMode: 'environment',
  });

  const restore = setNavigatorMediaDevices({
    enumerateDevices: async () =>
      permissionGranted
        ? ([
            { kind: 'videoinput', deviceId: 'ios-front', label: 'Фронтальная камера' },
            { kind: 'videoinput', deviceId: 'ios-back-ultra', label: 'Сверхширокоугольная задняя камера' },
            { kind: 'videoinput', deviceId: 'ios-back-main', label: 'Задняя камера' },
          ] as MediaDeviceInfo[])
        : [],
    getUserMedia: async (constraints?: MediaStreamConstraints) => {
      calls.push(constraints ?? {});
      const v = constraints?.video;
      // Simulate iOS WKWebView rejecting `exact: environment` before permission
      if (
        typeof v === 'object' &&
        v !== null &&
        typeof v.facingMode === 'object' &&
        v.facingMode !== null &&
        'exact' in v.facingMode
      ) {
        const err = new Error('Overconstrained');
        err.name = 'OverconstrainedError';
        throw err;
      }
      // Stage 4 explicit deviceId selection
      if (
        typeof v === 'object' &&
        v !== null &&
        typeof v.deviceId === 'object' &&
        v.deviceId !== null &&
        'exact' in v.deviceId &&
        v.deviceId.exact === 'ios-back-main'
      ) {
        return backMainStream;
      }
      // Simulate iOS returning front camera on initial ideal call
      permissionGranted = true;
      return frontStream;
    },
  });

  try {
    const res = await acquireCameraStream('environment', { cooldownMs: 0 });
    assert.equal(res.actualFacingMode, 'environment');
    assert.equal(res.deviceId, 'ios-back-main');
    assert.equal(frontStream._track.stopped, true);
    // Ensure we never passed 640x480 width/height constraints that bias iOS toward front camera
    for (const c of calls) {
      if (typeof c.video === 'object' && c.video !== null) {
        assert.equal('width' in c.video, false);
        assert.equal('height' in c.video, false);
      }
    }
  } finally {
    restore();
  }
});

test('iPhone switchCameraStream: stops front camera, switches to main Back Camera (skipping Ultra Wide)', async () => {
  const frontStream = makeFakeStream({
    deviceId: 'ios-front',
    label: 'Front Camera',
    facingMode: 'user',
  });
  const backMainStream = makeFakeStream({
    deviceId: 'ios-back-1x',
    label: 'Back Camera',
    facingMode: 'environment',
  });

  const restore = setNavigatorMediaDevices({
    enumerateDevices: async () =>
      [
        { kind: 'videoinput', deviceId: 'ios-front', label: 'Front Camera' },
        { kind: 'videoinput', deviceId: 'ios-back-uw', label: 'Back Ultra Wide Camera' },
        { kind: 'videoinput', deviceId: 'ios-back-tele', label: 'Back Telephoto Camera' },
        { kind: 'videoinput', deviceId: 'ios-back-1x', label: 'Back Camera' },
      ] as MediaDeviceInfo[],
    getUserMedia: async (constraints?: MediaStreamConstraints) => {
      const v = constraints?.video;
      if (
        typeof v === 'object' &&
        v !== null &&
        typeof v.facingMode === 'object' &&
        v.facingMode !== null &&
        'exact' in v.facingMode
      ) {
        const err = new Error('Overconstrained');
        err.name = 'OverconstrainedError';
        throw err;
      }
      if (
        typeof v === 'object' &&
        v !== null &&
        typeof v.deviceId === 'object' &&
        v.deviceId !== null &&
        'exact' in v.deviceId &&
        v.deviceId.exact === 'ios-back-1x'
      ) {
        return backMainStream;
      }
      return frontStream;
    },
  });

  try {
    const res = await switchCameraStream(frontStream, 'environment', {
      cooldownMs: 0,
      retryCooldownMs: 0,
    });
    assert.equal(frontStream._track.stopped, true);
    assert.equal(res.actualFacingMode, 'environment');
    assert.equal(res.deviceId, 'ios-back-1x');
    assert.equal(res.switchedDevice, true);
  } finally {
    restore();
  }
});
