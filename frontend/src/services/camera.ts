export type CameraFacingMode = 'environment' | 'user';

export interface CameraAcquisitionResult {
  stream: MediaStream;
  actualFacingMode: CameraFacingMode;
  deviceId: string | null;
  deviceLabel: string | null;
  usedFallback: boolean;
  switchedDevice: boolean;
}

const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 480;
const DEFAULT_FRAME_RATE: ConstrainDoubleRange = { ideal: 24, max: 30 };
const HARDWARE_RELEASE_RETRY_MS = 180;

const RETRYABLE_CONSTRAINT_ERRORS = new Set([
  'OverconstrainedError',
  'NotFoundError',
  'ConstraintNotSatisfiedError',
  'DevicesNotFoundError',
]);

const TRANSIENT_HARDWARE_ERRORS = new Set([
  'NotReadableError',
  'TrackStartError',
  'AbortError',
]);

/**
 * Stops every track on a MediaStream so hardware locks (especially iOS Safari's
 * single-active-camera limit) are released before opening another lens.
 */
export function stopMediaStream(stream: MediaStream | null): void {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

export function getOppositeFacingMode(
  current: CameraFacingMode
): CameraFacingMode {
  return current === 'environment' ? 'user' : 'environment';
}

export function getCameraLabel(mode: CameraFacingMode): string {
  return mode === 'environment' ? 'Back' : 'Front';
}

function getErrorName(err: unknown): string {
  if (!err || typeof err !== 'object') return '';
  return 'name' in err ? String((err as { name: unknown }).name) : '';
}

function isRetryableConstraintError(err: unknown): boolean {
  return RETRYABLE_CONSTRAINT_ERRORS.has(getErrorName(err));
}

function isTransientHardwareError(err: unknown): boolean {
  return TRANSIENT_HARDWARE_ERRORS.has(getErrorName(err));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function getVideoTrackDeviceId(stream: MediaStream | null): string | null {
  const track = stream?.getVideoTracks()[0];
  const deviceId = track?.getSettings?.()?.deviceId;
  return typeof deviceId === 'string' && deviceId.length > 0 ? deviceId : null;
}

function getVideoTrackLabel(stream: MediaStream | null): string | null {
  const label = stream?.getVideoTracks()[0]?.label?.trim();
  return label ? label : null;
}

function inferFacingMode(
  stream: MediaStream,
  requested: CameraFacingMode,
  strictMatched: boolean
): CameraFacingMode {
  const track = stream.getVideoTracks()[0];
  if (!track) return requested;

  const settings = track.getSettings?.();
  if (settings?.facingMode === 'environment' || settings?.facingMode === 'user') {
    return settings.facingMode;
  }

  if (strictMatched) {
    return requested;
  }

  const label = track.label.toLowerCase();
  if (/\b(front|user|facetime|truedepth|integrated|built-in|webcam)\b/.test(label)) {
    return 'user';
  }
  if (/\b(back|rear|environment|world)\b/.test(label)) {
    return 'environment';
  }

  return requested;
}

async function findNextVideoInputDeviceId(
  currentDeviceId: string | null
): Promise<string | null> {
  if (!navigator.mediaDevices?.enumerateDevices) return null;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const seen = new Set<string>();
    const videoInputs = devices.filter((d) => {
      if (d.kind !== 'videoinput' || !d.deviceId || seen.has(d.deviceId)) {
        return false;
      }
      seen.add(d.deviceId);
      return true;
    });

    if (videoInputs.length < 2) return null;
    if (!currentDeviceId) return videoInputs[0].deviceId;

    const currentIdx = videoInputs.findIndex(
      (d) => d.deviceId === currentDeviceId
    );
    if (currentIdx === -1) {
      return (
        videoInputs.find((d) => d.deviceId !== currentDeviceId)?.deviceId ??
        null
      );
    }
    return videoInputs[(currentIdx + 1) % videoInputs.length].deviceId;
  } catch {
    return null;
  }
}

function buildResult(
  stream: MediaStream,
  requested: CameraFacingMode,
  strictMatched: boolean,
  previousDeviceId: string | null | undefined
): CameraAcquisitionResult {
  const deviceId = getVideoTrackDeviceId(stream);
  const deviceLabel = getVideoTrackLabel(stream);
  const actualFacingMode = inferFacingMode(stream, requested, strictMatched);
  const switchedDevice = previousDeviceId
    ? deviceId !== null
      ? deviceId !== previousDeviceId
      : strictMatched
    : true;

  return {
    stream,
    actualFacingMode,
    deviceId,
    deviceLabel,
    usedFallback: !strictMatched,
    switchedDevice,
  };
}

async function tryAttempts(
  attempts: { constraints: MediaStreamConstraints; strict: boolean }[],
  requested: CameraFacingMode,
  previousDeviceId: string | null | undefined
): Promise<{ result: CameraAcquisitionResult | null; lastError: unknown }> {
  let lastError: unknown = null;

  for (const attempt of attempts) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        attempt.constraints
      );
      return {
        result: buildResult(
          stream,
          requested,
          attempt.strict,
          previousDeviceId
        ),
        lastError: null,
      };
    } catch (err) {
      lastError = err;
      // Permission denials or hardware-busy errors should surface immediately
      // rather than retrying constraint fallbacks and prompting repeatedly.
      if (!isRetryableConstraintError(err)) {
        throw err;
      }
    }
  }

  return { result: null, lastError };
}

/**
 * Requests a camera stream across iPhone (iOS Safari), Android, Windows, and macOS:
 *
 * 1. Tries `facingMode: { exact: preferredFacingMode }` first so iOS Safari and
 *    Android reliably open the main rear camera instead of scoring the front
 *    camera closer to 640x480.
 * 2. When switching on desktop OSes (Windows/macOS) whose webcams do not report
 *    `facingMode`, cycles through available `videoinput` deviceIds (e.g.,
 *    MacBook FaceTime <-> Continuity Camera / USB webcam).
 * 3. Falls back gracefully to `ideal` and `video: true` for single-camera laptops.
 */
export async function acquireCameraStream(
  preferredFacingMode: CameraFacingMode = 'environment',
  options?: { previousDeviceId?: string | null }
): Promise<CameraAcquisitionResult> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      'Camera API is unavailable in this browser context. Open over HTTPS or localhost.'
    );
  }

  const previousDeviceId = options?.previousDeviceId;

  // Stage 1: Exact facingMode (iPhone, Android, Windows 2-in-1 / Surface)
  const strictOutcome = await tryAttempts(
    [
      {
        strict: true,
        constraints: {
          video: {
            facingMode: { exact: preferredFacingMode },
            width: { ideal: DEFAULT_WIDTH },
            height: { ideal: DEFAULT_HEIGHT },
            frameRate: DEFAULT_FRAME_RATE,
          },
          audio: false,
        },
      },
      {
        strict: true,
        constraints: {
          video: {
            facingMode: { exact: preferredFacingMode },
          },
          audio: false,
        },
      },
    ],
    preferredFacingMode,
    previousDeviceId
  );

  if (strictOutcome.result) {
    return strictOutcome.result;
  }

  // Stage 2: Desktop multi-camera switch via deviceId (Windows/macOS webcams)
  if (previousDeviceId) {
    const nextDeviceId = await findNextVideoInputDeviceId(previousDeviceId);
    if (nextDeviceId) {
      const deviceOutcome = await tryAttempts(
        [
          {
            strict: false,
            constraints: {
              video: {
                deviceId: { exact: nextDeviceId },
                width: { ideal: DEFAULT_WIDTH },
                height: { ideal: DEFAULT_HEIGHT },
                frameRate: DEFAULT_FRAME_RATE,
              },
              audio: false,
            },
          },
          {
            strict: false,
            constraints: {
              video: {
                deviceId: { exact: nextDeviceId },
              },
              audio: false,
            },
          },
        ],
        preferredFacingMode,
        previousDeviceId
      );

      if (deviceOutcome.result) {
        return deviceOutcome.result;
      }
    }
  }

  // Stage 3: Universal fallback (single-camera MacBook / Windows laptop)
  const fallbackOutcome = await tryAttempts(
    [
      {
        strict: false,
        constraints: {
          video: {
            facingMode: { ideal: preferredFacingMode },
            width: { ideal: DEFAULT_WIDTH },
            height: { ideal: DEFAULT_HEIGHT },
            frameRate: DEFAULT_FRAME_RATE,
          },
          audio: false,
        },
      },
      {
        strict: false,
        constraints: {
          video: {
            facingMode: { ideal: preferredFacingMode },
          },
          audio: false,
        },
      },
      {
        strict: false,
        constraints: {
          video: true,
          audio: false,
        },
      },
    ],
    preferredFacingMode,
    previousDeviceId
  );

  if (fallbackOutcome.result) {
    return fallbackOutcome.result;
  }

  const finalError = fallbackOutcome.lastError ?? strictOutcome.lastError;
  throw finalError instanceof Error
    ? finalError
    : new Error('Unable to access any camera device.');
}

/**
 * Releases the current camera stream (required on iOS Safari before opening a
 * second lens), retries once if the camera HAL is still releasing, and opens
 * the opposite facingMode (or next desktop webcam on Windows/macOS).
 */
export async function switchCameraStream(
  currentStream: MediaStream | null,
  targetFacingMode: CameraFacingMode
): Promise<CameraAcquisitionResult> {
  const previousDeviceId = getVideoTrackDeviceId(currentStream);
  stopMediaStream(currentStream);

  try {
    return await acquireCameraStream(targetFacingMode, { previousDeviceId });
  } catch (err) {
    if (!isTransientHardwareError(err)) {
      throw err;
    }
    // Mobile Safari / Android HALs occasionally need a brief cooldown after track.stop().
    await sleep(HARDWARE_RELEASE_RETRY_MS);
    return acquireCameraStream(targetFacingMode, { previousDeviceId });
  }
}
