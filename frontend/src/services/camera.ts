export type CameraFacingMode = 'environment' | 'user';

export interface CameraAcquisitionResult {
  stream: MediaStream;
  actualFacingMode: CameraFacingMode;
  usedFallback: boolean;
}

const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 480;
const DEFAULT_FRAME_RATE: ConstrainDoubleRange = { ideal: 24, max: 30 };

const RETRYABLE_CONSTRAINT_ERRORS = new Set([
  'OverconstrainedError',
  'NotFoundError',
  'ConstraintNotSatisfiedError',
  'DevicesNotFoundError',
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
  if (/\b(front|user|facetime|truedepth)\b/.test(label)) {
    return 'user';
  }
  if (/\b(back|rear|environment|world)\b/.test(label)) {
    return 'environment';
  }

  return requested;
}

function isRetryableConstraintError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const name = 'name' in err ? String((err as { name: unknown }).name) : '';
  return RETRYABLE_CONSTRAINT_ERRORS.has(name);
}

/**
 * Requests a camera stream prioritizing an exact facingMode match first.
 *
 * Why: iOS Safari treats `facingMode: { ideal: 'environment' }` as a soft hint
 * and can pick the front camera if its native preset scores closer to 640x480.
 * Trying `{ exact: preferred }` first guarantees the rear camera on iPhone and
 * Android while gracefully falling back for single-camera laptops/desktops.
 */
export async function acquireCameraStream(
  preferredFacingMode: CameraFacingMode = 'environment'
): Promise<CameraAcquisitionResult> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera API is unavailable in this browser context.');
  }

  const attempts: {
    constraints: MediaStreamConstraints;
    strict: boolean;
  }[] = [
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
        video: true,
        audio: false,
      },
    },
  ];

  let lastError: unknown = null;

  for (const attempt of attempts) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        attempt.constraints
      );
      const actualFacingMode = inferFacingMode(
        stream,
        preferredFacingMode,
        attempt.strict
      );
      return {
        stream,
        actualFacingMode,
        usedFallback: !attempt.strict,
      };
    } catch (err) {
      lastError = err;
      // Permission denials or hardware-busy errors should surface immediately
      // rather than retrying and prompting repeatedly.
      if (!isRetryableConstraintError(err)) {
        throw err;
      }
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error('Unable to access any camera device.');
}

/**
 * Releases the current camera stream (required on iOS Safari before opening a
 * second lens) and opens the requested facingMode.
 */
export async function switchCameraStream(
  currentStream: MediaStream | null,
  targetFacingMode: CameraFacingMode
): Promise<CameraAcquisitionResult> {
  stopMediaStream(currentStream);
  return acquireCameraStream(targetFacingMode);
}
