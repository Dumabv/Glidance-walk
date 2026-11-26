export type CameraFacingMode = 'environment' | 'user';

export interface CameraAcquisitionResult {
  stream: MediaStream;
  actualFacingMode: CameraFacingMode;
  deviceId: string | null;
  deviceLabel: string | null;
  usedFallback: boolean;
  switchedDevice: boolean;
}

const SWITCH_COOLDOWN_MS = 140;
const HARDWARE_RELEASE_RETRY_MS = 240;

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

const FRONT_LABEL_RE =
  /(front|user|facetime|truedepth|selfie|integrated|built-in|webcam|фронтальн|передн|лицев|față|fata|avant)/i;
const BACK_LABEL_RE =
  /(back|rear|environment|world|задн|основн|тылов|spate|arrière|arriere|trasera|rück)/i;
const SECONDARY_REAR_LENS_RE =
  /(ultra\s*wide|telephoto|macro|depth|lidar|сверхширок|телефото|макро)/i;

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
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms));
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

export function classifyLabelFacingMode(label: string): CameraFacingMode | null {
  const trimmed = label.trim();
  if (!trimmed) return null;
  // Check back/rear before front because some labels include "Built-in Back Camera"
  if (BACK_LABEL_RE.test(trimmed)) return 'environment';
  if (FRONT_LABEL_RE.test(trimmed)) return 'user';
  return null;
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

  const fromLabel = classifyLabelFacingMode(track.label ?? '');
  if (fromLabel) {
    return fromLabel;
  }

  if (strictMatched) {
    return requested;
  }

  return requested;
}

async function listUniqueVideoInputs(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const seen = new Set<string>();
    return devices.filter((d) => {
      if (d.kind !== 'videoinput' || !d.deviceId || seen.has(d.deviceId)) {
        return false;
      }
      seen.add(d.deviceId);
      return true;
    });
  } catch {
    return [];
  }
}

/**
 * Picks the best matching `deviceId` for `targetFacingMode` (preferring the main
 * 1x rear wide lens over Ultra Wide / Telephoto on multi-lens iPhones), or cycles
 * to the next webcam on desktops that do not expose facingMode labels.
 */
async function findDeviceIdForFacingMode(
  targetFacingMode: CameraFacingMode,
  excludeDeviceId: string | null | undefined,
  allowGenericCycle: boolean
): Promise<string | null> {
  const videoInputs = await listUniqueVideoInputs();
  if (videoInputs.length === 0) return null;

  const matchingByLabel = videoInputs.filter(
    (d) => classifyLabelFacingMode(d.label) === targetFacingMode
  );

  if (matchingByLabel.length > 0) {
    const scored = matchingByLabel
      .map((d, idx) => {
        const isExcluded = excludeDeviceId ? d.deviceId === excludeDeviceId : false;
        const isSecondaryRear =
          targetFacingMode === 'environment' && SECONDARY_REAR_LENS_RE.test(d.label);
        const isMainBack =
          targetFacingMode === 'environment' &&
          /^(back camera|задняя камера|camera2 0, facing back)$/i.test(d.label.trim());
        const score =
          (isExcluded ? 100 : 0) +
          (isSecondaryRear ? 20 : 0) +
          (isMainBack ? -10 : 0);
        return { deviceId: d.deviceId, score, idx };
      })
      .sort((a, b) => a.score - b.score || a.idx - b.idx);

    const best = scored[0];
    if (best && (!excludeDeviceId || best.deviceId !== excludeDeviceId)) {
      return best.deviceId;
    }
  }

  if (!allowGenericCycle || videoInputs.length < 2) {
    return null;
  }

  if (!excludeDeviceId) {
    return videoInputs[0].deviceId;
  }

  const currentIdx = videoInputs.findIndex((d) => d.deviceId === excludeDeviceId);
  if (currentIdx === -1) {
    return videoInputs.find((d) => d.deviceId !== excludeDeviceId)?.deviceId ?? null;
  }
  return videoInputs[(currentIdx + 1) % videoInputs.length].deviceId;
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
      : actualFacingMode === requested
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
 * Requests a camera stream reliably across iPhone (iOS Safari / WKWebView),
 * Android, Windows, and macOS:
 *
 * 1. Requests bare `facingMode: { exact: preferredFacingMode }` WITHOUT width/height/fps
 *    constraints so iOS Safari never throws `OverconstrainedError` on 60fps/12MP rear sensors.
 * 2. If `exact` fails (WKWebView / desktop), matches a lens by `enumerateDevices()`
 *    label (`Back Camera` / `Задняя камера` / `Front Camera`) and requests `deviceId: { exact }`.
 * 3. Falls back to bare `facingMode: { ideal }` (WITHOUT `640x480` constraints that bias
 *    iOS WebKit's fitness-distance algorithm toward the front FaceTime camera!).
 * 4. Post-acquisition verification: if iOS still returned the wrong lens (e.g. front before
 *    initial permission unlocked device labels), stops it, waits for AVFoundation release,
 *    and re-acquires the exact matching `deviceId`.
 */
export async function acquireCameraStream(
  preferredFacingMode: CameraFacingMode = 'environment',
  options?: { previousDeviceId?: string | null; cooldownMs?: number }
): Promise<CameraAcquisitionResult> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      'Camera API is unavailable in this browser context. Open over HTTPS or localhost.'
    );
  }

  const previousDeviceId = options?.previousDeviceId;
  const cooldownMs = options?.cooldownMs ?? SWITCH_COOLDOWN_MS;

  // Stage 1: Clean exact facingMode (no width/height/frameRate to avoid iOS OverconstrainedError)
  const strictOutcome = await tryAttempts(
    [
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

  let acquired: CameraAcquisitionResult | null = strictOutcome.result;
  let lastError: unknown = strictOutcome.lastError;

  // Stage 2: Explicit deviceId matching preferredFacingMode (or next webcam on desktop switch)
  if (!acquired) {
    const matchedDeviceId = await findDeviceIdForFacingMode(
      preferredFacingMode,
      previousDeviceId,
      Boolean(previousDeviceId)
    );
    if (matchedDeviceId) {
      const deviceOutcome = await tryAttempts(
        [
          {
            strict: true,
            constraints: {
              video: {
                deviceId: { exact: matchedDeviceId },
              },
              audio: false,
            },
          },
        ],
        preferredFacingMode,
        previousDeviceId
      );
      acquired = deviceOutcome.result;
      lastError = deviceOutcome.lastError ?? lastError;
    }
  }

  // Stage 3: Bare `ideal` facingMode (NO 640x480 resolution bias toward iPhone front camera!)
  if (!acquired) {
    const fallbackOutcome = await tryAttempts(
      [
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
    acquired = fallbackOutcome.result;
    lastError = fallbackOutcome.lastError ?? lastError;
  }

  if (!acquired) {
    throw lastError instanceof Error
      ? lastError
      : new Error('Unable to access any camera device.');
  }

  // Stage 4: Post-acquisition wrong-lens correction (handles iOS Safari returning front camera
  // before first permission grant unlocked real device labels in enumerateDevices()).
  const wrongFacing = acquired.actualFacingMode !== preferredFacingMode;
  const stuckOnPrevious = Boolean(previousDeviceId && !acquired.switchedDevice);

  if (wrongFacing || stuckOnPrevious) {
    const correctedDeviceId = await findDeviceIdForFacingMode(
      preferredFacingMode,
      acquired.deviceId ?? previousDeviceId,
      stuckOnPrevious
    );

    if (correctedDeviceId && correctedDeviceId !== acquired.deviceId) {
      stopMediaStream(acquired.stream);
      await sleep(cooldownMs);
      try {
        const correctedOutcome = await tryAttempts(
          [
            {
              strict: true,
              constraints: {
                video: {
                  deviceId: { exact: correctedDeviceId },
                },
                audio: false,
              },
            },
          ],
          preferredFacingMode,
          previousDeviceId
        );
        if (correctedOutcome.result) {
          return correctedOutcome.result;
        }
      } catch {
        // Re-open with bare ideal fallback if exact deviceId failed unexpectedly
        const recovery = await tryAttempts(
          [{ strict: false, constraints: { video: true, audio: false } }],
          preferredFacingMode,
          previousDeviceId
        );
        if (recovery.result) {
          return recovery.result;
        }
      }
    }
  }

  return acquired;
}

/**
 * Releases the current camera stream with an iOS AVFoundation hardware cooldown,
 * retries once if the camera HAL is still releasing, and opens the opposite lens.
 */
export async function switchCameraStream(
  currentStream: MediaStream | null,
  targetFacingMode: CameraFacingMode,
  options?: { cooldownMs?: number; retryCooldownMs?: number }
): Promise<CameraAcquisitionResult> {
  const previousDeviceId = getVideoTrackDeviceId(currentStream);
  const hadActiveStream = Boolean(currentStream);
  const cooldownMs = options?.cooldownMs ?? SWITCH_COOLDOWN_MS;
  const retryCooldownMs = options?.retryCooldownMs ?? HARDWARE_RELEASE_RETRY_MS;

  stopMediaStream(currentStream);

  // iOS Safari's AVCaptureSession needs a brief pause after track.stop() before opening another lens.
  if (hadActiveStream && cooldownMs > 0) {
    await sleep(cooldownMs);
  }

  try {
    return await acquireCameraStream(targetFacingMode, {
      previousDeviceId,
      cooldownMs,
    });
  } catch (err) {
    if (!isTransientHardwareError(err) && !isRetryableConstraintError(err)) {
      throw err;
    }
    await sleep(retryCooldownMs);
    return acquireCameraStream(targetFacingMode, {
      previousDeviceId,
      cooldownMs,
    });
  }
}
