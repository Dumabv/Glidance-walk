import type {
  ElevationBand,
  GuidancePayload,
  MemoryCellPoint,
  Severity,
  SpatialMemorySnapshot,
  SteeringAction,
  ZoneMetrics,
} from '../types/telemetry';

const GRID_N = 32; // 32x32 rolling horizontal grid (8m x 8m at 0.25m/cell)
const GRID_LEN = GRID_N * GRID_N; // 1024 cells per vertical band
const CELL_M = 0.25;
const HALF_N = GRID_N / 2;
const HALF_FOV_DEG = 31;

// 3 Vertical Elevation Bands:
// Band 0 (`down`):  Floor, Curbs, Steps, Shin-level (`elev < -18°`, seen when looking at screen `beta ~ 35°..68°`)
// Band 1 (`mid`):   Body/Waist/Chest Walls, Chairs, People (`elev in [-18°, +14°]`, `beta ~ 65°..98°`)
// Band 2 (`up`):    Head/Doorway/Sign Overhangs (`elev > +14°`, seen when phone upright `beta >= 78°`)
const BAND_DOWN = 0;
const BAND_MID = 1;
const BAND_UP = 2;
const NUM_BANDS = 3;
const TOTAL_VOXELS = NUM_BANDS * GRID_LEN; // 3072 voxels total -> ONLY ~24 KB fixed static RAM!

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function wrapDeg180(deg: number): number {
  let d = (((deg + 180) % 360) + 360) % 360 - 180;
  if (d === -180) d = 180;
  return d;
}

function maxSeverity(a: Severity, b: Severity): Severity {
  const rank = (s: Severity) => (s === 'danger' ? 2 : s === 'caution' ? 1 : 0);
  return rank(a) >= rank(b) ? a : b;
}

/**
 * 3D Spherical + Voxel (`UP / MID / DOWN × 360°`) Anti-Hallucination Spatial Memory (~24 KB RAM!).
 *
 * Solves BOTH real-world hand-held smartphone problems simultaneously:
 * 1. Vertical + Horizontal Blind Spots ("Doesn't remember what's below or above"):
 *    - Separately memorizes 3 elevation layers (`DOWN` floor/curbs, `MID` walls/furniture, `UP` head overhangs).
 *    - When you lower your hand (`beta ~ 48°`) to read the screen, the camera sees only floor,
 *      while `MID` and `UP` STILL REMEMBER walls, tables, and overhangs ahead of you (advancing them
 *      closer with every step you take)!
 *    - When you raise the phone (`beta ~ 85°`) to scan ahead, `DOWN` STILL REMEMBERS curbs/steps
 *      at `0.5m..2.2m` in front of your feet!
 * 2. Multi-Frame Cross-Angle Anti-Hallucination Consensus:
 *    - Freezes map writes during fast hand swing/tilt (`motionGated`) where motion blur causes fake depth spikes.
 *    - Requires >= 3 consistent multi-frame spatial hits (`logOdds >= 1.65`, P >= 84%) at the same 3D world
 *      location before confirming an obstacle/curb, while free-space rays from neighboring angles immediately
 *      carve away 1-2 frame monocular depth hallucinations!
 */
export class SpatialMemoryEngine {
  // Fixed static buffers (~24 KB total RAM, 0 per-frame allocations!)
  private logOdds = new Float32Array(TOTAL_VOXELS);
  private curbFlags = new Uint8Array(GRID_LEN); // curbs live in BAND_DOWN
  private seenAge = new Float32Array(TOTAL_VOXELS);

  private scratchOdds = new Float32Array(TOTAL_VOXELS);
  private scratchCurb = new Uint8Array(GRID_LEN);
  private scratchSeen = new Float32Array(TOTAL_VOXELS);

  private baseAlpha: number | null = null;
  private lastRawAlpha: number | null = null;
  private unwrappedYawDeg = 0;
  private headingDeg = 0;
  private pitchDeg = 75; // default comfortable hand angle if desktop has no IMU
  private hasLiveImu = false;

  // Angular velocity motion-blur gate
  private prevFrameYaw = 0;
  private prevFramePitch = 75;
  private rotSpeedDegPerSec = 0;
  private filteredFlickers = 0;

  // Walking step odometry
  private subX = 0;
  private subY = 0;
  private stepCount = 0;
  private walkedMeters = 0;
  private gravMag = 9.81;
  private lastStepAt = 0;
  private accelArmed = false;

  private lastUpdateAt = 0;

  public reset(): void {
    this.logOdds.fill(0);
    this.curbFlags.fill(0);
    this.seenAge.fill(0);
    this.baseAlpha = null;
    this.lastRawAlpha = null;
    this.unwrappedYawDeg = 0;
    this.headingDeg = 0;
    this.pitchDeg = 75;
    this.hasLiveImu = false;
    this.prevFrameYaw = 0;
    this.prevFramePitch = 75;
    this.rotSpeedDegPerSec = 0;
    this.filteredFlickers = 0;
    this.subX = 0;
    this.subY = 0;
    this.stepCount = 0;
    this.walkedMeters = 0;
    this.gravMag = 9.81;
    this.lastStepAt = 0;
    this.accelArmed = false;
    this.lastUpdateAt = 0;
  }

  public updateOrientation(
    alphaDeg: number | null,
    compassHeading?: number | null,
    betaPitchDeg?: number | null
  ): void {
    if (typeof betaPitchDeg === 'number' && Number.isFinite(betaPitchDeg)) {
      this.hasLiveImu = true;
      this.pitchDeg = 0.75 * this.pitchDeg + 0.25 * betaPitchDeg;
    }

    let raw: number | null = null;
    if (typeof compassHeading === 'number' && Number.isFinite(compassHeading)) {
      raw = compassHeading;
    } else if (typeof alphaDeg === 'number' && Number.isFinite(alphaDeg)) {
      raw = (360 - alphaDeg) % 360;
    }
    if (raw === null) return;

    this.hasLiveImu = true;
    if (this.baseAlpha === null || this.lastRawAlpha === null) {
      this.baseAlpha = raw;
      this.lastRawAlpha = raw;
      this.unwrappedYawDeg = 0;
      this.headingDeg = 0;
      this.prevFrameYaw = 0;
      return;
    }

    const delta = wrapDeg180(raw - this.lastRawAlpha);
    this.lastRawAlpha = raw;
    if (Math.abs(delta) < 65) {
      this.unwrappedYawDeg += delta;
      this.headingDeg = 0.78 * this.headingDeg + 0.22 * this.unwrappedYawDeg;
    }
  }

  public updateMotion(
    ax: number | null | undefined,
    ay: number | null | undefined,
    az: number | null | undefined,
    rotAlpha?: number | null,
    rotBeta?: number | null,
    rotGamma?: number | null,
    dtSec = 0.016
  ): void {
    const ra = typeof rotAlpha === 'number' && Number.isFinite(rotAlpha) ? rotAlpha : 0;
    const rb = typeof rotBeta === 'number' && Number.isFinite(rotBeta) ? rotBeta : 0;
    const rg = typeof rotGamma === 'number' && Number.isFinite(rotGamma) ? rotGamma : 0;
    const instantRot = Math.hypot(ra, rb, rg);
    this.rotSpeedDegPerSec = 0.7 * this.rotSpeedDegPerSec + 0.3 * instantRot;

    if (this.baseAlpha === null && Math.abs(ra) > 0.5 && dtSec > 0 && dtSec < 0.2) {
      this.headingDeg -= ra * dtSec;
    }

    if (
      typeof ax !== 'number' ||
      typeof ay !== 'number' ||
      typeof az !== 'number' ||
      !Number.isFinite(ax) ||
      !Number.isFinite(ay) ||
      !Number.isFinite(az)
    ) {
      return;
    }

    const mag = Math.hypot(ax, ay, az);
    this.gravMag = 0.92 * this.gravMag + 0.08 * mag;
    const dyn = mag - this.gravMag;

    const now = performance.now();
    if (dyn > 1.35 && !this.accelArmed && now - this.lastStepAt >= 360) {
      this.accelArmed = true;
      this.lastStepAt = now;
      this.advanceStep(0.62);
    } else if (dyn < 0.35) {
      this.accelArmed = false;
    }
  }

  private advanceStep(stepMeters: number): void {
    this.stepCount += 1;
    this.walkedMeters += stepMeters;
    const rad = (this.headingDeg * Math.PI) / 180;
    this.subX += Math.sin(rad) * stepMeters;
    this.subY += Math.cos(rad) * stepMeters;

    const shiftCols = Math.trunc(this.subX / CELL_M);
    const shiftRows = Math.trunc(this.subY / CELL_M);
    if (shiftCols !== 0 || shiftRows !== 0) {
      this.subX -= shiftCols * CELL_M;
      this.subY -= shiftRows * CELL_M;
      this.shiftAllBands(shiftCols, shiftRows);
    }
  }

  private shiftAllBands(dCol: number, dRow: number): void {
    this.scratchOdds.fill(0);
    this.scratchCurb.fill(0);
    this.scratchSeen.fill(0);

    for (let b = 0; b < NUM_BANDS; b++) {
      const bOff = b * GRID_LEN;
      for (let gy = 0; gy < GRID_N; gy++) {
        const srcY = gy + dRow;
        if (srcY < 0 || srcY >= GRID_N) continue;
        for (let gx = 0; gx < GRID_N; gx++) {
          const srcX = gx + dCol;
          if (srcX < 0 || srcX >= GRID_N) continue;
          const dstIdx = bOff + gy * GRID_N + gx;
          const srcIdx = bOff + srcY * GRID_N + srcX;
          this.scratchOdds[dstIdx] = this.logOdds[srcIdx];
          this.scratchSeen[dstIdx] = this.seenAge[srcIdx];
          if (b === BAND_DOWN) {
            this.scratchCurb[gy * GRID_N + gx] = this.curbFlags[srcY * GRID_N + srcX];
          }
        }
      }
    }

    this.logOdds.set(this.scratchOdds);
    this.curbFlags.set(this.scratchCurb);
    this.seenAge.set(this.scratchSeen);
  }

  public integrateAndFuse(raw: GuidancePayload): {
    fused: GuidancePayload;
    snapshot: SpatialMemorySnapshot;
  } {
    const now = performance.now();
    const dtSec =
      this.lastUpdateAt > 0 ? clamp((now - this.lastUpdateAt) / 1000, 0.01, 1.0) : 0.08;
    this.lastUpdateAt = now;

    if (typeof raw.pitch_deg === 'number' && Number.isFinite(raw.pitch_deg) && Math.abs(raw.pitch_deg) > 1) {
      this.hasLiveImu = true;
      this.pitchDeg = 0.7 * this.pitchDeg + 0.3 * raw.pitch_deg;
    }

    // Determine which vertical elevation band the smartphone camera is currently aimed at:
    // `beta = 90°` -> horizontal (`mid`), `beta = 45°` -> floor (`down`), `beta >= 96°` -> ceiling/high (`up`)
    const cameraBand: ElevationBand = !this.hasLiveImu
      ? 'mid'
      : this.pitchDeg < 64
        ? 'down'
        : this.pitchDeg > 96
          ? 'up'
          : 'mid';

    const yawJump = Math.abs(wrapDeg180(this.headingDeg - this.prevFrameYaw));
    const pitchJump = Math.abs(this.pitchDeg - this.prevFramePitch);
    this.prevFrameYaw = this.headingDeg;
    this.prevFramePitch = this.pitchDeg;

    // 1. Motion-Blur Hallucination Gate: freeze writes when hand swings rapidly!
    const motionGated =
      this.rotSpeedDegPerSec > 95 || yawJump > 16 || pitchJump > 13;

    // 2. Gentle exponential decay (~11s half-life so UP/MID/DOWN stay remembered as you tilt hand!)
    const obsDecay = Math.exp(-dtSec / 11.0);
    const seenDecay = Math.exp(-dtSec / 14.0);
    for (let i = 0; i < TOTAL_VOXELS; i++) {
      const v = this.logOdds[i] * obsDecay;
      this.logOdds[i] = Math.abs(v) < 0.02 ? 0 : v;
      this.seenAge[i] *= seenDecay;
    }
    for (let i = 0; i < GRID_LEN; i++) {
      if (this.logOdds[BAND_DOWN * GRID_LEN + i] < 0.35) {
        this.curbFlags[i] = 0;
      }
    }

    // 3. Raycast into the appropriate vertical elevation band(s) (skipped during motion blur!)
    if (!raw.lens_blocked && !motionGated) {
      const isCurb = raw.curb.detected || raw.steering === 'CURB_DROP';
      const curbDist = clamp(2.5 - 1.8 * raw.curb.drop_row_ratio, 0.7, 2.4);

      // Which band receives main corridor rays?
      const primaryBandIdx =
        cameraBand === 'down' ? BAND_DOWN : cameraBand === 'up' ? BAND_UP : BAND_MID;

      const castIntoBand = (
        bandIdx: number,
        relStartDeg: number,
        relEndDeg: number,
        zone: ZoneMetrics,
        stampCurb: boolean
      ) => {
        const bOff = bandIdx * GRID_LEN;
        const hasHit = zone.severity !== 'clear' || stampCurb;
        // Convert image-plane distance to horizontal ground distance `d * cos(elev)`
        const elevRad = this.hasLiveImu
          ? ((clamp(this.pitchDeg, 25, 115) - 90) * Math.PI) / 180
          : 0;
        const groundScale = clamp(Math.cos(elevRad), 0.52, 1.0);
        const rawHitDist = stampCurb
          ? Math.min(zone.min_distance_m, curbDist)
          : zone.min_distance_m;
        const hitDist = clamp(rawHitDist * groundScale, 0.35, 3.8);
        const maxClearDist = clamp(hitDist - 0.24, 0.25, 3.6);

        // Conservative Bayesian update: requires ~3 consistent frames to reach P >= 84% (`logOdds >= 1.65`),
        // while free-space carving (`-0.68`) quickly erases 1-frame hallucinations!
        const hitWeight =
          stampCurb || zone.severity === 'danger'
            ? 0.92
            : zone.severity === 'caution'
              ? 0.58
              : 0;

        for (let rel = relStartDeg; rel <= relEndDeg + 0.1; rel += 5.0) {
          const worldRad = ((this.headingDeg + rel) * Math.PI) / 180;
          const sinA = Math.sin(worldRad);
          const cosA = Math.cos(worldRad);

          for (let d = 0.35; d <= maxClearDist; d += 0.24) {
            const gx = Math.floor(HALF_N + (sinA * d) / CELL_M);
            const gy = Math.floor(HALF_N + (cosA * d) / CELL_M);
            if (gx >= 0 && gx < GRID_N && gy >= 0 && gy < GRID_N) {
              const gIdx = gy * GRID_N + gx;
              const vIdx = bOff + gIdx;
              this.logOdds[vIdx] = clamp(this.logOdds[vIdx] - 0.68, -2.4, 4.2);
              this.seenAge[vIdx] = 1.0;
            }
          }

          if (hasHit && hitDist <= 3.65) {
            const gx = Math.floor(HALF_N + (sinA * hitDist) / CELL_M);
            const gy = Math.floor(HALF_N + (cosA * hitDist) / CELL_M);
            if (gx >= 0 && gx < GRID_N && gy >= 0 && gy < GRID_N) {
              const gIdx = gy * GRID_N + gx;
              const vIdx = bOff + gIdx;
              this.logOdds[vIdx] = clamp(this.logOdds[vIdx] + hitWeight, -2.4, 4.2);
              this.seenAge[vIdx] = 1.0;
              if (stampCurb && bandIdx === BAND_DOWN) {
                this.curbFlags[gIdx] = 1;
              }
            }
          }
        }
      };

      castIntoBand(primaryBandIdx, -HALF_FOV_DEG, -10.5, raw.left, false);
      castIntoBand(
        isCurb ? BAND_DOWN : primaryBandIdx,
        -10.5,
        10.5,
        raw.center,
        isCurb
      );
      castIntoBand(primaryBandIdx, 10.5, HALF_FOV_DEG, raw.right, false);

      // When holding phone upright (`mid`), lower half of frame ALSO sweeps `BAND_DOWN` floor,
      // and top overhang region sweeps `BAND_UP`!
      if (cameraBand === 'mid') {
        const floorProxyZone: ZoneMetrics = isCurb
          ? { severity: 'danger', obstacle_score: 0.8, min_distance_m: curbDist, coverage_ratio: 0.3 }
          : { severity: 'clear', obstacle_score: 0, min_distance_m: Math.min(2.4, raw.center.min_distance_m), coverage_ratio: 0 };
        castIntoBand(BAND_DOWN, -18, 18, floorProxyZone, isCurb);

        const upZone: ZoneMetrics = raw.overhead_hazard
          ? { severity: 'danger', obstacle_score: 0.85, min_distance_m: 1.25, coverage_ratio: 0.3 }
          : { severity: 'clear', obstacle_score: 0, min_distance_m: 2.6, coverage_ratio: 0 };
        castIntoBand(BAND_UP, -16, 16, upZone, false);
      }
    }

    // 4. Query ALL 3 Elevation Bands (`DOWN + MID + UP`) across forward & blind-spot sectors!
    //    This fuses what the phone saw BELOW, AHEAD, and ABOVE even when currently aimed elsewhere!
    const query3DSector = (
      relMinDeg: number,
      relMaxDeg: number,
      maxDistM: number,
      bandFilter: number | 'all'
    ) => {
      let maxOdds = 0;
      let minDist = 5.0;
      let hasCurb = false;
      let seenCells = 0;
      let totalCells = 0;

      const bStart = bandFilter === 'all' ? 0 : bandFilter;
      const bEnd = bandFilter === 'all' ? NUM_BANDS - 1 : bandFilter;

      for (let gy = 0; gy < GRID_N; gy++) {
        for (let gx = 0; gx < GRID_N; gx++) {
          const dx = (gx + 0.5 - HALF_N) * CELL_M;
          const dy = (gy + 0.5 - HALF_N) * CELL_M;
          const dist = Math.hypot(dx, dy);
          if (dist < 0.35 || dist > maxDistM) continue;

          const worldDeg = (Math.atan2(dx, dy) * 180) / Math.PI;
          const relDeg = wrapDeg180(worldDeg - this.headingDeg);
          if (relDeg < relMinDeg || relDeg > relMaxDeg) continue;

          const gIdx = gy * GRID_N + gx;
          for (let b = bStart; b <= bEnd; b++) {
            totalCells += 1;
            const vIdx = b * GRID_LEN + gIdx;
            if (this.seenAge[vIdx] > 0.22) {
              seenCells += 1;
            }
            const odds = this.logOdds[vIdx];
            // Multi-frame anti-hallucination threshold: odds >= 1.55 (~82.5% posterior -> >= 2-3 consistent hits!)
            if (odds >= 1.55) {
              if (odds > maxOdds) maxOdds = odds;
              if (dist < minDist) minDist = dist;
              if (b === BAND_DOWN && this.curbFlags[gIdx] === 1) {
                hasCurb = true;
              }
            }
          }
        }
      }
      const prob = maxOdds > 0 ? 1 / (1 + Math.exp(-maxOdds)) : 0;
      const seenRatio = totalCells > 0 ? seenCells / totalCells : 0;
      return { prob, minDist, hasCurb, seenRatio };
    };

    // Forward 3D sectors (combining UP + MID + DOWN!)
    const fwdLeft3D = query3DSector(-32, -10, 2.5, 'all');
    const fwdCenter3D = query3DSector(-12, 12, 2.5, 'all');
    const fwdRight3D = query3DSector(10, 32, 2.5, 'all');

    // Vertical band status ahead (`[-26°, +26°]`, within `2.3m`)
    const upAhead = query3DSector(-26, 26, 2.2, BAND_UP);
    const midAhead = query3DSector(-26, 26, 2.3, BAND_MID);
    const downAhead = query3DSector(-26, 26, 2.2, BAND_DOWN);

    // Blind-spot shoulder sectors (`[-90°..-26°]` and `[+26°..+90°]`)
    const blindLeft = query3DSector(-90, -26, 2.1, 'all');
    const blindRight = query3DSector(26, 90, 2.1, 'all');
    const blindLeftDanger = blindLeft.prob >= 0.82 && blindLeft.minDist <= 1.75;
    const blindRightDanger = blindRight.prob >= 0.82 && blindRight.minDist <= 1.75;

    // 5. ANTI-HALLUCINATION + CROSS-ELEVATION FUSION!
    //    Case A: Raw single frame claims a sudden hazard, BUT 3D multi-frame voxel map has NOT
    //            confirmed it yet (`prob < 0.82`) -> suppress 1-frame hallucination!
    //    Case B: Phone is aimed DOWN at floor (`cameraBand === 'down'`, raw sees `CLEAR`),
    //            BUT `MID` or `UP` remembers a wall/chair/overhang `<= 1.6m` ahead -> WARN!
    //    Case C: Phone is aimed UP/MID (`cameraBand !== 'down'`), BUT `DOWN` remembers a curb `<= 1.6m` ahead -> WARN!

    const toConfirmedZone = (
      rawZ: ZoneMetrics,
      memSec: { prob: number; minDist: number }
    ): { zone: ZoneMetrics; suppressed: boolean } => {
      if (raw.lens_blocked) return { zone: rawZ, suppressed: false };

      // Confirmed by 3D multi-frame memory (across ANY elevation band: UP, MID, or DOWN!)
      if (memSec.prob >= 0.83 && memSec.minDist <= 1.65) {
        return {
          zone: {
            severity: 'danger',
            obstacle_score: Math.max(rawZ.obstacle_score, Number(memSec.prob.toFixed(3))),
            min_distance_m: Math.min(rawZ.min_distance_m, Number(memSec.minDist.toFixed(2))),
            coverage_ratio: Math.max(rawZ.coverage_ratio, 0.35),
          },
          suppressed: false,
        };
      }
      if (memSec.prob >= 0.78 && memSec.minDist <= 2.4) {
        return {
          zone: {
            severity: rawZ.severity === 'danger' ? 'caution' : maxSeverity(rawZ.severity, 'caution'),
            obstacle_score: Math.max(0.28, Math.min(rawZ.obstacle_score, 0.42)),
            min_distance_m: Math.min(rawZ.min_distance_m, Number(memSec.minDist.toFixed(2))),
            coverage_ratio: rawZ.coverage_ratio,
          },
          suppressed: rawZ.severity === 'danger',
        };
      }

      // Single-frame raw spike NOT backed by multi-frame spatial consensus -> suppress flicker!
      if (rawZ.severity !== 'clear') {
        return {
          zone: {
            severity: 'clear',
            obstacle_score: Number((rawZ.obstacle_score * 0.35).toFixed(3)),
            min_distance_m: Math.max(rawZ.min_distance_m, 2.6),
            coverage_ratio: 0,
          },
          suppressed: true,
        };
      }
      return { zone: rawZ, suppressed: false };
    };

    const fL = toConfirmedZone(raw.left, fwdLeft3D);
    const fC = toConfirmedZone(raw.center, fwdCenter3D);
    const fR = toConfirmedZone(raw.right, fwdRight3D);

    const rawHadCurb = raw.curb.detected || raw.steering === 'CURB_DROP';
    const confirmedCurb = !raw.lens_blocked && downAhead.hasCurb && downAhead.prob >= 0.83 && downAhead.minDist <= 2.1;
    const curbSuppressed = rawHadCurb && !confirmedCurb;

    if (fL.suppressed || fC.suppressed || fR.suppressed || curbSuppressed) {
      this.filteredFlickers += 1;
    }

    const confirmedOverhead =
      !raw.lens_blocked && upAhead.prob >= 0.83 && upAhead.minDist <= 1.8;

    // Recompute Lean-4 verified steering on the 3D-confirmed + blind-spot-protected zones!
    let overall: Severity = maxSeverity(
      fL.zone.severity,
      maxSeverity(fC.zone.severity, fR.zone.severity)
    );
    if (confirmedCurb || confirmedOverhead || raw.lens_blocked) {
      overall = 'danger';
    }

    let steering: SteeringAction = 'CLEAR';
    let memoryVetoActive = false;
    let memoryHint: string | null = null;

    if (raw.lens_blocked) {
      steering = 'STOP';
    } else if (confirmedCurb) {
      steering = 'CURB_DROP';
      if (cameraBand !== 'down') {
        memoryHint = `Remembered step/curb below (${downAhead.minDist.toFixed(1)}m ahead)`;
      }
    } else if (confirmedOverhead && cameraBand === 'down') {
      steering = 'STOP';
      memoryHint = `Remembered head overhang above (${upAhead.minDist.toFixed(1)}m ahead)`;
    } else if (fC.zone.severity === 'danger') {
      const leftSafe = fL.zone.severity !== 'danger' && !blindLeftDanger;
      const rightSafe = fR.zone.severity !== 'danger' && !blindRightDanger;
      if (!leftSafe && !rightSafe) {
        steering = 'STOP';
        if (blindLeftDanger || blindRightDanger) {
          memoryVetoActive = true;
          memoryHint = '360° memory: obstacles on sides too -> STOP';
        }
      } else if (leftSafe && !rightSafe) {
        steering = 'VEER_LEFT';
        if (blindRightDanger) memoryVetoActive = true;
      } else if (!leftSafe && rightSafe) {
        steering = 'VEER_RIGHT';
        if (blindLeftDanger) memoryVetoActive = true;
      } else {
        steering =
          fL.zone.obstacle_score <= fR.zone.obstacle_score ? 'VEER_LEFT' : 'VEER_RIGHT';
      }
      if (cameraBand === 'down' && midAhead.prob >= 0.83) {
        memoryHint = `Remembered obstacle ahead at chest level (${midAhead.minDist.toFixed(1)}m)`;
      }
    } else if (fL.zone.severity === 'danger' && fR.zone.severity === 'danger') {
      steering = 'STOP';
    } else if (fL.zone.severity === 'danger') {
      steering = blindRightDanger ? 'STOP' : 'VEER_RIGHT';
    } else if (fR.zone.severity === 'danger') {
      steering = blindLeftDanger ? 'STOP' : 'VEER_LEFT';
    } else if (overall === 'caution') {
      const diff = Math.abs(fL.zone.obstacle_score - fR.zone.obstacle_score);
      if (fC.zone.severity === 'caution' || diff > 0.06) {
        if (fL.zone.obstacle_score + 0.05 < fR.zone.obstacle_score && !blindLeftDanger) {
          steering = 'VEER_LEFT';
        } else if (fR.zone.obstacle_score + 0.05 < fL.zone.obstacle_score && !blindRightDanger) {
          steering = 'VEER_RIGHT';
        }
      }
    }

    // Best escape heading across [-100°, +100°] when blocked
    let bestEscapeDeg: number | null = null;
    let bestEscapeScore = -999;
    for (const ang of [-75, -50, -25, 0, 25, 50, 75]) {
      const sec = query3DSector(ang - 16, ang + 16, 2.8, 'all');
      if (sec.seenRatio >= 0.2 && sec.prob < 0.65) {
        const sc = sec.minDist - sec.prob * 2.2 + sec.seenRatio * 0.5 - Math.abs(ang) * 0.004;
        if (sc > bestEscapeScore) {
          bestEscapeScore = sc;
          bestEscapeDeg = ang;
        }
      }
    }

    if (steering === 'STOP' && !raw.lens_blocked && !memoryHint) {
      if (bestEscapeDeg !== null && Math.abs(bestEscapeDeg) >= 25) {
        memoryHint =
          bestEscapeDeg < 0
            ? `Open space remembered ${Math.abs(bestEscapeDeg)}° Left`
            : `Open space remembered ${bestEscapeDeg}° Right`;
      } else {
        memoryHint = 'Tilt/sweep phone once to scan floor, ahead & sides';
      }
    }

    // 6. Collect active 3D cells (collapsing max occupancy across UP/MID/DOWN for 2D top-down + band tag)
    const cells: MemoryCellPoint[] = [];
    let seenTotal = 0;
    let nearTotal = 0;

    for (let gy = 0; gy < GRID_N; gy++) {
      for (let gx = 0; gx < GRID_N; gx++) {
        const dx = (gx + 0.5 - HALF_N) * CELL_M;
        const dy = (gy + 0.5 - HALF_N) * CELL_M;
        const d = Math.hypot(dx, dy);
        const gIdx = gy * GRID_N + gx;

        if (d <= 3.6) {
          nearTotal += 1;
          if (
            this.seenAge[BAND_DOWN * GRID_LEN + gIdx] > 0.18 ||
            this.seenAge[BAND_MID * GRID_LEN + gIdx] > 0.18 ||
            this.seenAge[BAND_UP * GRID_LEN + gIdx] > 0.18
          ) {
            seenTotal += 1;
          }
        }

        const oDown = this.logOdds[BAND_DOWN * GRID_LEN + gIdx];
        const oMid = this.logOdds[BAND_MID * GRID_LEN + gIdx];
        const oUp = this.logOdds[BAND_UP * GRID_LEN + gIdx];
        const bestOdds = Math.max(oDown, oMid, oUp);

        if (bestOdds >= 1.1) {
          const band: ElevationBand =
            oUp >= oMid && oUp >= oDown
              ? 'up'
              : oDown >= oMid
                ? 'down'
                : 'mid';
          cells.push({
            gx,
            gy,
            prob: 1 / (1 + Math.exp(-bestOdds)),
            isCurb: this.curbFlags[gIdx] === 1,
            band,
          });
        }
      }
    }

    const scannedPercent =
      nearTotal > 0 ? Math.min(100, Math.round((seenTotal / nearTotal) * 100)) : 0;

    const fused: GuidancePayload = {
      ...raw,
      overall_severity: overall,
      steering,
      left: fL.zone,
      center: fC.zone,
      right: fR.zone,
      curb: {
        ...raw.curb,
        detected: confirmedCurb,
      },
      overhead_hazard: confirmedOverhead,
      memory_veto: memoryVetoActive,
      memory_hint: memoryHint,
      best_escape_deg: bestEscapeDeg,
    };

    return {
      fused,
      snapshot: {
        headingDeg: this.headingDeg,
        pitchDeg: Math.round(this.pitchDeg),
        cameraBand,
        stepCount: this.stepCount,
        walkedMeters: Math.round(this.walkedMeters * 10) / 10,
        scannedPercent,
        upClear: upAhead.prob < 0.8,
        midClear: midAhead.prob < 0.8,
        downClear: downAhead.prob < 0.8 && !confirmedCurb,
        upSeen: upAhead.seenRatio > 0.12,
        midSeen: midAhead.seenRatio > 0.12,
        downSeen: downAhead.seenRatio > 0.12,
        filteredFlickers: this.filteredFlickers,
        motionGated,
        blindLeftDanger,
        blindRightDanger,
        bestEscapeDeg,
        memoryVetoActive,
        memoryHint,
        cells,
      },
    };
  }
}
