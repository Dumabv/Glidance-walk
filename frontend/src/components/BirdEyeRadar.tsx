import type {
  GuidancePayload,
  Severity,
  SpatialMemorySnapshot,
} from '../types/telemetry';

interface BirdEyeRadarProps {
  guidance: GuidancePayload | null;
  memory?: SpatialMemorySnapshot | null;
}

const SEVERITY_FILL: Record<Severity, string> = {
  clear: 'rgba(16, 185, 129, 0.18)',
  caution: 'rgba(245, 158, 11, 0.36)',
  danger: 'rgba(239, 68, 68, 0.52)',
};

const SEVERITY_STROKE: Record<Severity, string> = {
  clear: '#10b981',
  caution: '#f59e0b',
  danger: '#ef4444',
};

function sectorPath(
  cx: number,
  cy: number,
  rInner: number,
  rOuter: number,
  startDeg: number,
  endDeg: number
): string {
  const rad = (d: number) => (d * Math.PI) / 180;
  const x1 = cx + rOuter * Math.cos(rad(startDeg));
  const y1 = cy + rOuter * Math.sin(rad(startDeg));
  const x2 = cx + rOuter * Math.cos(rad(endDeg));
  const y2 = cy + rOuter * Math.sin(rad(endDeg));
  const x3 = cx + rInner * Math.cos(rad(endDeg));
  const y3 = cy + rInner * Math.sin(rad(endDeg));
  const x4 = cx + rInner * Math.cos(rad(startDeg));
  const y4 = cy + rInner * Math.sin(rad(startDeg));
  return [
    `M ${x1.toFixed(1)} ${y1.toFixed(1)}`,
    `A ${rOuter} ${rOuter} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`,
    `L ${x3.toFixed(1)} ${y3.toFixed(1)}`,
    `A ${rInner} ${rInner} 0 0 0 ${x4.toFixed(1)} ${y4.toFixed(1)}`,
    'Z',
  ].join(' ');
}

function distToRadius180(distMeters: number): number {
  const clamped = Math.max(0.4, Math.min(4.5, distMeters));
  return 34 + ((clamped - 0.4) / 4.1) * 116;
}

export function BirdEyeRadar({ guidance, memory }: BirdEyeRadarProps) {
  const left = guidance?.left;
  const center = guidance?.center;
  const right = guidance?.right;
  const curbDetected = guidance?.curb.detected ?? false;

  // Mode A: Full 360° Egocentric Spatial Memory Map (Walker centered, remembers out-of-FOV obstacles!)
  if (memory) {
    const cx = 130;
    const cy = 122;
    const maxRangeM = 4.0;
    const scalePxPerM = 96 / maxRangeM; // 4.0m -> 96px radius
    const headingRad = (memory.headingDeg * Math.PI) / 180;
    const cosH = Math.cos(headingRad);
    const sinH = Math.sin(headingRad);

    const distToR360 = (d: number) =>
      Math.max(14, Math.min(96, d * scalePxPerM));

    const leftR = left ? distToR360(left.min_distance_m) : 92;
    const centerR = center ? distToR360(center.min_distance_m) : 92;
    const rightR = right ? distToR360(right.min_distance_m) : 92;

    return (
      <div className="flex flex-col items-center justify-between rounded-xl border border-slate-800 bg-slate-900/85 p-3">
        <div className="mb-1.5 flex w-full items-center justify-between text-[11px] font-semibold text-slate-300">
          <span className="text-sky-300">
            3D Space Memory ({memory.scannedPercent}% · {memory.filteredFlickers} flickers blocked)
          </span>
          <span className="font-mono text-slate-400">
            {memory.stepCount} stp · {memory.pitchDeg}°
          </span>
        </div>

        {/* 3 Vertical Elevation Memory Pills: UP (Head) / MID (Ahead) / DOWN (Floor & Curbs) */}
        <div className="mb-1.5 grid w-full grid-cols-3 gap-1 text-center text-[10px] font-bold">
          {(
            [
              { id: 'up', label: 'UP · Head', ok: memory.upClear, seen: memory.upSeen },
              { id: 'mid', label: 'MID · Body', ok: memory.midClear, seen: memory.midSeen },
              { id: 'down', label: 'DOWN · Floor', ok: memory.downClear, seen: memory.downSeen },
            ] as const
          ).map((b) => {
            const activeCam = memory.cameraBand === b.id;
            const cls = !b.ok
              ? 'border-red-500/70 bg-red-950/70 text-red-200'
              : b.seen
                ? 'border-emerald-500/50 bg-emerald-950/50 text-emerald-200'
                : 'border-slate-700 bg-slate-800/70 text-slate-400';
            return (
              <div
                key={b.id}
                className={`rounded-md border px-1.5 py-1 ${cls} ${
                  activeCam ? 'ring-1 ring-sky-400' : ''
                }`}
              >
                {b.label}: {!b.ok ? 'HAZARD' : b.seen ? 'MEM OK' : 'UNSEEN'}
              </div>
            );
          })}
        </div>

        <svg role="img" viewBox="0 0 260 244" className="w-full max-w-[270px]">
          <title>360-degree spatial memory obstacle map around walker</title>

          {/* 360° Range Rings (1.5m, 3.0m, 4.0m) */}
          {[1.5, 3.0, 4.0].map((m) => {
            const r = m * scalePxPerM;
            return (
              <g key={m}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="1.2"
                  strokeDasharray="3 3"
                />
                <text
                  x={cx + 3}
                  y={cy - r + 10}
                  fill="#475569"
                  fontSize="8"
                  fontFamily="monospace"
                >
                  {m}m
                </text>
              </g>
            );
          })}

          {/* Crosshairs (Front / Back / Left / Right relative to walker's current heading) */}
          <line x1={cx} y1={cy - 98} x2={cx} y2={cy + 98} stroke="#1e293b" strokeWidth="1" />
          <line x1={cx - 98} y1={cy} x2={cx + 98} y2={cy} stroke="#1e293b" strokeWidth="1" />

          {/* Live 62° Narrow Camera Frustum (-121° to -59° in SVG screen coordinates where -90° is UP) */}
          <path
            d={sectorPath(cx, cy, 12, leftR, -121, -100.5)}
            fill={SEVERITY_FILL[left?.severity ?? 'clear']}
            stroke={SEVERITY_STROKE[left?.severity ?? 'clear']}
            strokeWidth="1.5"
          />
          <path
            d={sectorPath(cx, cy, 12, centerR, -100.5, -79.5)}
            fill={SEVERITY_FILL[center?.severity ?? 'clear']}
            stroke={SEVERITY_STROKE[center?.severity ?? 'clear']}
            strokeWidth="1.5"
          />
          <path
            d={sectorPath(cx, cy, 12, rightR, -79.5, -59)}
            fill={SEVERITY_FILL[right?.severity ?? 'clear']}
            stroke={SEVERITY_STROKE[right?.severity ?? 'clear']}
            strokeWidth="1.5"
          />

          {/* Remembered 360° Obstacles & Curbs (rotated into walker's ego-heading so UP = camera forward!) */}
          {memory.cells.map((c) => {
            const wx = (c.gx + 0.5 - 16) * 0.25;
            const wy = (c.gy + 0.5 - 16) * 0.25;
            // Rotate world (wx, wy) by -headingDeg into body frame (bx = right, by = forward)
            const bx = wx * cosH - wy * sinH;
            const by = wx * sinH + wy * cosH;
            const sx = cx + bx * scalePxPerM;
            const sy = cy - by * scalePxPerM;
            if (Math.hypot(sx - cx, sy - cy) > 98) return null;

            const fill = c.isCurb
              ? `rgba(34, 211, 238, ${0.45 + 0.5 * c.prob})`
              : c.band === 'sonar'
                ? `rgba(232, 121, 249, ${0.55 + 0.45 * c.prob})`
                : c.prob > 0.78
                  ? `rgba(239, 68, 68, ${0.45 + 0.5 * c.prob})`
                  : `rgba(245, 158, 11, ${0.4 + 0.45 * c.prob})`;

            return (
              <rect
                key={`${c.gx}-${c.gy}`}
                x={sx - 3.2}
                y={sy - 3.2}
                width="6.4"
                height="6.4"
                rx="1.5"
                fill={fill}
              />
            );
          })}

          {/* Best Remembered Escape Vector Arrow */}
          {memory.bestEscapeDeg !== null && Math.abs(memory.bestEscapeDeg) >= 20 && (
            <line
              x1={cx}
              y1={cy}
              x2={cx + Math.sin((memory.bestEscapeDeg * Math.PI) / 180) * 68}
              y2={cy - Math.cos((memory.bestEscapeDeg * Math.PI) / 180) * 68}
              stroke="#34d399"
              strokeWidth="3"
              strokeDasharray="5 3"
              strokeLinecap="round"
            />
          )}

          {/* Blind-spot danger arcs on Left / Right shoulders */}
          {memory.blindLeftDanger && (
            <path
              d={sectorPath(cx, cy, 20, 48, -180, -122)}
              fill="rgba(239, 68, 68, 0.28)"
              stroke="#ef4444"
              strokeWidth="1.5"
              strokeDasharray="3 2"
            />
          )}
          {memory.blindRightDanger && (
            <path
              d={sectorPath(cx, cy, 20, 48, -58, 0)}
              fill="rgba(239, 68, 68, 0.28)"
              stroke="#ef4444"
              strokeWidth="1.5"
              strokeDasharray="3 2"
            />
          )}

          {/* Walker center */}
          <circle cx={cx} cy={cy} r="7" fill="#38bdf8" stroke="#0f172a" strokeWidth="2" />
          <text x={cx} y="13" textAnchor="middle" fill="#94a3b8" fontSize="9" fontWeight="700">
            62° CAM FOV
          </text>
          <text x="16" y={cy + 3} fill="#64748b" fontSize="8" fontWeight="700">
            BLIND L
          </text>
          <text x="206" y={cy + 3} fill="#64748b" fontSize="8" fontWeight="700">
            BLIND R
          </text>
          <text x={cx} y="236" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="700">
            REMEMBERED BEHIND (12 KB RAM)
          </text>
        </svg>

        {memory.memoryHint && (
          <p className="mt-1 w-full rounded-lg bg-sky-950/60 px-2.5 py-1 text-center text-[11px] font-semibold text-sky-200">
            {memory.memoryHint}
          </p>
        )}
      </div>
    );
  }

  // Mode B: Classic 180° Forward Sector Radar (used in Mission Control static frame preview)
  const cx = 160;
  const cy = 178;
  const summaryTitle = guidance
    ? `Obstacle radar: Left ${guidance.left.severity} at ${guidance.left.min_distance_m}m, Center ${guidance.center.severity} at ${guidance.center.min_distance_m}m, Right ${guidance.right.severity} at ${guidance.right.min_distance_m}m${curbDetected ? ', curb drop detected ahead' : ''}.`
    : 'Obstacle radar waiting for live camera stream.';

  const leftR = left ? distToRadius180(left.min_distance_m) : 145;
  const centerR = center ? distToRadius180(center.min_distance_m) : 145;
  const rightR = right ? distToRadius180(right.min_distance_m) : 145;

  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-slate-800 bg-slate-900/80 p-3">
      <svg role="img" viewBox="0 0 320 205" className="w-full max-w-[320px]">
        <title>{summaryTitle}</title>
        {[58, 102, 146].map((r, idx) => (
          <g key={r}>
            <path
              d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
              fill="none"
              stroke="#1e293b"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
            <text
              x={cx + r - 22}
              y={cy - 4}
              fill="#64748b"
              fontSize="9"
              fontFamily="monospace"
            >
              {idx === 0 ? '1m' : idx === 1 ? '2m' : '3.5m'}
            </text>
          </g>
        ))}

        <path
          d={sectorPath(cx, cy, 24, leftR, -160, -113)}
          fill={SEVERITY_FILL[left?.severity ?? 'clear']}
          stroke={SEVERITY_STROKE[left?.severity ?? 'clear']}
          strokeWidth="2"
        />
        <path
          d={sectorPath(cx, cy, 24, centerR, -113, -67)}
          fill={SEVERITY_FILL[center?.severity ?? 'clear']}
          stroke={SEVERITY_STROKE[center?.severity ?? 'clear']}
          strokeWidth="2"
        />
        <path
          d={sectorPath(cx, cy, 24, rightR, -67, -20)}
          fill={SEVERITY_FILL[right?.severity ?? 'clear']}
          stroke={SEVERITY_STROKE[right?.severity ?? 'clear']}
          strokeWidth="2"
        />

        {curbDetected && (
          <path
            d={`M ${cx - 64} ${cy - 36} A 74 74 0 0 1 ${cx + 64} ${cy - 36}`}
            fill="none"
            stroke="#22d3ee"
            strokeWidth="5"
            strokeLinecap="round"
          />
        )}

        <text x="58" y="88" fill="#e2e8f0" fontSize="11" fontWeight="700">
          L {left ? `${left.min_distance_m.toFixed(1)}m` : '--'}
        </text>
        <text x="142" y="42" fill="#e2e8f0" fontSize="11" fontWeight="700">
          C {center ? `${center.min_distance_m.toFixed(1)}m` : '--'}
        </text>
        <text x="224" y="88" fill="#e2e8f0" fontSize="11" fontWeight="700">
          R {right ? `${right.min_distance_m.toFixed(1)}m` : '--'}
        </text>

        <circle cx={cx} cy={cy} r="10" fill="#38bdf8" />
        <text
          x={cx}
          y={cy + 22}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="10"
          fontWeight="600"
        >
          WALKER
        </text>
      </svg>
    </div>
  );
}
