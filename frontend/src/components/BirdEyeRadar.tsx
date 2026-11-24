import type { GuidancePayload, Severity } from '../types/telemetry';

interface BirdEyeRadarProps {
  guidance: GuidancePayload | null;
}

const SEVERITY_FILL: Record<Severity, string> = {
  clear: 'rgba(16, 185, 129, 0.22)',
  caution: 'rgba(245, 158, 11, 0.42)',
  danger: 'rgba(239, 68, 68, 0.58)',
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

function distToRadius(distMeters: number): number {
  const clamped = Math.max(0.4, Math.min(4.5, distMeters));
  return 34 + ((clamped - 0.4) / 4.1) * 116;
}

export function BirdEyeRadar({ guidance }: BirdEyeRadarProps) {
  const cx = 160;
  const cy = 178;

  const left = guidance?.left;
  const center = guidance?.center;
  const right = guidance?.right;
  const curbDetected = guidance?.curb.detected ?? false;

  const summaryTitle = guidance
    ? `Obstacle radar: Left ${guidance.left.severity} at ${guidance.left.min_distance_m}m, Center ${guidance.center.severity} at ${guidance.center.min_distance_m}m, Right ${guidance.right.severity} at ${guidance.right.min_distance_m}m${curbDetected ? ', curb drop detected ahead' : ''}.`
    : 'Obstacle radar waiting for live camera stream.';

  const leftR = left ? distToRadius(left.min_distance_m) : 145;
  const centerR = center ? distToRadius(center.min_distance_m) : 145;
  const rightR = right ? distToRadius(right.min_distance_m) : 145;

  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-slate-800 bg-slate-900/80 p-3">
      <svg
        role="img"
        viewBox="0 0 320 205"
        className="w-full max-w-[320px]"
      >
        <title>{summaryTitle}</title>

        {/* Distance rings (1m, 2m, 3.5m) */}
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

        {/* Left Sector (-160° to -113°) */}
        <path
          d={sectorPath(cx, cy, 24, leftR, -160, -113)}
          fill={SEVERITY_FILL[left?.severity ?? 'clear']}
          stroke={SEVERITY_STROKE[left?.severity ?? 'clear']}
          strokeWidth="2"
        />

        {/* Center Sector (-113° to -67°) */}
        <path
          d={sectorPath(cx, cy, 24, centerR, -113, -67)}
          fill={SEVERITY_FILL[center?.severity ?? 'clear']}
          stroke={SEVERITY_STROKE[center?.severity ?? 'clear']}
          strokeWidth="2"
        />

        {/* Right Sector (-67° to -20°) */}
        <path
          d={sectorPath(cx, cy, 24, rightR, -67, -20)}
          fill={SEVERITY_FILL[right?.severity ?? 'clear']}
          stroke={SEVERITY_STROKE[right?.severity ?? 'clear']}
          strokeWidth="2"
        />

        {/* Curb / Drop-off cyan alert arc */}
        {curbDetected && (
          <path
            d={`M ${cx - 64} ${cy - 36} A 74 74 0 0 1 ${cx + 64} ${cy - 36}`}
            fill="none"
            stroke="#22d3ee"
            strokeWidth="5"
            strokeLinecap="round"
          />
        )}

        {/* Zone labels */}
        <text x="58" y="88" fill="#e2e8f0" fontSize="11" fontWeight="700">
          L {left ? `${left.min_distance_m.toFixed(1)}m` : '--'}
        </text>
        <text x="142" y="42" fill="#e2e8f0" fontSize="11" fontWeight="700">
          C {center ? `${center.min_distance_m.toFixed(1)}m` : '--'}
        </text>
        <text x="224" y="88" fill="#e2e8f0" fontSize="11" fontWeight="700">
          R {right ? `${right.min_distance_m.toFixed(1)}m` : '--'}
        </text>

        {/* Walker marker */}
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
