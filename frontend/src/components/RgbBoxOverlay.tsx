import type { DetectedObject, Severity } from '../types/telemetry';

interface RgbBoxOverlayProps {
  rawB64: string;
  objects?: DetectedObject[];
  showBoxes?: boolean;
  showCorridors?: boolean;
  large?: boolean;
  className?: string;
}

const BOX_STYLE: Record<
  Severity,
  { border: string; fill: string; pill: string }
> = {
  danger: {
    border: 'border-red-500 shadow-[0_0_16px_rgba(239,68,68,0.55)]',
    fill: 'bg-red-500/15',
    pill: 'bg-red-600 text-white',
  },
  caution: {
    border: 'border-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.45)]',
    fill: 'bg-amber-400/12',
    pill: 'bg-amber-400 text-slate-950',
  },
  clear: {
    border: 'border-emerald-400',
    fill: 'bg-emerald-400/10',
    pill: 'bg-emerald-500 text-slate-950',
  },
};

export function RgbBoxOverlay({
  rawB64,
  objects = [],
  showBoxes = true,
  showCorridors = true,
  large = false,
  className = '',
}: RgbBoxOverlayProps) {
  return (
    <div
      className={`relative flex items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-slate-950 ${className}`}
    >
      {/* Inner wrapper hugs the rendered image so normalized [0..1] boxes stay pixel-accurate across portrait & landscape */}
      <div className="relative inline-block max-h-full max-w-full leading-none">
        <img
          src={`data:image/jpeg;base64,${rawB64}`}
          alt="Live phone camera view with detected obstacle bounding boxes"
          className="block max-h-full max-w-full rounded-lg object-contain"
        />

        {showCorridors && (
          <div className="pointer-events-none absolute inset-0">
            <div
              className="absolute bottom-0 top-0 border-l border-dashed border-white/25"
              style={{ left: '33.333%' }}
            />
            <div
              className="absolute bottom-0 top-0 border-l border-dashed border-white/25"
              style={{ left: '66.666%' }}
            />
            <div className="absolute inset-x-0 top-1.5 grid grid-cols-3 text-center">
              {(['L', 'C', 'R'] as const).map((lane) => (
                <span
                  key={lane}
                  className="mx-auto rounded bg-slate-950/60 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-slate-200/80"
                >
                  {lane}
                </span>
              ))}
            </div>
          </div>
        )}

        {showBoxes &&
          objects.map((obj, idx) => {
            const [rawX1, rawY1, rawX2, rawY2] = obj.bbox_norm;
            const x1 = Math.max(0, Math.min(1, rawX1));
            const y1 = Math.max(0, Math.min(1, rawY1));
            const x2 = Math.max(x1, Math.min(1, rawX2));
            const y2 = Math.max(y1, Math.min(1, rawY2));
            const w = x2 - x1;
            const h = y2 - y1;
            if (w <= 0.005 || h <= 0.005) return null;

            const style = BOX_STYLE[obj.severity];
            const labelInside = y1 < 0.14;

            return (
              <div
                key={`${obj.label}-${idx}-${x1.toFixed(2)}`}
                className={`pointer-events-none absolute rounded-md border-2 transition-all duration-75 ${style.border} ${style.fill}`}
                style={{
                  left: `${x1 * 100}%`,
                  top: `${y1 * 100}%`,
                  width: `${w * 100}%`,
                  height: `${h * 100}%`,
                }}
              >
                <span
                  className={`absolute left-0 whitespace-nowrap rounded px-1.5 py-0.5 font-extrabold uppercase tracking-wide shadow ${
                    large ? 'text-xs' : 'text-[10px]'
                  } ${style.pill} ${
                    labelInside ? 'top-0.5 left-0.5' : '-top-5'
                  }`}
                >
                  {obj.label} {obj.distance_m.toFixed(1)}m
                </span>
              </div>
            );
          })}
      </div>
    </div>
  );
}
