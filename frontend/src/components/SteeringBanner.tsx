import {
  AlertTriangle,
  ArrowBigLeft,
  ArrowBigRight,
  CheckCircle2,
  OctagonAlert,
  Radio,
} from 'lucide-react';
import type { GuidancePayload, Severity, SteeringAction } from '../types/telemetry';

interface SteeringBannerProps {
  guidance: GuidancePayload | null;
  giant?: boolean;
}

interface BannerVisual {
  title: string;
  container: string;
  iconBg: string;
  accentText: string;
  Icon: typeof CheckCircle2;
}

const STEERING_VISUAL: Record<SteeringAction, BannerVisual> = {
  CLEAR: {
    title: 'PATH CLEAR',
    container:
      'border-emerald-500/60 bg-gradient-to-r from-emerald-950/90 via-emerald-900/45 to-slate-950/95 shadow-[0_0_32px_rgba(16,185,129,0.18)]',
    iconBg: 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/40',
    accentText: 'text-emerald-300',
    Icon: CheckCircle2,
  },
  VEER_LEFT: {
    title: 'VEER LEFT',
    container:
      'border-amber-400/75 bg-gradient-to-r from-amber-950/90 via-amber-900/45 to-slate-950/95 shadow-[0_0_36px_rgba(245,158,11,0.22)]',
    iconBg: 'bg-amber-400/20 text-amber-300 ring-1 ring-amber-400/50',
    accentText: 'text-amber-200',
    Icon: ArrowBigLeft,
  },
  VEER_RIGHT: {
    title: 'VEER RIGHT',
    container:
      'border-amber-400/75 bg-gradient-to-r from-amber-950/90 via-amber-900/45 to-slate-950/95 shadow-[0_0_36px_rgba(245,158,11,0.22)]',
    iconBg: 'bg-amber-400/20 text-amber-300 ring-1 ring-amber-400/50',
    accentText: 'text-amber-200',
    Icon: ArrowBigRight,
  },
  STOP: {
    title: 'STOP',
    container:
      'border-red-500/85 bg-gradient-to-r from-red-950/95 via-red-900/55 to-slate-950/95 shadow-[0_0_44px_rgba(239,68,68,0.32)]',
    iconBg: 'bg-red-500/25 text-red-200 ring-1 ring-red-400/60',
    accentText: 'text-red-200',
    Icon: OctagonAlert,
  },
  CURB_DROP: {
    title: 'CURB DROP',
    container:
      'border-fuchsia-400/85 bg-gradient-to-r from-fuchsia-950/95 via-red-950/65 to-slate-950/95 shadow-[0_0_44px_rgba(232,121,249,0.28)]',
    iconBg: 'bg-fuchsia-500/25 text-fuchsia-200 ring-1 ring-fuchsia-400/60',
    accentText: 'text-fuchsia-200',
    Icon: AlertTriangle,
  },
};

const ZONE_PILL: Record<Severity, string> = {
  clear: 'border-emerald-500/50 bg-emerald-500/15 text-emerald-200',
  caution: 'border-amber-400/60 bg-amber-500/20 text-amber-100',
  danger: 'border-red-500/75 bg-red-500/25 text-red-100',
};

export function SteeringBanner({ guidance, giant = false }: SteeringBannerProps) {
  if (!guidance) {
    return (
      <div
        role="status"
        aria-live="polite"
        className={`flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-slate-800 bg-slate-900/80 ${
          giant ? 'px-6 py-4' : 'px-4 py-3'
        }`}
      >
        <div className="flex items-center gap-4">
          <div
            className={`flex items-center justify-center rounded-2xl bg-slate-800 text-slate-400 ${
              giant ? 'h-16 w-16' : 'h-11 w-11'
            }`}
          >
            <Radio className={giant ? 'h-9 w-9' : 'h-5 w-5'} aria-hidden="true" />
          </div>
          <div>
            <div
              className={`font-black uppercase tracking-tight text-slate-200 ${
                giant ? 'text-3xl lg:text-5xl' : 'text-xl'
              }`}
            >
              STANDBY
            </div>
            <p className={giant ? 'text-base text-slate-400' : 'text-xs text-slate-400'}>
              Connect a phone via QR code or trigger a stage demo scene to begin live guidance.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const visual = STEERING_VISUAL[guidance.steering];
  const { Icon } = visual;
  const headline =
    guidance.lens_blocked
      ? 'LENS COVERED · STOP'
      : guidance.steering === 'CURB_DROP' && guidance.curb.stairs_likely
        ? 'STAIRS / CURB DROP'
        : visual.title;

  return (
    <div
      role="status"
      aria-live="assertive"
      className={`flex flex-wrap items-center justify-between gap-4 rounded-2xl border-2 transition-colors duration-150 ${
        visual.container
      } ${giant ? 'px-6 py-4' : 'px-4 py-3'}`}
    >
      <div className="flex min-w-0 items-center gap-4">
        <div
          className={`flex shrink-0 items-center justify-center rounded-2xl ${
            visual.iconBg
          } ${giant ? 'h-16 w-16 lg:h-20 lg:w-20' : 'h-12 w-12'}`}
        >
          <Icon
            className={giant ? 'h-10 w-10 lg:h-12 lg:w-12' : 'h-6 w-6'}
            aria-hidden="true"
          />
        </div>

        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2.5">
            <span
              className={`font-black uppercase leading-none tracking-tight text-white ${
                giant ? 'text-4xl sm:text-5xl lg:text-6xl' : 'text-2xl sm:text-3xl'
              }`}
            >
              {headline}
            </span>
            {guidance.lens_blocked && (
              <span className="rounded-lg border border-red-400/70 bg-red-500/30 px-2 py-0.5 text-xs font-extrabold uppercase text-red-100">
                Lens Blocked
              </span>
            )}
            {guidance.overhead_hazard && (
              <span className="rounded-lg border border-red-400/60 bg-red-500/25 px-2 py-0.5 text-xs font-extrabold uppercase text-red-100">
                Overhead
              </span>
            )}
            {guidance.wall_override && (
              <span className="rounded-lg border border-red-400/60 bg-red-500/25 px-2 py-0.5 text-xs font-extrabold uppercase text-red-100">
                Wall Ahead
              </span>
            )}
            {typeof guidance.pitch_deg === 'number' && (
              <span className="rounded-lg border border-sky-400/50 bg-sky-500/20 px-2 py-0.5 font-mono text-xs font-bold text-sky-200">
                Pitch {guidance.pitch_deg.toFixed(0)}°
              </span>
            )}
          </div>
          <p
            className={`mt-1 truncate font-semibold ${visual.accentText} ${
              giant ? 'text-lg lg:text-xl' : 'text-sm'
            }`}
          >
            {guidance.spoken_cue}
          </p>
        </div>
      </div>

      {/* Right-hand L / C / R metric distance pills */}
      <div className="flex shrink-0 items-center gap-2">
        {(['left', 'center', 'right'] as const).map((side) => {
          const z = guidance[side];
          return (
            <div
              key={side}
              className={`flex flex-col items-center rounded-xl border text-center ${
                ZONE_PILL[z.severity]
              } ${giant ? 'min-w-[84px] px-3.5 py-2' : 'min-w-[66px] px-2.5 py-1.5'}`}
            >
              <span
                className={`font-extrabold uppercase tracking-wider opacity-85 ${
                  giant ? 'text-xs' : 'text-[10px]'
                }`}
              >
                {side}
              </span>
              <span
                className={`font-mono font-black leading-tight ${
                  giant ? 'text-xl lg:text-2xl' : 'text-sm'
                }`}
              >
                {z.min_distance_m.toFixed(1)}m
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
