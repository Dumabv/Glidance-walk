import { useCallback, useEffect, useId, useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import {
  Activity,
  Box,
  Cpu,
  Gauge,
  Maximize2,
  Minimize2,
  Play,
  Presentation,
  RotateCcw,
  SlidersHorizontal,
  Smartphone,
  Wifi,
  X,
} from 'lucide-react';
import { BirdEyeRadar } from '../components/BirdEyeRadar';
import {
  LatencySparkline,
  type LatencySample,
} from '../components/LatencySparkline';
import { RgbBoxOverlay } from '../components/RgbBoxOverlay';
import { SteeringBanner } from '../components/SteeringBanner';
import type {
  CalibrationConfig,
  DashboardMessage,
  DemoSceneSummary,
  GuidancePayload,
  NetworkInfo,
  Severity,
} from '../types/telemetry';

const SEVERITY_BADGE: Record<Severity, string> = {
  clear: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  caution: 'bg-amber-500/20 text-amber-200 border-amber-500/40',
  danger: 'bg-red-500/25 text-red-200 border-red-500/60',
};

function getInitialStageMode(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('stage') === '1';
}

export function MissionControl() {
  const [phoneConnected, setPhoneConnected] = useState(false);
  const [guidance, setGuidance] = useState<GuidancePayload | null>(null);
  const [rawB64, setRawB64] = useState<string | null>(null);
  const [depthB64, setDepthB64] = useState<string | null>(null);
  const [simulatedNote, setSimulatedNote] = useState<string | null>(null);
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);
  const [demoScenes, setDemoScenes] = useState<DemoSceneSummary[]>([]);
  const [calibration, setCalibration] = useState<CalibrationConfig | null>(null);
  const [latencyHistory, setLatencyHistory] = useState<LatencySample[]>([]);
  const [savingCalib, setSavingCalib] = useState(false);
  const [stageMode, setStageMode] = useState<boolean>(getInitialStageMode);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(
    typeof document !== 'undefined' && Boolean(document.fullscreenElement)
  );
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  const idPrefix = useId();

  const toggleFullscreen = useCallback(async () => {
    if (typeof document === 'undefined') return;
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch {
      // Ignore fullscreen rejections (e.g., unsupported browser / gesture restriction)
    }
  }, []);

  const enterStageWithFullscreen = useCallback(() => {
    setStageMode(true);
    if (typeof document !== 'undefined' && !document.fullscreenElement) {
      void document.documentElement.requestFullscreen().catch(() => {});
    }
  }, []);

  const exitStageMode = useCallback(() => {
    setStageMode(false);
    if (typeof document !== 'undefined' && document.fullscreenElement) {
      void document.exitFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const key = e.key.toLowerCase();
      if (key === 'p') {
        e.preventDefault();
        setStageMode((prev) => !prev);
      } else if (key === 'f') {
        e.preventDefault();
        void toggleFullscreen();
      } else if (key === 'b') {
        e.preventDefault();
        setShowBoxes((prev) => !prev);
      } else if (e.key === 'Escape' && stageMode && !document.fullscreenElement) {
        setStageMode(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [stageMode, toggleFullscreen]);

  useEffect(() => {
    void fetch('/api/network-info')
      .then((r) => r.json())
      .then(setNetworkInfo)
      .catch(() => {});

    void fetch('/api/calibration')
      .then((r) => r.json())
      .then(setCalibration)
      .catch(() => {});

    void fetch('/api/demo-scenes')
      .then((r) => r.json())
      .then(setDemoScenes)
      .catch(() => {});
  }, []);

  useEffect(() => {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let active = true;

    const connect = () => {
      if (!active) return;
      ws = new WebSocket(`${wsProto}//${window.location.host}/ws/dashboard`);
      ws.onmessage = (event) => {
        try {
          const msg: DashboardMessage = JSON.parse(event.data);
          if (typeof msg.phone_connected === 'boolean') {
            setPhoneConnected(msg.phone_connected);
          }
          if (msg.guidance) {
            setGuidance(msg.guidance);
            const srvMs = msg.guidance.latency.server_total_ms;
            const infMs = msg.guidance.latency.inference_ms;
            if (srvMs > 0) {
              setLatencyHistory((prev) => [
                ...prev.slice(-59),
                { total: srvMs, inference: infMs },
              ]);
            }
          }
          if (msg.type === 'telemetry') {
            setRawB64(msg.raw_b64 ?? null);
            setDepthB64(msg.depth_b64 ?? null);
            setSimulatedNote(msg.simulated_note ?? null);
          }
        } catch {
          // Ignore malformed message
        }
      };
      ws.onclose = () => {
        if (active) {
          reconnectTimer = window.setTimeout(connect, 1500);
        }
      };
    };

    connect();
    return () => {
      active = false;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  const triggerDemoScene = async (sceneId: string) => {
    await fetch('/api/demo-scenes/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: sceneId }),
    });
  };

  const calibTimerRef = useState<{ id: number | null }>({ id: null })[0];

  const applyCalibration = (next: CalibrationConfig) => {
    const safeNext: CalibrationConfig = {
      ...next,
      caution_threshold: Math.min(
        next.caution_threshold,
        Math.max(0.05, Number((next.danger_threshold - 0.04).toFixed(2)))
      ),
    };
    setCalibration(safeNext);
    setSavingCalib(true);
    if (calibTimerRef.id !== null) {
      window.clearTimeout(calibTimerRef.id);
    }
    calibTimerRef.id = window.setTimeout(async () => {
      try {
        await fetch('/api/calibration', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(safeNext),
        });
      } finally {
        setSavingCalib(false);
      }
    }, 120);
  };

  const resetMetrics = () => {
    setLatencyHistory([]);
  };

  const sortedTotals =
    latencyHistory.length > 0
      ? latencyHistory.map((s) => s.total).sort((a, b) => a - b)
      : [];
  const p50 =
    sortedTotals.length > 0
      ? sortedTotals[Math.floor(sortedTotals.length * 0.5)]
      : 0;
  const p95 =
    sortedTotals.length > 0
      ? sortedTotals[
          Math.min(sortedTotals.length - 1, Math.floor(sortedTotals.length * 0.95))
        ]
      : 0;

  const detectedCount = guidance?.objects?.length ?? 0;

  if (stageMode) {
    return (
      <div className="fixed inset-0 z-50 flex flex-col gap-3 overflow-y-auto bg-[#050811] p-4 text-slate-100 lg:overflow-hidden lg:p-5">
        {/* Giant Projector Steering Banner */}
        <SteeringBanner guidance={guidance} giant />

        {/* Cinema 3-Panel Vision Grid */}
        <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Stage Panel 1: Live RGB + YOLOv8n Bounding Boxes (or Large QR) */}
          <section className="flex min-h-[280px] flex-col rounded-2xl border border-slate-800 bg-slate-900/75 p-4 lg:col-span-4 lg:min-h-0">
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <h2 className="text-base font-extrabold tracking-tight text-white">
                1. Live RGB + Obstacle Boxes
              </h2>
              <div className="flex items-center gap-2">
                {detectedCount > 0 && (
                  <span className="rounded-full border border-sky-400/40 bg-sky-500/15 px-2.5 py-0.5 text-xs font-bold text-sky-200">
                    {detectedCount} {detectedCount === 1 ? 'box' : 'boxes'}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setShowBoxes((v) => !v)}
                  className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-xs font-bold transition ${
                    showBoxes
                      ? 'border-sky-400/60 bg-sky-500/20 text-sky-200'
                      : 'border-slate-700 bg-slate-800 text-slate-400'
                  }`}
                >
                  <Box className="h-3.5 w-3.5" aria-hidden="true" />
                  {showBoxes ? 'Boxes ON' : 'Boxes OFF'}
                </button>
              </div>
            </div>

            {rawB64 ? (
              <RgbBoxOverlay
                rawB64={rawB64}
                objects={guidance?.objects}
                showBoxes={showBoxes}
                large
                className="min-h-0 flex-1"
              />
            ) : (
              <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-950/75 p-4 text-center">
                {networkInfo ? (
                  <>
                    <div className="rounded-2xl bg-white p-3.5 shadow-lg">
                      <QRCodeSVG
                        value={networkInfo.mobile_url}
                        size={180}
                        level="M"
                      />
                    </div>
                    <p className="text-sm font-medium text-slate-200">
                      Scan on Phone (Same Wi-Fi / Hotspot)
                    </p>
                    <code className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-sky-300">
                      {networkInfo.mobile_url}
                    </code>
                  </>
                ) : (
                  <p className="text-sm text-slate-400">
                    Loading local network URL...
                  </p>
                )}
                {simulatedNote && (
                  <p className="rounded-lg bg-sky-500/15 px-3 py-1 text-xs font-semibold text-sky-300">
                    Simulating: {simulatedNote}
                  </p>
                )}
              </div>
            )}
          </section>

          {/* Stage Panel 2: Metric Depth + Corridor Overlay */}
          <section className="flex min-h-[280px] flex-col rounded-2xl border border-slate-800 bg-slate-900/75 p-4 lg:col-span-4 lg:min-h-0">
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <h2 className="text-base font-extrabold tracking-tight text-white">
                2. Metric Depth + Corridors
              </h2>
              <span className="text-xs font-semibold text-slate-300">
                {guidance
                  ? `${guidance.latency.inference_ms.toFixed(0)}ms depth${
                      guidance.latency.yolo_ms
                        ? ` · ${guidance.latency.yolo_ms.toFixed(0)}ms YOLO`
                        : ''
                    }`
                  : 'Inferno + Boxes'}
              </span>
            </div>

            {depthB64 ? (
              <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
                <img
                  src={`data:image/jpeg;base64,${depthB64}`}
                  alt="Colorized depth map showing near obstacles in bright yellow and far areas in dark purple"
                  className="max-h-full max-w-full rounded-lg object-contain"
                />
              </div>
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center rounded-xl border border-slate-800 bg-slate-950/75 p-6 text-center text-sm text-slate-400">
                Connect a phone or click a Stage Scene below to display live
                metric depth &amp; ground-plane overlays.
              </div>
            )}
          </section>

          {/* Stage Panel 3: Bird's-Eye Radar + Semantic Objects */}
          <section className="flex min-h-[280px] flex-col justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/75 p-4 lg:col-span-4 lg:min-h-0">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-extrabold tracking-tight text-white">
                3. Bird&apos;s-Eye Radar
              </h2>
              <span className="font-mono text-xs text-slate-300">
                Curb Δ:{' '}
                <strong className="text-white">
                  {guidance ? guidance.curb.gradient_peak.toFixed(3) : '0.000'}
                </strong>
              </span>
            </div>

            <div className="flex min-h-0 flex-1 items-center justify-center">
              <BirdEyeRadar guidance={guidance} />
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/85 p-3">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="inline-flex items-center gap-1.5 font-semibold text-slate-300">
                  <Activity className="h-3.5 w-3.5 text-sky-400" aria-hidden="true" />
                  Detected Obstacles
                </span>
                <span className="font-mono text-slate-400">
                  {guidance ? `Frame #${guidance.frame_id}` : 'Waiting'}
                </span>
              </div>
              {guidance?.objects && guidance.objects.length > 0 ? (
                <ul className="mt-2 flex flex-wrap gap-1.5">
                  {guidance.objects.map((obj, idx) => (
                    <li
                      key={`${obj.label}-${idx}`}
                      className={`rounded-md border px-2.5 py-1 text-xs font-extrabold uppercase ${
                        SEVERITY_BADGE[obj.severity]
                      }`}
                    >
                      {obj.label} · {obj.distance_m.toFixed(1)}m · {obj.corridor}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-1.5 text-xs text-slate-400">
                  {guidance?.curb.detected
                    ? 'Curb / step-down edge detected in lower ground band.'
                    : 'No foreground obstacles flagged by YOLOv8n.'}
                </p>
              )}
            </div>
          </section>
        </div>

        {/* Bottom Projector Stage Control Bar */}
        <footer className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/90 px-4 py-2.5">
          <div className="flex flex-wrap items-center gap-2 text-xs font-semibold">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-950 px-3 py-1 text-sky-300">
              <Cpu className="h-3.5 w-3.5" aria-hidden="true" />
              {(guidance?.device ?? 'AUTO').toUpperCase()} ·{' '}
              {(guidance?.kernel_backend ?? 'RUST-SIMD').toUpperCase()}
            </span>
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 ${
                phoneConnected
                  ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-300'
                  : 'border-slate-700 bg-slate-950 text-slate-400'
              }`}
            >
              <Smartphone className="h-3.5 w-3.5" aria-hidden="true" />
              {phoneConnected ? 'Phone Live' : 'Waiting for Phone'}
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950 px-3 py-1 text-slate-200">
              <Gauge className="h-3.5 w-3.5 text-amber-400" aria-hidden="true" />
              <span>
                P50 {p50.toFixed(1)}ms · P95 {p95.toFixed(1)}ms
              </span>
              <LatencySparkline samples={latencyHistory} width={76} height={18} />
              <button
                type="button"
                onClick={resetMetrics}
                disabled={latencyHistory.length === 0}
                title="Reset P50/P95 latency metrics"
                aria-label="Reset latency metrics"
                className="rounded-full p-0.5 text-slate-400 transition hover:bg-slate-800 hover:text-white disabled:opacity-40"
              >
                <RotateCcw className="h-3 w-3" aria-hidden="true" />
              </button>
            </span>
          </div>

          {/* Quick Stage Simulator Triggers */}
          {demoScenes.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Stage Sim:
              </span>
              {demoScenes.slice(0, 5).map((s) => (
                <button
                  key={s.scene_id}
                  type="button"
                  onClick={() => void triggerDemoScene(s.scene_id)}
                  className="rounded-lg border border-slate-700 bg-slate-800/90 px-2.5 py-1 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:bg-slate-700"
                >
                  {s.note}
                </button>
              ))}
            </div>
          )}

          {/* Stage Display Controls */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void toggleFullscreen()}
              title="Toggle Fullscreen (F)"
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-100 hover:border-sky-400 hover:bg-slate-700"
            >
              {isFullscreen ? (
                <Minimize2 className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </button>
            <button
              type="button"
              onClick={exitStageMode}
              title="Exit Projector Stage Mode (P or Esc)"
              className="inline-flex items-center gap-1.5 rounded-xl bg-slate-100 px-3 py-1.5 text-xs font-extrabold text-slate-950 hover:bg-white"
            >
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              Exit Stage
            </button>
          </div>
        </footer>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 pb-12 pt-4">
      {/* Top Status & KPI Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Glidance Mission Control
          </h1>
          <p className="text-sm text-slate-400">
            Live monocular depth perception, ground-plane subtraction &amp; curb
            hazard telemetry
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs font-semibold">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-sky-300">
            <Cpu className="h-3.5 w-3.5" aria-hidden="true" />
            Engine: {(guidance?.device ?? 'AUTO').toUpperCase()} · {(guidance?.kernel_backend ?? 'RUST-SIMD').toUpperCase()}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 ${
              phoneConnected
                ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-300'
                : 'border-slate-700 bg-slate-900 text-slate-400'
            }`}
          >
            <Smartphone className="h-3.5 w-3.5" aria-hidden="true" />
            {phoneConnected ? 'Phone Streaming Live' : 'Waiting for Phone'}
          </span>
          <span className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">
            <Gauge className="h-3.5 w-3.5 text-amber-400" aria-hidden="true" />
            <span>
              P50: {p50.toFixed(1)} ms · P95: {p95.toFixed(1)} ms
            </span>
            <LatencySparkline samples={latencyHistory} width={88} height={20} />
            <button
              type="button"
              onClick={resetMetrics}
              disabled={latencyHistory.length === 0}
              title="Clear P50/P95 & sparkline history between demo runs"
              className="inline-flex items-center gap-1 rounded-full border border-slate-700 bg-slate-800 px-2 py-0.5 text-[11px] font-bold text-slate-200 transition hover:border-amber-400 hover:text-white disabled:opacity-40"
            >
              <RotateCcw className="h-3 w-3" aria-hidden="true" />
              Reset Metrics
            </button>
          </span>

          <button
            type="button"
            onClick={enterStageWithFullscreen}
            title="Enter Projector Stage Mode (Shortcut: P)"
            className="inline-flex items-center gap-1.5 rounded-full bg-sky-500 px-3.5 py-1.5 text-xs font-extrabold text-slate-950 shadow-sm transition hover:bg-sky-400"
          >
            <Presentation className="h-3.5 w-3.5" aria-hidden="true" />
            Projector Stage Mode
          </button>

          <button
            type="button"
            onClick={() => void toggleFullscreen()}
            title="Toggle Fullscreen (Shortcut: F)"
            aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
            className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-bold text-slate-200 transition hover:border-sky-400 hover:text-white"
          >
            {isFullscreen ? (
              <Minimize2 className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
      </div>

      {/* Live Steering Banner */}
      <SteeringBanner guidance={guidance} />

      {/* Main 3-Panel Vision Grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Panel 1: Raw Camera + Live RGB Boxes or QR Join */}
        <section className="flex flex-col rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              1. Phone Camera Feed
            </h2>
            <div className="flex items-center gap-2">
              {rawB64 && (
                <button
                  type="button"
                  onClick={() => setShowBoxes((v) => !v)}
                  title="Toggle Live RGB Bounding Boxes (Shortcut: B)"
                  className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-bold transition ${
                    showBoxes
                      ? 'border-sky-400/50 bg-sky-500/15 text-sky-200'
                      : 'border-slate-700 bg-slate-800 text-slate-400'
                  }`}
                >
                  <Box className="h-3 w-3" aria-hidden="true" />
                  {showBoxes
                    ? `Boxes (${detectedCount})`
                    : 'Boxes OFF'}
                </button>
              )}
              <span className="text-xs text-slate-400">
                {phoneConnected ? 'Live RGB' : 'Scan to Connect'}
              </span>
            </div>
          </div>

          {rawB64 ? (
            <RgbBoxOverlay
              rawB64={rawB64}
              objects={guidance?.objects}
              showBoxes={showBoxes}
              className="aspect-[4/3] w-full"
            />
          ) : (
            <div className="flex aspect-[4/3] flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-950/70 p-4 text-center">
              {networkInfo ? (
                <>
                  <div className="rounded-xl bg-white p-2.5">
                    <QRCodeSVG
                      value={networkInfo.mobile_url}
                      size={132}
                      level="M"
                    />
                  </div>
                  <p className="text-xs text-slate-300">
                    Scan on Android/iPhone (same Wi-Fi or Hotspot):
                  </p>
                  <code className="rounded bg-slate-900 px-2 py-1 text-[11px] text-sky-300">
                    {networkInfo.mobile_url}
                  </code>
                </>
              ) : (
                <p className="text-xs text-slate-400">
                  Loading local network URL...
                </p>
              )}
            </div>
          )}

          {simulatedNote && !rawB64 && (
            <p className="mt-2 rounded-lg bg-sky-500/10 px-3 py-1.5 text-xs text-sky-300">
              Simulating: {simulatedNote}
            </p>
          )}
        </section>

        {/* Panel 2: Colorized Depth + Corridor Overlays */}
        <section className="flex flex-col rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-100">
              2. Metric Depth + YOLOv8n Overlay
            </h2>
            <span className="text-xs text-slate-400">
              {guidance
                ? `${guidance.latency.inference_ms.toFixed(0)}ms depth${guidance.latency.yolo_ms ? ` · ${guidance.latency.yolo_ms.toFixed(0)}ms YOLO` : ''}`
                : 'Inferno + Boxes'}
            </span>
          </div>

          {depthB64 ? (
            <div className="flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
              <img
                src={`data:image/jpeg;base64,${depthB64}`}
                alt="Colorized depth map showing near obstacles in bright yellow and far areas in dark purple"
                className="max-h-full max-w-full rounded-lg object-contain"
              />
            </div>
          ) : (
            <div className="flex aspect-[4/3] items-center justify-center rounded-xl border border-slate-800 bg-slate-950/70 p-6 text-center text-xs text-slate-400">
              Connect a phone or click a Demo Scene below to inspect live depth
              maps and corridor overlays.
            </div>
          )}

          {guidance && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
              {(['left', 'center', 'right'] as const).map((side) => (
                <div
                  key={side}
                  className={`rounded-lg border px-2 py-1.5 ${
                    SEVERITY_BADGE[guidance[side].severity]
                  }`}
                >
                  <div className="font-bold uppercase">{side}</div>
                  <div>{guidance[side].min_distance_m.toFixed(1)}m</div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Panel 3: Bird's-Eye Radar + Guidance Directive */}
        <section className="flex flex-col justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-100">
              3. Bird&apos;s-Eye Radar &amp; Cue
            </h2>
            {guidance && (
              <span
                className={`rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase ${
                  SEVERITY_BADGE[guidance.overall_severity]
                }`}
              >
                {guidance.steering.replace('_', ' ')}
              </span>
            )}
          </div>

          <BirdEyeRadar guidance={guidance} />

          <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="inline-flex items-center gap-1">
                <Activity className="h-3.5 w-3.5 text-sky-400" aria-hidden="true" />
                Curb Gradient Peak
              </span>
              <span className="font-mono text-slate-200">
                {guidance ? guidance.curb.gradient_peak.toFixed(3) : '0.000'}
              </span>
            </div>
            <p className="mt-1.5 text-sm font-semibold text-white">
              {guidance?.spoken_cue ?? 'Awaiting first frame...'}
            </p>
            {guidance?.objects && guidance.objects.length > 0 && (
              <ul className="mt-2 flex flex-wrap gap-1.5">
                {guidance.objects.map((obj, idx) => (
                  <li
                    key={`${obj.label}-${idx}`}
                    className={`rounded-md border px-2 py-0.5 text-[11px] font-bold uppercase ${
                      SEVERITY_BADGE[obj.severity]
                    }`}
                  >
                    {obj.label} · {obj.distance_m.toFixed(1)}m · {obj.corridor}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>

      {/* Bottom Row: Instant Stage Simulator + Live Threshold Calibration */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-7">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-bold text-white">
              <Play className="h-4 w-4 text-emerald-400" aria-hidden="true" />
              Instant Stage Scene Simulator (No Phone Needed)
            </h2>
            <span className="inline-flex items-center gap-1 text-xs text-slate-400">
              <Wifi className="h-3.5 w-3.5" aria-hidden="true" />
              Tests ground-plane &amp; curb math live
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {demoScenes.slice(0, 9).map((s) => (
              <button
                key={s.scene_id}
                type="button"
                onClick={() => void triggerDemoScene(s.scene_id)}
                className="rounded-lg border border-slate-700 bg-slate-800/90 px-3 py-2 text-xs font-medium text-slate-100 hover:border-sky-400 hover:bg-slate-700"
              >
                {s.note}
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 lg:col-span-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-bold text-white">
              <SlidersHorizontal
                className="h-4 w-4 text-sky-400"
                aria-hidden="true"
              />
              Live Geometry Calibration
            </h2>
            <span className="text-xs text-slate-400">
              {savingCalib ? 'Saving...' : 'Applies instantly'}
            </span>
          </div>

          {calibration && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {[
                {
                  key: 'caution_threshold' as const,
                  label: 'Caution Threshold',
                  min: 0.1,
                  max: 0.6,
                  step: 0.02,
                },
                {
                  key: 'danger_threshold' as const,
                  label: 'Danger Threshold',
                  min: 0.25,
                  max: 0.8,
                  step: 0.02,
                },
                {
                  key: 'curb_gradient_threshold' as const,
                  label: 'Curb Drop Sensitivity',
                  min: 0.06,
                  max: 0.35,
                  step: 0.01,
                },
                {
                  key: 'floor_margin' as const,
                  label: 'Floor Slope Margin',
                  min: 0.05,
                  max: 0.3,
                  step: 0.01,
                },
              ].map((ctrl) => {
                const inputId = `${idPrefix}-${ctrl.key}`;
                return (
                  <div key={ctrl.key} className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor={inputId} className="text-slate-300">
                        {ctrl.label}
                      </label>
                      <span className="font-mono text-sky-300">
                        {calibration[ctrl.key].toFixed(2)}
                      </span>
                    </div>
                    <input
                      id={inputId}
                      type="range"
                      min={ctrl.min}
                      max={ctrl.max}
                      step={ctrl.step}
                      value={calibration[ctrl.key]}
                      onChange={(e) =>
                        void applyCalibration({
                          ...calibration,
                          [ctrl.key]: parseFloat(e.target.value),
                        })
                      }
                      className="w-full accent-sky-400"
                    />
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
