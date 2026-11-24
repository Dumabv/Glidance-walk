import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  Radio,
  Save,
  Smartphone,
  Vibrate,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { BirdEyeRadar } from '../components/BirdEyeRadar';
import { HapticsEngine, type HapticPulseEvent } from '../services/haptics';
import { SpatialAudioEngine } from '../services/spatialAudio';
import type {
  GuidancePayload,
  SampleLabel,
  SteeringAction,
} from '../types/telemetry';

const STEERING_META: Record<
  SteeringAction,
  { title: string; bg: string; border: string }
> = {
  CLEAR: {
    title: 'PATH CLEAR',
    bg: 'bg-emerald-950/80 text-emerald-200',
    border: 'border-emerald-500/70',
  },
  VEER_LEFT: {
    title: 'STEP LEFT',
    bg: 'bg-amber-950/90 text-amber-100',
    border: 'border-amber-400',
  },
  VEER_RIGHT: {
    title: 'STEP RIGHT',
    bg: 'bg-amber-950/90 text-amber-100',
    border: 'border-amber-400',
  },
  STOP: {
    title: 'STOP — BLOCKED',
    bg: 'bg-red-950/95 text-red-100',
    border: 'border-red-500',
  },
  CURB_DROP: {
    title: 'CURB / STEP DOWN',
    bg: 'bg-cyan-950/95 text-cyan-100',
    border: 'border-cyan-400',
  },
};

const QUICK_LABELS: { label: SampleLabel; text: string }[] = [
  { label: 'clear', text: 'Clear Floor' },
  { label: 'obstacle_left', text: 'Obs Left' },
  { label: 'obstacle_center', text: 'Obs Center' },
  { label: 'obstacle_right', text: 'Obs Right' },
  { label: 'curb_drop', text: 'Curb / Step' },
];

export function MobileWalker() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafTimerRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const lastSendAtRef = useRef(0);
  const iosSwitchLabelRef = useRef<HTMLLabelElement | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [guidance, setGuidance] = useState<GuidancePayload | null>(null);
  const [rttMs, setRttMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [hapticsOn, setHapticsOn] = useState(true);
  const [sonarOn, setSonarOn] = useState(true);
  const [voiceOn, setVoiceOn] = useState(false);
  const [lastPulse, setLastPulse] = useState<HapticPulseEvent | null>(null);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const [liveAnnouncement, setLiveAnnouncement] = useState('');

  const frameTimestampsRef = useRef<number[]>([]);
  const lastAnnouncedSteeringRef = useRef<SteeringAction | null>(null);

  const hapticsRef = useRef<HapticsEngine | null>(null);
  const audioRef = useRef<SpatialAudioEngine | null>(null);

  if (!hapticsRef.current) {
    hapticsRef.current = new HapticsEngine((evt) => setLastPulse(evt));
  }
  if (!audioRef.current) {
    audioRef.current = new SpatialAudioEngine();
  }

  useEffect(() => {
    hapticsRef.current?.attachIosSwitchFallback(iosSwitchLabelRef.current);
  }, []);

  useEffect(() => {
    hapticsRef.current?.setEnabled(hapticsOn);
  }, [hapticsOn]);

  useEffect(() => {
    audioRef.current?.setSonarEnabled(sonarOn);
  }, [sonarOn]);

  useEffect(() => {
    audioRef.current?.setVoiceEnabled(voiceOn);
  }, [voiceOn]);

  const stopSession = useCallback(() => {
    if (rafTimerRef.current !== null) {
      window.clearInterval(rafTimerRef.current);
      rafTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    inFlightRef.current = false;
    setStreaming(false);
    setWsConnected(false);
  }, []);

  useEffect(() => () => stopSession(), [stopSession]);

  const sendFrame = useCallback(() => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas) return;
    if (video.readyState < 2) return;

    const now = performance.now();
    // Backpressure: wait for previous reply (safety release after 450ms)
    if (inFlightRef.current && now - lastSendAtRef.current < 450) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = 320;
    canvas.height = 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    inFlightRef.current = true;
    lastSendAtRef.current = now;
    const clientTs = Date.now();

    canvas.toBlob(
      async (blob) => {
        if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          inFlightRef.current = false;
          return;
        }
        const imgBuf = await blob.arrayBuffer();
        const packet = new ArrayBuffer(8 + imgBuf.byteLength);
        new DataView(packet).setFloat64(0, clientTs, false);
        new Uint8Array(packet, 8).set(new Uint8Array(imgBuf));
        wsRef.current.send(packet);
      },
      'image/jpeg',
      0.66
    );
  }, []);

  const startSession = async () => {
    setErrorMsg(null);
    audioRef.current?.unlock();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      setErrorMsg(
        `Camera access failed (${err instanceof Error ? err.message : 'permission denied'}). Make sure you opened via HTTPS.`
      );
      return;
    }

    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/phone`);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      setStreaming(true);
      rafTimerRef.current = window.setInterval(sendFrame, 75);
    };

    ws.onmessage = (event) => {
      inFlightRef.current = false;
      try {
        const data: GuidancePayload = JSON.parse(event.data);
        setGuidance(data);
        const now = Date.now();
        setRttMs(Math.max(0, Math.round(now - data.client_ts)));

        const stamp = performance.now();
        frameTimestampsRef.current = [
          ...frameTimestampsRef.current.filter((t) => stamp - t < 1500),
          stamp,
        ];
        setFps(
          Math.round((frameTimestampsRef.current.length / 1.5) * 10) / 10
        );

        hapticsRef.current?.handleGuidance(data);
        audioRef.current?.handleGuidance(data);

        if (lastAnnouncedSteeringRef.current !== data.steering) {
          lastAnnouncedSteeringRef.current = data.steering;
          setLiveAnnouncement(data.spoken_cue);
        }
      } catch {
        // Ignore malformed frame
      }
    };

    ws.onerror = () => {
      setErrorMsg('WebSocket connection error. Check that the M4 backend is running.');
    };

    ws.onclose = () => {
      setWsConnected(false);
    };
  };

  const quickCapture = async (label: SampleLabel) => {
    setCaptureNotice('Saving snapshot...');
    try {
      const res = await fetch('/api/samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          note: `Phone field capture (${label.replace('_', ' ')})`,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Start camera stream first');
      }
      setCaptureNotice(`Saved real sample: ${label.replace('_', ' ')}`);
    } catch (err) {
      setCaptureNotice(
        err instanceof Error ? err.message : 'Failed to capture snapshot'
      );
    }
    window.setTimeout(() => setCaptureNotice(null), 2600);
  };

  const activeSteering: SteeringAction = guidance?.steering ?? 'CLEAR';
  const meta = STEERING_META[activeSteering];
  const nativeVib = hapticsRef.current.isNativeVibrationSupported();

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-4 px-3 pb-24 pt-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            Mobile Walker HUD
          </h1>
          <p className="text-xs text-slate-400">
            {nativeVib
              ? 'Android native vibration active'
              : 'iOS mode: 3D stereo sonar + Taptic switch fallback'}
          </p>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
            wsConnected
              ? 'bg-emerald-500/20 text-emerald-300'
              : 'bg-slate-800 text-slate-400'
          }`}
        >
          <Radio className="h-3.5 w-3.5" aria-hidden="true" />
          {wsConnected ? `${fps.toFixed(0)} FPS · ${rttMs}ms` : 'Offline'}
        </span>
      </div>

      {/* Centralized debounced live announcement for screen readers */}
      <div aria-live="assertive" aria-atomic="true" className="visually-hidden">
        {liveAnnouncement}
      </div>

      {/* Off-screen iOS 17.4+ switch Taptic bridge (non-focusable) */}
      <span className="visually-hidden" aria-hidden="true">
        <label ref={iosSwitchLabelRef}>
          <input
            type="checkbox"
            tabIndex={-1}
            {...({ switch: '' } as Record<string, string>)}
          />
        </label>
      </span>

      {errorMsg && (
        <div
          role="alert"
          className="rounded-xl border border-red-500/70 bg-red-950/80 p-3 text-sm text-red-100"
        >
          {errorMsg}
        </div>
      )}

      {/* High-contrast guidance banner */}
      <section
        className={`rounded-2xl border-2 p-5 text-center transition-colors ${meta.bg} ${meta.border}`}
      >
        <p className="text-xs font-semibold uppercase tracking-widest opacity-80">
          {guidance
            ? `Server ${guidance.latency.inference_ms.toFixed(0)} ms (${guidance.device.toUpperCase()})`
            : 'Ready to Walk'}
        </p>
        <p className="mt-1 text-3xl font-black tracking-tight">{meta.title}</p>
        <p className="mt-1 text-sm opacity-90">
          {guidance?.spoken_cue ?? 'Tap Start Walking to stream rear camera.'}
        </p>
      </section>

      {/* Primary Action Button */}
      <button
        type="button"
        onClick={streaming ? stopSession : startSession}
        className={`flex w-full items-center justify-center gap-2 rounded-2xl px-6 py-4 text-lg font-bold shadow-lg transition ${
          streaming
            ? 'bg-red-600 text-white hover:bg-red-500'
            : 'bg-sky-500 text-slate-950 hover:bg-sky-400'
        }`}
      >
        {streaming ? (
          <>
            <CameraOff className="h-6 w-6" aria-hidden="true" />
            Stop Walking Stream
          </>
        ) : (
          <>
            <Camera className="h-6 w-6" aria-hidden="true" />
            Start Walking Stream
          </>
        )}
      </button>

      {/* Tactile & Audio controls */}
      <div className="grid grid-cols-2 gap-2.5">
        <button
          type="button"
          aria-pressed={hapticsOn}
          onClick={() => setHapticsOn((v) => !v)}
          className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-semibold ${
            hapticsOn
              ? 'border-sky-400 bg-sky-500/20 text-sky-200'
              : 'border-slate-700 bg-slate-900 text-slate-400'
          }`}
        >
          <Vibrate className="h-4 w-4" aria-hidden="true" />
          Haptics: {hapticsOn ? 'ON' : 'OFF'}
        </button>

        <button
          type="button"
          aria-pressed={sonarOn}
          onClick={() => {
            audioRef.current?.unlock();
            setSonarOn((v) => !v);
          }}
          className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-semibold ${
            sonarOn
              ? 'border-sky-400 bg-sky-500/20 text-sky-200'
              : 'border-slate-700 bg-slate-900 text-slate-400'
          }`}
        >
          {sonarOn ? (
            <Volume2 className="h-4 w-4" aria-hidden="true" />
          ) : (
            <VolumeX className="h-4 w-4" aria-hidden="true" />
          )}
          3D Sonar: {sonarOn ? 'ON' : 'OFF'}
        </button>

        <button
          type="button"
          aria-pressed={voiceOn}
          onClick={() => {
            audioRef.current?.unlock();
            setVoiceOn((v) => !v);
          }}
          className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-semibold ${
            voiceOn
              ? 'border-emerald-400 bg-emerald-500/20 text-emerald-200'
              : 'border-slate-700 bg-slate-900 text-slate-400'
          }`}
        >
          Voice Cues: {voiceOn ? 'ON' : 'OFF'}
        </button>

        <button
          type="button"
          onClick={() => {
            hapticsRef.current?.triggerTestBuzz();
            audioRef.current?.playTestSonar();
          }}
          className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-3 py-3 text-sm font-semibold text-slate-100 hover:bg-slate-700"
        >
          <Smartphone className="h-4 w-4" aria-hidden="true" />
          Test Buzz + Ping
        </button>
      </div>

      {/* Live Camera Preview & Radar */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
          <video
            ref={videoRef}
            playsInline
            muted
            className="aspect-[4/3] w-full object-cover"
          />
          <canvas ref={canvasRef} className="hidden" />
          <div className="flex items-center justify-between px-3 py-2 text-xs text-slate-400">
            <span>Last Pulse: {lastPulse?.patternLabel ?? 'Idle'}</span>
            <span>RTT: {rttMs} ms</span>
          </div>
        </div>
        <BirdEyeRadar guidance={guidance} />
      </div>

      {/* Quick Hallway Dataset Capture for Midterm Benchmark */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="flex items-center gap-1.5 text-sm font-bold text-slate-100">
            <Save className="h-4 w-4 text-sky-400" aria-hidden="true" />
            Record Labeled Hallway Snapshot
          </h2>
          {captureNotice && (
            <span className="text-xs font-medium text-emerald-300">
              {captureNotice}
            </span>
          )}
        </div>
        <p className="mb-2.5 text-xs text-slate-400">
          Tap a ground-truth label while streaming to save the current hallway
          frame into the Midterm Benchmark dataset.
        </p>
        <div className="flex flex-wrap gap-2">
          {QUICK_LABELS.map((item) => (
            <button
              key={item.label}
              type="button"
              onClick={() => void quickCapture(item.label)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:bg-slate-700"
            >
              + {item.text}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
