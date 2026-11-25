import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  ChevronDown,
  ChevronUp,
  Cpu,
  Lock,
  Radio,
  Save,
  Smartphone,
  Unlock,
  Vibrate,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { BirdEyeRadar } from '../components/BirdEyeRadar';
import { HapticsEngine, type HapticPulseEvent } from '../services/haptics';
import {
  SpatialAudioEngine,
  type GuidanceProfile,
  type VoiceLang,
} from '../services/spatialAudio';
import type {
  ExecutionMode,
  GuidancePayload,
  SampleLabel,
  SteeringAction,
} from '../types/telemetry';

const STEERING_META: Record<
  SteeringAction | 'READY' | 'SIGNAL_LOST',
  { title: string; bg: string; border: string }
> = {
  READY: {
    title: 'READY TO WALK',
    bg: 'bg-slate-900/90 text-slate-100',
    border: 'border-slate-600',
  },
  SIGNAL_LOST: {
    title: 'STOP — SIGNAL LOST',
    bg: 'bg-red-950/95 text-red-100',
    border: 'border-red-500',
  },
  CLEAR: {
    title: 'PATH CLEAR',
    bg: 'bg-emerald-950/85 text-emerald-200',
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
  const workerRef = useRef<Worker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const rvfcHandleRef = useRef<number | null>(null);
  const watchdogRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  const sessionActiveRef = useRef(false);
  const inFlightRef = useRef(false);
  const encodingRef = useRef(false);
  const lastSendAtRef = useRef(0);
  const lastValidFrameAtRef = useRef(0);
  const activeRuntimeRef = useRef<'ws' | 'worker'>('ws');
  const workerReadyRef = useRef(false);
  const pitchDegRef = useRef(0);
  const rollDegRef = useRef(0);
  const iosSwitchLabelRef = useRef<HTMLLabelElement | null>(null);

  const [execMode, setExecMode] = useState<ExecutionMode>('auto');
  const [streaming, setStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [workerStatus, setWorkerStatus] = useState<string>('idle');
  const [signalLost, setSignalLost] = useState(false);
  const [tiltWarning, setTiltWarning] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [guidance, setGuidance] = useState<GuidancePayload | null>(null);
  const [rttMs, setRttMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [videoAspect, setVideoAspect] = useState<'portrait' | 'landscape'>('portrait');

  const [hapticsOn, setHapticsOn] = useState(true);
  const [sonarOn, setSonarOn] = useState(true);
  const [voiceOn, setVoiceOn] = useState(true);
  const [guidanceProfile, setGuidanceProfile] = useState<GuidanceProfile>('smart_calm');
  const [voiceLang, setVoiceLang] = useState<VoiceLang>('ru');
  const [eyesFreeLock, setEyesFreeLock] = useState(false);
  const [devOpen, setDevOpen] = useState(false);
  const [lastPulse, setLastPulse] = useState<HapticPulseEvent | null>(null);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const [liveAnnouncement, setLiveAnnouncement] = useState('');

  const frameTimestampsRef = useRef<number[]>([]);
  const smoothedRttRef = useRef<number>(250);
  const videoAspectRef = useRef<'portrait' | 'landscape'>('portrait');
  const dispatchFrameRef = useRef<() => void>(() => {});
  const lastAnnouncedSteeringRef = useRef<string | null>(null);
  const execModeRef = useRef<ExecutionMode>(execMode);
  execModeRef.current = execMode;

  const audioRef = useRef<SpatialAudioEngine | null>(null);
  const hapticsRef = useRef<HapticsEngine | null>(null);

  if (!audioRef.current) {
    audioRef.current = new SpatialAudioEngine();
  }
  if (!hapticsRef.current) {
    hapticsRef.current = new HapticsEngine(
      (evt) => setLastPulse(evt),
      (rank) => audioRef.current?.playIosTactileClick(rank)
    );
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

  useEffect(() => {
    audioRef.current?.setProfile(guidanceProfile);
    hapticsRef.current?.setCalmBackoff(guidanceProfile !== 'sonar_3d');
  }, [guidanceProfile]);

  useEffect(() => {
    audioRef.current?.setVoiceLang(voiceLang);
  }, [voiceLang]);

  const handleIncomingGuidance = useCallback((data: GuidancePayload) => {
    inFlightRef.current = false;
    lastValidFrameAtRef.current = performance.now();
    setSignalLost(false);
    setGuidance(data);

    const now = Date.now();
    const measuredRtt = Math.max(1, Math.round(now - data.client_ts));
    smoothedRttRef.current = 0.75 * smoothedRttRef.current + 0.25 * measuredRtt;
    setRttMs(measuredRtt);

    const stamp = performance.now();
    frameTimestampsRef.current = [
      ...frameTimestampsRef.current.filter((t) => stamp - t < 1500),
      stamp,
    ];
    setFps(Math.round((frameTimestampsRef.current.length / 1.5) * 10) / 10);

    hapticsRef.current?.handleGuidance(data);
    audioRef.current?.handleGuidance(data);

    const key = `${data.overall_severity}:${data.steering}`;
    if (lastAnnouncedSteeringRef.current !== key) {
      lastAnnouncedSteeringRef.current = key;
      setLiveAnnouncement(data.spoken_cue);
    }

    // Zero-wait turnaround: immediately queue next frame if >= 30ms elapsed since previous dispatch
    if (sessionActiveRef.current && performance.now() - lastSendAtRef.current >= 30) {
      dispatchFrameRef.current();
    }
  }, []);

  const ensureWorker = useCallback(() => {
    if (workerRef.current) return workerRef.current;
    const worker = new Worker(
      new URL('../workers/vision.worker.ts', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (evt) => {
      const msg = evt.data;
      if (msg?.type === 'worker_status') {
        setWorkerStatus(`${msg.status}: ${msg.detail}`);
        if (msg.status === 'ready') {
          workerReadyRef.current = true;
        }
      } else if (msg?.type === 'guidance' && msg.guidance) {
        handleIncomingGuidance(msg.guidance as GuidancePayload);
      } else if (msg?.type === 'worker_error') {
        inFlightRef.current = false;
        setWorkerStatus(`error: ${msg.detail}`);
      }
    };
    worker.postMessage({ type: 'init' });
    workerRef.current = worker;
    return worker;
  }, [handleIncomingGuidance]);

  const acquireWakeLock = useCallback(async () => {
    try {
      if ('wakeLock' in navigator && document.visibilityState === 'visible') {
        wakeLockRef.current = await navigator.wakeLock.request('screen');
      }
    } catch {
      // Ignore gracefully if battery-saver blocks WakeLock
    }
  }, []);

  const stopSession = useCallback(() => {
    sessionActiveRef.current = false;
    if (rvfcHandleRef.current !== null) {
      const v = videoRef.current as (HTMLVideoElement & {
        cancelVideoFrameCallback?: (h: number) => void;
      }) | null;
      if (v && typeof v.cancelVideoFrameCallback === 'function') {
        v.cancelVideoFrameCallback(rvfcHandleRef.current);
      } else {
        window.cancelAnimationFrame(rvfcHandleRef.current);
      }
      rvfcHandleRef.current = null;
    }
    if (watchdogRef.current !== null) {
      window.clearInterval(watchdogRef.current);
      watchdogRef.current = null;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (wakeLockRef.current) {
      void wakeLockRef.current.release().catch(() => {});
      wakeLockRef.current = null;
    }
    inFlightRef.current = false;
    encodingRef.current = false;
    setStreaming(false);
    setWsConnected(false);
    setSignalLost(false);
    setGuidance(null);
  }, []);

  useEffect(() => {
    const onVisibility = () => {
      if (!sessionActiveRef.current) return;
      if (document.visibilityState === 'visible') {
        void acquireWakeLock();
      } else {
        audioRef.current?.speakStatusImmediate(null, true);
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      stopSession();
      workerRef.current?.terminate();
      workerRef.current = null;
      audioRef.current?.dispose();
    };
  }, [acquireWakeLock, stopSession]);

  // IMU DeviceOrientation Pitch/Roll Compensation & Tilt Guard
  useEffect(() => {
    const onOrientation = (e: DeviceOrientationEvent) => {
      if (e.beta === null || e.gamma === null) return;
      pitchDegRef.current = 0.8 * pitchDegRef.current + 0.2 * e.beta;
      rollDegRef.current = 0.8 * rollDegRef.current + 0.2 * e.gamma;

      if (!sessionActiveRef.current) return;
      const p = pitchDegRef.current;
      if (p > 5 && p < 34) {
        setTiltWarning('Camera aimed too far down at shoes — tilt phone slightly up.');
      } else if (p > 108 && p < 165) {
        setTiltWarning('Camera aimed too high at ceiling — level phone forward.');
      } else {
        setTiltWarning(null);
      }
    };
    window.addEventListener('deviceorientation', onOrientation, { passive: true });
    return () => window.removeEventListener('deviceorientation', onOrientation);
  }, []);

  const dispatchFrame = useCallback(() => {
    if (!sessionActiveRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    const now = performance.now();
    if (encodingRef.current) return;
    if (inFlightRef.current && now - lastSendAtRef.current < 520) return;

    const vw = video.videoWidth || 480;
    const vh = video.videoHeight || 640;
    const isPortrait = vh > vw;

    // Aspect-preserving ViT-14 aligned dimensions (eliminates 1.78x portrait squash!)
    const maxSide = activeRuntimeRef.current === 'worker' ? 252 : 322;
    const scale = Math.min(1, maxSide / Math.max(vw, vh));
    const targetW = Math.max(112, Math.round((vw * scale) / 14) * 14);
    const targetH = Math.max(112, Math.round((vh * scale) / 14) * 14);

    const nextAspect = isPortrait ? 'portrait' : 'landscape';
    if (nextAspect !== videoAspectRef.current) {
      videoAspectRef.current = nextAspect;
      setVideoAspect(nextAspect);
    }

    const useWorker =
      execModeRef.current === 'on_phone' ||
      (execModeRef.current === 'auto' && activeRuntimeRef.current === 'worker');

    // Path A: 100% Standalone On-Phone Worker (`ImageBitmap` zero-JPEG transfer -> WebGPU/WASM + Rust `.wasm`)
    if (useWorker) {
      const worker = ensureWorker();
      inFlightRef.current = true;
      encodingRef.current = true;
      lastSendAtRef.current = now;
      const clientTs = Date.now();
      createImageBitmap(video, {
        resizeWidth: targetW,
        resizeHeight: targetH,
        resizeQuality: 'low',
      })
        .then((bitmap) => {
          encodingRef.current = false;
          worker.postMessage(
            {
              type: 'frame',
              bitmap,
              clientTs,
              pitchDeg: pitchDegRef.current,
              rollDeg: rollDegRef.current,
            },
            [bitmap]
          );
        })
        .catch(() => {
          encodingRef.current = false;
          inFlightRef.current = false;
        });
      return;
    }

    // Path B: Tethered Low-Latency Binary WebSocket (`[f64 ts][f32 pitch][f32 roll] + JPEG`)
    const ws = wsRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !canvas) return;

    // Only set canvas dimensions when orientation/resolution actually changes
    if (canvas.width !== targetW || canvas.height !== targetH || !canvasCtxRef.current) {
      canvas.width = targetW;
      canvas.height = targetH;
      canvasCtxRef.current = canvas.getContext('2d', {
        alpha: false,
        desynchronized: true,
      });
    }
    const ctx = canvasCtxRef.current;
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, targetW, targetH);

    inFlightRef.current = true;
    encodingRef.current = true;
    lastSendAtRef.current = now;
    const clientTs = Date.now();
    const pitch = pitchDegRef.current;
    const roll = rollDegRef.current;

    canvas.toBlob(
      async (blob) => {
        encodingRef.current = false;
        if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          inFlightRef.current = false;
          return;
        }
        const imgBuf = await blob.arrayBuffer();
        const packet = new ArrayBuffer(16 + imgBuf.byteLength);
        const dv = new DataView(packet);
        dv.setFloat64(0, clientTs, false);
        dv.setFloat32(8, pitch, false);
        dv.setFloat32(12, roll, false);
        new Uint8Array(packet, 16).set(new Uint8Array(imgBuf));
        wsRef.current.send(packet);
      },
      'image/jpeg',
      0.68
    );
  }, [ensureWorker]);

  dispatchFrameRef.current = dispatchFrame;

  const connectWebSocket = useCallback(
    (attempt = 0) => {
      if (!sessionActiveRef.current || execModeRef.current === 'on_phone') return;
      const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/phone`);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setErrorMsg(null);
        activeRuntimeRef.current = 'ws';
        lastValidFrameAtRef.current = performance.now();
      };

      ws.onmessage = (event) => {
        try {
          const data: GuidancePayload = JSON.parse(event.data);
          handleIncomingGuidance(data);
        } catch {
          inFlightRef.current = false;
        }
      };

      ws.onerror = () => {
        if (execModeRef.current === 'auto') {
          // Seamless hot-failover to Standalone On-Phone WebWorker + Rust `.wasm`!
          activeRuntimeRef.current = 'worker';
          ensureWorker();
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (!sessionActiveRef.current) return;
        if (execModeRef.current === 'auto') {
          activeRuntimeRef.current = 'worker';
          ensureWorker();
        }
        const delay = Math.min(2500, 300 * Math.pow(1.7, attempt));
        reconnectTimerRef.current = window.setTimeout(
          () => connectWebSocket(attempt + 1),
          delay
        );
      };
    },
    [ensureWorker, handleIncomingGuidance]
  );

  const startSession = async () => {
    setErrorMsg(null);
    hapticsRef.current?.triggerTapFeedback();
    audioRef.current?.unlock();

    // Request iOS 13+ IMU orientation permission inside user gesture if available
    const devOrient = DeviceOrientationEvent as unknown as {
      requestPermission?: () => Promise<string>;
    };
    if (typeof devOrient.requestPermission === 'function') {
      void devOrient.requestPermission().catch(() => {});
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 24, max: 30 },
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
        `Camera access failed (${err instanceof Error ? err.message : 'permission denied'}). Open over HTTPS or localhost.`
      );
      return;
    }

    await acquireWakeLock();
    audioRef.current?.resetSessionState();
    hapticsRef.current?.resetSessionState();
    sessionActiveRef.current = true;
    lastValidFrameAtRef.current = performance.now();
    setStreaming(true);

    if (execMode === 'on_phone') {
      activeRuntimeRef.current = 'worker';
      ensureWorker();
    } else {
      activeRuntimeRef.current = 'ws';
      connectWebSocket(0);
      if (execMode === 'auto') {
        // Warm up standalone worker in background for instant zero-drop failover
        ensureWorker();
      }
    }

    // Beat-free camera-synced frame pacing (`requestVideoFrameCallback` up to 30 FPS):
    // Eliminates the 0-69ms idle beat-aliasing lag of `setInterval(..., 70)`!
    const vExt = videoRef.current as (HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
    }) | null;
    const stepLoop = () => {
      if (!sessionActiveRef.current) return;
      if (performance.now() - lastSendAtRef.current >= 33) {
        dispatchFrame();
      }
      if (vExt && typeof vExt.requestVideoFrameCallback === 'function') {
        rvfcHandleRef.current = vExt.requestVideoFrameCallback(stepLoop);
      } else {
        rvfcHandleRef.current = window.requestAnimationFrame(stepLoop);
      }
    };
    stepLoop();

    // Adaptive RTT-Aware Watchdog: avoids false alarms on 4G / Cloudflare tunnels while catching real stalls!
    watchdogRef.current = window.setInterval(() => {
      if (!sessionActiveRef.current) return;
      const track = streamRef.current?.getVideoTracks()[0];
      const trackDead = !track || track.readyState === 'ended';
      const graceMs =
        activeRuntimeRef.current === 'worker' && !workerReadyRef.current
          ? 14000
          : Math.max(3600, Math.min(8500, smoothedRttRef.current * 3.5));
      const stale = performance.now() - lastValidFrameAtRef.current > graceMs;

      if (trackDead || stale) {
        setSignalLost(true);
        hapticsRef.current?.triggerFailsafeDisconnect();
        audioRef.current?.triggerDisconnectAlarm(
          trackDead ? 'camera_dead' : 'signal_delayed'
        );
      }
    }, 350);
  };

  const quickCapture = async (label: SampleLabel) => {
    hapticsRef.current?.triggerTapFeedback();
    setCaptureNotice('Saving snapshot...');
    try {
      let image_b64: string | undefined;
      if (videoRef.current) {
        const v = videoRef.current;
        const snapCanvas = document.createElement('canvas');
        snapCanvas.width = v.videoWidth || 320;
        snapCanvas.height = v.videoHeight || 240;
        snapCanvas.getContext('2d')?.drawImage(v, 0, 0, snapCanvas.width, snapCanvas.height);
        image_b64 = snapCanvas.toDataURL('image/jpeg', 0.8);
      }
      const res = await fetch('/api/samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          note: `Phone field capture (${label.replace('_', ' ')})`,
          image_b64,
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

  // Fail-closed visual banner state (never defaults to green `PATH CLEAR` before first frame!)
  const bannerStateKey: keyof typeof STEERING_META = !streaming
    ? 'READY'
    : signalLost
      ? 'SIGNAL_LOST'
      : guidance
        ? guidance.steering
        : 'READY';
  const meta = STEERING_META[bannerStateKey];
  const nativeVib = hapticsRef.current.isNativeVibrationSupported();

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-3.5 px-3 pb-24 pt-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            Mobile Walker HUD
          </h1>
          <p className="text-xs text-slate-400">
            {nativeVib
              ? 'Android LRA tactile patterns + 3D HRTF sonar'
              : 'iOS 3D HRTF binaural sonar + tactile sub-bass'}
          </p>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
            streaming && !signalLost
              ? 'bg-emerald-500/20 text-emerald-300'
              : signalLost
                ? 'bg-red-500/25 text-red-200'
                : 'bg-slate-800 text-slate-400'
          }`}
        >
          <Radio className="h-3.5 w-3.5" aria-hidden="true" />
          {streaming
            ? `${fps.toFixed(1)} FPS · ${rttMs}ms (${(guidance?.kernel_backend ?? activeRuntimeRef.current).toUpperCase()})`
            : 'Standby'}
        </span>
      </div>

      {/* Screen-reader live region muted automatically when app TTS is active to prevent double-talk */}
      <div
        aria-live={voiceOn ? 'off' : 'assertive'}
        aria-atomic="true"
        className="visually-hidden"
      >
        {liveAnnouncement}
      </div>

      {/* Off-screen iOS 17.4+ switch Taptic bridge for physical button taps */}
      <span className="visually-hidden" aria-hidden="true">
        <label ref={iosSwitchLabelRef}>
          <input
            type="checkbox"
            tabIndex={-1}
            {...({ switch: '' } as Record<string, string>)}
          />
        </label>
      </span>

      {/* Execution Mode Selector: Auto Failover vs Tethered Server vs 100% On-Phone Standalone */}
      <div
        role="group"
        aria-label="Inference execution mode"
        className="grid grid-cols-3 gap-1.5 rounded-xl border border-slate-800 bg-slate-900/90 p-1.5 text-xs font-semibold"
      >
        {(
          [
            { id: 'auto', label: 'Auto (Failover)' },
            { id: 'tethered', label: 'Laptop Server' },
            { id: 'on_phone', label: 'On-Phone WASM' },
          ] as const
        ).map((m) => (
          <button
            key={m.id}
            type="button"
            disabled={eyesFreeLock}
            aria-pressed={execMode === m.id}
            onClick={() => {
              hapticsRef.current?.triggerTapFeedback();
              setExecMode(m.id);
              if (m.id === 'on_phone') {
                activeRuntimeRef.current = 'worker';
                ensureWorker();
              } else if (m.id === 'tethered') {
                activeRuntimeRef.current = 'ws';
              }
            }}
            className={`rounded-lg px-2 py-2 transition ${
              execMode === m.id
                ? 'bg-sky-500 text-slate-950 font-bold'
                : 'text-slate-300 hover:bg-slate-800'
            } disabled:opacity-50`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {workerStatus !== 'idle' && (execMode === 'on_phone' || activeRuntimeRef.current === 'worker') && (
        <p className="flex items-center gap-1.5 rounded-lg border border-sky-500/30 bg-sky-950/40 px-3 py-1.5 text-xs text-sky-200">
          <Cpu className="h-3.5 w-3.5 shrink-0 text-sky-400" aria-hidden="true" />
          <span>{workerStatus}</span>
        </p>
      )}

      {tiltWarning && (
        <div
          role="status"
          className="rounded-xl border border-amber-500/60 bg-amber-950/70 px-3 py-2 text-xs font-semibold text-amber-100"
        >
          {tiltWarning}
        </div>
      )}

      {errorMsg && (
        <div
          role="alert"
          className="rounded-xl border border-red-500/70 bg-red-950/80 p-3 text-sm text-red-100"
        >
          {errorMsg}
        </div>
      )}

      {/* High-contrast guidance banner (doubles as single-tap spoken status surface in Eyes-Free mode) */}
      <button
        type="button"
        onClick={() => {
          hapticsRef.current?.triggerTapFeedback();
          audioRef.current?.speakStatusImmediate(guidance, signalLost);
        }}
        className={`w-full rounded-2xl border-2 p-5 text-center transition-colors ${meta.bg} ${meta.border}`}
      >
        <span className="block text-xs font-semibold uppercase tracking-widest opacity-80">
          {signalLost
            ? 'Fail-Closed Watchdog Engaged'
            : guidance
              ? `${guidance.latency.inference_ms.toFixed(0)}ms infer · ${guidance.latency.hazard_ms.toFixed(2)}ms ${guidance.kernel_backend ?? 'rust'} (${guidance.device.toUpperCase()})`
              : wsConnected
                ? 'Connected — Awaiting First Frame'
                : 'Tap Banner Anytime to Hear Status'}
        </span>
        <span className="mt-1 block text-3xl font-black tracking-tight">
          {meta.title}
        </span>
        <span className="mt-1 block text-sm opacity-90">
          {signalLost
            ? 'Connection or camera interrupted — stop walking.'
            : guidance?.spoken_cue ?? 'Tap Start Walking Stream to engage rear camera.'}
        </span>
      </button>

      {/* Primary Action Button + Palm-Safe Eyes-Free Lock */}
      <div className="flex gap-2">
        <button
          type="button"
          disabled={eyesFreeLock && streaming}
          onClick={streaming ? stopSession : () => void startSession()}
          className={`flex flex-1 items-center justify-center gap-2 rounded-2xl px-5 py-4 text-base font-bold shadow-lg transition ${
            streaming
              ? 'bg-red-600 text-white hover:bg-red-500 disabled:opacity-45'
              : 'bg-sky-500 text-slate-950 hover:bg-sky-400'
          }`}
        >
          {streaming ? (
            <>
              <CameraOff className="h-5 w-5" aria-hidden="true" />
              {eyesFreeLock ? 'Locked (Unlock to Stop)' : 'Stop Walking Stream'}
            </>
          ) : (
            <>
              <Camera className="h-5 w-5" aria-hidden="true" />
              Start Walking Stream
            </>
          )}
        </button>

        {streaming && (
          <button
            type="button"
            aria-pressed={eyesFreeLock}
            aria-label={eyesFreeLock ? 'Unlock touch controls' : 'Lock controls for pocket or chest carry'}
            onClick={() => {
              hapticsRef.current?.triggerTapFeedback();
              setEyesFreeLock((v) => !v);
            }}
            className={`flex items-center justify-center gap-1.5 rounded-2xl border px-4 py-4 text-xs font-bold ${
              eyesFreeLock
                ? 'border-amber-400 bg-amber-500/25 text-amber-100'
                : 'border-slate-700 bg-slate-900 text-slate-300'
            }`}
          >
            {eyesFreeLock ? (
              <Lock className="h-5 w-5" aria-hidden="true" />
            ) : (
              <Unlock className="h-5 w-5" aria-hidden="true" />
            )}
            {eyesFreeLock ? 'Locked' : 'Lock'}
          </button>
        )}
      </div>

      {/* Calm Guidance Profile Selector (eliminates audio fatigue & stationary nagging!) */}
      <div
        role="group"
        aria-label="Audio guidance profile"
        className="grid grid-cols-3 gap-1.5 rounded-xl border border-slate-800 bg-slate-900/90 p-1.5 text-xs font-semibold"
      >
        {(
          [
            { id: 'smart_calm', label: 'Smart Calm' },
            { id: 'sonar_3d', label: '3D Radar' },
            { id: 'tactile_only', label: 'Silent Vibro' },
          ] as const
        ).map((p) => (
          <button
            key={p.id}
            type="button"
            disabled={eyesFreeLock}
            aria-pressed={guidanceProfile === p.id}
            onClick={() => {
              audioRef.current?.unlock();
              hapticsRef.current?.triggerTapFeedback();
              setGuidanceProfile(p.id);
            }}
            className={`rounded-lg px-2 py-2 transition ${
              guidanceProfile === p.id
                ? 'bg-emerald-500 text-slate-950 font-bold'
                : 'text-slate-300 hover:bg-slate-800'
            } disabled:opacity-50`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Tactile & Audio controls */}
      <div className="grid grid-cols-2 gap-2.5">
        <button
          type="button"
          disabled={eyesFreeLock}
          aria-pressed={hapticsOn}
          onClick={() => setHapticsOn((v) => !v)}
          className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-semibold ${
            hapticsOn
              ? 'border-sky-400 bg-sky-500/20 text-sky-200'
              : 'border-slate-700 bg-slate-900 text-slate-400'
          } disabled:opacity-50`}
        >
          <Vibrate className="h-4 w-4" aria-hidden="true" />
          Haptics: {hapticsOn ? 'ON' : 'OFF'}
        </button>

        <button
          type="button"
          disabled={eyesFreeLock || guidanceProfile === 'tactile_only'}
          aria-pressed={sonarOn && guidanceProfile !== 'tactile_only'}
          onClick={() => {
            audioRef.current?.unlock();
            setSonarOn((v) => !v);
          }}
          className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-semibold ${
            sonarOn && guidanceProfile !== 'tactile_only'
              ? 'border-sky-400 bg-sky-500/20 text-sky-200'
              : 'border-slate-700 bg-slate-900 text-slate-400'
          } disabled:opacity-50`}
        >
          {sonarOn && guidanceProfile !== 'tactile_only' ? (
            <Volume2 className="h-4 w-4" aria-hidden="true" />
          ) : (
            <VolumeX className="h-4 w-4" aria-hidden="true" />
          )}
          3D Sonar: {sonarOn && guidanceProfile !== 'tactile_only' ? 'ON' : 'OFF'}
        </button>

        <button
          type="button"
          disabled={guidanceProfile === 'tactile_only'}
          aria-pressed={voiceOn && guidanceProfile !== 'tactile_only'}
          onClick={() => {
            audioRef.current?.unlock();
            setVoiceOn((v) => !v);
          }}
          className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-semibold ${
            voiceOn && guidanceProfile !== 'tactile_only'
              ? 'border-emerald-400 bg-emerald-500/20 text-emerald-200'
              : 'border-slate-700 bg-slate-900 text-slate-400'
          } disabled:opacity-50`}
        >
          Voice: {voiceOn && guidanceProfile !== 'tactile_only' ? `ON (${voiceLang.toUpperCase()})` : 'OFF'}
        </button>

        <button
          type="button"
          onClick={() => {
            audioRef.current?.unlock();
            hapticsRef.current?.triggerTapFeedback();
            setVoiceLang((l) => (l === 'ru' ? 'en' : 'ru'));
          }}
          className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-3 py-3 text-sm font-semibold text-slate-100 hover:bg-slate-700"
        >
          <Smartphone className="h-4 w-4" aria-hidden="true" />
          Lang: {voiceLang === 'ru' ? 'Русский (1 слово)' : 'English (1 word)'}
        </button>
      </div>

      {/* Live Camera Preview (Aspect-Ratio Faithful!) & Radar */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
          <video
            ref={videoRef}
            playsInline
            muted
            className={`w-full bg-black object-contain ${
              videoAspect === 'portrait' ? 'aspect-[3/4] max-h-72' : 'aspect-[4/3]'
            }`}
          />
          <canvas ref={canvasRef} className="hidden" />
          <div className="flex items-center justify-between px-3 py-2 text-xs text-slate-400">
            <span className="truncate">Pulse: {lastPulse?.patternLabel ?? 'Idle'}</span>
            <span>RTT: {rttMs} ms</span>
          </div>
        </div>
        <BirdEyeRadar guidance={guidance} />
      </div>

      {/* Collapsible Developer Hallway Dataset Recorder (prevents accidental palm taps during walking) */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
        <button
          type="button"
          aria-expanded={devOpen}
          onClick={() => setDevOpen((v) => !v)}
          className="flex w-full items-center justify-between text-left text-sm font-bold text-slate-100"
        >
          <span className="flex items-center gap-1.5">
            <Save className="h-4 w-4 text-sky-400" aria-hidden="true" />
            Developer Dataset Capture
          </span>
          {devOpen ? (
            <ChevronUp className="h-4 w-4 text-slate-400" aria-hidden="true" />
          ) : (
            <ChevronDown className="h-4 w-4 text-slate-400" aria-hidden="true" />
          )}
        </button>

        {captureNotice && (
          <p role="status" className="mt-1.5 text-xs font-medium text-emerald-300">
            {captureNotice}
          </p>
        )}

        {devOpen && (
          <div className="mt-2.5">
            <p className="mb-2.5 text-xs text-slate-400">
              Tap a ground-truth label while streaming to save the current hallway
              frame into the Benchmark Lab dataset.
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
          </div>
        )}
      </section>
    </div>
  );
}
