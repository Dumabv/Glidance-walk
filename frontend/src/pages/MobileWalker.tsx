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
  SwitchCamera,
  Unlock,
  Vibrate,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { BirdEyeRadar } from '../components/BirdEyeRadar';
import {
  acquireCameraStream,
  getCameraLabel,
  getOppositeFacingMode,
  stopMediaStream,
  switchCameraStream,
  type CameraFacingMode,
} from '../services/camera';
import { HapticsEngine, type HapticPulseEvent } from '../services/haptics';
import {
  SpatialAudioEngine,
  type GuidanceProfile,
  type VoiceLang,
} from '../services/spatialAudio';
import { SpatialMemoryEngine } from '../services/spatialMemory';
import { UltrasonicBatSonar } from '../services/ultrasonicSonar';
import type {
  ExecutionMode,
  GuidancePayload,
  SampleLabel,
  SpatialMemorySnapshot,
  SteeringAction,
  UltrasonicReading,
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
  const sessionIdRef = useRef(0);
  const switchingCameraRef = useRef(false);
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
  const [facingMode, setFacingMode] = useState<CameraFacingMode>('environment');
  const [cameraDeviceLabel, setCameraDeviceLabel] = useState<string | null>(null);
  const [switchingCamera, setSwitchingCamera] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [workerStatus, setWorkerStatus] = useState<string>('idle');
  const [signalLost, setSignalLost] = useState(false);
  const [tiltWarning, setTiltWarning] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [guidance, setGuidance] = useState<GuidancePayload | null>(null);
  const [memorySnap, setMemorySnap] = useState<SpatialMemorySnapshot | null>(null);
  const [batSonarOn, setBatSonarOn] = useState(false);
  const [batReading, setBatReading] = useState<UltrasonicReading | null>(null);
  const [batError, setBatError] = useState<string | null>(null);
  const batReadingRef = useRef<UltrasonicReading | null>(null);
  const [rttMs, setRttMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [videoAspect, setVideoAspect] = useState<'portrait' | 'landscape'>('portrait');

  const [hapticsOn, setHapticsOn] = useState(true);
  const [sonarOn, setSonarOn] = useState(true);
  const [voiceOn, setVoiceOn] = useState(true);
  const [guidanceProfile, setGuidanceProfile] = useState<GuidanceProfile>('smart_calm');
  const [voiceLang, setVoiceLang] = useState<VoiceLang>('en');
  const [eyesFreeLock, setEyesFreeLock] = useState(false);
  const [devOpen, setDevOpen] = useState(false);
  const [lastPulse, setLastPulse] = useState<HapticPulseEvent | null>(null);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const [liveAnnouncement, setLiveAnnouncement] = useState('');
  const [statusAnnouncement, setStatusAnnouncement] = useState('');

  const frameTimestampsRef = useRef<number[]>([]);
  const smoothedRttRef = useRef<number>(250);
  const videoAspectRef = useRef<'portrait' | 'landscape'>('portrait');
  const dispatchFrameRef = useRef<() => void>(() => {});
  const lastAnnouncedSteeringRef = useRef<string | null>(null);
  const execModeRef = useRef<ExecutionMode>(execMode);
  execModeRef.current = execMode;

  const audioRef = useRef<SpatialAudioEngine | null>(null);
  const hapticsRef = useRef<HapticsEngine | null>(null);
  const memoryRef = useRef<SpatialMemoryEngine | null>(null);
  const batSonarRef = useRef<UltrasonicBatSonar | null>(null);

  if (!audioRef.current) {
    audioRef.current = new SpatialAudioEngine();
  }
  if (!hapticsRef.current) {
    hapticsRef.current = new HapticsEngine(
      (evt) => setLastPulse(evt),
      (rank) => audioRef.current?.playIosTactileClick(rank)
    );
  }
  if (!memoryRef.current) {
    memoryRef.current = new SpatialMemoryEngine();
  }
  if (!batSonarRef.current) {
    batSonarRef.current = new UltrasonicBatSonar((r) => {
      batReadingRef.current = r;
      setBatReading(r);
      memoryRef.current?.stampUltrasonicEcho(r);
    });
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

  const handleIncomingGuidance = useCallback((raw: GuidancePayload) => {
    inFlightRef.current = false;
    lastValidFrameAtRef.current = performance.now();
    setSignalLost(false);

    // Fuse narrow 62° frame into 360° rolling Spatial Memory (24 KB RAM, ~0.03ms CPU)
    const memOut = memoryRef.current?.integrateAndFuse(raw);
    let data = memOut ? memOut.fused : raw;

    // Multimodal Sensor Fusion: Camera Depth + 20kHz Ultrasonic Bat-Sonar -> GLASS & WHITE-WALL SHIELD!
    const ultra = batReadingRef.current;
    if (ultra && ultra.active && !ultra.calibrating && ultra.detected && ultra.confidence >= 0.48) {
      const camMissedBarrier =
        !data.lens_blocked &&
        data.center.severity !== 'danger' &&
        data.center.min_distance_m > 1.55 &&
        ultra.distanceM <= 1.4;

      if (camMissedBarrier) {
        const isSpecular = ultra.surface === 'flat_wall_or_glass';
        data = {
          ...data,
          overall_severity: 'danger',
          steering: 'STOP',
          glass_barrier: isSpecular,
          wall_override: true,
          ultrasonic_dist_m: ultra.distanceM,
          center: {
            ...data.center,
            severity: 'danger',
            obstacle_score: Math.max(data.center.obstacle_score, ultra.confidence),
            min_distance_m: ultra.distanceM,
          },
          spoken_cue: isSpecular
            ? `Smooth wall or glass at ${ultra.distanceM.toFixed(1)}m. Stop.`
            : `Acoustic obstacle at ${ultra.distanceM.toFixed(1)}m. Stop.`,
          memory_hint: `20kHz Sonar (${ultra.surfaceLabel}): ${ultra.distanceM.toFixed(2)}m · SNR ${ultra.snr.toFixed(1)}x`,
        };
      } else if (data.lens_blocked && ultra.distanceM <= 1.75) {
        data = {
          ...data,
          ultrasonic_dist_m: ultra.distanceM,
          spoken_cue: `Dark camera — 20kHz sonar ${ultra.surfaceLabel.toLowerCase()} at ${ultra.distanceM.toFixed(1)}m.`,
          memory_hint: `0-Lux Bat-Sonar: ${ultra.surfaceLabel} at ${ultra.distanceM.toFixed(2)}m`,
        };
      } else {
        data = { ...data, ultrasonic_dist_m: ultra.distanceM };
      }
    }

    if (memOut) {
      setMemorySnap(
        data.glass_barrier && data.memory_hint
          ? { ...memOut.snapshot, memoryHint: data.memory_hint }
          : memOut.snapshot
      );
    }
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

  const cancelFrameLoop = useCallback(() => {
    if (rvfcHandleRef.current === null) return;
    const v = videoRef.current as (HTMLVideoElement & {
      cancelVideoFrameCallback?: (h: number) => void;
    }) | null;
    if (v && typeof v.cancelVideoFrameCallback === 'function') {
      v.cancelVideoFrameCallback(rvfcHandleRef.current);
    } else {
      window.cancelAnimationFrame(rvfcHandleRef.current);
    }
    rvfcHandleRef.current = null;
  }, []);

  const startFrameLoop = useCallback(() => {
    cancelFrameLoop();
    // Beat-free camera-synced frame pacing (`requestVideoFrameCallback` up to 30 FPS):
    // Eliminates the 0-69ms idle beat-aliasing lag of `setInterval(..., 70)`!
    const vExt = videoRef.current as (HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
    }) | null;
    const stepLoop = () => {
      if (!sessionActiveRef.current) return;
      if (performance.now() - lastSendAtRef.current >= 33) {
        dispatchFrameRef.current();
      }
      if (vExt && typeof vExt.requestVideoFrameCallback === 'function') {
        rvfcHandleRef.current = vExt.requestVideoFrameCallback(stepLoop);
      } else {
        rvfcHandleRef.current = window.requestAnimationFrame(stepLoop);
      }
    };
    stepLoop();
  }, [cancelFrameLoop]);

  const attachStreamToVideo = useCallback(async (stream: MediaStream) => {
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          throw err;
        }
      }
    }
  }, []);

  const stopSession = useCallback(() => {
    sessionIdRef.current += 1;
    sessionActiveRef.current = false;
    cancelFrameLoop();
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
    stopMediaStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (wakeLockRef.current) {
      void wakeLockRef.current.release().catch(() => {});
      wakeLockRef.current = null;
    }
    inFlightRef.current = false;
    encodingRef.current = false;
    switchingCameraRef.current = false;
    setSwitchingCamera(false);
    setCameraDeviceLabel(null);
    setStreaming(false);
    setWsConnected(false);
    setSignalLost(false);
    setGuidance(null);
    setMemorySnap(null);
  }, [cancelFrameLoop]);

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
      batSonarRef.current?.stop();
      workerRef.current?.terminate();
      workerRef.current = null;
      audioRef.current?.dispose();
    };
  }, [acquireWakeLock, stopSession]);

  // IMU DeviceOrientation + DeviceMotion (Pitch/Roll Tilt Guard + 360° Spatial Memory Heading & Step Odometry)
  useEffect(() => {
    const onOrientation = (e: DeviceOrientationEvent) => {
      const compass = (e as DeviceOrientationEvent & { webkitCompassHeading?: number })
        .webkitCompassHeading;
      memoryRef.current?.updateOrientation(e.alpha, compass ?? null, e.beta);

      if (e.beta === null || e.gamma === null) return;
      pitchDegRef.current = 0.8 * pitchDegRef.current + 0.2 * e.beta;
      rollDegRef.current = 0.8 * rollDegRef.current + 0.2 * e.gamma;

      if (!sessionActiveRef.current) return;
      const p = pitchDegRef.current;
      if (p > 5 && p < 24) {
        setTiltWarning('Camera aimed straight down at shoes — tilt slightly forward.');
      } else if (p > 115 && p < 165) {
        setTiltWarning('Camera aimed at ceiling — level phone slightly down.');
      } else {
        setTiltWarning(null);
      }
    };

    const onMotion = (e: DeviceMotionEvent) => {
      if (!sessionActiveRef.current) return;
      const acc = e.accelerationIncludingGravity;
      const rot = e.rotationRate;
      const dtSec = e.interval ? e.interval / 1000 : 0.016;
      memoryRef.current?.updateMotion(
        acc?.x,
        acc?.y,
        acc?.z,
        rot?.alpha ?? null,
        rot?.beta ?? null,
        rot?.gamma ?? null,
        dtSec
      );
    };

    window.addEventListener('deviceorientation', onOrientation, { passive: true });
    window.addEventListener('devicemotion', onMotion, { passive: true });
    return () => {
      window.removeEventListener('deviceorientation', onOrientation);
      window.removeEventListener('devicemotion', onMotion);
    };
  }, []);

  const dispatchFrame = useCallback(() => {
    if (!sessionActiveRef.current || switchingCameraRef.current) return;
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

    // Request iOS 13+ IMU orientation & motion permissions inside user gesture if available
    const devOrient = DeviceOrientationEvent as unknown as {
      requestPermission?: () => Promise<string>;
    };
    if (typeof devOrient.requestPermission === 'function') {
      void devOrient.requestPermission().catch(() => {});
    }
    const devMotion = DeviceMotionEvent as unknown as {
      requestPermission?: () => Promise<string>;
    };
    if (typeof devMotion.requestPermission === 'function') {
      void devMotion.requestPermission().catch(() => {});
    }

    const activeSessionId = ++sessionIdRef.current;

    try {
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      stopMediaStream(streamRef.current);
      streamRef.current = null;
      const { stream, actualFacingMode, deviceLabel, usedFallback } =
        await acquireCameraStream(facingMode);
      if (sessionIdRef.current !== activeSessionId) {
        stopMediaStream(stream);
        return;
      }
      setFacingMode(actualFacingMode);
      setCameraDeviceLabel(usedFallback ? deviceLabel : null);
      await attachStreamToVideo(stream);
    } catch (err) {
      setErrorMsg(
        `Camera access failed (${err instanceof Error ? err.message : 'permission denied'}). Open over HTTPS or localhost.`
      );
      return;
    }

    await acquireWakeLock();
    audioRef.current?.resetSessionState();
    hapticsRef.current?.resetSessionState();
    memoryRef.current?.reset();
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

    startFrameLoop();

    // Adaptive RTT-Aware Watchdog: avoids false alarms on 4G / Cloudflare tunnels while catching real stalls!
    watchdogRef.current = window.setInterval(() => {
      if (!sessionActiveRef.current || switchingCameraRef.current) return;
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

  const handleSwitchCamera = async () => {
    if (eyesFreeLock || switchingCameraRef.current) return;
    hapticsRef.current?.triggerTapFeedback();
    const targetMode = getOppositeFacingMode(facingMode);

    // Before streaming starts, simply toggle the preferred default camera.
    if (!streaming && !streamRef.current) {
      setFacingMode(targetMode);
      setStatusAnnouncement(
        `Selected ${getCameraLabel(targetMode).toLowerCase()} camera.`
      );
      return;
    }

    const activeSessionId = sessionIdRef.current;
    switchingCameraRef.current = true;
    setSwitchingCamera(true);
    setErrorMsg(null);
    cancelFrameLoop();

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    try {
      const { stream, actualFacingMode, deviceLabel, usedFallback, switchedDevice } =
        await switchCameraStream(streamRef.current, targetMode);

      if (sessionIdRef.current !== activeSessionId) {
        stopMediaStream(stream);
        return;
      }

      await attachStreamToVideo(stream);
      setFacingMode(actualFacingMode);
      setCameraDeviceLabel(usedFallback ? deviceLabel : null);
      inFlightRef.current = false;
      encodingRef.current = false;
      lastValidFrameAtRef.current = performance.now();
      setSignalLost(false);
      startFrameLoop();

      if (!switchedDevice) {
        setStatusAnnouncement('Only one camera is available on this device.');
      } else if (usedFallback && deviceLabel) {
        setStatusAnnouncement(`Switched camera to ${deviceLabel}.`);
      } else {
        setStatusAnnouncement(
          `Switched to ${getCameraLabel(actualFacingMode).toLowerCase()} camera.`
        );
      }
    } catch (err) {
      setErrorMsg(
        `Unable to switch camera (${err instanceof Error ? err.message : 'hardware busy'}). Tap Start Walking Stream to reconnect.`
      );
      stopSession();
    } finally {
      switchingCameraRef.current = false;
      setSwitchingCamera(false);
    }
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
  const cameraLabel = getCameraLabel(facingMode);
  const nextCameraLabel = getCameraLabel(getOppositeFacingMode(facingMode));

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
      <div aria-live="polite" aria-atomic="true" className="visually-hidden">
        {statusAnnouncement}
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
        {!signalLost && guidance?.objects && guidance.objects.length > 0 && (
          <span className="mt-2 flex flex-wrap justify-center gap-1.5">
            {guidance.objects.slice(0, 3).map((obj, idx) => (
              <span
                key={`${obj.label}-${idx}`}
                className="rounded-full bg-slate-950/60 px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide"
              >
                {obj.label} · {obj.distance_m.toFixed(1)}m
              </span>
            ))}
          </span>
        )}
      </button>

      {/* Primary Action Button + Camera Switch + Palm-Safe Eyes-Free Lock */}
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          disabled={eyesFreeLock && streaming}
          onClick={streaming ? stopSession : () => void startSession()}
          className={`flex min-w-[11rem] flex-1 items-center justify-center gap-2 rounded-2xl px-5 py-4 text-base font-bold shadow-lg transition ${
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

        <button
          type="button"
          aria-disabled={eyesFreeLock || switchingCamera || undefined}
          aria-busy={switchingCamera || undefined}
          aria-label={`Camera: ${cameraLabel}, switch to ${nextCameraLabel.toLowerCase()}`}
          onClick={() => void handleSwitchCamera()}
          className={`flex items-center justify-center gap-1.5 rounded-2xl border px-3.5 py-4 text-xs font-bold transition ${
            eyesFreeLock || switchingCamera
              ? 'border-slate-700 bg-slate-900 text-slate-400 opacity-55'
              : 'border-slate-700 bg-slate-800 text-slate-100 hover:border-sky-400 hover:bg-slate-700'
          }`}
        >
          <SwitchCamera className="h-5 w-5" aria-hidden="true" />
          <span>Camera: {cameraLabel}</span>
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

        {/* Active 17.5–21.0 kHz FMCW Ultrasonic Bat-Sonar (Glass Door & Pitch-Darkness Shield!) */}
        <button
          type="button"
          disabled={eyesFreeLock}
          aria-pressed={batSonarOn}
          onClick={() => {
            hapticsRef.current?.triggerTapFeedback();
            setBatError(null);
            if (batSonarOn) {
              batSonarRef.current?.stop();
              batReadingRef.current = null;
              setBatReading(null);
              setBatSonarOn(false);
            } else {
              void batSonarRef.current
                ?.start()
                .then(() => setBatSonarOn(true))
                .catch((err) => {
                  setBatError(
                    err instanceof Error ? err.message : 'Microphone permission required for 20kHz sonar'
                  );
                  setBatSonarOn(false);
                });
            }
          }}
          className={`col-span-2 flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-bold transition ${
            batSonarOn
              ? 'border-fuchsia-400 bg-fuchsia-500/20 text-fuchsia-200'
              : 'border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800'
          } disabled:opacity-50`}
        >
          <Radio className="h-4 w-4" aria-hidden="true" />
          {batSonarOn
            ? batReading?.calibrating
              ? '20kHz Bat-Sonar: Calibrating Chassis...'
              : batReading?.detected
                ? `20kHz Bat-Sonar: ECHO ${batReading.distanceM.toFixed(2)}m (SNR ${batReading.snr}x)`
                : '20kHz Ultrasonic Glass Shield: ACTIVE'
            : 'Enable 20kHz Ultrasonic Bat-Sonar (Glass & Darkness)'}
        </button>
      </div>

      {batError && (
        <p role="alert" className="rounded-lg border border-red-500/50 bg-red-950/60 px-3 py-1.5 text-xs text-red-200">
          20kHz Sonar error: {batError}
        </p>
      )}

      {/* Smart 20kHz FMCW Acoustic Analyzer: 3-Zone Colored A-Scope + Surface / Velocity / Room Classification */}
      {batSonarOn && batReading && (
        <div className="rounded-xl border border-fuchsia-500/40 bg-slate-900/90 p-3">
          <div className="mb-1.5 flex items-center justify-between text-[11px] font-semibold">
            <span className="text-fuchsia-300">
              20kHz Acoustic Room Analyzer ({Math.round(batReading.sampleRate / 1000)}kHz ·{' '}
              {batReading.signalOk ? 'Mic Locked' : 'Low Mic Signal'})
            </span>
            <button
              type="button"
              onClick={() => batSonarRef.current?.recalibrate()}
              className="rounded border border-fuchsia-500/50 bg-fuchsia-950/60 px-2 py-0.5 text-[10px] text-fuchsia-200 hover:bg-fuchsia-900/60"
            >
              Re-Zero
            </button>
          </div>

          {/* 3 Smart Acoustic Telemetry Cards */}
          <div className="mb-2 grid grid-cols-3 gap-1.5 text-center text-[10px] font-bold">
            <div className="rounded-md border border-slate-700 bg-slate-950/80 px-1.5 py-1 text-slate-200">
              <div className="text-[9px] font-normal text-slate-400">Surface Type</div>
              <div className={batReading.detected ? 'text-fuchsia-300' : 'text-emerald-300'}>
                {batReading.calibrating ? 'Calibrating...' : batReading.surfaceLabel}
              </div>
            </div>
            <div className="rounded-md border border-slate-700 bg-slate-950/80 px-1.5 py-1 text-slate-200">
              <div className="text-[9px] font-normal text-slate-400">Echo / Speed</div>
              <div
                className={
                  batReading.zone === 'danger'
                    ? 'text-red-400'
                    : batReading.zone === 'approach'
                      ? 'text-amber-300'
                      : 'text-slate-300'
                }
              >
                {batReading.detected
                  ? `${batReading.distanceM.toFixed(2)}m (${batReading.approachMps < -0.15 ? `${batReading.approachMps}m/s` : 'static'})`
                  : 'Clear >1.8m'}
              </div>
            </div>
            <div className="rounded-md border border-slate-700 bg-slate-950/80 px-1.5 py-1 text-slate-200">
              <div className="text-[9px] font-normal text-slate-400">Acoustics / Width</div>
              <div className="text-cyan-300">
                {batReading.roomReverb.toUpperCase()} · {batReading.peakWidthCm}cm
              </div>
            </div>
          </div>

          {/* 3-Zone Color-Coded A-Scope: Danger (0.26-0.75m) | Approach (0.75-1.30m) | Far/Wall (1.30-1.86m) */}
          <svg viewBox="0 0 320 72" className="h-18 w-full rounded bg-slate-950/90">
            <rect x="0" y="0" width={((0.75 - 0.26) / 1.6) * 320} height="56" fill="rgba(239, 68, 68, 0.14)" />
            <rect
              x={((0.75 - 0.26) / 1.6) * 320}
              y="0"
              width={((1.3 - 0.75) / 1.6) * 320}
              height="56"
              fill="rgba(245, 158, 11, 0.10)"
            />
            <rect
              x={((1.3 - 0.26) / 1.6) * 320}
              y="0"
              width={((1.86 - 1.3) / 1.6) * 320}
              height="56"
              fill="rgba(34, 211, 238, 0.08)"
            />
            <text x="6" y="10" fill="#fca5a5" fontSize="7.5" fontWeight="700">DANGER</text>
            <text x="106" y="10" fill="#fcd34d" fontSize="7.5" fontWeight="700">APPROACH</text>
            <text x="216" y="10" fill="#67e8f9" fontSize="7.5" fontWeight="700">ROOM WALL</text>
            {[0.5, 0.75, 1.0, 1.3, 1.6].map((m) => {
              const x = ((m - 0.26) / 1.6) * 320;
              return (
                <g key={m}>
                  <line x1={x} y1="0" x2={x} y2="56" stroke="#1e293b" strokeDasharray="2 2" />
                  <text x={x + 2} y="67" fill="#64748b" fontSize="8" fontFamily="monospace">
                    {m.toFixed(1)}m
                  </text>
                </g>
              );
            })}
            <polyline
              fill="none"
              stroke={
                batReading.zone === 'danger'
                  ? '#f43f5e'
                  : batReading.zone === 'approach'
                    ? '#f59e0b'
                    : '#e879f9'
              }
              strokeWidth="2.2"
              points={batReading.profile64
                .map((v, i) => `${((i / 63) * 320).toFixed(1)},${(54 - v * 46).toFixed(1)}`)
                .join(' ')}
            />
            {batReading.detected && (
              <circle
                cx={((batReading.distanceM - 0.26) / 1.6) * 320}
                cy="16"
                r="4.5"
                fill={batReading.zone === 'danger' ? '#f43f5e' : '#e879f9'}
              />
            )}
          </svg>
        </div>
      )}

      {/* Live Camera Preview (Aspect-Ratio Faithful!) & Radar */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full bg-black object-contain ${
              videoAspect === 'portrait' ? 'aspect-[3/4] max-h-72' : 'aspect-[4/3]'
            }`}
          />
          <canvas ref={canvasRef} className="hidden" />
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-2 text-xs text-slate-400">
            <span className="truncate">
              Lens: {cameraDeviceLabel ?? cameraLabel}
            </span>
            <span className="truncate">Pulse: {lastPulse?.patternLabel ?? 'Idle'}</span>
            <span>RTT: {rttMs} ms</span>
          </div>
        </div>
        <BirdEyeRadar guidance={guidance} memory={memorySnap} />
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
