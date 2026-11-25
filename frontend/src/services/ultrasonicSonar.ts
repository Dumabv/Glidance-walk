import type {
  UltrasonicReading,
  UltrasonicSurface,
  UltrasonicZone,
} from '../types/telemetry';

interface WasmSonarExports {
  memory: WebAssembly.Memory;
  glidance_sonar_init: (sampleRateHz: number) => void;
  glidance_sonar_reset_cal: () => void;
  glidance_sonar_tx_ptr: () => number;
  glidance_sonar_mic_ptr: () => number;
  glidance_sonar_out_ptr: () => number;
  glidance_sonar_process: () => number;
}

const SONAR_CHIRP_LEN = 256;
const SONAR_PERIOD = 2048;
const SONAR_FRAME_LEN = 4096;
const SONAR_BINS = 64;

/**
 * Inline AudioWorklet processor:
 * - Emits our exact Rust-synthesized Hann-windowed 17.5–21.0 kHz FMCW chirp every 2048 samples (~42.7 ms).
 * - Captures raw 4096-sample microphone frames and posts them back every ~85 ms (`~11.7 Hz`) with zero UI jitter.
 */
const WORKLET_CODE = `
class GlidanceBatSonarProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chirp = new Float32Array(${SONAR_CHIRP_LEN});
    this.gain = 0.22;
    this.txPhase = 0;
    this.rxRing = new Float32Array(${SONAR_FRAME_LEN});
    this.rxPos = 0;
    this.rxCounter = 0;
    this.ready = false;

    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'chirp' && e.data.chirp) {
        this.chirp.set(e.data.chirp);
        this.ready = true;
      } else if (e.data && e.data.type === 'gain') {
        this.gain = e.data.gain;
      }
    };
  }

  process(inputs, outputs) {
    const outChan = outputs[0] && outputs[0][0];
    const inChan = inputs[0] && inputs[0][0];

    if (outChan) {
      for (let i = 0; i < outChan.length; i++) {
        if (this.ready && this.txPhase < ${SONAR_CHIRP_LEN}) {
          outChan[i] = this.chirp[this.txPhase] * this.gain;
        } else {
          outChan[i] = 0;
        }
        this.txPhase = (this.txPhase + 1) % ${SONAR_PERIOD};
      }
    }

    if (inChan && this.ready) {
      for (let i = 0; i < inChan.length; i++) {
        this.rxRing[this.rxPos] = inChan[i];
        this.rxPos = (this.rxPos + 1) % ${SONAR_FRAME_LEN};
      }
      this.rxCounter += inChan.length;
      if (this.rxCounter >= ${SONAR_FRAME_LEN}) {
        this.rxCounter = 0;
        const ordered = new Float32Array(${SONAR_FRAME_LEN});
        const tail = ${SONAR_FRAME_LEN} - this.rxPos;
        ordered.set(this.rxRing.subarray(this.rxPos), 0);
        ordered.set(this.rxRing.subarray(0, this.rxPos), tail);
        this.port.postMessage({ type: 'rx', samples: ordered }, [ordered.buffer]);
      }
    }
    return true;
  }
}
registerProcessor('glidance-bat-sonar', GlidanceBatSonarProcessor);
`;

/**
 * Active 17.5–21.0 kHz Phase-Aligned FMCW Ultrasonic Bat-Sonar Driver.
 *
 * Uses raw studio mic capture (`echoCancellation: false, noiseSuppression: false, autoGainControl: false`)
 * + AudioWorklet + `glidance_core.wasm` SIMD128 Complex Matched Filter to detect transparent GLASS doors,
 * mirrors, and flat walls even in 0-lux pitch darkness (`0.26m – 1.86m` range, `2.5cm` bin resolution).
 */
export class UltrasonicBatSonar {
  private ctx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private wasm: WasmSonarExports | null = null;
  private workletUrl: string | null = null;
  private active = false;
  private onReading?: (reading: UltrasonicReading) => void;
  private distHistory: { t: number; d: number }[] = [];

  constructor(onReading?: (reading: UltrasonicReading) => void) {
    this.onReading = onReading;
  }

  public isActive(): boolean {
    return this.active;
  }

  public recalibrate(): void {
    this.distHistory = [];
    this.wasm?.glidance_sonar_reset_cal();
  }

  public async start(): Promise<void> {
    if (this.active) return;

    // 1. Load Rust SIMD128 WASM DSP engine
    if (!this.wasm) {
      const resp = await fetch('/glidance_core.wasm');
      if (!resp.ok) throw new Error('Could not load glidance_core.wasm');
      const bytes = await resp.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(bytes);
      const exp = instance.exports as unknown as WasmSonarExports;
      if (typeof exp.glidance_sonar_process !== 'function') {
        throw new Error('glidance_core.wasm missing ultrasonic sonar exports');
      }
      this.wasm = exp;
    }

    // 2. Request RAW unprocessed microphone audio (CRITICAL: disable echoCancellation & noiseSuppression!)
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: { ideal: 48000 },
      },
      video: false,
    });

    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    this.ctx = new AudioCtx({ sampleRate: 48000, latencyHint: 'interactive' });
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }

    const fs = this.ctx.sampleRate;
    this.wasm.glidance_sonar_init(fs);

    // Extract bit-exact chirp generated by Rust WASM
    const txPtr = this.wasm.glidance_sonar_tx_ptr();
    const txChirp = new Float32Array(
      this.wasm.memory.buffer,
      txPtr,
      SONAR_CHIRP_LEN
    ).slice();

    // 3. Spin up inline AudioWorklet (zero main-thread audio dropouts)
    const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
    this.workletUrl = URL.createObjectURL(blob);
    await this.ctx.audioWorklet.addModule(this.workletUrl);

    this.workletNode = new AudioWorkletNode(this.ctx, 'glidance-bat-sonar', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });

    this.sourceNode = this.ctx.createMediaStreamSource(this.micStream);
    this.sourceNode.connect(this.workletNode);
    this.workletNode.connect(this.ctx.destination);

    this.workletNode.port.onmessage = (evt) => {
      if (!this.active || !this.wasm || evt.data?.type !== 'rx') return;
      const samples = evt.data.samples as Float32Array;
      if (!samples || samples.length !== SONAR_FRAME_LEN) return;

      const micPtr = this.wasm.glidance_sonar_mic_ptr();
      new Float32Array(this.wasm.memory.buffer, micPtr, SONAR_FRAME_LEN).set(
        samples
      );

      this.wasm.glidance_sonar_process();

      const outPtr = this.wasm.glidance_sonar_out_ptr();
      const u8 = new Uint8Array(this.wasm.memory.buffer, outPtr, 4);
      const f32 = new Float32Array(
        this.wasm.memory.buffer,
        outPtr + 4,
        4 + SONAR_BINS
      );

      const profile64: number[] = new Array(SONAR_BINS);
      let peakIdx = 0;
      let peakVal = 0;
      let tailSum = 0;
      let tailCount = 0;
      for (let i = 0; i < SONAR_BINS; i++) {
        const v = f32[4 + i];
        profile64[i] = v;
        if (v > peakVal) {
          peakVal = v;
          peakIdx = i;
        }
        if (i >= 32) {
          tailSum += v;
          tailCount += 1;
        }
      }

      const detected = u8[0] === 1;
      const signalOk = u8[1] === 1;
      const calibrating = u8[2] === 1;
      const distanceM = Math.round(f32[0] * 100) / 100;
      const snr = Math.round(f32[1] * 10) / 10;
      const confidence = Math.round(f32[2] * 100) / 100;
      const directLevel = f32[3];

      // Smart Acoustic Feature Extraction:
      // 1. Peak FWHM Width (bins above 55% of peak):
      //    Sharp narrow spike (<= 4 bins = <=10cm) => Specular FLAT WALL or GLASS/MIRROR!
      //    Wide/multi-lobe echo (>= 5 bins)       => Diffuse FURNITURE / CORNER!
      let widthBins = 1;
      if (peakVal > 0.15) {
        const halfThresh = peakVal * 0.55;
        for (let l = peakIdx - 1; l >= 0 && profile64[l] >= halfThresh; l--) widthBins++;
        for (let r = peakIdx + 1; r < SONAR_BINS && profile64[r] >= halfThresh; r++) widthBins++;
      }
      const peakWidthCm = Math.round(widthBins * 2.5);

      // 2. Room Reverberation Envelope (far tail energy 1.06m..1.86m)
      const tailMean = tailCount > 0 ? tailSum / tailCount : 0;
      const roomReverb: 'open' | 'room' | 'enclosed' =
        tailMean > 0.36 ? 'enclosed' : tailMean > 0.17 ? 'room' : 'open';

      // 3. Approach Velocity (m/s, negative = closing in toward obstacle!)
      const now = performance.now();
      let approachMps = 0;
      if (detected) {
        this.distHistory.push({ t: now, d: distanceM });
        if (this.distHistory.length > 6) this.distHistory.shift();
        if (this.distHistory.length >= 3) {
          const oldest = this.distHistory[0];
          const dt = (now - oldest.t) / 1000;
          if (dt > 0.12) {
            approachMps = Math.round(((distanceM - oldest.d) / dt) * 10) / 10;
          }
        }
      } else if (this.distHistory.length > 0 && now - this.distHistory[this.distHistory.length - 1].t > 600) {
        this.distHistory = [];
      }

      // 4. Meaningful Acoustic Surface & Zone Classification
      let surface: UltrasonicSurface = 'open_space';
      let surfaceLabel = 'Open Space';
      if (detected) {
        if (widthBins <= 4 && snr >= 5.0) {
          surface = 'flat_wall_or_glass';
          surfaceLabel = 'Flat Wall / Glass';
        } else if (roomReverb === 'enclosed' && distanceM >= 1.15) {
          surface = 'narrow_hallway';
          surfaceLabel = 'Narrow Hallway';
        } else {
          surface = 'furniture_or_corner';
          surfaceLabel = 'Furniture / Corner';
        }
      } else if (roomReverb === 'enclosed') {
        surface = 'narrow_hallway';
        surfaceLabel = 'Enclosed Space';
      }

      const zone: UltrasonicZone = !detected
        ? 'clear'
        : distanceM <= 0.75
          ? 'danger'
          : distanceM <= 1.3
            ? 'approach'
            : 'far';

      this.onReading?.({
        active: true,
        detected,
        signalOk,
        calibrating,
        distanceM,
        snr,
        confidence,
        directLevel,
        sampleRate: fs,
        profile64,
        surface,
        surfaceLabel,
        zone,
        approachMps,
        roomReverb,
        peakWidthCm,
      });
    };

    this.workletNode.port.postMessage({ type: 'chirp', chirp: txChirp });
    this.active = true;
  }

  public stop(): void {
    this.active = false;
    try {
      this.sourceNode?.disconnect();
    } catch {
      // ignore
    }
    try {
      this.workletNode?.disconnect();
    } catch {
      // ignore
    }
    this.sourceNode = null;
    this.workletNode = null;

    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }
    if (this.ctx) {
      void this.ctx.close().catch(() => {});
      this.ctx = null;
    }
    if (this.workletUrl) {
      URL.revokeObjectURL(this.workletUrl);
      this.workletUrl = null;
    }
  }
}
