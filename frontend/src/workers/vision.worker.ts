import type {
  CalibrationConfig,
  GuidancePayload,
  Severity,
  SteeringAction,
  ZoneMetrics,
} from '../types/telemetry';

/**
 * Dedicated On-Phone Standalone Vision + Rust WebAssembly Safety Worker.
 *
 * Runs completely off the main UI thread so touch gestures, 3D WebAudio sonar,
 * and Android/iOS haptics stay at 60 FPS with zero jank:
 * 1. Loads `glidance_core.wasm` (15 KB `#![no_std]` Rust binary compiled from the exact
 *    Lean-4 verified `GlidanceSafety` kernel).
 * 2. Dynamically loads `@huggingface/transformers` (`onnx-community/depth-anything-v2-small`)
 *    via WebGPU FP16 (Android Chrome / iOS Safari WebGPU) or WASM quantized INT8 fallback.
 * 3. Processes transferred `ImageBitmap` frames with zero JPEG encode/decode overhead!
 */

interface WasmExports {
  memory: WebAssembly.Memory;
  glidance_wasm_input_ptr: () => number;
  glidance_wasm_cfg_ptr: () => number;
  glidance_wasm_out_ptr: () => number;
  glidance_wasm_reset_state: () => void;
  glidance_wasm_analyze: (
    h: number,
    w: number,
    rawSpreadMilli: number,
    lumaMean: number,
    lumaStd: number,
    pitchDeg: number,
    stateful: number,
    frameFresh: number
  ) => number;
  glidance_wasm_normalize_and_analyze: (
    h: number,
    w: number,
    lumaMean: number,
    lumaStd: number,
    pitchDeg: number,
    stateful: number,
    frameFresh: number
  ) => number;
}

const SEVERITY_BY_CODE: Severity[] = ['clear', 'caution', 'danger'];
const STEERING_BY_CODE: SteeringAction[] = [
  'CLEAR',
  'VEER_LEFT',
  'VEER_RIGHT',
  'STOP',
  'CURB_DROP',
];
const CUE_BY_CODE: Record<number, string> = {
  0: 'Path clear.',
  1: 'Curb or step down ahead. Slow down.',
  2: 'Stairs going down ahead. Stop.',
  3: 'Obstacle ahead. Step left.',
  4: 'Obstacle ahead. Step right.',
  5: 'Path blocked ahead. Stop.',
  6: 'Object ahead. Slight left.',
  7: 'Object ahead. Slight right.',
  8: 'Objects on both sides. Stay centered.',
  9: 'Close obstacle on left. Move right.',
  10: 'Close obstacle on right. Move left.',
  11: 'Narrow gap. Stop and check.',
  12: 'Obstacle on left.',
  13: 'Obstacle on right.',
  14: 'Low overhead obstacle ahead. Stop.',
  15: 'Camera blocked or too dark. Stop.',
  16: 'Signal lost. Stop and wait.',
};

let wasmExports: WasmExports | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let depthPipeline: any = null;
let modelLoadingPromise: Promise<void> | null = null;
let activeDeviceLabel = 'phone-wasm';
let frameCounter = 0;

// EMA scale-lock anchors matching `backend/vision/depth_engine.py`
let emaP2: number | null = null;
let emaP98: number | null = null;
let emaSpread: number | null = null;

let offscreen: OffscreenCanvas | null = null;
let offCtx: OffscreenCanvasRenderingContext2D | null = null;
// Tiny 16x16 scratch surface avoids full-frame GPU->CPU getImageData pipeline stalls!
let lumaCanvas: OffscreenCanvas | null = null;
let lumaCtx: OffscreenCanvasRenderingContext2D | null = null;

async function ensureWasm(): Promise<WasmExports> {
  if (wasmExports) return wasmExports;
  const resp = await fetch('/glidance_core.wasm');
  let instance: WebAssembly.Instance;
  if ('instantiateStreaming' in WebAssembly) {
    try {
      const res = await WebAssembly.instantiateStreaming(resp.clone());
      instance = res.instance;
    } catch {
      const bytes = await resp.arrayBuffer();
      const res = await WebAssembly.instantiate(bytes);
      instance = res.instance;
    }
  } else {
    const bytes = await resp.arrayBuffer();
    const res = await WebAssembly.instantiate(bytes);
    instance = res.instance;
  }
  wasmExports = instance.exports as unknown as WasmExports;
  wasmExports.glidance_wasm_reset_state();
  return wasmExports;
}

async function ensureDepthModel(): Promise<void> {
  if (depthPipeline) return;
  if (modelLoadingPromise) return modelLoadingPromise;

  modelLoadingPromise = (async () => {
    self.postMessage({
      type: 'worker_status',
      status: 'loading',
      detail: 'Loading on-phone Depth-Anything-V2 + Rust WASM kernel...',
    });

    await ensureWasm();

    // Dynamic ESM import from jsDelivr keeps production repo light & works standalone on GitHub Pages/Vercel!
    const hfUrl = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.2/+esm';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const hf: any = await import(/* @vite-ignore */ hfUrl);
    hf.env.allowLocalModels = false;

    const hasWebGPU =
      typeof navigator !== 'undefined' && 'gpu' in navigator;
    // Detect low-RAM phones (e.g. <= 4GB deviceMemory or iOS WebKit) to pick compact quantization
    const devMemGb =
      (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 4;

    if (hasWebGPU) {
      try {
        depthPipeline = await hf.pipeline(
          'depth-estimation',
          'onnx-community/depth-anything-v2-small',
          {
            device: 'webgpu',
            // q4f16 (~18 MB VRAM) on <=4GB mobile devices prevents iOS/Android tab OOM kills; fp16 (~49 MB) on >=6GB
            dtype: devMemGb <= 4 ? 'q4f16' : 'fp16',
          }
        );
        activeDeviceLabel = 'phone-webgpu';
      } catch {
        depthPipeline = null;
      }
    }

    if (!depthPipeline) {
      depthPipeline = await hf.pipeline(
        'depth-estimation',
        'onnx-community/depth-anything-v2-small',
        { device: 'wasm', dtype: 'q8' }
      );
      activeDeviceLabel = 'phone-wasm';
    }

    // CRITICAL MOBILE SPEEDUP + BUFFER SAFETY:
    // Default DPTImageProcessor upscales inputs to 518x518 (1,369 ViT patches = 28x more attention!).
    // Locking to 224x224 (256 ViT patches) speeds up mobile WebGPU/WASM by ~5-7x AND fits strictly
    // inside Rust WASM's 384x384 linear memory with zero overflow!
    const fe = depthPipeline?.processor?.feature_extractor;
    if (fe) {
      fe.size = { height: 224, width: 224 };
      fe.keep_aspect_ratio = false;
      fe.ensure_multiple_of = 14;
    }

    self.postMessage({
      type: 'worker_status',
      status: 'ready',
      device: activeDeviceLabel,
      detail: `Standalone on-phone model ready (${activeDeviceLabel.toUpperCase()})`,
    });
  })();

  try {
    await modelLoadingPromise;
  } catch (err) {
    modelLoadingPromise = null;
    throw err;
  }
}

function writeWasmConfig(wasm: WasmExports, cfg?: CalibrationConfig): void {
  if (!cfg) return;
  const ptr = wasm.glidance_wasm_cfg_ptr();
  const dv = new DataView(wasm.memory.buffer, ptr, 24);
  const dangerM = Math.round(cfg.danger_threshold * 1000);
  const cautionM = Math.min(
    Math.round(cfg.caution_threshold * 1000),
    Math.max(50, dangerM - 40)
  );
  dv.setUint32(0, cautionM, true);
  dv.setUint32(4, dangerM, true);
  dv.setUint32(8, Math.round(cfg.curb_gradient_threshold * 1000), true);
  dv.setUint32(12, Math.round(cfg.floor_margin * 1000), true);
  dv.setUint32(16, Math.round(cfg.ema_alpha * 1000), true);
  dv.setUint32(20, Math.round(cfg.confirm_frames), true);
}

function readZone(dv: DataView, offset: number): ZoneMetrics {
  const sevCode = Math.min(2, dv.getUint8(offset));
  return {
    severity: SEVERITY_BY_CODE[sevCode],
    obstacle_score: Number(dv.getFloat32(offset + 4, true).toFixed(4)),
    min_distance_m: Number(dv.getFloat32(offset + 8, true).toFixed(2)),
    coverage_ratio: Number(dv.getFloat32(offset + 12, true).toFixed(4)),
  };
}

function normalizeDepthWithWallLock(raw: Float32Array): {
  closeness: Float32Array;
  spreadMilli: number;
} {
  // Fast 256-bin min/max + percentile pass
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  const range = Math.max(maxV - minV, 1e-6);
  const hist = new Uint32Array(256);
  for (let i = 0; i < raw.length; i++) {
    const b = Math.min(255, Math.max(0, Math.floor(((raw[i] - minV) / range) * 255)));
    hist[b]++;
  }
  const findQ = (q: number): number => {
    const target = raw.length * q;
    let acc = 0;
    for (let b = 0; b < 256; b++) {
      acc += hist[b];
      if (acc >= target) return minV + (b / 255) * range;
    }
    return maxV;
  };

  let p2 = findQ(0.02);
  let p98 = findQ(0.98);
  const spread = Math.max(0, p98 - p2);
  let spreadMilli = 600;

  if (emaSpread === null || emaP2 === null || emaP98 === null) {
    if (spread > 1e-4) {
      emaP2 = p2;
      emaP98 = p98;
      emaSpread = spread;
    } else {
      spreadMilli = 0;
    }
  } else {
    spreadMilli = Math.min(
      1000,
      Math.max(0, Math.round((spread / Math.max(emaSpread, 1e-4)) * 550))
    );
    if (spreadMilli > 120) {
      emaP2 = 0.85 * emaP2 + 0.15 * p2;
      emaP98 = 0.85 * emaP98 + 0.15 * p98;
      emaSpread = Math.max(1e-4, emaP98 - emaP2);
    } else {
      p2 = emaP2;
      p98 = emaP98;
    }
  }

  const denom = Math.max(p98 - p2, 1e-5);
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const c = (raw[i] - p2) / denom;
    out[i] = c < 0 ? 0 : c > 1 ? 1 : c;
  }
  return { closeness: out, spreadMilli };
}

self.onmessage = async (evt: MessageEvent) => {
  const msg = evt.data;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'init') {
    try {
      await ensureDepthModel();
    } catch (err) {
      self.postMessage({
        type: 'worker_status',
        status: 'error',
        detail:
          err instanceof Error
            ? err.message
            : 'Failed to load on-phone depth model',
      });
    }
    return;
  }

  if (msg.type === 'reset') {
    wasmExports?.glidance_wasm_reset_state();
    emaP2 = null;
    emaP98 = null;
    emaSpread = null;
    return;
  }

  if (msg.type === 'frame') {
    const bitmap = msg.bitmap as ImageBitmap;
    const clientTs = Number(msg.clientTs || Date.now());
    const pitchDeg = Number(msg.pitchDeg || 0);

    try {
      const tTotal = performance.now();
      await ensureDepthModel();
      const wasm = await ensureWasm();
      if (msg.config) {
        writeWasmConfig(wasm, msg.config as CalibrationConfig);
      }

      const tDec = performance.now();
      const w = bitmap.width;
      const h = bitmap.height;
      if (!offscreen || offscreen.width !== w || offscreen.height !== h) {
        offscreen = new OffscreenCanvas(w, h);
        offCtx = offscreen.getContext('2d', { alpha: false });
      }
      if (!lumaCanvas) {
        lumaCanvas = new OffscreenCanvas(16, 16);
        lumaCtx = lumaCanvas.getContext('2d', { willReadFrequently: true });
      }
      offCtx!.drawImage(bitmap, 0, 0, w, h);
      lumaCtx!.drawImage(bitmap, 0, 0, 16, 16);
      const rgba = lumaCtx!.getImageData(0, 0, 16, 16).data;
      bitmap.close();

      // Zero-stall 16x16 luminance mean & std (256 pixels = ~0.01ms)
      let sumY = 0;
      let sumSqY = 0;
      for (let i = 0; i < rgba.length; i += 4) {
        const y = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
        sumY += y;
        sumSqY += y * y;
      }
      const lumaMean = sumY * (1 / 256);
      const lumaStd = Math.sqrt(Math.max(0, sumSqY * (1 / 256) - lumaMean * lumaMean));
      const decodeMs = performance.now() - tDec;

      const tInf = performance.now();
      const result = await depthPipeline(offscreen);
      const rawTensor = result.predicted_depth;
      const rawData: Float32Array =
        rawTensor?.data instanceof Float32Array
          ? rawTensor.data
          : new Float32Array(rawTensor.data);
      const dims: number[] = rawTensor.dims || [h, w];
      const dh = dims[dims.length - 2] || h;
      const dw = dims[dims.length - 1] || w;
      const inferMs = performance.now() - tInf;

      // Fused zero-copy WASM SIMD128 normalization + V-disparity safety analysis!
      // Safety net: if any future model ever outputs > 384px, stride-downsample into WASM buffer!
      const tHaz = performance.now();
      const inPtr = wasm.glidance_wasm_input_ptr();
      let targetH = dh;
      let targetW = dw;
      if (dh > 384 || dw > 384 || dh * dw > 384 * 384) {
        const stride = Math.ceil(Math.max(dh, dw) / 256);
        targetH = Math.floor(dh / stride);
        targetW = Math.floor(dw / stride);
        const dstView = new Float32Array(wasm.memory.buffer, inPtr, targetH * targetW);
        for (let r = 0; r < targetH; r++) {
          const srcRow = r * stride * dw;
          const dstRow = r * targetW;
          for (let c = 0; c < targetW; c++) {
            dstView[dstRow + c] = rawData[srcRow + c * stride];
          }
        }
      } else {
        new Float32Array(wasm.memory.buffer, inPtr, dh * dw).set(
          rawData.subarray(0, dh * dw)
        );
      }
      // Immediately release WebGPU/WASM tensor buffers to prevent mobile Safari/Chrome RAM leaks!
      rawTensor?.dispose?.();
      if (typeof wasm.glidance_wasm_normalize_and_analyze === 'function') {
        wasm.glidance_wasm_normalize_and_analyze(
          targetH,
          targetW,
          lumaMean,
          lumaStd,
          pitchDeg,
          1,
          1
        );
      } else {
        const sub = new Float32Array(wasm.memory.buffer, inPtr, targetH * targetW);
        const { closeness, spreadMilli } = normalizeDepthWithWallLock(sub);
        sub.set(closeness);
        wasm.glidance_wasm_analyze(
          targetH,
          targetW,
          spreadMilli,
          lumaMean,
          lumaStd,
          pitchDeg,
          1,
          1
        );
      }
      const outPtr = wasm.glidance_wasm_out_ptr();
      const dv = new DataView(wasm.memory.buffer, outPtr, 76);

      const overallCode = Math.min(2, dv.getUint8(0));
      const steerCode = Math.min(4, dv.getUint8(1));
      const cueCode = dv.getUint8(2);
      const flags = dv.getUint8(3);

      const left = readZone(dv, 4);
      const center = readZone(dv, 20);
      const right = readZone(dv, 36);

      const curbDetected = dv.getUint8(52) !== 0;
      const stairsLikely = dv.getUint8(53) !== 0;
      const curbConf = Number(dv.getFloat32(56, true).toFixed(4));
      const dropRow = Number(dv.getFloat32(60, true).toFixed(3));
      const gradPeak = Number(dv.getFloat32(64, true).toFixed(4));
      const hazardMs = performance.now() - tHaz;
      const totalMs = performance.now() - tTotal;

      frameCounter += 1;
      const guidance: GuidancePayload = {
        frame_id: frameCounter,
        client_ts: clientTs,
        server_ts: Date.now(),
        overall_severity: SEVERITY_BY_CODE[overallCode],
        steering: STEERING_BY_CODE[steerCode],
        spoken_cue: CUE_BY_CODE[cueCode] ?? 'Path clear.',
        left,
        center,
        right,
        curb: {
          detected: curbDetected,
          stairs_likely: stairsLikely,
          confidence: curbConf,
          drop_row_ratio: dropRow,
          gradient_peak: gradPeak,
        },
        latency: {
          decode_ms: Number(decodeMs.toFixed(2)),
          inference_ms: Number(inferMs.toFixed(2)),
          hazard_ms: Number(hazardMs.toFixed(3)),
          server_total_ms: Number(totalMs.toFixed(2)),
        },
        device: activeDeviceLabel,
        model_name: 'onnx-community/depth-anything-v2-small',
        kernel_backend: 'rust-wasm',
        wall_override: (flags & 0x01) !== 0,
        overhead_hazard: (flags & 0x02) !== 0,
        lens_blocked: (flags & 0x04) !== 0,
        pitch_deg: pitchDeg !== 0 ? Number(pitchDeg.toFixed(1)) : null,
      };

      self.postMessage({ type: 'guidance', guidance });
    } catch (err) {
      try {
        bitmap.close();
      } catch {
        // Ignore
      }
      self.postMessage({
        type: 'worker_error',
        detail: err instanceof Error ? err.message : 'Worker inference failed',
      });
    }
  }
};
