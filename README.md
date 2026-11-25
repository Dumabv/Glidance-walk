# Glidance v2.0 — Formally Verified Phone-Camera Obstacle & Curb Navigator

A real-time assistive navigation prototype powered by **`Depth-Anything-V2-Small`**, a **3-band Theil-Sen V-Disparity ground-plane & curb analyzer** written in **Rust** (`cdylib` + `wasm32-unknown-unknown`), **34 machine-checked Lean 4 safety theorems**, **3D binaural HRTF spatial sonar**, and **priority-preempting Android/iOS haptics**.

Supports **two seamless execution modes** on smartphones:
1. **Tethered Low-Latency Mode (`wss://`):** Streams aspect-preserved frames + 16-byte IMU pitch/roll telemetry to the FastAPI backend (`CUDA/MPS FP16` or `ONNX Runtime C++ SIMD` + zero-copy `glidance_core` Rust `.dll`/`.dylib`), simultaneously driving the **Mission Control** & **Benchmark Lab** projector dashboards.
2. **100% Standalone On-Phone Mode (`WebWorker` + `WebGPU/WASM` + `glidance_core.wasm`):** Runs `Depth-Anything-V2-Small` + the exact same 15 KB `#![no_std]` Rust safety kernel directly inside the mobile browser (zero laptop/backend required, automatic hot-failover if Wi-Fi drops!).

> **Disclaimer:** Hallway research prototype only — not a certified medical or real-world mobility aid.

---

## Architecture & Data Flow

```text
[Mode A: Tethered Mission Control]
Phone Browser (/?mode=mobile) --[wss:// 16B (ts+pitch+roll) + Aspect JPEG]--> Vite HTTPS (:5173)
  --> FastAPI + Depth-Anything-V2 (CUDA/MPS FP16 or ONNX Runtime INT8, ViT-14 patch-aligned)
  --> Zero-Copy Rust `glidance_core` (.dll/.dylib/.so, ~0.04 ms/frame)
  --> Lean-4 Verified Safety Kernel (Never-Steer-Into-Danger + Curb/Box Mutual Exclusion)
  --> Priority Haptics + 3D HRTF Binaural Sonar on Phone + Throttled Heatmap to Projector

[Mode B: Standalone On-Phone PWA / Automatic Offline Failover]
Phone Camera --> ImageBitmap Zero-Copy Transfer --> Dedicated WebWorker (`vision.worker.ts`)
  --> Transformers.js WebGPU FP16 / WASM INT8 (`onnx-community/depth-anything-v2-small`)
  --> `glidance_core.wasm` (15 KB Rust WebAssembly safety kernel, ~0.08 ms/frame)
  --> Identical `GuidancePayload` --> Priority Haptics + 3D HRTF Binaural Sonar
```

---

## Formal Mathematical Verification (Lean 4)

See [`formal_verification/README.md`](formal_verification/README.md) for the **34 machine-checked theorems (`149` declarations, `0 sorryAx`, `0` custom axioms)** verified by `SafeVerify` + `leanchecker` in Lean 4 (`v4.35.0`), proving:
- **Fail-Closed Telemetry (`decideGuidance_failClosed`):** Stale/dropped streams or covered camera lenses deterministically trigger `STOP` (`danger`), never leaving a blind walker on `PATH CLEAR`.
- **Strict Severity Dominance (`decideGuidance_severity_exact`):** `overall_severity == max(L, C, R, curb)` across all states.
- **Never Steer Into Danger (`decideGuidance_never_veer_left_danger` / `_right_danger`):** `VEER_LEFT` mathematically guarantees `left != danger`; `VEER_RIGHT` guarantees `right != danger`.
- **Curb-vs-Obstacle Mutual Exclusion (`curb_obstacle_mutual_exclusion`):** Low crates, shin blocks, and shiny linoleum reflections can never trigger false `CURB_DROP` alerts.

---

## Quick Start

### 1. Build Dual-Target Rust Safety Kernel (Optional — prebuilt `.wasm` & automatic verified fallback included)
```bash
cd rust_core
cargo test --release
cargo build --release
cargo build --target wasm32-unknown-unknown --release
```

### 2. Backend & Frontend Setup
```powershell
# Install Python dependencies (ONNX Runtime works out-of-the-box without a 2.5GB PyTorch install!)
pip install -r backend/requirements.txt

# Install Frontend dependencies
cd frontend && npm install && cd ..

# Terminal 1 — Start FastAPI backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Start HTTPS frontend & WebSocket proxy
cd frontend && npm run dev
```

### 3. Automated Verification Suite
```powershell
# Rust kernel unit & counterexample tests
cd rust_core && cargo test --release && cd ..

# Python unit, adversarial edge-case & WebSocket protocol tests
python -m pytest -v backend/tests/test_hazard_and_api.py

# Strict TypeScript check + Vite production bundle
cd frontend && npm run build
```
