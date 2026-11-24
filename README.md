# Glidance — Phone-Camera Obstacle & Curb Detector with Haptics

A 48-hour hackathon prototype that streams a smartphone camera feed to a real-time monocular depth backend (`Depth-Anything-V2-Small`), subtracts the expected flat-floor slope to detect **Left / Center / Right obstacles AND downward curbs/stairs**, delivers **Android native multi-pattern haptics + 3D stereo spatial sonar**, and projects a live **Mission Control & Midterm Benchmark Dashboard**.

> **Disclaimer:** Hallway research prototype only — not a certified medical or real-world mobility aid.

---

## Architecture & Data Flow

```text
Phone Browser (/mobile) --[wss:// 8-byte client_ts + JPEG]--> Vite HTTPS Proxy (:5173)
  --> FastAPI + Depth-Anything-V2-Small (M4 MPS / CUDA / CPU auto-detect)
  --> Ground-Plane & Curb Gradient Analyzer + Multi-Frame Debounce
  --> Tiny Guidance JSON back to Phone (Android Haptics + 3D Spatial Audio)
  --> Rich Telemetry + Neon Depth Heatmap to Projector (/dashboard & /benchmark)
```

---

## Setup & Run

### Requirements
- **Python 3.10 – 3.12**
- **Node.js 20+**

### macOS (Apple Silicon M4 — ~50 ms / ~20 FPS on MPS)
```bash
# 1. Backend setup (once)
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt

# 2. Frontend setup (once)
cd frontend && npm install && cd ..

# 3. Terminal 1 — Start backend
backend/.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 4. Terminal 2 — Start HTTPS frontend & WebSocket proxy
cd frontend && npm run dev
```

### Windows (PowerShell — Auto CPU/CUDA Fallback)
```powershell
# 1. Backend setup (once)
python -m venv backend\.venv
backend\.venv\Scripts\pip install -r backend\requirements.txt

# 2. Frontend setup (once)
cd frontend
npm install
cd ..

# 3. Terminal 1 — Start backend
backend\.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Optional instant UI-dev mode (skips downloading/running the depth model):
# $env:GLIDANCE_MOCK_DEPTH="1"; backend\.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 4. Terminal 2 — Start HTTPS frontend & WebSocket proxy
cd frontend
npm run dev
```

---

## Connecting Your Phone (Android / iPhone)
1. Open **`https://localhost:5173`** on your laptop and click **Advanced → Proceed to localhost (unsafe)** (one-time self-signed local HTTPS cert required for mobile cameras).
2. Connect your phone and laptop to the **same Wi-Fi** (or a **Phone Personal Hotspot** during the live presentation for `< 8 ms` LAN latency).
3. Scan the **QR Code** in **Mission Control** → accept the local certificate once → tap **Start Walking Stream**!
   - **Android Chrome:** Full native multi-pattern `navigator.vibrate()` haptics + 3D spatial sonar.
   - **iPhone Safari:** 3D stereo spatial sonar (`Web Audio API`) + voice cues + iOS switch Taptic fallback.

---

## Teammate Guide & Task Split

See **[TEAMMATE_GUIDE.md](TEAMMATE_GUIDE.md)** for the exact Windows + Android workflow, files you own (`haptics.ts`, `spatialAudio.ts`, `MobileWalker.tsx`, real hallway benchmark dataset), and Git branching steps so we can work in parallel with zero merge conflicts.

---

## Verification & Automated Tests
```bash
# Backend unit + WebSocket integration tests
backend/.venv/bin/python -m pytest -v backend/tests/test_hazard_and_api.py

# Frontend strict TypeScript check + production bundle
cd frontend && npm run build
```
