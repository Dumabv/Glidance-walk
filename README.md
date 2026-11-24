# Glidance — Phone-Camera Obstacle & Curb Detector with Haptics

A 48-hour hackathon prototype that streams a smartphone camera feed to a real-time monocular depth backend (`Depth-Anything-V2-Small`), subtracts the expected flat-floor slope to detect **Left / Center / Right obstacles AND downward curbs/stairs**, delivers **Android native multi-pattern haptics + 3D stereo spatial sonar**, and projects a live **Mission Control & Midterm Benchmark Dashboard**.

> **Disclaimer:** Hallway research prototype only — not a certified medical or real-world mobility aid.

## Architecture

```text
Phone Browser (/mobile) --[wss:// JPEG + client_ts]--> Vite HTTPS Proxy (:5173)
  --> FastAPI + Depth-Anything-V2-Small (M4 MPS / CUDA / CPU)
  --> Ground-Plane & Curb Gradient Analyzer + 3-Frame Debounce
  --> Tiny Guidance JSON back to Phone (Haptics + 3D Spatial Audio)
  --> Rich Telemetry + Neon Depth Map to Classroom Projector (/dashboard)
```

## Quick Start (MacBook Pro M4 / Windows)

### 1. Start the Backend (Terminal 1)
```bash
# macOS / Linux
backend/.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Windows (after creating venv & pip install -r backend/requirements.txt)
backend\.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start the HTTPS Frontend & Proxy (Terminal 2)
```bash
cd frontend
npm run dev
```

### 3. Open Projector & Connect Phone
1. Open **`https://localhost:5173`** on your laptop (accept the local dev certificate once).
2. Connect your **Android or iPhone** to the same Wi-Fi (or your **Personal Hotspot** on stage for `< 8 ms` LAN latency).
3. Scan the QR code shown in **Mission Control** → accept the local certificate once → tap **Start Walking Stream**!

## Running Automated Tests
```bash
backend/.venv/bin/python -m pytest -v backend/tests/test_hazard_and_api.py
cd frontend && npm run build
```
