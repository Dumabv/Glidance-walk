# Teammate Handoff Guide (v2 Core & Windows Presentation Setup)

Welcome! The full v2 core (FastAPI backend + Metric Depth FP16 + YOLOv8n + Rust SIMD/WASM safety kernel + Lean 4 proofs + Smart Calm Haptics/3D Audio + Rear Camera Switcher) is already merged and running on `main`.

The **midterm presentation server & projector Mission Control** will run on the **Windows laptop (CUDA GPU)**.

---

## 1. Sync `main` & Create Your Feature Branch
```powershell
git checkout main
git pull origin main
git checkout -b feat/hallway-dataset
```

---

## 2. Active Priorities Before the Presentation

### Task A — Capture 25–30 Real Hallway & Stair Photos (Issue #6, P0)
Our **Benchmark Lab** already runs 15 synthetic calibration scenes, but adding **real university hallway & staircase photos** makes our midterm evaluation undeniable!
1. Start the backend + HTTPS frontend on the **Windows laptop (CUDA server)** and connect your phone over the same Wi-Fi or **Personal Hotspot** (ensure Windows Defender Firewall allows inbound connections on Vite port `:5173`).
2. Scan the QR code on your phone → tap **Start Walking Stream** (the rear camera is selected automatically).
3. Expand **Developer Dataset Capture** at the bottom of `MobileWalker` and record at least **25 labeled scenes**:
   - **6× `Clear Floor`** (empty hallways, varied lighting & natural downward tilt)
   - **5× `Obs Left`** (chair, person, wall corner on the left ~1–2m away)
   - **6× `Obs Center`** (chair, doorway edge, pillar, or close wall directly ahead)
   - **5× `Obs Right`** (desk, backpack, person on the right)
   - **5× `Curb / Step`** (standing ~1–1.5m before stairs going down or an outdoor sidewalk curb)
4. Captures are saved inside `backend/data/samples/` (`metadata.json` + `real-*.jpg`). Commit that folder via PR so both laptops share the exact same benchmark dataset.

### Task B — Stage Demo Choreography & Backup Video (Issue #8, P2)
1. Verify **Personal Hotspot** connection between the **Windows laptop (CUDA server)** and phone (`< 15 ms` network RTT).
2. Plan a clean **60-second live obstacle walk** for the classroom presentation (e.g., clear walk → obstacle left → chair center → step/curb edge).
3. Record a **30-second split-screen backup video** (Projector `Mission Control` + phone walker) in case of venue Wi-Fi or projector issues.
4. Capture the final `BenchmarkLab` screenshot (Naive vs. Glidance F1, Curb Recall, False-Alarm Rate, P50/P95 Latency) on real scenes for the slides.

---

## 3. Files Ownership (Zero Merge Conflicts)
- **Windows Laptop / Field Dataset & Demo Sphere:**
  - `backend/data/samples/*`
  - Field tuning (only if needed during rehearsal): `frontend/src/services/haptics.ts`, `frontend/src/services/spatialAudio.ts`
- **Dashboard & Benchmark Polish Sphere (Issue #4):**
  - `frontend/src/pages/MissionControl.tsx`
  - `frontend/src/pages/BenchmarkLab.tsx`

---

## 4. Before Pushing Your Work
Run both verification checks so `main` stays green:
```powershell
backend\.venv\Scripts\python -m pytest -v backend/tests/test_hazard_and_api.py
cd frontend && npm run build
```
Then push your branch and open a Pull Request:
```powershell
git add .
git commit -m "feat(dataset): add real university hallway and stair benchmark samples"
git push -u origin feat/hallway-dataset
```
