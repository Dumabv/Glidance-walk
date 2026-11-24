# Teammate Handoff Guide (Android + Windows Sphere)

Welcome! The full working core (FastAPI backend + Depth-Anything-V2 + Ground/Curb Math + React HUD + Benchmark Lab) is already tested and running on `main`.

To avoid Git merge conflicts while we polish in parallel, here is **your sphere**, **your files**, and **how to push your changes cleanly**.

---

## 1. Create Your Feature Branch First
```bash
git pull origin main
git checkout -b feat/android-haptics-and-dataset
```

---

## 2. Your 3 High-Impact Tasks

### Task A — Capture 20–25 Real Hallway & Stair Photos (Biggest Midterm Boost!)
Our **Benchmark Lab** already runs 15 synthetic calibration scenes, but showing **real university hallway & staircase photos** in the midterm charts will make our evaluation unbeatable!
1. Run the app on your Windows laptop (or connect your Android phone to the M4 MacBook over hotspot).
2. Open the QR code on your **Android Chrome** browser → tap **Start Walking Stream**.
3. Walk through a hallway/corridor/stairs and tap the **Record Labeled Hallway Snapshot** buttons:
   - **5× `Clear Floor`** (hold phone at normal walking angle looking slightly down!)
   - **5× `Obs Left`** (chair, person, wall corner on the left ~1–2m away)
   - **5× `Obs Center`** (chair, doorway, pillar, or close wall directly ahead)
   - **5× `Obs Right`** (desk, person, wall on the right)
   - **5× `Curb / Step`** (standing ~1–1.5m before stairs going down or a sidewalk curb!)
4. Your captures are automatically saved inside `backend/data/samples/` (`metadata.json` + `real-*.jpg`). Commit that folder so both laptops have the exact same benchmark dataset!

### Task B — Tune Android Haptics & 3D Spatial Audio Feel
Walk around holding your Android phone (try closing your eyes with a spotter!) and polish the real-world feel in:
- **`frontend/src/services/haptics.ts`**
  - Adjust vibration durations (`[95, 55, 95, 55, 160]`) and cooldown intervals (`420ms` vs `700ms`) so `CURB`, `STOP`, `VEER_LEFT`, `VEER_RIGHT`, and `CAUTION` feel instantly distinguishable in your hand without buzzing annoyingly.
- **`frontend/src/services/spatialAudio.ts`**
  - Tune sonar frequencies, stereo pan intensity, and voice cue phrasing/speed.
- **`frontend/src/pages/MobileWalker.tsx`**
  - Polish button sizes, readability in bright hallways, or add any extra walking UX controls (e.g., an instant Mute All tap area).

### Task C — Stage Demo Choreography & Backup Video
1. Plan a clean **60-second live obstacle walk** for the classroom presentation (e.g., clear walk → chair left → chair center → step/curb edge).
2. Record a **30-second screen-recording backup video** of Mission Control + Android walking just in case the classroom projector has an issue.

---

## 3. Files You Own vs. Files I Own (Zero Conflicts!)
- **Your Sphere:**
  - `frontend/src/services/haptics.ts`
  - `frontend/src/services/spatialAudio.ts`
  - `frontend/src/pages/MobileWalker.tsx`
  - `backend/data/samples/*`
- **My Sphere (M4 Vision Tuning + Projector Dashboard):**
  - `backend/vision/hazard_detector.py` & `backend/schemas.py`
  - `frontend/src/pages/MissionControl.tsx` & `frontend/src/pages/BenchmarkLab.tsx`

---

## 4. Before Pushing Your Work
Run both quick checks so `main` stays green:
```powershell
backend\.venv\Scripts\python -m pytest -v backend/tests/test_hazard_and_api.py
cd frontend && npm run build
```
Then push your branch and open a Pull Request (or push straight to your branch):
```bash
git add .
git commit -m "feat: tune android haptics and add real hallway benchmark dataset"
git push -u origin feat/android-haptics-and-dataset
```
