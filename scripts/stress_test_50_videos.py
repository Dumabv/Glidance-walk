"""50-Video Full-Pipeline Stress Test for Glidance (RTX 3050 Small-FP16 + Rust SIMD + Dual-Head ONNX + Wall/Door Shield)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
from typing import Any

import cv2
import numpy as np

from backend.evaluation.video_diagnostics import diagnose_video_run
from backend.schemas import CalibrationConfig
from backend.vision.depth_engine import DepthEngine
from backend.vision.hazard_detector import (
    GroundAwareHazardDetector,
    detect_close_flat_wall_or_door,
)
from backend.vision.object_detector import SemanticObjectDetector

HDR = {"User-Agent": "GlidanceWalkStress50/1.0 (https://github.com/Dumabv/Glidance-walk)"}

SEARCH_TOPICS = [
    "walking street",
    "walking indoor",
    "walking stairs",
    "walking bridge",
    "walking market",
    "walking subway",
    "walking campus",
    "walking mall",
    "walking corridor",
    "walking sidewalk",
    "walking park",
    "walking station",
    "walking city",
    "walking room",
    "walking museum",
    "walking alley",
    "walking crowd",
    "walking plaza",
]

BANNED_WORDS = ("police", "bodycam", "shoot", "gun", "election", "cerebellar", "surgery", "microscope")


def download_wikimedia_videos(target_dir: Path, need_count: int) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".webm", ".mp4", ".ogv", ".mpg"} and p.stat().st_size > 80_000
    )
    if len(existing) >= need_count:
        return existing[:need_count]

    seen_titles: set[str] = set()
    collected: list[tuple[str, str, str]] = []  # (title, url, ext)

    for q in SEARCH_TOPICS:
        if len(existing) + len(collected) >= need_count + 8:
            break
        time.sleep(0.6)
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query&list=search&srnamespace=6"
            "&srlimit=8&format=json&srsearch=" + urllib.parse.quote(q + " filetype:video")
        )
        try:
            res = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=12).read())
            titles = [
                r["title"]
                for r in res.get("query", {}).get("search", [])
                if r["title"] not in seen_titles
                and not any(b in r["title"].lower() for b in BANNED_WORDS)
            ]
            if not titles:
                continue
            time.sleep(0.6)
            u2 = (
                "https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo"
                "&iiprop=url|size&format=json&titles=" + urllib.parse.quote("|".join(titles[:6]))
            )
            pages = json.loads(urllib.request.urlopen(urllib.request.Request(u2, headers=HDR), timeout=12).read()).get("query", {}).get("pages", {})
            for p in pages.values():
                t = p.get("title", "")
                if t in seen_titles:
                    continue
                ii = (p.get("imageinfo") or [{}])[0]
                mb = (ii.get("size") or 0) / 1e6
                dl = ii.get("url", "")
                ext = Path(urllib.parse.urlparse(dl).path).suffix.lower()
                if 0.3 <= mb <= 14.0 and ext in (".webm", ".mp4", ".ogv"):
                    seen_titles.add(t)
                    collected.append((t, dl, ext))
        except Exception:
            continue

    idx = len(existing) + 1
    out_paths = list(existing)
    for title, dl, ext in collected:
        if len(out_paths) >= need_count:
            break
        dst = target_dir / f"wm_{idx:02d}{ext}"
        try:
            time.sleep(0.5)
            with urllib.request.urlopen(urllib.request.Request(dl, headers=HDR), timeout=45) as r:
                data = r.read()
            if len(data) > 80_000:
                dst.write_bytes(data)
                out_paths.append(dst)
                print(f"  [{len(out_paths)}/{need_count}] downloaded {dst.name} ({len(data)/1e6:.1f} MB) <- {title[:60]}")
                idx += 1
        except Exception:
            continue
    return out_paths


def build_extra_real_clips(target_dir: Path, need_more: int) -> list[Path]:
    """Builds short 3-second real-photo walkthrough MP4s from held-out indoor/stairs/street datasets if Wikimedia < 50."""
    if need_more <= 0:
        return []
    pools = [
        ("indoor", sorted(Path(r"D:\glidance_datasets\glidance_indoor_v2\valid\images").glob("sun_*.jpg"))),
        ("stairs", sorted(Path(r"D:\glidance_datasets\stairs_obstacles\valid\images").glob("*.jpg"))),
        ("street", sorted(Path(r"D:\glidance_datasets\non_stairs_test").glob("*.jpg"))),
    ]
    created: list[Path] = []
    w, h, fps = 640, 480, 10
    clip_i = 1
    for p_idx in range(need_more):
        kind, imgs = pools[p_idx % len(pools)]
        if len(imgs) < 3:
            continue
        start = (p_idx * 2) % max(1, len(imgs) - 2)
        chunk = imgs[start : start + 3]
        dst = target_dir / f"realclip_{clip_i:02d}_{kind}.mp4"
        if not dst.exists():
            wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for im_p in chunk:
                im = cv2.imread(str(im_p))
                if im is None:
                    continue
                im = cv2.resize(im, (w + 48, h + 36))
                for k in range(6):
                    dx = int((k / 5.0) * 32)
                    dy = int((k / 5.0) * 24)
                    wr.write(im[dy : dy + h, dx : dx + w])
            wr.release()
        created.append(dst)
        clip_i += 1
    return created


def evaluate_video(
    vp: Path,
    depth: DepthEngine,
    obj_det: SemanticObjectDetector,
    sample_fps: float = 2.0,
    max_frames: int = 16,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return {"video": vp.name, "error": "cannot_open", "frames": 0}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native_fps / sample_fps)))
    cfg = CalibrationConfig()
    hdet = GroundAwareHazardDetector(cfg)

    timeline: list[dict[str, Any]] = []
    sev_counts = {"clear": 0, "caution": 0, "danger": 0}
    steer_counts: dict[str, int] = {}
    obj_counts: dict[str, int] = {}
    lats: list[float] = []
    dir_contradictions = 0

    fi = 0
    sampled = 0
    while sampled < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi % stride != 0:
            fi += 1
            continue
        if bgr.shape[1] > 640:
            nh = int(round(bgr.shape[0] * (640.0 / bgr.shape[1])))
            bgr = cv2.resize(bgr, (640, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t_sec = round(fi / max(native_fps, 1.0), 2)

        t0 = time.perf_counter()
        pred = depth.predict_rgb(rgb)
        haz = hdet.analyze(
            pred.closeness,
            stateful=True,
            raw_spread_milli=pred.raw_spread_milli,
            luma_mean=pred.luma_mean,
            luma_std=pred.luma_std,
        )
        ores = obj_det.detect(rgb, pred.depth_meters)
        # Apply same flat-floor guard as main.py
        flat_floor = haz.center.severity == "clear" and haz.curb.gradient_peak < 0.055
        objs = [
            o for o in ores.objects
            if not (o.label in ("stairs", "step", "curb") and flat_floor and o.confidence < 0.65 and o.distance_m > 1.8)
        ]
        haz = hdet.fuse_objects(haz, objs, stateful=True)
        haz, wobj = detect_close_flat_wall_or_door(rgb, haz)
        if wobj is not None and not any(o.label in ("wall", "door") for o in objs):
            objs.insert(0, wobj)

        ms = (time.perf_counter() - t0) * 1000.0
        lats.append(ms)
        sev_counts[haz.overall_severity] = sev_counts.get(haz.overall_severity, 0) + 1
        steer_counts[haz.steering] = steer_counts.get(haz.steering, 0) + 1
        for o in objs:
            obj_counts[o.label] = obj_counts.get(o.label, 0) + 1

        c_low = haz.spoken_cue.lower()
        if ("on right" in c_low and "step right" in c_low) or ("on left" in c_low and "step left" in c_low):
            dir_contradictions += 1

        timeline.append(
            {
                "time_sec": t_sec,
                "steering": haz.steering,
                "severity": haz.overall_severity,
                "curb_detected": haz.curb.detected,
                "wall_override": haz.wall_override,
                "spoken_cue": haz.spoken_cue,
                "objects": [o.model_dump() for o in objs],
                "total_ms": round(ms, 2),
            }
        )
        sampled += 1
        fi += 1
    cap.release()

    # Contiguous alert segments
    segs: list[dict[str, Any]] = []
    cur = None
    for row in timeline:
        if row["steering"] != "CLEAR":
            if cur is None or cur["steering"] != row["steering"]:
                if cur is not None:
                    segs.append(cur)
                cur = {"steering": row["steering"], "start_sec": row["time_sec"], "end_sec": row["time_sec"], "frames": 1}
            else:
                cur["end_sec"] = row["time_sec"]
                cur["frames"] += 1
        else:
            if cur is not None:
                segs.append(cur)
                cur = None
    if cur is not None:
        segs.append(cur)

    summary = {
        "video": vp.name,
        "frames": sampled,
        "kernel_backend": hdet.kernel_backend,
        "p50_ms": round(float(np.median(lats)) if lats else 0.0, 2),
        "p95_ms": round(float(np.percentile(lats, 95)) if lats else 0.0, 2),
        "severity_counts": sev_counts,
        "steering_counts": steer_counts,
        "object_counts": obj_counts,
        "dir_contradictions": dir_contradictions,
        "alert_segments": segs,
        "timeline": timeline,
    }
    summary["diagnostics"] = diagnose_video_run(summary)
    return summary


def main() -> None:
    stress_dir = Path(r"D:\glidance_videos\stress50")
    stress_dir.mkdir(parents=True, exist_ok=True)

    # Start with our 8 existing videos + IMG_7723.MOV
    base_videos = sorted(
        p for p in Path(r"D:\glidance_videos").iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".ogv", ".mov"}
    )
    cls_vid = Path(r"C:\Users\PC\Downloads\IMG_7723.MOV")
    if cls_vid.exists():
        base_videos.append(cls_vid)

    need_wm = max(0, 50 - len(base_videos))
    print(f"Have {len(base_videos)} base videos; fetching up to {need_wm} Wikimedia videos...")
    wm_videos = download_wikimedia_videos(stress_dir, need_count=min(need_wm, 32))
    still_need = max(0, 50 - len(base_videos) - len(wm_videos))
    extra_clips = build_extra_real_clips(stress_dir, still_need)

    all_videos = (base_videos + wm_videos + extra_clips)[:50]
    print(f"\n=== RUNNING 50-VIDEO STRESS TEST ({len(all_videos)} videos) ===")

    os.environ["GLIDANCE_RUNTIME"] = "onnx"
    os.environ["GLIDANCE_ONNX_TIER"] = "small-fp16"
    depth = DepthEngine(use_mock=False)
    obj_det = SemanticObjectDetector()
    print(f"Engine: {depth.model_id} ({depth.device}) + {obj_det.weights_name} ({obj_det.device})")

    reports: list[dict[str, Any]] = []
    total_frames = 0
    all_p50: list[float] = []
    agg_objs: dict[str, int] = {}
    agg_findings: dict[str, int] = {}
    total_dir_contra = 0
    total_cue_contra = 0

    for idx, vp in enumerate(all_videos, 1):
        rep = evaluate_video(vp, depth, obj_det, sample_fps=2.0, max_frames=16)
        reports.append(rep)
        if rep.get("error"):
            print(f"  [{idx:02d}/50] {vp.name:34s} ERROR: {rep['error']}")
            continue
        total_frames += rep["frames"]
        all_p50.append(rep["p50_ms"])
        total_dir_contra += rep["dir_contradictions"]
        for k, v in rep["object_counts"].items():
            agg_objs[k] = agg_objs.get(k, 0) + v
        codes = []
        for f in rep["diagnostics"].get("findings", []):
            code = f"{f['severity']}:{f['code']}"
            codes.append(code)
            agg_findings[code] = agg_findings.get(code, 0) + 1
            if f["code"] == "CUE_CONTRADICTION":
                total_cue_contra += 1
        hs = rep["diagnostics"].get("health_score", 100)
        print(
            f"  [{idx:02d}/50] {vp.name:34s} | f={rep['frames']:2d} | p50={rep['p50_ms']:5.1f}ms | "
            f"health={hs:3d} | steer={rep['steering_counts']} | objs={list(rep['object_counts'].keys())[:4]}"
        )

    out_json = stress_dir / "report_50.json"
    out_json.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    ok_reps = [r for r in reports if not r.get("error") and r["frames"] > 0]
    avg_health = float(np.mean([r["diagnostics"]["health_score"] for r in ok_reps])) if ok_reps else 0.0
    worst = sorted(ok_reps, key=lambda r: r["diagnostics"]["health_score"])[:5]

    print("\n================ 50-VIDEO STRESS TEST FINAL REPORT ================")
    print(f"Videos evaluated:        {len(ok_reps)} / {len(all_videos)} ({total_frames} total frames)")
    print(f"Median latency (p50):    {np.median(all_p50):.1f} ms (~{1000.0/max(1.0, float(np.median(all_p50))):.1f} FPS)")
    print(f"Average health score:    {avg_health:.1f} / 100")
    print(f"CRITICAL cue mismatches: {total_cue_contra}")
    print(f"Direction contradictions:{total_dir_contra}")
    print(f"Findings across 50 vids: {agg_findings}")
    print(f"Top detected classes:    {dict(sorted(agg_objs.items(), key=lambda kv: -kv[1])[:12])}")
    print("Worst 5 videos:")
    for w in worst:
        fc = [f["code"] for f in w["diagnostics"]["findings"]]
        print(f"  - {w['video']}: health={w['diagnostics']['health_score']}, findings={fc}, steer={w['steering_counts']}")
    print(f"Full report saved to: {out_json}")


if __name__ == "__main__":
    main()
