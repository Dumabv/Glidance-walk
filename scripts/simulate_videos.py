"""Multi-system frame-by-frame full-pipeline video simulator for Glidance.

Runs real videos through multiple system profiles:
  1. rtx_full_v2      : Depth-Anything-V2-Large-FP16 (ONNX CUDA) + Rust SIMD + glidance_indoor_v2.onnx + yolo11n.onnx
  2. edge_fast_v2     : Depth-Anything-V2-Small-Q8  (ONNX)      + Rust SIMD + glidance_indoor_v2.onnx + yolo11n.onnx
  3. depth_only_large : Depth-Anything-V2-Large-FP16 (ONNX CUDA) + Rust SIMD (no YOLO)

For every sampled frame, logs exact steering, severity, curb/wall/overhead flags,
detected objects + metric distances, spoken cue, depth_ms, yolo_ms, total_ms,
runs `diagnose_video_run()`, and renders annotated MP4 + JSON reports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.evaluation.video_diagnostics import diagnose_video_run
from backend.schemas import CalibrationConfig
from backend.vision.depth_engine import DepthEngine
from backend.vision.hazard_detector import (
    GroundAwareHazardDetector,
    fuse_yolo_with_verified_hazard,
)
from backend.vision.object_detector import SemanticObjectDetector


def build_ground_truth_walkthrough_video(out_path: Path) -> Path:
    """Stitches held-out real photos (indoor rooms + stairs + street) with smooth pan/zoom into an MP4."""
    if out_path.exists() and out_path.stat().st_size > 100_000:
        return out_path

    indoor = sorted(Path(r"D:\glidance_datasets\glidance_indoor_v2\test\images").glob("sun_*.jpg"))[:5]
    stairs = sorted(Path(r"D:\glidance_datasets\stairs_obstacles\test\images").glob("*.jpg"))[:5]
    street = sorted(Path(r"D:\glidance_datasets\non_stairs_test").glob("*.jpg"))[:4]
    seq = indoor + stairs + street
    if not seq:
        return out_path

    w, h, fps = 640, 480, 10
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in seq:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (w + 60, h + 45))
        # 10 frames (1.0s) of smooth walking-forward zoom/pan per real photo
        for k in range(10):
            dx = int((k / 9.0) * 40)
            dy = int((k / 9.0) * 30)
            crop = img[dy : dy + h, dx : dx + w]
            writer.write(crop)
    writer.release()
    return out_path


def _draw_overlay(
    frame_bgr: np.ndarray,
    steering: str,
    severity: str,
    spoken: str,
    objects: list[dict[str, Any]],
    total_ms: float,
    curb: bool,
    wall: bool,
) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    color_map = {"clear": (60, 200, 80), "caution": (0, 200, 255), "danger": (40, 50, 240)}
    col = color_map.get(severity, (200, 200, 200))

    for obj in objects:
        bx1, by1, bx2, by2 = obj["bbox_norm"]
        x1, y1, x2, y2 = int(bx1 * w), int(by1 * h), int(bx2 * w), int(by2 * h)
        ocol = color_map.get(obj["severity"], (255, 220, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), ocol, 2)
        tag = f"{obj['label']} {obj['confidence']:.2f} ({obj['distance_m']:.1f}m)"
        cv2.putText(vis, tag, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ocol, 2)

    cv2.rectangle(vis, (0, 0), (w, 58), (18, 18, 18), -1)
    flags = []
    if curb:
        flags.append("CURB/STAIRS")
    if wall:
        flags.append("WALL")
    flag_str = f" [{'+'.join(flags)}]" if flags else ""
    cv2.putText(
        vis,
        f"{steering} ({severity.upper()}){flag_str}  {total_ms:.1f}ms",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        col,
        2,
    )
    cv2.putText(vis, spoken[:72], (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)
    return vis


def run_video_profile(
    video_path: Path,
    profile_name: str,
    depth_engine: DepthEngine,
    obj_detector: SemanticObjectDetector | None,
    sample_fps: float,
    max_frames: int,
    save_annotated: bool,
    out_dir: Path,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native_fps / sample_fps)))
    cfg = CalibrationConfig()
    hazard_det = GroundAwareHazardDetector(cfg)

    writer = None
    ann_path = out_dir / f"{video_path.stem}_{profile_name}.mp4"

    timeline: list[dict[str, Any]] = []
    severity_counts = {"clear": 0, "caution": 0, "danger": 0}
    steering_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    latencies: list[float] = []
    depth_times: list[float] = []
    yolo_times: list[float] = []

    frame_idx = 0
    sampled = 0

    print(f"\n--- [{profile_name}] {video_path.name} (native {native_fps:.1f}fps, stride={stride}) ---")
    while sampled < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        t_sec = round(frame_idx / max(native_fps, 1.0), 2)
        if frame_bgr.shape[1] > 640:
            nh = int(round(frame_bgr.shape[0] * (640.0 / frame_bgr.shape[1])))
            frame_bgr = cv2.resize(frame_bgr, (640, nh), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        depth_res = depth_engine.predict_rgb(frame_rgb)
        t_depth = depth_res.inference_ms

        haz = hazard_det.analyze(
            depth_res.closeness,
            stateful=True,
            raw_spread_milli=depth_res.raw_spread_milli,
            luma_mean=depth_res.luma_mean,
            luma_std=depth_res.luma_std,
        )

        yolo_ms = 0.0
        objs = []
        if obj_detector is not None and obj_detector.is_active:
            det_res = obj_detector.detect(frame_rgb, depth_res.depth_meters)
            yolo_ms = det_res.yolo_ms
            objs = det_res.objects
            haz = hazard_det.fuse_objects(haz, objs, stateful=True)

        total_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(total_ms)
        depth_times.append(t_depth)
        yolo_times.append(yolo_ms)

        severity_counts[haz.overall_severity] = severity_counts.get(haz.overall_severity, 0) + 1
        steering_counts[haz.steering] = steering_counts.get(haz.steering, 0) + 1
        obj_dicts = [o.model_dump() for o in objs]
        for o in objs:
            object_counts[o.label] = object_counts.get(o.label, 0) + 1

        timeline.append(
            {
                "frame": sampled,
                "time_sec": t_sec,
                "steering": haz.steering,
                "severity": haz.overall_severity,
                "curb_detected": haz.curb.detected,
                "stairs_likely": haz.curb.stairs_likely,
                "wall_override": haz.wall_override,
                "overhead_hazard": haz.overhead_hazard,
                "spoken_cue": haz.spoken_cue,
                "objects": obj_dicts,
                "depth_ms": round(t_depth, 2),
                "yolo_ms": round(yolo_ms, 2),
                "total_ms": round(total_ms, 2),
            }
        )

        # Print every frame cleanly so we can monitor frame-by-frame behavior
        obj_str = ", ".join(f"{o.label}({o.confidence:.2f}@{o.distance_m:.1f}m)" for o in objs[:3]) or "-"
        flags = ("C" if haz.curb.detected else "") + ("S" if haz.curb.stairs_likely else "") + ("W" if haz.wall_override else "")
        print(
            f"  t={t_sec:5.2f}s | {haz.steering:10s} | {haz.overall_severity:7s} | "
            f"flg={flags or '-':3s} | {total_ms:5.1f}ms (d={t_depth:4.1f},y={yolo_ms:4.1f}) | "
            f"objs=[{obj_str}] | {haz.spoken_cue}"
        )

        if save_annotated:
            vis = _draw_overlay(
                frame_bgr,
                haz.steering,
                haz.overall_severity,
                haz.spoken_cue,
                obj_dicts,
                total_ms,
                haz.curb.detected,
                haz.wall_override,
            )
            if writer is None:
                vh, vw = vis.shape[:2]
                writer = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (vw, vh))
            writer.write(vis)

        sampled += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Build contiguous alert segments for `diagnose_video_run`
    segments: list[dict[str, Any]] = []
    cur_seg: dict[str, Any] | None = None
    for row in timeline:
        if row["steering"] != "CLEAR":
            if cur_seg is None or cur_seg["steering"] != row["steering"]:
                if cur_seg is not None:
                    segments.append(cur_seg)
                cur_seg = {
                    "steering": row["steering"],
                    "start_sec": row["time_sec"],
                    "end_sec": row["time_sec"],
                    "frames": 1,
                }
            else:
                cur_seg["end_sec"] = row["time_sec"]
                cur_seg["frames"] += 1
        else:
            if cur_seg is not None:
                segments.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        segments.append(cur_seg)

    summary = {
        "video": video_path.name,
        "profile": profile_name,
        "frames": sampled,
        "kernel_backend": hazard_det.kernel_backend,
        "p50_ms": round(float(np.median(latencies)) if latencies else 0.0, 2),
        "p95_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "avg_depth_ms": round(float(np.mean(depth_times)) if depth_times else 0.0, 2),
        "avg_yolo_ms": round(float(np.mean(yolo_times)) if yolo_times else 0.0, 2),
        "severity_counts": severity_counts,
        "steering_counts": steering_counts,
        "object_counts": object_counts,
        "alert_segments": segments,
        "timeline": timeline,
    }
    summary["diagnostics"] = diagnose_video_run(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", default=r"D:\glidance_videos")
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=45)
    parser.add_argument("--profiles", default="rtx_full_v2", help="Comma-separated profiles or 'all'")
    args = parser.parse_args()

    vdir = Path(args.videos_dir)
    out_dir = vdir / "sim_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure ground-truth stitched walkthrough video exists
    build_ground_truth_walkthrough_video(vdir / "04_indoor_stairs_street_mix.mp4")

    video_files = sorted(
        p for p in vdir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".mpg", ".ogv"}
    )
    print(f"Found {len(video_files)} videos: {[p.name for p in video_files]}")

    wanted = (
        {"rtx_full_v2", "edge_fast_v2", "depth_only_large"}
        if args.profiles.strip().lower() == "all"
        else {x.strip() for x in args.profiles.split(",") if x.strip()}
    )

    os.environ["GLIDANCE_RUNTIME"] = "onnx"
    depth_large = None
    depth_small = None
    obj_det = None
    if wanted & {"rtx_full_v2", "depth_only_large"}:
        os.environ["GLIDANCE_ONNX_TIER"] = "large"
        depth_large = DepthEngine(use_mock=False)
        print(f"Loaded Depth Large: {depth_large.model_id} ({depth_large.device})")
    if "edge_fast_v2" in wanted:
        os.environ["GLIDANCE_ONNX_TIER"] = "small"
        depth_small = DepthEngine(use_mock=False)
        print(f"Loaded Depth Small: {depth_small.model_id} ({depth_small.device})")
    if wanted & {"rtx_full_v2", "edge_fast_v2"}:
        obj_det = SemanticObjectDetector()
        print(f"Loaded Object Detector: {obj_det.weights_name} ({obj_det.device})")

    profiles = [
        p for p in [
            ("rtx_full_v2", depth_large, obj_det, True),
            ("edge_fast_v2", depth_small, obj_det, False),
            ("depth_only_large", depth_large, None, False),
        ]
        if p[0] in wanted
    ]

    all_reports: list[dict[str, Any]] = []
    for vp in video_files:
        for pname, deng, odet, save_ann in profiles:
            rep = run_video_profile(
                vp,
                pname,
                deng,
                odet,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                save_annotated=save_ann,
                out_dir=out_dir,
            )
            all_reports.append(rep)

    report_path = out_dir / "simulation_report.json"
    report_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")

    print("\n================ MULTI-SYSTEM SIMULATION SUMMARY ================")
    for rep in all_reports:
        if "error" in rep:
            print("ERROR:", rep["error"])
            continue
        findings = rep["diagnostics"].get("findings", [])
        f_codes = [f"{f['severity']}:{f['code']}" for f in findings] or ["OK"]
        print(
            f"{rep['video']:32s} | {rep['profile']:16s} | frames={rep['frames']:2d} | "
            f"p50={rep['p50_ms']:5.1f}ms p95={rep['p95_ms']:5.1f}ms | "
            f"steer={rep['steering_counts']} | objs={rep['object_counts']} | diag={f_codes}"
        )
    print(f"Full frame-by-frame JSON & annotated MP4s saved in: {out_dir}")


if __name__ == "__main__":
    main()
