"""Offline classroom/hallway video analyzer & calibration harness for Glidance v2.

Runs a local MP4/MOV video directly through the production Depth-Anything-V2 Metric +
YOLOv8n + stateful Rust SIMD safety kernel pipeline, printing hazard segments,
object statistics, and P50/P95 latencies. Optionally writes a split-screen annotated
MP4 (raw camera | metric depth + YOLO boxes) and a JSON timeline report.

Usage:
    python scripts/analyze_video.py /path/to/classroom.mp4 --stride 3
    python scripts/analyze_video.py /path/to/classroom.mp4 --save-mp4 out_demo.mp4 --save-json report.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

# Allow running either as `python scripts/analyze_video.py` or `python -m scripts.analyze_video`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.evaluation.video_diagnostics import diagnose_video_run
from backend.schemas import CalibrationConfig, GuidancePayload, LatencyBreakdown
from backend.vision.depth_engine import DepthEngine, render_depth_overlay_jpeg
from backend.vision.hazard_detector import GroundAwareHazardDetector
from backend.vision.object_detector import SemanticObjectDetector


def _format_ts(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:04.1f}"


def _resize_vit14(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(w, h)))
    target_w = max(112, round((w * scale) / 14.0) * 14)
    target_h = max(112, round((h * scale) / 14.0) * 14)
    if target_w == w and target_h == h:
        return frame_bgr
    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def analyze_video(
    video_path: Path,
    *,
    stride: int = 3,
    max_side: int = 322,
    disable_yolo: bool = False,
    mock_depth: bool = False,
    save_mp4: Path | None = None,
    save_json: Path | None = None,
    caution_threshold: float | None = None,
    danger_threshold: float | None = None,
    curb_threshold: float | None = None,
) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_fps = max(1.0, src_fps / max(1, stride))

    cfg_kwargs: dict[str, Any] = {}
    if caution_threshold is not None:
        cfg_kwargs["caution_threshold"] = caution_threshold
    if danger_threshold is not None:
        cfg_kwargs["danger_threshold"] = danger_threshold
    if curb_threshold is not None:
        cfg_kwargs["curb_gradient_threshold"] = curb_threshold
    config = CalibrationConfig(**cfg_kwargs)

    print(f"[Glidance Video Eval] Loading engines (mock={mock_depth}, yolo={not disable_yolo})...")
    depth_engine = DepthEngine(use_mock=mock_depth)
    obj_detector = SemanticObjectDetector(use_mock=(mock_depth or disable_yolo))
    hazard_detector = GroundAwareHazardDetector(config=config)
    print(
        f"[Glidance Video Eval] Device={depth_engine.device.upper()} | "
        f"Model={depth_engine.model_id} | Kernel={hazard_detector.kernel_backend} | "
        f"YOLO={'ON' if obj_detector.is_active else 'OFF'}"
    )
    print(
        f"[Glidance Video Eval] Source: {video_path.name} "
        f"({total_src_frames} frames @ {src_fps:.1f} FPS, stride={stride} -> {out_fps:.1f} FPS)"
    )

    writer: cv2.VideoWriter | None = None
    steering_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    latencies_ms: list[float] = []
    timeline: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    active_seg: dict[str, Any] | None = None
    src_idx = 0
    processed = 0
    t_start_wall = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if src_idx % stride != 0:
                src_idx += 1
                continue

            ts_sec = src_idx / src_fps
            resized_bgr = _resize_vit14(frame_bgr, max_side=max_side)
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
            luma_mean = float(np.mean(gray))
            luma_std = float(np.std(gray))

            t0 = time.perf_counter()
            pred = depth_engine.predict_rgb(rgb, luma_mean, luma_std)
            haz = hazard_detector.analyze(
                pred.closeness,
                stateful=True,
                raw_spread_milli=pred.raw_spread_milli,
                luma_mean=pred.luma_mean,
                luma_std=pred.luma_std,
                frame_fresh=True,
            )

            detected_objects = []
            yolo_ms = 0.0
            if obj_detector.is_active and pred.depth_meters is not None:
                obj_res = obj_detector.detect(rgb, pred.depth_meters)
                haz = hazard_detector.fuse_objects(haz, obj_res.objects, stateful=True)
                detected_objects = obj_res.objects
                yolo_ms = obj_res.yolo_ms
            else:
                haz = hazard_detector.fuse_objects(haz, [], stateful=True)

            total_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(total_ms)
            processed += 1

            steering_counts[haz.steering] += 1
            severity_counts[haz.overall_severity] += 1
            for obj in detected_objects:
                object_counts[obj.label] += 1

            frame_record = {
                "frame": src_idx,
                "time_sec": round(ts_sec, 2),
                "severity": haz.overall_severity,
                "steering": haz.steering,
                "spoken_cue": haz.spoken_cue,
                "min_dist_m": {
                    "left": haz.left.min_distance_m,
                    "center": haz.center.min_distance_m,
                    "right": haz.right.min_distance_m,
                },
                "curb_detected": haz.curb.detected,
                "objects": [
                    {
                        "label": o.label,
                        "distance_m": o.distance_m,
                        "corridor": o.corridor,
                        "severity": o.severity,
                    }
                    for o in detected_objects
                ],
                "total_ms": round(total_ms, 2),
            }
            timeline.append(frame_record)

            # Group consecutive non-CLEAR alerts into readable segments
            if haz.steering != "CLEAR":
                if active_seg is not None and active_seg["steering"] == haz.steering:
                    active_seg["end_sec"] = ts_sec
                    active_seg["frames"] += 1
                else:
                    if active_seg is not None:
                        segments.append(active_seg)
                    active_seg = {
                        "start_sec": ts_sec,
                        "end_sec": ts_sec,
                        "steering": haz.steering,
                        "severity": haz.overall_severity,
                        "cue": haz.spoken_cue,
                        "frames": 1,
                    }
            elif active_seg is not None:
                segments.append(active_seg)
                active_seg = None

            if save_mp4 is not None:
                guidance = GuidancePayload.model_construct(
                    frame_id=processed,
                    client_ts=ts_sec * 1000.0,
                    server_ts=ts_sec * 1000.0,
                    overall_severity=haz.overall_severity,
                    steering=haz.steering,
                    spoken_cue=haz.spoken_cue,
                    left=haz.left,
                    center=haz.center,
                    right=haz.right,
                    curb=haz.curb,
                    objects=detected_objects,
                    latency=LatencyBreakdown.model_construct(
                        decode_ms=0.0,
                        inference_ms=round(pred.inference_ms, 2),
                        yolo_ms=round(yolo_ms, 2),
                        hazard_ms=round(haz.hazard_ms, 3),
                        server_total_ms=round(total_ms, 2),
                    ),
                    device=depth_engine.device,
                    model_name=depth_engine.model_id,
                    kernel_backend=haz.kernel_backend,
                    overhead_hazard=haz.overhead_hazard,
                    wall_override=haz.wall_override,
                    lens_blocked=haz.lens_blocked,
                )
                depth_jpg = render_depth_overlay_jpeg(pred.closeness, guidance, jpeg_quality=90)
                depth_bgr = cv2.imdecode(np.frombuffer(depth_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if depth_bgr is not None:
                    dh, dw = depth_bgr.shape[:2]
                    left_panel = cv2.resize(resized_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
                    combined = np.hstack([left_panel, depth_bgr])
                    banner_h = 34
                    canvas = np.zeros((dh + banner_h, dw * 2, 3), dtype=np.uint8)
                    canvas[banner_h:, :] = combined
                    banner_color = (
                        (60, 60, 220)
                        if haz.overall_severity == "danger"
                        else ((0, 175, 240) if haz.overall_severity == "caution" else (45, 165, 90))
                    )
                    cv2.rectangle(canvas, (0, 0), (dw * 2, banner_h), banner_color, -1)
                    hud_text = f"{_format_ts(ts_sec)} | {haz.steering} | {haz.spoken_cue[:58]}"
                    cv2.putText(
                        canvas,
                        hud_text,
                        (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    if writer is None:
                        save_mp4.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(save_mp4), fourcc, out_fps, (canvas.shape[1], canvas.shape[0])
                        )
                    writer.write(canvas)

            if processed % 50 == 0:
                print(
                    f"  ... processed {processed} frames (video {_format_ts(ts_sec)}) | "
                    f"last={haz.steering} ({total_ms:.1f} ms)"
                )

            src_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if active_seg is not None:
        segments.append(active_seg)

    wall_sec = time.perf_counter() - t_start_wall
    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    summary: dict[str, Any] = {
        "video": str(video_path),
        "processed_frames": processed,
        "stride": stride,
        "wall_seconds": round(wall_sec, 2),
        "device": depth_engine.device,
        "model": depth_engine.model_id,
        "kernel_backend": hazard_detector.kernel_backend,
        "yolo_active": obj_detector.is_active,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "steering_counts": dict(steering_counts),
        "severity_counts": dict(severity_counts),
        "top_objects": dict(object_counts.most_common(10)),
        "alert_segments": segments,
        "timeline": timeline,
    }
    diagnostics = diagnose_video_run(summary)
    summary["diagnostics"] = diagnostics

    print("\n================ GLIDANCE VIDEO REPORT ================")
    print(f"Processed : {processed} frames in {wall_sec:.1f}s (P50={p50:.1f}ms, P95={p95:.1f}ms)")
    print(f"Steering  : {dict(steering_counts)}")
    print(f"Severity  : {dict(severity_counts)}")
    if object_counts:
        print(f"Top YOLO  : {dict(object_counts.most_common(8))}")

    print(f"\n=========== SELF-DIAGNOSTICS (Score: {diagnostics['health_score']}/100) ===========")
    for item in diagnostics["findings"]:
        ts_str = (
            f" | t={', '.join(_format_ts(t) for t in item['timestamps'][:5])}"
            if item["timestamps"]
            else ""
        )
        print(f"  [{item['severity']:<8}] {item['title']} ({item['evidence']}{ts_str})")
        print(f"             -> Fix: {item['fix_hint']}")

    print(f"\nAlert Segments ({len(segments)} total):")
    for seg in segments[:25]:
        print(
            f"  {_format_ts(seg['start_sec'])} -> {_format_ts(seg['end_sec'])} "
            f"[{seg['steering']:<10} | {seg['severity']:<7} | {seg['frames']:2d}f] {seg['cue']}"
        )
    if len(segments) > 25:
        print(f"  ... and {len(segments) - 25} more segments (see --save-json)")

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report -> {save_json}")
    if save_mp4 is not None:
        print(f"Saved annotated split-screen MP4 -> {save_mp4}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a recorded classroom/hallway video with Glidance v2.")
    parser.add_argument("video", type=Path, help="Path to local .mp4 / .mov file")
    parser.add_argument("--stride", type=int, default=3, help="Analyze every Nth frame (default: 3)")
    parser.add_argument("--max-side", type=int, default=322, help="Max frame dimension aligned to ViT-14 (default: 322)")
    parser.add_argument("--disable-yolo", action="store_true", help="Disable YOLOv8n fusion (depth + Rust only)")
    parser.add_argument("--mock", action="store_true", help="Use mock depth engine for fast smoke testing")
    parser.add_argument("--save-mp4", type=Path, default=None, help="Optional output path for split-screen annotated MP4")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional output path for timeline JSON report")
    parser.add_argument("--caution", type=float, default=None, help="Override caution_threshold")
    parser.add_argument("--danger", type=float, default=None, help="Override danger_threshold")
    parser.add_argument("--curb", type=float, default=None, help="Override curb_gradient_threshold")
    args = parser.parse_args()

    analyze_video(
        args.video,
        stride=max(1, args.stride),
        max_side=args.max_side,
        disable_yolo=args.disable_yolo,
        mock_depth=args.mock,
        save_mp4=args.save_mp4,
        save_json=args.save_json,
        caution_threshold=args.caution,
        danger_threshold=args.danger,
        curb_threshold=args.curb,
    )


if __name__ == "__main__":
    main()
