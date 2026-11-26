"""Bake a privacy-safe, compact depth + YOLO telemetry trace from a local video for fast CI replay.

Runs `DepthEngine` and `SemanticObjectDetector` ONCE locally (GPU/MPS/CUDA) on a walking video
and saves only quantized depth maps + YOLO bounding-box metadata (zero RGB pixels, zero faces)
into `backend/tests/fixtures/` (~1-2 MB) so GitHub Actions can replay the entire walk through
the real Rust SIMD kernel & fusion logic in <0.5s with no GPU or PyTorch dependency.

Usage:
    python scripts/bake_video_trace.py /Users/Apple/Downloads/IMG_7723.MOV --stride 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vision.depth_engine import DepthEngine
from backend.vision.object_detector import SemanticObjectDetector

DEFAULT_OUT_DIR = REPO_ROOT / "backend" / "tests" / "fixtures"


def _resize_vit14(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(w, h)))
    target_w = max(112, round((w * scale) / 14.0) * 14)
    target_h = max(112, round((h * scale) / 14.0) * 14)
    if target_w == w and target_h == h:
        return frame_bgr
    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def bake_trace(
    video_path: Path,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    name: str = "classroom_walk",
    stride: int = 6,
    max_side: int = 322,
    trace_max_side: int = 128,
) -> tuple[Path, Path]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[Bake Trace] Loading DepthEngine + SemanticObjectDetector...")
    depth_engine = DepthEngine(use_mock=False)
    obj_detector = SemanticObjectDetector(use_mock=False)
    print(
        f"[Bake Trace] Device={depth_engine.device.upper()} | "
        f"Model={depth_engine.model_id} | YOLO={'ON' if obj_detector.is_active else 'OFF'}"
    )

    closeness_frames: list[np.ndarray] = []
    depth_m_frames: list[np.ndarray] = []
    spreads: list[int] = []
    luma_means: list[float] = []
    luma_stds: list[float] = []
    times_sec: list[float] = []
    objects_per_frame: list[list[dict]] = []

    src_idx = 0
    t0 = time.perf_counter()
    trace_shape: tuple[int, int] | None = None

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

            pred = depth_engine.predict_rgb(rgb, luma_mean, luma_std)
            h, w = pred.closeness.shape[:2]
            if trace_shape is None:
                sc = min(1.0, trace_max_side / float(max(h, w)))
                th = max(48, int(round(h * sc)))
                tw = max(48, int(round(w * sc)))
                trace_shape = (th, tw)

            th, tw = trace_shape
            cl_small = cv2.resize(pred.closeness, (tw, th), interpolation=cv2.INTER_AREA)
            closeness_frames.append(np.clip(np.round(cl_small * 255.0), 0, 255).astype(np.uint8))

            # Quantize metric depth [0.25m..6.0m] into uint8 (~2.3cm precision) for compact Git storage
            dm = pred.depth_meters if pred.depth_meters is not None else (4.8 - pred.closeness * 4.45)
            dm_small = cv2.resize(dm.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
            dm_q = np.clip(np.round((dm_small - 0.25) / 5.75 * 255.0), 0, 255).astype(np.uint8)
            depth_m_frames.append(dm_q)

            spreads.append(int(pred.raw_spread_milli))
            luma_means.append(round(luma_mean, 2))
            luma_stds.append(round(luma_std, 2))
            times_sec.append(round(ts_sec, 3))

            objs: list[dict] = []
            if obj_detector.is_active and pred.depth_meters is not None:
                obj_res = obj_detector.detect(rgb, pred.depth_meters)
                objs = [o.model_dump() for o in obj_res.objects]
            objects_per_frame.append(objs)

            if len(closeness_frames) % 75 == 0:
                print(f"  ... baked {len(closeness_frames)} frames (t={ts_sec:.1f}s)")
            src_idx += 1
    finally:
        cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}_trace.npz"
    meta_path = out_dir / f"{name}_meta.json"

    np.savez_compressed(
        npz_path,
        closeness_u8=np.stack(closeness_frames, axis=0),
        depth_m_u8=np.stack(depth_m_frames, axis=0),
        raw_spread_milli=np.asarray(spreads, dtype=np.int16),
        luma_mean=np.asarray(luma_means, dtype=np.float32),
        luma_std=np.asarray(luma_stds, dtype=np.float32),
        time_sec=np.asarray(times_sec, dtype=np.float32),
    )

    # Preserve existing golden scenarios if re-baking an existing trace
    existing_scenarios: list[dict] = []
    if meta_path.exists():
        try:
            existing_scenarios = json.loads(meta_path.read_text(encoding="utf-8")).get("scenarios", [])
        except (json.JSONDecodeError, OSError):
            existing_scenarios = []

    meta = {
        "trace_name": name,
        "source_video": video_path.name,
        "source_fps": round(src_fps, 2),
        "stride": stride,
        "frame_count": len(closeness_frames),
        "duration_sec": round(times_sec[-1], 2) if times_sec else 0.0,
        "trace_shape_hw": list(trace_shape) if trace_shape else [0, 0],
        "depth_model": depth_engine.model_id,
        "is_metric_model": depth_engine._is_metric_model,
        "yolo_active": obj_detector.is_active,
        "scenarios": existing_scenarios,
        "objects_per_frame": objects_per_frame,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    npz_kb = npz_path.stat().st_size / 1024.0
    meta_kb = meta_path.stat().st_size / 1024.0
    print(
        f"[Bake Trace] Done in {elapsed:.1f}s! Frames={len(closeness_frames)} ({meta['duration_sec']}s)\n"
        f"  -> {npz_path} ({npz_kb:.1f} KB)\n"
        f"  -> {meta_path} ({meta_kb:.1f} KB)"
    )
    return npz_path, meta_path


def bake_samples_trace(
    samples_dir: Path,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    name: str = "real_photos",
    trace_max_side: int = 160,
) -> tuple[Path, Path]:
    """Bakes labeled real photos from `backend/data/samples/` into a compact GPU-free CI fixture."""
    meta_file = samples_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Samples metadata not found: {meta_file}")

    rows = json.loads(meta_file.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError(f"No samples in {meta_file}")

    print(f"[Bake Samples] Loading DepthEngine + SemanticObjectDetector for {len(rows)} real photos...")
    depth_engine = DepthEngine(use_mock=False)
    obj_detector = SemanticObjectDetector(use_mock=False)

    closeness_list: list[np.ndarray] = []
    depth_m_list: list[np.ndarray] = []
    spreads: list[int] = []
    luma_means: list[float] = []
    luma_stds: list[float] = []
    sample_entries: list[dict] = []

    t0 = time.perf_counter()
    # Normalize all sample maps onto a common 160x120 or aspect-consistent padded/resized grid?
    # Keep each photo's exact portrait/landscape shape by storing per-sample keys in .npz!
    npz_payload: dict[str, np.ndarray] = {}

    for idx, row in enumerate(rows):
        sid = row["sample_id"]
        img_path = samples_dir / f"{sid}.jpg"
        if not img_path.exists():
            continue
        raw_bytes = img_path.read_bytes()
        dec = depth_engine.decode_frame(raw_bytes)
        pred = depth_engine.predict_rgb(dec.rgb, dec.luma_mean, dec.luma_std)

        h, w = pred.closeness.shape[:2]
        sc = min(1.0, trace_max_side / float(max(h, w)))
        th = max(48, int(round(h * sc)))
        tw = max(48, int(round(w * sc)))

        cl_small = cv2.resize(pred.closeness, (tw, th), interpolation=cv2.INTER_AREA)
        dm = pred.depth_meters if pred.depth_meters is not None else (4.8 - pred.closeness * 4.45)
        dm_small = cv2.resize(dm.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)

        cl_u8 = np.clip(np.round(cl_small * 255.0), 0, 255).astype(np.uint8)
        dm_u8 = np.clip(np.round((dm_small - 0.25) / 5.75 * 255.0), 0, 255).astype(np.uint8)
        npz_payload[f"cl_{idx}"] = cl_u8
        npz_payload[f"dm_{idx}"] = dm_u8

        objs: list[dict] = []
        if obj_detector.is_active and pred.depth_meters is not None:
            obj_res = obj_detector.detect(dec.rgb, pred.depth_meters)
            objs = [o.model_dump() for o in obj_res.objects]

        sample_entries.append(
            {
                "index": idx,
                "sample_id": sid,
                "label": row["label"],
                "note": row.get("note", ""),
                "raw_spread_milli": int(pred.raw_spread_milli),
                "luma_mean": round(float(pred.luma_mean), 2),
                "luma_std": round(float(pred.luma_std), 2),
                "objects": objs,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}_trace.npz"
    meta_path = out_dir / f"{name}_meta.json"
    np.savez_compressed(npz_path, **npz_payload)

    meta = {
        "trace_name": name,
        "sample_count": len(sample_entries),
        "depth_model": depth_engine.model_id,
        "is_metric_model": depth_engine._is_metric_model,
        "yolo_active": obj_detector.is_active,
        "samples": sample_entries,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(
        f"[Bake Samples] Done in {elapsed:.1f}s! Samples={len(sample_entries)}\n"
        f"  -> {npz_path} ({npz_path.stat().st_size / 1024.0:.1f} KB)\n"
        f"  -> {meta_path} ({meta_path.stat().st_size / 1024.0:.1f} KB)"
    )
    return npz_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake compact depth+YOLO CI replay trace from video or samples.")
    parser.add_argument("video", type=Path, nargs="?", default=None, help="Path to local .mp4 / .mov file")
    parser.add_argument("--samples-dir", type=Path, default=None, help="Bake labeled photos from backend/data/samples/")
    parser.add_argument("--name", type=str, default="classroom_walk", help="Fixture base name")
    parser.add_argument("--stride", type=int, default=6, help="Sample every Nth frame (default: 6 -> ~5 FPS)")
    parser.add_argument("--trace-max-side", type=int, default=128, help="Quantized depth map max side (default: 128)")
    args = parser.parse_args()

    if args.samples_dir is not None:
        bake_samples_trace(
            args.samples_dir,
            name="real_photos" if args.name == "classroom_walk" else args.name,
        )
    if args.video is not None:
        bake_trace(
            args.video,
            name=args.name,
            stride=max(1, args.stride),
            trace_max_side=args.trace_max_side,
        )
    if args.video is None and args.samples_dir is None:
        parser.error("Provide a video path and/or --samples-dir")


if __name__ == "__main__":
    main()
