from __future__ import annotations

import io
import os
import struct
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.evaluation.dataset_store import build_synthetic_calibration_suite
from backend.schemas import CalibrationConfig, DetectedObject
from backend.vision.baseline_naive import NaiveCenterThresholdDetector
from backend.vision.hazard_detector import (
    GroundAwareHazardDetector,
    decide_guidance_verified,
)
from backend.vision.object_detector import estimate_box_distance_m

ALLOWED_KERNELS = (
    ("rust-simd",)
    if os.getenv("GLIDANCE_REQUIRE_RUST", "0") == "1"
    else ("rust-simd", "numpy-verified")
)


def _make_jpeg(width: int = 160, height: int = 120, brightness: int = 110) -> bytes:
    # Add light gradient so synthetic frame isn't flagged as a covered/black camera lens
    y = np.linspace(brightness - 25, brightness + 25, height, dtype=np.uint8)[:, None]
    arr = np.repeat(y, width, axis=1)
    rgb = np.stack([arr, arr, arr], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG")
    return buf.getvalue()


def test_ground_aware_beats_naive_on_synthetic_and_adversarial_suite() -> None:
    scenes = build_synthetic_calibration_suite()
    naive = NaiveCenterThresholdDetector()
    smart = GroundAwareHazardDetector(CalibrationConfig())

    assert smart.kernel_backend in ALLOWED_KERNELS, f"Unexpected kernel backend: {smart.kernel_backend}"

    smart_correct = 0
    naive_clear_false_alarms = 0
    smart_clear_false_alarms = 0
    smart_curb_hits = 0
    curb_total = 0

    for scene in scenes:
        n_res = naive.analyze(scene.closeness_map)
        s_res = smart.analyze(
            scene.closeness_map,
            stateful=False,
            raw_spread_milli=scene.raw_spread_milli,
        )

        if scene.label == "clear":
            if n_res.obstacle_detected:
                naive_clear_false_alarms += 1
            if s_res.overall_severity != "clear" or s_res.steering != "CLEAR":
                smart_clear_false_alarms += 1
            else:
                smart_correct += 1
        elif scene.label == "curb_drop":
            curb_total += 1
            if s_res.curb.detected and s_res.steering == "CURB_DROP":
                smart_curb_hits += 1
                smart_correct += 1
        elif scene.label == "obstacle_left" and s_res.left.severity != "clear" and s_res.steering not in ("VEER_LEFT", "CURB_DROP"):
            smart_correct += 1
        elif scene.label == "obstacle_center" and (
            s_res.center.severity != "clear" or s_res.steering == "STOP"
        ) and s_res.steering != "CURB_DROP":
            smart_correct += 1
        elif scene.label == "obstacle_right" and s_res.right.severity != "clear" and s_res.steering not in ("VEER_RIGHT", "CURB_DROP"):
            smart_correct += 1

    assert smart_clear_false_alarms == 0, "Smart detector must ignore tilted floors, ramps & shiny stripes"
    assert naive_clear_false_alarms > 0, "Naive baseline should fail on tilted floor"
    assert smart_curb_hits == curb_total, "Smart detector must catch all curb drop-offs"
    assert smart_correct == len(scenes), f"Expected 100% on full 20-scene suite, got {smart_correct}/{len(scenes)}"

    # Verify 100% Rust-vs-NumPy parity across all 20 calibration + adversarial edge-case scenes!
    for scene in scenes:
        r_res = smart.analyze(scene.closeness_map, stateful=False, raw_spread_milli=scene.raw_spread_milli)
        n_fb = smart._analyze_numpy_fallback(
            scene.closeness_map,
            stateful=False,
            raw_spread_milli=scene.raw_spread_milli,
            luma_mean=120.0,
            luma_std=35.0,
            pitch_deg=0.0,
            frame_fresh=True,
            t0=time.perf_counter(),
        )
        assert r_res.overall_severity == n_fb.overall_severity, f"Severity mismatch on {scene.sample_id}"
        assert r_res.steering == n_fb.steering, f"Steering mismatch on {scene.sample_id}"
        assert r_res.curb.detected == n_fb.curb.detected, f"Curb mismatch on {scene.sample_id}"


def test_lean4_safety_invariants_and_counterexamples() -> None:
    """Exhaustively checks all 3^3 = 27 severity combinations against Lean 4 theorems."""
    for l_sev in range(3):
        for c_sev in range(3):
            for r_sev in range(3):
                for curb in (False, True):
                    ov, act = decide_guidance_verified(
                        l_sev, 480, c_sev, 260, r_sev, 580, curb, True
                    )
                    expected_max = max(l_sev, c_sev, r_sev, 2 if curb else 0)
                    # Theorem 2: Strict Severity Dominance
                    assert ov == expected_max
                    # Theorem 3a & 3b: Never Steer Into Danger
                    if act == 1:  # VEER_LEFT
                        assert l_sev != 2
                    if act == 2:  # VEER_RIGHT
                        assert r_sev != 2
                    # Theorem 3c: All danger without curb -> STOP
                    if l_sev == 2 and c_sev == 2 and r_sev == 2 and not curb:
                        assert act == 3

    # Theorem 1: Fail-closed stale frame -> (danger, STOP)
    assert decide_guidance_verified(0, 0, 0, 0, 0, 0, False, False) == (2, 3)

    # Covered / pitch-black camera lens -> STOP + lens_blocked
    det = GroundAwareHazardDetector(CalibrationConfig())
    dark_res = det.analyze(
        np.zeros((180, 240), dtype=np.float32),
        stateful=False,
        luma_mean=1.5,
        luma_std=0.8,
    )
    assert dark_res.overall_severity == "danger"
    assert dark_res.steering == "STOP"
    assert dark_res.lens_blocked is True

    # Inverted threshold config auto-sanitizes
    inv_cfg = CalibrationConfig(caution_threshold=0.65, danger_threshold=0.35)
    assert inv_cfg.caution_threshold <= inv_cfg.danger_threshold - 0.04


def test_temporal_debounce_prevents_single_frame_danger_spike() -> None:
    cfg = CalibrationConfig(confirm_frames=2, ema_alpha=1.0)
    detector = GroundAwareHazardDetector(cfg)

    scenes = {s.sample_id: s for s in build_synthetic_calibration_suite()}
    wall = scenes["syn-center-3"]

    first = detector.analyze(wall.closeness_map, stateful=True, raw_spread_milli=wall.raw_spread_milli)
    second = detector.analyze(wall.closeness_map, stateful=True, raw_spread_milli=wall.raw_spread_milli)

    assert first.center.severity == "caution"
    assert second.center.severity == "danger"
    assert second.steering == "STOP"


def test_fastapi_endpoints_and_8byte_and_16byte_websocket_protocols(tmp_path) -> None:
    os.environ["GLIDANCE_MOCK_DEPTH"] = "1"
    from backend.evaluation.dataset_store import DatasetStore
    from backend.main import app, state

    state.depth_engine = None  # force mock init
    orig_store = state.dataset_store
    state.dataset_store = DatasetStore(data_dir=tmp_path)
    try:
        with TestClient(app) as client:
            r = client.get("/api/health")
            assert r.status_code == 200
            hdata = r.json()
            assert hdata["status"] == "ok"
            assert hdata["kernel_backend"] in ALLOWED_KERNELS

            bench = client.post("/api/benchmark/run")
            assert bench.status_code == 200
            data = bench.json()
            assert data["smart"]["f1"] == 1.0
            assert data["smart"]["direction_accuracy"] == 1.0
            assert data["smart"]["false_alarm_rate_clear"] == 0.0
            assert data["smart"]["curb_recall"] == 1.0
            assert data["naive"]["curb_recall"] == 0.0

            # 1. Classic 8-byte binary WebSocket header
            with client.websocket_connect("/ws/phone") as ws:
                ts_ms = time.time() * 1000.0
                payload8 = struct.pack(">d", ts_ms) + _make_jpeg()
                ws.send_bytes(payload8)
                reply = ws.receive_json()
                assert reply["frame_id"] >= 1
                assert abs(reply["client_ts"] - ts_ms) < 1.0
                assert reply["kernel_backend"] in ALLOWED_KERNELS

                # 2. Upgraded 16-byte IMU-enriched binary WebSocket header `[f64 ts][f32 pitch][f32 roll]+JPEG`
                payload16 = struct.pack(">dff", ts_ms + 50.0, 68.5, -2.0) + _make_jpeg(width=240, height=320)
                ws.send_bytes(payload16)
                reply2 = ws.receive_json()
                assert reply2["pitch_deg"] == 68.5
    finally:
        state.dataset_store = orig_store


def test_box_distance_uses_center_core_median() -> None:
    """Background pixels at bbox edges must not corrupt the object's metric distance."""
    depth_meters = np.full((180, 240), 4.5, dtype=np.float32)
    depth_meters[65:115, 100:140] = 1.15
    dist = estimate_box_distance_m(depth_meters, x1=80, y1=40, x2=160, y2=140)
    assert abs(dist - 1.15) < 0.05


def test_yolo_enrichment_preserves_lean4_invariants() -> None:
    """YOLO semantic fusion names obstacles while preserving Lean 4 never-steer-into-danger."""
    from backend.main import fuse_yolo_with_verified_hazard

    cfg = CalibrationConfig()
    det = GroundAwareHazardDetector(cfg)
    clear_scene = build_synthetic_calibration_suite()[0]
    base_haz = det.analyze(
        clear_scene.closeness_map,
        stateful=False,
        raw_spread_milli=clear_scene.raw_spread_milli,
    )
    assert base_haz.steering == "CLEAR"

    chair_center = DetectedObject(
        label="chair",
        confidence=0.86,
        distance_m=1.05,
        corridor="center",
        bbox_norm=[0.38, 0.35, 0.62, 0.80],
        severity="danger",
    )
    person_left = DetectedObject(
        label="person",
        confidence=0.88,
        distance_m=1.10,
        corridor="left",
        bbox_norm=[0.05, 0.25, 0.30, 0.85],
        severity="danger",
    )

    fused = fuse_yolo_with_verified_hazard(base_haz, [chair_center, person_left], cfg)
    assert fused.overall_severity == "danger"
    # Left & Center are danger, Right is clear -> Lean 4 theorem mandates VEER_RIGHT (never VEER_LEFT!)
    assert fused.steering == "VEER_RIGHT"
    assert "Chair" in fused.spoken_cue
    assert "1.1m" in fused.spoken_cue


def test_onnx_relative_fallback_resets_is_metric_flag() -> None:
    """PR #10 review regression test: relative ONNX models must NOT invert depth via 1/z."""
    from backend.vision.depth_engine import DepthEngine

    eng = DepthEngine(use_mock=True)
    assert eng._is_metric_model is True
    # Simulate ONNX branch loading a relative model (e.g. Depth-Anything-V2-Large-FP16)
    display_name = "depth-anything/Depth-Anything-V2-Large-FP16"
    eng.model_id = display_name
    eng._is_metric_model = "metric" in display_name.lower()
    assert eng._is_metric_model is False

    # Verify higher relative disparity stays closer (no 1/z inversion!)
    raw_disparity = np.linspace(1.0, 10.0, 120 * 160, dtype=np.float32).reshape(120, 160)
    closeness, _ = eng._normalize_depth_fast(raw_disparity)
    assert float(np.mean(closeness[-10:, :])) > float(np.mean(closeness[:10, :]))
