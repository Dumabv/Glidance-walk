from __future__ import annotations

import io
import os
import struct
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.evaluation.dataset_store import build_synthetic_calibration_suite
from backend.schemas import CalibrationConfig
from backend.vision.baseline_naive import NaiveCenterThresholdDetector
from backend.vision.hazard_detector import GroundAwareHazardDetector


def _make_jpeg(width: int = 160, height: int = 120, brightness: int = 100) -> bytes:
    arr = np.full((height, width, 3), brightness, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_ground_aware_beats_naive_on_synthetic_suite() -> None:
    scenes = build_synthetic_calibration_suite()
    naive = NaiveCenterThresholdDetector()
    smart = GroundAwareHazardDetector(CalibrationConfig())

    smart_correct = 0
    naive_clear_false_alarms = 0
    smart_clear_false_alarms = 0
    smart_curb_hits = 0
    curb_total = 0

    for scene in scenes:
        n_res = naive.analyze(scene.closeness_map)
        s_res = smart.analyze(scene.closeness_map, stateful=False)

        if scene.label == "clear":
            if n_res.obstacle_detected:
                naive_clear_false_alarms += 1
            if s_res.overall_severity != "clear":
                smart_clear_false_alarms += 1
            else:
                smart_correct += 1
        elif scene.label == "curb_drop":
            curb_total += 1
            if s_res.curb.detected:
                smart_curb_hits += 1
                smart_correct += 1
        elif scene.label == "obstacle_left" and s_res.left.severity != "clear":
            smart_correct += 1
        elif scene.label == "obstacle_center" and (
            s_res.center.severity != "clear" or s_res.steering == "STOP"
        ):
            smart_correct += 1
        elif scene.label == "obstacle_right" and s_res.right.severity != "clear":
            smart_correct += 1

    assert smart_clear_false_alarms == 0, "Smart detector should ignore flat tilted floors"
    assert naive_clear_false_alarms > 0, "Naive baseline should fail on tilted floor"
    assert smart_curb_hits == curb_total, "Smart detector must catch all curb drop-offs"
    assert smart_correct == len(scenes), f"Expected 100% on calibration suite, got {smart_correct}/{len(scenes)}"


def test_temporal_debounce_prevents_single_frame_danger_spike() -> None:
    cfg = CalibrationConfig(confirm_frames=2, ema_alpha=1.0)
    detector = GroundAwareHazardDetector(cfg)

    scenes = {s.sample_id: s.closeness_map for s in build_synthetic_calibration_suite()}
    wall = scenes["syn-center-3"]

    first = detector.analyze(wall, stateful=True)
    second = detector.analyze(wall, stateful=True)

    assert first.center.severity == "caution"
    assert second.center.severity == "danger"
    assert second.steering == "STOP"


def test_fastapi_endpoints_and_websocket_roundtrip() -> None:
    os.environ["GLIDANCE_MOCK_DEPTH"] = "1"
    from backend.main import app, state

    state.depth_engine = None  # force mock init

    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        bench = client.post("/api/benchmark/run")
        assert bench.status_code == 200
        data = bench.json()
        assert data["smart"]["f1"] > data["naive"]["f1"]
        assert data["smart"]["curb_recall"] == 1.0
        assert data["naive"]["curb_recall"] == 0.0

        # Binary WebSocket frame protocol test
        with client.websocket_connect("/ws/phone") as ws:
            ts_ms = time.time() * 1000.0
            payload = struct.pack(">d", ts_ms) + _make_jpeg()
            ws.send_bytes(payload)
            reply = ws.receive_json()
            assert reply["frame_id"] >= 1
            assert abs(reply["client_ts"] - ts_ms) < 1.0
            assert reply["overall_severity"] in ("clear", "caution", "danger")
