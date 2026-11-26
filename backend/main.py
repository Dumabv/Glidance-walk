from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager
import json
import os
import socket
import struct
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.evaluation.benchmark_runner import run_benchmark
from backend.evaluation.dataset_store import (
    DatasetStore,
    build_synthetic_calibration_suite,
)
from backend.schemas import (
    BenchmarkReport,
    CalibrationConfig,
    GuidancePayload,
    LatencyBreakdown,
    SampleLabel,
    SampleRecord,
)
from backend.vision.depth_engine import DepthEngine, render_depth_overlay_jpeg
from backend.vision.hazard_detector import (
    GroundAwareHazardDetector,
    detect_close_flat_wall_or_door,
    fuse_close_range_metric_depth,
    fuse_yolo_with_verified_hazard,
)
from backend.vision.object_detector import SemanticObjectDetector


def get_lan_ip() -> str:
    """Finds the active Wi-Fi / Personal Hotspot IPv4 address for mobile QR connect."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


class SaveSampleRequest(BaseModel):
    label: SampleLabel
    note: str = ""
    image_b64: str | None = None


class SimulateSceneRequest(BaseModel):
    scene_id: str


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.depth_engine: DepthEngine | None = None
        self.object_detector: SemanticObjectDetector | None = None
        self.config = CalibrationConfig()
        self.sim_detector = GroundAwareHazardDetector(config=self.config)
        self.active_phone_detectors: dict[int, GroundAwareHazardDetector] = {}
        self.next_conn_id: int = 1
        self.dataset_store = DatasetStore()
        self.dashboard_clients: set[WebSocket] = set()
        self.frame_counter: int = 0
        self.last_raw_jpeg: bytes | None = None
        self.last_guidance: GuidancePayload | None = None
        self.recent_latencies_ms: deque[float] = deque(maxlen=120)
        self.last_dashboard_push_at: float = 0.0
        self.dashboard_push_in_flight: bool = False

    @property
    def phone_connected(self) -> bool:
        with self._lock:
            return len(self.active_phone_detectors) > 0


state = AppState()


def ensure_engine() -> DepthEngine:
    if state.depth_engine is None:
        use_mock = os.getenv("GLIDANCE_MOCK_DEPTH", "0") == "1"
        state.depth_engine = DepthEngine(use_mock=use_mock)
    return state.depth_engine


def ensure_object_detector() -> SemanticObjectDetector:
    if state.object_detector is None:
        use_mock = (
            os.getenv("GLIDANCE_MOCK_DEPTH", "0") == "1"
            or os.getenv("GLIDANCE_DISABLE_YOLO", "0") == "1"
        )
        state.object_detector = SemanticObjectDetector(use_mock=use_mock)
    return state.object_detector


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_engine()
    ensure_object_detector()
    yield


app = FastAPI(
    title="Glidance Vision & Haptics Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast_dashboard(message: dict[str, Any]) -> None:
    if not state.dashboard_clients:
        return
    payload = json.dumps(message)
    stale: list[WebSocket] = []
    for ws in list(state.dashboard_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        state.dashboard_clients.discard(ws)


@app.get("/api/health")
def health() -> dict[str, Any]:
    engine = ensure_engine()
    obj_det = ensure_object_detector()
    with state._lock:
        lat = list(state.recent_latencies_ms)
        frames = state.frame_counter
    return {
        "status": "ok",
        "device": engine.device,
        "runtime": engine.runtime,
        "kernel_backend": state.sim_detector.kernel_backend,
        "model_name": engine.model_id,
        "yolo_enabled": obj_det.is_active,
        "phone_connected": state.phone_connected,
        "frames_processed": frames,
        "p50_server_ms": round(float(np.percentile(lat, 50)), 2) if lat else 0.0,
        "p95_server_ms": round(float(np.percentile(lat, 95)), 2) if lat else 0.0,
    }


@app.get("/api/network-info")
def network_info() -> dict[str, Any]:
    lan_ip = get_lan_ip()
    frontend_port = int(os.getenv("GLIDANCE_FRONTEND_PORT", "5173"))
    return {
        "lan_ip": lan_ip,
        "mobile_url": f"https://{lan_ip}:{frontend_port}/?mode=mobile",
        "dashboard_url": f"https://{lan_ip}:{frontend_port}/",
    }


@app.get("/api/calibration", response_model=CalibrationConfig)
def get_calibration() -> CalibrationConfig:
    return state.config


@app.put("/api/calibration", response_model=CalibrationConfig)
def update_calibration(cfg: CalibrationConfig) -> CalibrationConfig:
    with state._lock:
        state.config = cfg
        state.sim_detector.update_config(cfg)
        detectors = list(state.active_phone_detectors.values())
    for det in detectors:
        det.update_config(cfg)
    return cfg


@app.get("/api/samples", response_model=list[SampleRecord])
def list_samples() -> list[SampleRecord]:
    return state.dataset_store.list_samples(include_thumbnails=True)


@app.post("/api/samples", response_model=SampleRecord)
def save_sample(req: SaveSampleRequest) -> SampleRecord:
    raw_bytes: bytes | None = None
    if req.image_b64:
        b64_clean = req.image_b64.split(",")[-1]
        raw_bytes = base64.b64decode(b64_clean)
    else:
        with state._lock:
            raw_bytes = state.last_raw_jpeg

    if not raw_bytes:
        raise HTTPException(
            status_code=400,
            detail="No frame available yet. Stream from a phone/webcam or attach image_b64.",
        )

    return state.dataset_store.save_sample(raw_bytes, req.label, req.note)


@app.delete("/api/samples/{sample_id}")
def delete_sample(sample_id: str) -> dict[str, bool]:
    deleted = state.dataset_store.delete_sample(sample_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"deleted": True}


@app.post("/api/benchmark/run", response_model=BenchmarkReport)
def execute_benchmark(include_synthetic: bool = True) -> BenchmarkReport:
    engine = ensure_engine()
    obj_detector = ensure_object_detector()
    return run_benchmark(
        depth_engine=engine,
        dataset_store=state.dataset_store,
        config=state.config,
        include_synthetic=include_synthetic,
        object_detector=obj_detector,
    )


@app.get("/api/demo-scenes")
def list_demo_scenes() -> list[dict[str, str]]:
    return [
        {"scene_id": s.sample_id, "label": s.label, "note": s.note}
        for s in build_synthetic_calibration_suite()
    ]


@app.post("/api/demo-scenes/simulate", response_model=GuidancePayload)
async def simulate_demo_scene(req: SimulateSceneRequest) -> GuidancePayload:
    """Lets Mission Control preview any hallway/curb/edge scenario live without needing a phone."""
    engine = ensure_engine()
    scenes = {s.sample_id: s for s in build_synthetic_calibration_suite()}
    scene = scenes.get(req.scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Demo scene not found")

    now = time.time()
    res = state.sim_detector.analyze(
        scene.closeness_map,
        stateful=False,
        raw_spread_milli=scene.raw_spread_milli,
    )
    with state._lock:
        state.frame_counter += 1
        fid = state.frame_counter

    guidance = GuidancePayload(
        frame_id=fid,
        client_ts=now * 1000.0,
        server_ts=now * 1000.0,
        overall_severity=res.overall_severity,
        steering=res.steering,
        spoken_cue=res.spoken_cue,
        left=res.left,
        center=res.center,
        right=res.right,
        curb=res.curb,
        latency=LatencyBreakdown(
            decode_ms=0.0,
            inference_ms=0.0,
            hazard_ms=res.hazard_ms,
            server_total_ms=res.hazard_ms,
        ),
        device=engine.device,
        model_name=engine.model_id,
        kernel_backend=res.kernel_backend,
        overhead_hazard=res.overhead_hazard,
        wall_override=res.wall_override,
        lens_blocked=res.lens_blocked,
    )
    depth_jpg = render_depth_overlay_jpeg(scene.closeness_map, guidance)
    with state._lock:
        state.last_guidance = guidance
    await broadcast_dashboard(
        {
            "type": "telemetry",
            "guidance": guidance.model_dump(),
            "depth_b64": base64.b64encode(depth_jpg).decode("ascii"),
            "raw_b64": None,
            "simulated_note": scene.note,
            "phone_connected": state.phone_connected,
        }
    )
    return guidance


def _parse_phone_ws_message(msg: dict[str, Any]) -> tuple[bytes | None, float, float]:
    """Parses 8-byte classic header `[f64 ts]+JPEG`, 16-byte IMU header `[f64 ts][f32 pitch][f32 roll]+JPEG`, or JSON."""
    client_ts = time.time() * 1000.0
    pitch_deg = 0.0

    if "bytes" in msg and msg["bytes"]:
        raw: bytes = msg["bytes"]
        # 16-byte IMU-enriched binary frame: JPEG SOI `0xFF 0xD8` starts at offset 16
        if len(raw) > 18 and raw[16:18] == b"\xff\xd8":
            client_ts, pitch_deg, _roll_deg = struct.unpack(">dff", raw[:16])
            return raw[16:], client_ts, pitch_deg
        # Classic 8-byte binary frame: `[f64 client_ts] + JPEG`
        if len(raw) > 8:
            client_ts = struct.unpack(">d", raw[:8])[0]
            return raw[8:], client_ts, 0.0
    elif "text" in msg and msg["text"]:
        parsed = json.loads(msg["text"])
        if "frame_b64" in parsed:
            jpeg_bytes = base64.b64decode(parsed["frame_b64"].split(",")[-1])
            client_ts = float(parsed.get("client_ts", client_ts))
            pitch_deg = float(parsed.get("pitch_deg", 0.0))
            return jpeg_bytes, client_ts, pitch_deg

    return None, client_ts, pitch_deg


def _process_frame_critical_path(
    detector: GroundAwareHazardDetector,
    jpeg_bytes: bytes,
    client_ts: float,
    pitch_deg: float,
) -> tuple[GuidancePayload, np.ndarray]:
    """Critical phone haptic loop — skips dashboard JPEG overlay encoding completely!"""
    engine = ensure_engine()
    obj_detector = ensure_object_detector()
    t_total = time.perf_counter()

    dec = engine.decode_frame(jpeg_bytes)
    pred = engine.predict_rgb(dec.rgb, dec.luma_mean, dec.luma_std)
    haz = detector.analyze(
        pred.closeness,
        stateful=True,
        raw_spread_milli=pred.raw_spread_milli,
        luma_mean=pred.luma_mean,
        luma_std=pred.luma_std,
        pitch_deg=pitch_deg,
        frame_fresh=True,
    )
    depth_m = pred.depth_meters
    if engine._is_metric_model and not engine.use_mock and depth_m is not None:
        haz = fuse_close_range_metric_depth(haz, depth_m, pitch_deg, detector.config)
    if obj_detector.is_active and depth_m is not None:
        obj_res = obj_detector.detect(dec.rgb, depth_m)
        raw_objects = list(obj_res.objects)
        # IMU + Depth Flat-Floor Guard: tiled/striped floors viewed downward (0 < pitch < 48 deg)
        # or completely flat floors (clear center + low curb gradient) should not trigger false `stairs`.
        looking_at_floor = 0.0 < pitch_deg < 48.0
        flat_floor_depth = haz.center.severity == "clear" and haz.curb.gradient_peak < 0.055
        filtered_objects = []
        for obj in raw_objects:
            if obj.label in ("stairs", "step", "curb"):
                if looking_at_floor and obj.confidence < 0.72 and haz.curb.gradient_peak < 0.06:
                    continue
                if flat_floor_depth and obj.confidence < 0.65 and obj.distance_m > 1.8:
                    continue
            filtered_objects.append(obj)

        haz = detector.fuse_objects(haz, filtered_objects, stateful=True)
        detected_objects = filtered_objects
        yolo_ms = obj_res.yolo_ms
    else:
        haz = detector.fuse_objects(haz, [], stateful=True)
        detected_objects = []
        yolo_ms = 0.0

    haz, close_wall_obj = detect_close_flat_wall_or_door(dec.rgb, haz)
    if close_wall_obj is not None and not any(o.label in ("wall", "door") for o in detected_objects):
        detected_objects.insert(0, close_wall_obj)

    server_total_ms = (time.perf_counter() - t_total) * 1000.0

    with state._lock:
        state.frame_counter += 1
        fid = state.frame_counter
        state.last_raw_jpeg = jpeg_bytes
        state.recent_latencies_ms.append(server_total_ms)

    guidance = GuidancePayload.model_construct(
        frame_id=fid,
        client_ts=client_ts,
        server_ts=time.time() * 1000.0,
        overall_severity=haz.overall_severity,
        steering=haz.steering,
        spoken_cue=haz.spoken_cue,
        left=haz.left,
        center=haz.center,
        right=haz.right,
        curb=haz.curb,
        objects=detected_objects,
        latency=LatencyBreakdown.model_construct(
            decode_ms=round(dec.decode_ms, 2),
            inference_ms=round(pred.inference_ms, 2),
            yolo_ms=round(yolo_ms, 2),
            hazard_ms=round(haz.hazard_ms, 3),
            server_total_ms=round(server_total_ms, 2),
        ),
        device=engine.device,
        model_name=engine.model_id,
        kernel_backend=haz.kernel_backend,
        overhead_hazard=haz.overhead_hazard,
        wall_override=haz.wall_override,
        lens_blocked=haz.lens_blocked,
        pitch_deg=round(pitch_deg, 1) if pitch_deg != 0.0 else None,
    )
    with state._lock:
        state.last_guidance = guidance
    return guidance, pred.closeness


@app.websocket("/ws/phone")
async def phone_stream_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    with state._lock:
        conn_id = state.next_conn_id
        state.next_conn_id += 1
        detector = GroundAwareHazardDetector(config=state.config)
        state.active_phone_detectors[conn_id] = detector

    await broadcast_dashboard({"type": "status", "phone_connected": True})

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # Latest-frame-wins drain: discard queued older frames if network jittered
            disconnected = False
            while True:
                try:
                    newer = await asyncio.wait_for(websocket.receive(), timeout=0.0004)
                    if newer.get("type") == "websocket.disconnect":
                        disconnected = True
                        break
                    if ("bytes" in newer and newer["bytes"]) or ("text" in newer and newer["text"]):
                        msg = newer
                except asyncio.TimeoutError:
                    break
            if disconnected:
                break

            jpeg_bytes, client_ts, pitch_deg = _parse_phone_ws_message(msg)
            if not jpeg_bytes:
                continue

            guidance, closeness = await asyncio.to_thread(
                _process_frame_critical_path, detector, jpeg_bytes, client_ts, pitch_deg
            )

            # 1. Immediately reply to phone BEFORE any dashboard image work!
            await websocket.send_text(guidance.model_dump_json())

            # 2. Throttled, memory-bounded fan-out to projector dashboards (at most 1 frame in flight!)
            now_mono = time.perf_counter()
            if (
                state.dashboard_clients
                and not state.dashboard_push_in_flight
                and (now_mono - state.last_dashboard_push_at >= 0.080)
            ):
                state.last_dashboard_push_at = now_mono
                state.dashboard_push_in_flight = True
                depth_jpeg = await asyncio.to_thread(
                    render_depth_overlay_jpeg, closeness, guidance
                )

                async def _push_dash(raw_jpg: bytes, d_jpg: bytes, g_dict: dict[str, Any]) -> None:
                    try:
                        await broadcast_dashboard(
                            {
                                "type": "telemetry",
                                "guidance": g_dict,
                                "raw_b64": base64.b64encode(raw_jpg).decode("ascii"),
                                "depth_b64": base64.b64encode(d_jpg).decode("ascii"),
                                "phone_connected": True,
                            }
                        )
                    finally:
                        state.dashboard_push_in_flight = False

                asyncio.create_task(_push_dash(jpeg_bytes, depth_jpeg, guidance.model_dump()))
    except WebSocketDisconnect:
        pass
    finally:
        with state._lock:
            state.active_phone_detectors.pop(conn_id, None)
            still_connected = len(state.active_phone_detectors) > 0
        await broadcast_dashboard({"type": "status", "phone_connected": still_connected})


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    state.dashboard_clients.add(websocket)
    with state._lock:
        last_g = state.last_guidance.model_dump() if state.last_guidance else None
    await websocket.send_text(
        json.dumps(
            {
                "type": "status",
                "phone_connected": state.phone_connected,
                "guidance": last_g,
            }
        )
    )
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.dashboard_clients.discard(websocket)
