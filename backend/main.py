from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import struct
import time
from contextlib import asynccontextmanager
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
from backend.vision.hazard_detector import GroundAwareHazardDetector


def get_lan_ip() -> str:
    """Finds the Mac's active Wi-Fi / Personal Hotspot IPv4 address for mobile QR connect."""
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
        self.depth_engine: DepthEngine | None = None
        self.config = CalibrationConfig()
        self.detector = GroundAwareHazardDetector(config=self.config)
        self.dataset_store = DatasetStore()
        self.dashboard_clients: set[WebSocket] = set()
        self.frame_counter: int = 0
        self.phone_connected: bool = False
        self.last_raw_jpeg: bytes | None = None
        self.last_guidance: GuidancePayload | None = None
        self.recent_latencies_ms: list[float] = []


state = AppState()


def ensure_engine() -> DepthEngine:
    if state.depth_engine is None:
        use_mock = os.getenv("GLIDANCE_MOCK_DEPTH", "0") == "1"
        state.depth_engine = DepthEngine(use_mock=use_mock)
    return state.depth_engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_engine()
    yield


app = FastAPI(
    title="Glidance Vision & Haptics Backend",
    version="1.0.0",
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
    lat = state.recent_latencies_ms
    return {
        "status": "ok",
        "device": engine.device,
        "model_name": engine.model_id,
        "phone_connected": state.phone_connected,
        "frames_processed": state.frame_counter,
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
    state.config = cfg
    state.detector.update_config(cfg)
    return state.config


@app.get("/api/samples", response_model=list[SampleRecord])
def list_samples() -> list[SampleRecord]:
    return state.dataset_store.list_samples(include_thumbnails=True)


@app.post("/api/samples", response_model=SampleRecord)
def save_sample(req: SaveSampleRequest) -> SampleRecord:
    raw_bytes: bytes | None = None
    if req.image_b64:
        b64_clean = req.image_b64.split(",")[-1]
        raw_bytes = base64.b64decode(b64_clean)
    elif state.last_raw_jpeg is not None:
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
    return run_benchmark(
        depth_engine=engine,
        dataset_store=state.dataset_store,
        config=state.config,
        include_synthetic=include_synthetic,
    )


@app.get("/api/demo-scenes")
def list_demo_scenes() -> list[dict[str, str]]:
    return [
        {"scene_id": s.sample_id, "label": s.label, "note": s.note}
        for s in build_synthetic_calibration_suite()
    ]


@app.post("/api/demo-scenes/simulate", response_model=GuidancePayload)
async def simulate_demo_scene(req: SimulateSceneRequest) -> GuidancePayload:
    """Lets Mission Control preview any hallway/curb scenario live without needing a phone."""
    engine = ensure_engine()
    scenes = {s.sample_id: s for s in build_synthetic_calibration_suite()}
    scene = scenes.get(req.scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Demo scene not found")

    now = time.time()
    state.frame_counter += 1
    res = state.detector.analyze(scene.closeness_map, stateful=False)
    guidance = GuidancePayload(
        frame_id=state.frame_counter,
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
    )
    depth_jpg = render_depth_overlay_jpeg(scene.closeness_map, guidance)
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


def _process_frame_sync(
    jpeg_bytes: bytes, client_ts: float
) -> tuple[GuidancePayload, bytes]:
    engine = ensure_engine()
    t_total = time.perf_counter()

    img, decode_ms = engine.decode_jpeg(jpeg_bytes)
    closeness, inference_ms = engine.predict_pil(img)
    haz = state.detector.analyze(closeness, stateful=True)
    server_total_ms = (time.perf_counter() - t_total) * 1000.0

    state.frame_counter += 1
    state.last_raw_jpeg = jpeg_bytes
    state.recent_latencies_ms.append(server_total_ms)
    if len(state.recent_latencies_ms) > 120:
        state.recent_latencies_ms.pop(0)

    guidance = GuidancePayload(
        frame_id=state.frame_counter,
        client_ts=client_ts,
        server_ts=time.time() * 1000.0,
        overall_severity=haz.overall_severity,
        steering=haz.steering,
        spoken_cue=haz.spoken_cue,
        left=haz.left,
        center=haz.center,
        right=haz.right,
        curb=haz.curb,
        latency=LatencyBreakdown(
            decode_ms=round(decode_ms, 2),
            inference_ms=round(inference_ms, 2),
            hazard_ms=round(haz.hazard_ms, 2),
            server_total_ms=round(server_total_ms, 2),
        ),
        device=engine.device,
        model_name=engine.model_id,
    )
    state.last_guidance = guidance
    depth_jpeg = render_depth_overlay_jpeg(closeness, guidance)
    return guidance, depth_jpeg


@app.websocket("/ws/phone")
async def phone_stream_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    state.phone_connected = True
    state.detector.reset()
    await broadcast_dashboard({"type": "status", "phone_connected": True})

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            jpeg_bytes: bytes | None = None
            client_ts = time.time() * 1000.0

            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                # Binary protocol: 8-byte big-endian float64 client_ts + raw JPEG payload
                if len(raw) > 8:
                    client_ts = struct.unpack(">d", raw[:8])[0]
                    jpeg_bytes = raw[8:]
            elif "text" in msg and msg["text"]:
                parsed = json.loads(msg["text"])
                if "frame_b64" in parsed:
                    jpeg_bytes = base64.b64decode(parsed["frame_b64"].split(",")[-1])
                    client_ts = float(parsed.get("client_ts", client_ts))

            if not jpeg_bytes:
                continue

            guidance, depth_jpeg = await asyncio.to_thread(
                _process_frame_sync, jpeg_bytes, client_ts
            )

            # Immediately send lightweight guidance JSON back to the phone for haptics
            await websocket.send_text(guidance.model_dump_json())

            # Fan out rich telemetry + visual overlays to connected projector dashboards
            if state.dashboard_clients:
                await broadcast_dashboard(
                    {
                        "type": "telemetry",
                        "guidance": guidance.model_dump(),
                        "raw_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
                        "depth_b64": base64.b64encode(depth_jpeg).decode("ascii"),
                        "phone_connected": True,
                    }
                )
    except WebSocketDisconnect:
        pass
    finally:
        state.phone_connected = False
        await broadcast_dashboard({"type": "status", "phone_connected": False})


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    state.dashboard_clients.add(websocket)
    await websocket.send_text(
        json.dumps(
            {
                "type": "status",
                "phone_connected": state.phone_connected,
                "guidance": (
                    state.last_guidance.model_dump() if state.last_guidance else None
                ),
            }
        )
    )
    try:
        while True:
            # Keep socket alive and ignore ping frames from clients
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.dashboard_clients.discard(websocket)
