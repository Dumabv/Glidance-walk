from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["clear", "caution", "danger"]
SteeringAction = Literal["CLEAR", "VEER_LEFT", "VEER_RIGHT", "STOP", "CURB_DROP"]
SampleLabel = Literal[
    "clear",
    "obstacle_left",
    "obstacle_center",
    "obstacle_right",
    "curb_drop",
]


class ZoneMetrics(BaseModel):
    severity: Severity
    obstacle_score: float = Field(ge=0.0, le=1.0)
    min_distance_m: float = Field(ge=0.2, le=10.0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)


class CurbMetrics(BaseModel):
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    drop_row_ratio: float = Field(ge=0.0, le=1.0)
    gradient_peak: float = Field(ge=0.0)


class LatencyBreakdown(BaseModel):
    decode_ms: float
    inference_ms: float
    hazard_ms: float
    server_total_ms: float


class GuidancePayload(BaseModel):
    frame_id: int
    client_ts: float
    server_ts: float
    overall_severity: Severity
    steering: SteeringAction
    spoken_cue: str
    left: ZoneMetrics
    center: ZoneMetrics
    right: ZoneMetrics
    curb: CurbMetrics
    latency: LatencyBreakdown
    device: str
    model_name: str


class CalibrationConfig(BaseModel):
    caution_threshold: float = Field(default=0.24, ge=0.05, le=0.8)
    danger_threshold: float = Field(default=0.44, ge=0.15, le=0.95)
    curb_gradient_threshold: float = Field(default=0.14, ge=0.04, le=0.6)
    floor_margin: float = Field(default=0.13, ge=0.02, le=0.4)
    ema_alpha: float = Field(default=0.55, ge=0.1, le=1.0)
    confirm_frames: int = Field(default=2, ge=1, le=5)


class SampleRecord(BaseModel):
    sample_id: str
    label: SampleLabel
    note: str = ""
    source: Literal["real", "synthetic"] = "real"
    created_at: float
    image_b64: str | None = None


class EngineBenchmarkMetrics(BaseModel):
    engine_name: str
    precision: float
    recall: float
    f1: float
    false_alarm_rate_clear: float
    curb_recall: float
    direction_accuracy: float
    p50_ms: float
    p95_ms: float


class PerSampleComparison(BaseModel):
    sample_id: str
    label: SampleLabel
    note: str
    source: Literal["real", "synthetic"]
    naive_detected: bool
    naive_correct: bool
    smart_steering: SteeringAction
    smart_severity: Severity
    smart_curb: bool
    smart_correct: bool
    smart_ms: float


class BenchmarkReport(BaseModel):
    total_samples: int
    real_samples: int
    synthetic_samples: int
    device: str
    model_name: str
    naive: EngineBenchmarkMetrics
    smart: EngineBenchmarkMetrics
    samples: list[PerSampleComparison]
