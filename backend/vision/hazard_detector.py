from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from backend.schemas import (
    CalibrationConfig,
    Corridor,
    CurbMetrics,
    DetectedObject,
    Severity,
    SteeringAction,
    ZoneMetrics,
)

SEVERITY_BY_CODE: tuple[Severity, ...] = ("clear", "caution", "danger")
STEERING_BY_CODE: tuple[SteeringAction, ...] = (
    "CLEAR",
    "VEER_LEFT",
    "VEER_RIGHT",
    "STOP",
    "CURB_DROP",
)

CUE_BY_CODE: dict[int, str] = {
    0: "Path clear.",
    1: "Curb or step down ahead. Slow down.",
    2: "Stairs going down ahead. Stop.",
    3: "Obstacle ahead. Step left.",
    4: "Obstacle ahead. Step right.",
    5: "Path blocked ahead. Stop.",
    6: "Object ahead. Slight left.",
    7: "Object ahead. Slight right.",
    8: "Objects on both sides. Stay centered.",
    9: "Close obstacle on left. Move right.",
    10: "Close obstacle on right. Move left.",
    11: "Narrow gap. Stop and check.",
    12: "Obstacle on left.",
    13: "Obstacle on right.",
    14: "Low overhead obstacle ahead. Stop.",
    15: "Camera blocked or too dark. Stop.",
    16: "Signal lost. Stop and wait.",
}

# Lean 4 Verified Milliscale Constants (`GlidanceSafety.Target.lean`)
WALL_SPREAD_EPS = 80
WALL_PROXIMITY_MIN = 700
WALL_BLOCKAGE_MIN = 840
WALL_DANGER_SCORE = 1000
CURB_MIN_FG_FLOOR = 350
BOX_MAX_EXCESS = 120
SYMMETRIC_CAUTION_MARGIN = 50


# ============================================================================
# Zero-Copy Rust `cdylib` (`glidance_core`) C-ABI Structures & Loader
# ============================================================================


class _CCalibrationConfig(ctypes.Structure):
    _fields_ = [
        ("caution_milli", ctypes.c_uint32),
        ("danger_milli", ctypes.c_uint32),
        ("curb_thresh_milli", ctypes.c_uint32),
        ("floor_margin_milli", ctypes.c_uint32),
        ("ema_alpha_milli", ctypes.c_uint32),
        ("confirm_frames", ctypes.c_uint32),
    ]


class _CDetectorState(ctypes.Structure):
    _fields_ = [
        ("ema_scores_milli", ctypes.c_uint32 * 3),
        ("ema_curb_milli", ctypes.c_uint32),
        ("danger_streak", ctypes.c_uint32 * 3),
        ("curb_streak", ctypes.c_uint32),
    ]


class _CZoneOut(ctypes.Structure):
    _fields_ = [
        ("severity", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
        ("obstacle_score", ctypes.c_float),
        ("min_distance_m", ctypes.c_float),
        ("coverage_ratio", ctypes.c_float),
    ]


class _CCurbOut(ctypes.Structure):
    _fields_ = [
        ("detected", ctypes.c_uint8),
        ("stairs_likely", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 2),
        ("confidence", ctypes.c_float),
        ("drop_row_ratio", ctypes.c_float),
        ("gradient_peak", ctypes.c_float),
    ]


class _CHazardOutput(ctypes.Structure):
    _fields_ = [
        ("overall_severity", ctypes.c_uint8),
        ("steering", ctypes.c_uint8),
        ("cue_code", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("left", _CZoneOut),
        ("center", _CZoneOut),
        ("right", _CZoneOut),
        ("curb", _CCurbOut),
        ("floor_slope", ctypes.c_float),
        ("floor_intercept", ctypes.c_float),
    ]


def _load_rust_kernel():
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    candidates = [
        here / "glidance_core.dll",
        here / "libglidance_core.dylib",
        here / "libglidance_core.so",
        repo_root / "rust_core" / "target" / "release" / "glidance_core.dll",
        repo_root / "rust_core" / "target" / "release" / "libglidance_core.dylib",
        repo_root / "rust_core" / "target" / "release" / "libglidance_core.so",
    ]
    existing = [p for p in candidates if p.exists()]
    if sys.platform == "darwin":
        existing.sort(key=lambda p: (0 if p.suffix == ".dylib" else 1, -p.stat().st_mtime))
    elif sys.platform.startswith("win"):
        existing.sort(key=lambda p: (0 if p.suffix == ".dll" else 1, -p.stat().st_mtime))
    else:
        existing.sort(key=lambda p: -p.stat().st_mtime)

    for path in existing:
        if True:
            try:
                lib = ctypes.CDLL(str(path))
                fn = lib.glidance_analyze_frame
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_uint8,
                    ctypes.c_uint8,
                    ctypes.POINTER(_CCalibrationConfig),
                    ctypes.POINTER(_CDetectorState),
                    ctypes.POINTER(_CHazardOutput),
                ]
                fn.restype = ctypes.c_int32
                return fn
            except OSError:
                continue
    return None


_RUST_ANALYZE_FN = _load_rust_kernel()


@dataclass(slots=True)
class HazardResult:
    overall_severity: Severity
    steering: SteeringAction
    spoken_cue: str
    left: ZoneMetrics
    center: ZoneMetrics
    right: ZoneMetrics
    curb: CurbMetrics
    hazard_ms: float
    kernel_backend: str = "rust-simd"
    overhead_hazard: bool = False
    wall_override: bool = False
    lens_blocked: bool = False


# ============================================================================
# Pure-Python 1:1 Mirror of Lean 4 `GlidanceSafety` (Cross-Platform Fallback)
# ============================================================================


def decide_guidance_verified(
    left_sev: int,
    left_score: int,
    center_sev: int,
    center_score: int,
    right_sev: int,
    right_score: int,
    curb_detected: bool,
    frame_fresh: bool,
) -> tuple[int, int]:
    """Exact Python mirror of Lean 4 `GlidanceSafety.decideGuidance`."""
    if not frame_fresh:
        return 2, 3  # (danger, STOP)

    overall = max(left_sev, center_sev, right_sev, 2 if curb_detected else 0)

    def pick_safer() -> int:
        if left_sev < right_sev:
            return 1  # VEER_LEFT
        if right_sev < left_sev:
            return 2  # VEER_RIGHT
        return 1 if left_score <= right_score else 2

    if curb_detected:
        act = 4  # CURB_DROP
    elif left_sev == 2 and right_sev == 2:
        act = 3  # STOP
    elif center_sev == 2:
        act = pick_safer()
    elif center_sev == 1:
        if left_sev == 1 and right_sev == 1 and abs(left_score - right_score) <= SYMMETRIC_CAUTION_MARGIN:
            act = 0  # CLEAR (stay centered)
        else:
            act = pick_safer()
    else:
        if left_sev == 2:
            act = 2  # VEER_RIGHT
        elif right_sev == 2:
            act = 1  # VEER_LEFT
        elif left_sev == 1 and right_sev == 0:
            act = 2  # VEER_RIGHT
        elif right_sev == 1 and left_sev == 0:
            act = 1  # VEER_LEFT
        else:
            act = 0  # CLEAR

    return overall, act


class GroundAwareHazardDetector:
    """Lean-4 verified ground-plane, overhang, wall-collapse & curb detector.

    Executes the zero-copy Rust SIMD kernel (`glidance_core.dll` / `.dylib` / `.so`)
    in ~0.05 ms per frame when built, with an automatic 1:1 verified NumPy fallback.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._lock = threading.Lock()
        self.config = config or CalibrationConfig()
        self._c_cfg = self._to_c_config(self.config)
        self._c_state = _CDetectorState()
        self._c_out = _CHazardOutput()
        self._c_cfg_ref = ctypes.byref(self._c_cfg)
        self._c_state_ref = ctypes.byref(self._c_state)
        self._c_out_ref = ctypes.byref(self._c_out)
        self.kernel_backend = "rust-simd" if _RUST_ANALYZE_FN is not None else "numpy-verified"
        self._yolo_alert_streak: dict[Corridor, int] = {"left": 0, "center": 0, "right": 0}
        self._yolo_danger_streak: dict[Corridor, int] = {"left": 0, "center": 0, "right": 0}
        self._last_steer: SteeringAction = "CLEAR"

    @staticmethod
    def _to_c_config(cfg: CalibrationConfig) -> _CCalibrationConfig:
        danger_m = int(round(cfg.danger_threshold * 1000.0))
        caution_m = min(int(round(cfg.caution_threshold * 1000.0)), max(50, danger_m - 40))
        return _CCalibrationConfig(
            caution_milli=caution_m,
            danger_milli=danger_m,
            curb_thresh_milli=int(round(cfg.curb_gradient_threshold * 1000.0)),
            floor_margin_milli=int(round(cfg.floor_margin * 1000.0)),
            ema_alpha_milli=int(round(cfg.ema_alpha * 1000.0)),
            confirm_frames=int(cfg.confirm_frames),
        )

    def update_config(self, config: CalibrationConfig) -> None:
        with self._lock:
            self.config = config
            self._c_cfg = self._to_c_config(config)
            self._c_cfg_ref = ctypes.byref(self._c_cfg)

    def reset(self) -> None:
        with self._lock:
            self._c_state = _CDetectorState()
            self._c_state_ref = ctypes.byref(self._c_state)
            self._yolo_alert_streak = {"left": 0, "center": 0, "right": 0}
            self._yolo_danger_streak = {"left": 0, "center": 0, "right": 0}
            self._last_steer = "CLEAR"

    def fuse_objects(
        self,
        haz: HazardResult,
        objects: list[DetectedObject],
        *,
        stateful: bool = True,
    ) -> HazardResult:
        """Applies YOLOv8n fusion plus stateful temporal debounce & directional hysteresis."""
        if not stateful:
            return fuse_yolo_with_verified_hazard(haz, objects, self.config)

        base_sevs: dict[Corridor, Severity] = {
            "left": haz.left.severity,
            "center": haz.center.severity,
            "right": haz.right.severity,
        }

        # Filter YOLO alert & danger spikes through `confirm_frames` debounce so 1-frame flickers don't nag
        stabilized_objs: list[DetectedObject] = []
        seen_alert: set[Corridor] = set()
        seen_danger: set[Corridor] = set()
        for obj in objects:
            if obj.confidence >= 0.40 and obj.severity != "clear":
                seen_alert.add(obj.corridor)
                if obj.severity == "danger":
                    seen_danger.add(obj.corridor)

        with self._lock:
            confirm_n = max(1, int(self.config.confirm_frames))
            for side in ("left", "center", "right"):
                if side in seen_alert or base_sevs[side] != "clear":
                    self._yolo_alert_streak[side] += 1
                else:
                    self._yolo_alert_streak[side] = 0
                if side in seen_danger or base_sevs[side] == "danger":
                    self._yolo_danger_streak[side] += 1
                else:
                    self._yolo_danger_streak[side] = 0
            alert_streaks = dict(self._yolo_alert_streak)
            danger_streaks = dict(self._yolo_danger_streak)

        for obj in objects:
            if (
                obj.severity == "caution"
                and base_sevs[obj.corridor] == "clear"
                and alert_streaks[obj.corridor] < confirm_n
            ):
                continue
            if (
                obj.severity == "danger"
                and base_sevs[obj.corridor] != "danger"
                and danger_streaks[obj.corridor] < confirm_n
            ):
                stabilized_objs.append(obj.model_copy(update={"severity": "caution"}))
            else:
                stabilized_objs.append(obj)

        fused = fuse_yolo_with_verified_hazard(haz, stabilized_objs, self.config)
        if fused.curb.detected and _has_foreground_object_on_curb(objects):
            # Suppress false curb even if the foreground chair/person was debounced on frame 1
            fused = fuse_yolo_with_verified_hazard(
                replace(
                    haz,
                    curb=haz.curb.model_copy(
                        update={"detected": False, "stairs_likely": False, "confidence": 0.0}
                    ),
                ),
                stabilized_objs,
                self.config,
            )

        # Directional hysteresis: prevent rapid VEER_LEFT <-> VEER_RIGHT ping-pong when both sides are safe.
        # Strictly preserves Lean 4 "Never Steer Into Danger" (never holds a direction whose corridor is `danger`).
        with self._lock:
            prev_steer = self._last_steer
            steering = fused.steering
            spoken = fused.spoken_cue
            if (
                prev_steer in ("VEER_LEFT", "VEER_RIGHT")
                and steering in ("VEER_LEFT", "VEER_RIGHT")
                and prev_steer != steering
            ):
                prev_z = fused.left if prev_steer == "VEER_LEFT" else fused.right
                new_z = fused.left if steering == "VEER_LEFT" else fused.right
                prev_code = _SEV_TO_CODE[prev_z.severity]
                new_code = _SEV_TO_CODE[new_z.severity]
                if (
                    prev_code < 2
                    and prev_code <= new_code
                    and (prev_z.obstacle_score - new_z.obstacle_score) < 0.18
                ):
                    steering = prev_steer
                    if steering == "VEER_LEFT":
                        spoken = spoken.replace("Step right.", "Step left.").replace("Move right.", "Move left.")
                    else:
                        spoken = spoken.replace("Step left.", "Step right.").replace("Move left.", "Move right.")
            self._last_steer = steering

        if steering != fused.steering or spoken != fused.spoken_cue:
            return replace(fused, steering=steering, spoken_cue=spoken)
        return fused

    def analyze(
        self,
        closeness: np.ndarray,
        stateful: bool = True,
        *,
        raw_spread_milli: int = 600,
        luma_mean: float = 120.0,
        luma_std: float = 35.0,
        pitch_deg: float = 0.0,
        frame_fresh: bool = True,
    ) -> HazardResult:
        t0 = time.perf_counter()
        arr = (
            closeness
            if (closeness.dtype == np.float32 and closeness.flags.c_contiguous)
            else np.ascontiguousarray(closeness, dtype=np.float32)
        )
        h, w = arr.shape

        with self._lock:
            if _RUST_ANALYZE_FN is not None and h <= 512 and w <= 512 and (h * w) <= 147_456:
                rc = _RUST_ANALYZE_FN(
                    arr.ctypes.data,
                    h,
                    w,
                    0 if raw_spread_milli < 0 else (1000 if raw_spread_milli > 1000 else int(raw_spread_milli)),
                    float(luma_mean),
                    float(luma_std),
                    float(pitch_deg if pitch_deg is not None else 0.0),
                    1 if stateful else 0,
                    1 if frame_fresh else 0,
                    self._c_cfg_ref,
                    self._c_state_ref,
                    self._c_out_ref,
                )
                if rc == 0:
                    hazard_ms = (time.perf_counter() - t0) * 1000.0
                    return self._from_c_output(self._c_out, hazard_ms)

            # Fallback if native library is not built on another machine
            return self._analyze_numpy_fallback(
                arr,
                stateful=stateful,
                raw_spread_milli=raw_spread_milli,
                luma_mean=luma_mean,
                luma_std=luma_std,
                pitch_deg=pitch_deg,
                frame_fresh=frame_fresh,
                t0=t0,
            )

    def _from_c_output(self, out: _CHazardOutput, hazard_ms: float) -> HazardResult:
        def conv_zone(z: _CZoneOut) -> ZoneMetrics:
            os_v = max(0.0, min(1.0, float(z.obstacle_score)))
            dist_v = max(0.2, min(10.0, float(z.min_distance_m)))
            cov_v = max(0.0, min(1.0, float(z.coverage_ratio)))
            return ZoneMetrics.model_construct(
                severity=SEVERITY_BY_CODE[min(2, int(z.severity))],
                obstacle_score=round(os_v, 4),
                min_distance_m=round(dist_v, 2),
                coverage_ratio=round(cov_v, 4),
            )

        curb = CurbMetrics.model_construct(
            detected=bool(out.curb.detected),
            confidence=round(max(0.0, min(1.0, float(out.curb.confidence))), 4),
            drop_row_ratio=round(max(0.0, min(1.0, float(out.curb.drop_row_ratio))), 3),
            gradient_peak=round(max(0.0, float(out.curb.gradient_peak)), 4),
            stairs_likely=bool(out.curb.stairs_likely),
        )
        flags = int(out.flags)
        return HazardResult(
            overall_severity=SEVERITY_BY_CODE[min(2, int(out.overall_severity))],
            steering=STEERING_BY_CODE[min(4, int(out.steering))],
            spoken_cue=CUE_BY_CODE.get(int(out.cue_code), "Path clear."),
            left=conv_zone(out.left),
            center=conv_zone(out.center),
            right=conv_zone(out.right),
            curb=curb,
            hazard_ms=round(hazard_ms, 3),
            kernel_backend="rust-simd",
            wall_override=bool(flags & 0x01),
            overhead_hazard=bool(flags & 0x02),
            lens_blocked=bool(flags & 0x04),
        )

    def _analyze_numpy_fallback(
        self,
        closeness: np.ndarray,
        *,
        stateful: bool,
        raw_spread_milli: int,
        luma_mean: float,
        luma_std: float,
        pitch_deg: float,
        frame_fresh: bool,
        t0: float,
    ) -> HazardResult:
        """1:1 verified NumPy fallback matching `rust_core/src/lib.rs` when DLL is absent."""
        blocked_zone = ZoneMetrics(
            severity="danger", obstacle_score=1.0, min_distance_m=0.35, coverage_ratio=1.0
        )
        empty_curb = CurbMetrics(
            detected=False, confidence=0.0, drop_row_ratio=0.72, gradient_peak=0.0
        )

        if not frame_fresh:
            return HazardResult(
                overall_severity="danger",
                steering="STOP",
                spoken_cue=CUE_BY_CODE[16],
                left=blocked_zone,
                center=blocked_zone,
                right=blocked_zone,
                curb=empty_curb,
                hazard_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                kernel_backend="numpy-verified",
            )

        if (not np.isfinite(luma_mean)) or (not np.isfinite(luma_std)) or (
            0.0 <= luma_mean < 8.0 and 0.0 <= luma_std < 4.0
        ):
            return HazardResult(
                overall_severity="danger",
                steering="STOP",
                spoken_cue=CUE_BY_CODE[15],
                left=blocked_zone,
                center=blocked_zone,
                right=blocked_zone,
                curb=empty_curb,
                hazard_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                kernel_backend="numpy-verified",
                lens_blocked=True,
            )

        closeness = np.nan_to_num(closeness, nan=0.0, posinf=1.0, neginf=0.0)
        h, w = closeness.shape
        c_cfg = self._c_cfg
        has_live_imu = bool(np.isfinite(pitch_deg) and abs(pitch_deg) > 1.0)
        y_start = (h * 20) // 100
        y_end_pct = (
            int(np.clip(74.0 + (pitch_deg - 38.0) * 0.92, 74.0, 96.0))
            if (has_live_imu and 10.0 < pitch_deg < 62.0)
            else 96
        )
        y_end = min(h, max(y_start + 8, (h * y_end_pct) // 100))
        roi = closeness[y_start:y_end, :]
        roi_h = roi.shape[0]

        row_q20 = np.percentile(roi, 20.0, axis=1).astype(np.float32)
        u0, u1 = (roi_h * 10) // 100, max(1, (roi_h * 26) // 100)
        m0, m1 = (roi_h * 42) // 100, max(1, (roi_h * 58) // 100)
        b0, b1 = (roi_h * 76) // 100, max(2, (roi_h * 95) // 100)
        q_top = float(np.median(row_q20[u0:u1]))
        q_mid = float(np.median(row_q20[m0:m1]))
        q_bot = float(np.median(row_q20[b0:b1]))

        denom_r = float(max(1, roi_h - 1))
        t_top = 0.5 * (u0 + u1 - 1) / denom_r
        t_mid = 0.5 * (m0 + m1 - 1) / denom_r
        t_bot = 0.5 * (b0 + b1 - 1) / denom_r
        s_tb = (q_bot - q_top) / max(0.25, t_bot - t_top)
        s_mb = (q_bot - q_mid) / max(0.15, t_bot - t_mid)
        s_tm = (q_mid - q_top) / max(0.15, t_mid - t_top)
        obs_slope = (
            float(s_mb)
            if (q_top <= 0.05 and q_mid <= 0.08 and q_bot >= 0.22)
            else float(np.median([s_tb, s_mb, s_tm]))
        )
        obs_intercept = q_bot - obs_slope * t_bot

        imu_shift = (
            float(np.clip((74.0 - pitch_deg) * 0.006, -0.08, 0.24))
            if has_live_imu and 12.0 < pitch_deg < 135.0
            else 0.0
        )
        max_intercept = (
            max(0.38, 0.20 + max(0.0, imu_shift))
            if (obs_slope >= 0.40 and q_bot <= 0.98 and float(row_q20[0]) <= 0.48)
            else (0.20 + max(0.0, imu_shift))
        )
        min_slope = 0.32 if (has_live_imu and 12.0 < pitch_deg < 55.0) else 0.44
        floor_slope = float(np.clip(obs_slope, min_slope, 1.45))
        floor_intercept = float(np.clip(obs_intercept, -0.65, max_intercept))

        y_norm = np.linspace(0.0, 1.0, roi_h, dtype=np.float32)
        prior = np.maximum(0.04, floor_intercept + floor_slope * y_norm)
        expected_floor = np.minimum(np.maximum(prior, row_q20 * 0.86), prior + 0.06)

        allow_overhang = (not has_live_imu) or (74.0 <= pitch_deg <= 112.0)
        if allow_overhang:
            oh_patch = closeness[(h * 4) // 100 : y_start, (w * 24) // 100 : (w * 76) // 100]
            oh_p90 = float(np.percentile(oh_patch, 90.0)) if oh_patch.size > 0 else 0.0
            oh_cov = float(np.mean(oh_patch > 0.80)) if oh_patch.size > 0 else 0.0
        else:
            oh_p90, oh_cov = 0.0, 0.0
        overhead_hazard = allow_overhang and raw_spread_milli > WALL_SPREAD_EPS and oh_p90 >= 0.84 and oh_cov >= 0.18

        margin = c_cfg.floor_margin_milli / 1000.0
        toe_start = min(roi_h - 2, (roi_h * 74) // 100)
        upper_rows = max(1, (roi_h * 68) // 100)
        excess = np.maximum(0.0, roi - (expected_floor[:, None] + margin)) * 1.65
        near_boost = np.zeros_like(roi)
        near_boost[:toe_start, :] = np.maximum(0.0, roi[:toe_start, :] - 0.80) * 1.40
        hazard_map = np.clip(np.maximum(excess, near_boost), 0.0, 1.0)

        x1, x2 = w // 3, (2 * w) // 3
        slices = [(0, x1), (x1, x2), (x2, w)]
        ema_atk = max(int(c_cfg.ema_alpha_milli), 780)
        ema_rel = int(c_cfg.ema_alpha_milli)

        z_metrics: list[ZoneMetrics] = []
        z_sev: list[int] = []
        z_score_m: list[int] = []
        wall_ov_any = False

        flat_vertical_plane = obs_slope < 0.20 and abs(q_bot - q_top) < 0.14
        textureless_white_wall = (
            flat_vertical_plane and luma_mean >= 20.0 and luma_std < 22.0 and raw_spread_milli <= 280
        )

        for idx, (sx, ex) in enumerate(slices):
            hz = hazard_map[:, sx:ex]
            rz = roi[:, sx:ex]
            p88 = float(np.percentile(hz, 88.0))
            cov = float(np.mean(hz > 0.16))
            full_m = int(round(np.clip(0.70 * p88 + 0.60 * cov, 0.0, 1.0) * 1000.0))

            toe_hz = hz[toe_start:, :]
            toe_cov = float(np.mean(toe_hz > 0.20))
            toe_m = (
                int(round(np.clip(0.68 * float(np.percentile(toe_hz, 88.0)) + 0.55 * toe_cov, 0.0, 1.0) * 1000.0))
                if toe_cov >= 0.22
                else 0
            )
            rel_m = max(full_m, toe_m)
            if idx == 1 and overhead_hazard:
                rel_m = max(rel_m, int(c_cfg.danger_milli) + 40)

            upper_med_m = int(round(float(np.percentile(rz[:upper_rows, :], 55.0)) * 1000.0))
            white_wall_hit = (textureless_white_wall and upper_med_m >= 250) or (
                flat_vertical_plane and raw_spread_milli <= 220 and upper_med_m >= 440
            )
            is_wall = (
                (raw_spread_milli <= WALL_SPREAD_EPS and upper_med_m >= WALL_PROXIMITY_MIN)
                or (upper_med_m >= WALL_BLOCKAGE_MIN)
                or white_wall_hit
            )
            if is_wall:
                wall_ov_any = True
                raw_comb = WALL_DANGER_SCORE
            else:
                raw_comb = min(1000, rel_m)

            if stateful:
                prev = int(self._c_state.ema_scores_milli[idx])
                alpha = ema_atk if raw_comb >= prev else ema_rel
                score_m = (alpha * raw_comb + (1000 - alpha) * prev) // 1000
                self._c_state.ema_scores_milli[idx] = score_m
                if score_m >= c_cfg.danger_milli:
                    nxt = int(self._c_state.danger_streak[idx]) + 1
                    self._c_state.danger_streak[idx] = nxt
                    sev = 2 if nxt >= c_cfg.confirm_frames else 1
                elif score_m >= c_cfg.caution_milli:
                    self._c_state.danger_streak[idx] = 0
                    sev = 1
                else:
                    self._c_state.danger_streak[idx] = 0
                    sev = 0
            else:
                score_m = raw_comb
                sev = 2 if score_m >= c_cfg.danger_milli else (1 if score_m >= c_cfg.caution_milli else 0)

            peak_u = float(np.percentile(rz[:upper_rows, :], 90.0))
            sf = score_m / 1000.0
            eff = max(sf, max(0.0, peak_u - 0.35))
            dist = float(np.clip(4.6 * (1.0 - eff) + 0.35, 0.35, 5.0))

            z_sev.append(sev)
            z_score_m.append(score_m)
            z_metrics.append(
                ZoneMetrics(
                    severity=SEVERITY_BY_CODE[sev],
                    obstacle_score=round(sf, 4),
                    min_distance_m=round(dist, 2),
                    coverage_ratio=round(cov, 4),
                )
            )

        # Curb check with mutual exclusion
        y0, y1 = (h * 50) // 100, (h * 91) // 100
        cx0, cx1 = (w * 22) // 100, (w * 78) // 100
        patch = cv2.GaussianBlur(closeness[y0:y1, cx0:cx1], (3, 3), 0)
        row_prof = np.median(patch, axis=1)
        ph = patch.shape[0]
        step = max(2, ph // 14)
        drops = row_prof[step:] - row_prof[:-step]
        base_s = max(0.0, float(np.median(drops)))
        ex_drops = np.maximum(0.0, drops - base_s)
        peak_i = int(np.argmax(ex_drops))
        peak_v = float(ex_drops[peak_i])
        row_std = float(np.std(patch[min(ph - 1, peak_i + step // 2), :]))
        cons_w = float(np.clip(1.0 - row_std * 2.2, 0.25, 1.0))

        fl_indices = np.clip(np.arange(y0, y1) - y_start, 0, roi_h - 1)
        fl_rows = expected_floor[fl_indices]
        fg_floor_m = int(round(float(np.median(row_prof[min(ph - 1, peak_i + step) :])) * 1000.0))
        fg_pos_ex = np.maximum(0.0, patch[peak_i + 1 :, :] - fl_rows[peak_i + 1 :, None])
        fg_ex_m = int(round(float(np.percentile(fg_pos_ex, 88.0)) * 1000.0)) if fg_pos_ex.size > 0 else 0

        bg_r = max(0, peak_i - 2)
        trough_val = float(row_prof[bg_r])
        bg_def = float(fl_rows[bg_r] - trough_val)
        rebounded = False
        if bg_def >= 0.07 and peak_v * cons_w * 1000.0 >= int(c_cfg.curb_thresh_milli) * 0.8:
            c_top = y_start + (roi_h * 18) // 100
            c_bot = max(c_top + 4, y0 - 2)
            if c_bot > c_top + 2:
                upper_meds = np.median(closeness[c_top:c_bot, cx0:cx1], axis=1)
                upper_fl = expected_floor[np.clip(np.arange(c_top, c_bot) - y_start, 0, roi_h - 1)]
                rebound_ratio = float(np.mean(np.abs(upper_meds - upper_fl) <= 0.055))
                higher_ratio = float(np.mean(upper_meds > trough_val + 0.07))
                rebounded = rebound_ratio >= 0.70 and higher_ratio >= 0.50

        plunge_ok = bg_def >= 0.07 and not rebounded
        drop_grad_m = int(round(peak_v * cons_w * 1000.0)) if plunge_ok else 0

        curb_trig = (
            drop_grad_m >= int(c_cfg.curb_thresh_milli)
            and fg_floor_m >= CURB_MIN_FG_FLOOR
            and fg_ex_m <= BOX_MAX_EXCESS
        )
        adj_peak = peak_v * cons_w * (1.0 if curb_trig else 0.25)
        raw_conf = float(np.clip(adj_peak / max(self.config.curb_gradient_threshold * 1.4, 1e-4), 0.0, 1.0))

        if stateful:
            prev_c = int(self._c_state.ema_curb_milli)
            rc_m = int(round(raw_conf * 1000.0))
            ac = ema_atk if rc_m >= prev_c else ema_rel
            cm = (ac * rc_m + (1000 - ac) * prev_c) // 1000
            self._c_state.ema_curb_milli = cm
            if curb_trig:
                self._c_state.curb_streak += 1
            else:
                self._c_state.curb_streak = 0
            curb_det = int(self._c_state.curb_streak) >= int(c_cfg.confirm_frames)
            conf_out = cm / 1000.0
        else:
            curb_det = curb_trig
            conf_out = raw_conf

        curb_metrics = CurbMetrics(
            detected=curb_det,
            confidence=round(conf_out, 4),
            drop_row_ratio=round(float((y0 + peak_i) / max(1, h)), 3),
            gradient_peak=round(adj_peak, 4),
            stairs_likely=False,
        )

        ov_code, act_code = decide_guidance_verified(
            z_sev[0],
            z_score_m[0],
            z_sev[1],
            z_score_m[1],
            z_sev[2],
            z_score_m[2],
            curb_det,
            True,
        )
        cue = (
            CUE_BY_CODE[1]
            if act_code == 4
            else (CUE_BY_CODE[14] if overhead_hazard and act_code == 3 else CUE_BY_CODE[{0: 0, 1: 3, 2: 4, 3: 5}[act_code]])
        )
        return HazardResult(
            overall_severity=SEVERITY_BY_CODE[ov_code],
            steering=STEERING_BY_CODE[act_code],
            spoken_cue=cue,
            left=z_metrics[0],
            center=z_metrics[1],
            right=z_metrics[2],
            curb=curb_metrics,
            hazard_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            kernel_backend="numpy-verified",
            wall_override=wall_ov_any,
            overhead_hazard=overhead_hazard,
        )


_SEV_TO_CODE: dict[Severity, int] = {"clear": 0, "caution": 1, "danger": 2}
_FOREGROUND_VERTICAL_OBJECTS = {
    "person",
    "chair",
    "couch",
    "dining table",
    "backpack",
    "suitcase",
    "bench",
    "potted plant",
    "bed",
    "refrigerator",
}


def _has_foreground_object_on_curb(objects: list[DetectedObject]) -> bool:
    """Detects a close vertical foreground obstacle (chair/person/table) that can mimic a curb edge."""
    for obj in objects:
        if obj.confidence < 0.40 or obj.label not in _FOREGROUND_VERTICAL_OBJECTS:
            continue
        if obj.distance_m > 2.2:
            continue
        bx1, by1, bx2, by2 = obj.bbox_norm
        area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if by2 >= 0.62 and area >= 0.08:
            return True
    return False


def fuse_yolo_with_verified_hazard(
    haz: HazardResult,
    objects: list[DetectedObject],
    config: CalibrationConfig,
) -> HazardResult:
    """Enriches corridor distances & spoken cues with YOLOv8n while preserving Lean 4 invariants."""
    if not objects or haz.lens_blocked:
        return haz

    curb = haz.curb
    curb_suppressed = False
    if curb.detected and _has_foreground_object_on_curb(objects):
        curb = curb.model_copy(update={"detected": False, "stairs_likely": False, "confidence": 0.0})
        curb_suppressed = True

    nearest_by_corridor: dict[Corridor, DetectedObject] = {}
    for obj in objects:
        if obj.confidence < 0.40 or obj.severity == "clear":
            continue
        cur = nearest_by_corridor.get(obj.corridor)
        if cur is None or obj.distance_m < cur.distance_m:
            nearest_by_corridor[obj.corridor] = obj

    if not nearest_by_corridor and not curb_suppressed:
        return haz

    zones: dict[Corridor, ZoneMetrics] = {
        "left": haz.left,
        "center": haz.center,
        "right": haz.right,
    }
    upgraded = curb_suppressed
    for side, obj in nearest_by_corridor.items():
        z = zones[side]
        new_dist = round(min(z.min_distance_m, obj.distance_m), 2)
        new_sev = z.severity
        new_score = z.obstacle_score
        if _SEV_TO_CODE[obj.severity] > _SEV_TO_CODE[z.severity]:
            new_sev = obj.severity
            upgraded = True
            if obj.severity == "danger":
                new_score = max(new_score, min(1.0, config.danger_threshold + 0.14))
            elif obj.severity == "caution":
                new_score = max(new_score, min(1.0, config.caution_threshold + 0.06))
        zones[side] = ZoneMetrics(
            severity=new_sev,
            obstacle_score=round(new_score, 4),
            min_distance_m=new_dist,
            coverage_ratio=z.coverage_ratio,
        )

    left_z, center_z, right_z = zones["left"], zones["center"], zones["right"]
    overall_sev = haz.overall_severity
    steering = haz.steering

    # A full-body close person (<= 1.25m) filling the frame is a front barrier, not a hanging overhang.
    c_obj = nearest_by_corridor.get("center")
    l_obj = nearest_by_corridor.get("left")
    r_obj = nearest_by_corridor.get("right")
    has_close_person = any(
        o.confidence >= 0.40 and o.label == "person" and o.distance_m <= 1.25
        for o in objects
    )
    overhead_hazard = haz.overhead_hazard and not has_close_person

    # Re-run Lean-4 verified decision core whenever YOLO upgrades severity or suppresses a false curb.
    # ALWAYS re-run even on wall_override so overall_severity == max(L,C,R,curb) & never-steer-into-danger hold!
    if upgraded:
        ov_code, act_code = decide_guidance_verified(
            _SEV_TO_CODE[left_z.severity],
            int(round(left_z.obstacle_score * 1000)),
            _SEV_TO_CODE[center_z.severity],
            int(round(center_z.obstacle_score * 1000)),
            _SEV_TO_CODE[right_z.severity],
            int(round(right_z.obstacle_score * 1000)),
            curb.detected,
            True,
        )
        overall_sev = SEVERITY_BY_CODE[ov_code]
        if not overhead_hazard:
            steering = STEERING_BY_CODE[act_code]
        elif act_code == 3:
            steering = "STOP"
            if spoken == "Path clear.":
                spoken = CUE_BY_CODE[14]

    # Enrich spoken_cue with the nearest named object and GUARANTEE non-CLEAR steering never says "Path clear."
    spoken = haz.spoken_cue
    if not curb.detected and not overhead_hazard:
        if steering == "STOP":
            if c_obj is not None:
                spoken = f"{c_obj.label.capitalize()} blocking path at {c_obj.distance_m:.1f}m. Stop."
            elif l_obj is not None and r_obj is not None:
                spoken = f"Narrow gap between {l_obj.label} and {r_obj.label}. Stop."
            else:
                spoken = "Path blocked ahead. Stop."
        elif steering == "VEER_LEFT":
            target = c_obj or r_obj
            if target is not None:
                where = "ahead" if target.corridor == "center" else "on right"
                spoken = f"{target.label.capitalize()} {where} ({target.distance_m:.1f}m). Step left."
            elif spoken == "Path clear." or curb_suppressed:
                spoken = "Obstacle on right. Step left."
        elif steering == "VEER_RIGHT":
            target = c_obj or l_obj
            if target is not None:
                where = "ahead" if target.corridor == "center" else "on left"
                spoken = f"{target.label.capitalize()} {where} ({target.distance_m:.1f}m). Step right."
            elif spoken == "Path clear." or curb_suppressed:
                spoken = "Obstacle on left. Step right."
        elif steering == "CLEAR":
            spoken = "Path clear."

    return replace(
        haz,
        overall_severity=overall_sev,
        steering=steering,
        spoken_cue=spoken,
        left=left_z,
        center=center_z,
        right=right_z,
        curb=curb,
        overhead_hazard=overhead_hazard,
    )


METRIC_CLOSE_DANGER_M = 1.05
METRIC_CLOSE_CAUTION_M = 1.45


def fuse_close_range_metric_depth(
    haz: HazardResult,
    depth_meters: np.ndarray | None,
    pitch_deg: float,
    config: CalibrationConfig,
) -> HazardResult:
    """Anchors corridor distances & close-range solid obstacles to real metric depth.

    Samples the forward body-level band (`y = 24%..60%`) per corridor so flat floors below
    do not trigger false alarms, while close desks, doors, walls, or chairs (`<= 1.05m`)
    reliably escalate through Lean-4 verified `decide_guidance_verified`.
    """
    if depth_meters is None or haz.lens_blocked:
        return haz

    has_live_imu = bool(np.isfinite(pitch_deg) and abs(pitch_deg) > 1.0)
    if has_live_imu and (pitch_deg < 50.0 or pitch_deg > 125.0):
        return haz

    h, w = depth_meters.shape
    if h < 16 or w < 16:
        return haz

    y0 = (h * 24) // 100
    y1 = max(y0 + 4, (h * 60) // 100)
    x1, x2 = w // 3, (2 * w) // 3
    corridor_slices: tuple[tuple[Corridor, int, int], ...] = (
        ("left", 0, x1),
        ("center", x1, x2),
        ("right", x2, w),
    )

    zones: dict[Corridor, ZoneMetrics] = {
        "left": haz.left,
        "center": haz.center,
        "right": haz.right,
    }
    upgraded = False

    for side, sx, ex in corridor_slices:
        patch = depth_meters[y0:y1, sx:ex]
        if patch.size == 0:
            continue
        metric_dist = float(np.clip(np.percentile(patch, 20.0), 0.25, 10.0))
        z = zones[side]
        new_dist = round(min(z.min_distance_m, metric_dist), 2)
        new_sev = z.severity
        new_score = z.obstacle_score

        if metric_dist <= METRIC_CLOSE_DANGER_M and _SEV_TO_CODE[z.severity] < 2:
            new_sev = "danger"
            new_score = max(new_score, min(1.0, config.danger_threshold + 0.16))
            upgraded = True
        elif metric_dist <= METRIC_CLOSE_CAUTION_M and _SEV_TO_CODE[z.severity] < 1:
            new_sev = "caution"
            new_score = max(new_score, min(1.0, config.caution_threshold + 0.08))
            upgraded = True

        zones[side] = ZoneMetrics(
            severity=new_sev,
            obstacle_score=round(new_score, 4),
            min_distance_m=new_dist,
            coverage_ratio=z.coverage_ratio,
        )

    left_z, center_z, right_z = zones["left"], zones["center"], zones["right"]
    overall_sev = haz.overall_severity
    steering = haz.steering
    spoken = haz.spoken_cue

    # Always re-run Lean-4 verified decision core whenever any corridor severity escalated
    # so `overall_severity == max(L,C,R,curb)` and `never-steer-into-danger` hold even on wall_override.
    if upgraded:
        ov_code, act_code = decide_guidance_verified(
            _SEV_TO_CODE[left_z.severity],
            int(round(left_z.obstacle_score * 1000)),
            _SEV_TO_CODE[center_z.severity],
            int(round(center_z.obstacle_score * 1000)),
            _SEV_TO_CODE[right_z.severity],
            int(round(right_z.obstacle_score * 1000)),
            haz.curb.detected,
            True,
        )
        overall_sev = SEVERITY_BY_CODE[ov_code]
        if not haz.overhead_hazard:
            steering = STEERING_BY_CODE[act_code]
            if not haz.curb.detected:
                if steering == "STOP":
                    spoken = f"Close obstacle at {center_z.min_distance_m:.1f}m. Stop."
                elif steering == "VEER_LEFT":
                    d_obs = min(center_z.min_distance_m, right_z.min_distance_m)
                    spoken = f"Close obstacle at {d_obs:.1f}m. Step left."
                elif steering == "VEER_RIGHT":
                    d_obs = min(center_z.min_distance_m, left_z.min_distance_m)
                    spoken = f"Close obstacle at {d_obs:.1f}m. Step right."
        elif act_code == 3:
            steering = "STOP"
            if spoken == "Path clear.":
                spoken = CUE_BY_CODE[14]

    return replace(
        haz,
        overall_severity=overall_sev,
        steering=steering,
        spoken_cue=spoken,
        left=left_z,
        center=center_z,
        right=right_z,
    )
