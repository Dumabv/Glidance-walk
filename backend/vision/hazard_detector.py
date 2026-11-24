from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from backend.schemas import (
    CalibrationConfig,
    CurbMetrics,
    Severity,
    SteeringAction,
    ZoneMetrics,
)


@dataclass
class HazardResult:
    overall_severity: Severity
    steering: SteeringAction
    spoken_cue: str
    left: ZoneMetrics
    center: ZoneMetrics
    right: ZoneMetrics
    curb: CurbMetrics
    hazard_ms: float


class GroundAwareHazardDetector:
    """Detects L/C/R obstacles and downward curbs while ignoring flat tilted floors.

    Key engineering ideas:
    1. Row-wise physical floor prior + conservative 20th-percentile envelope so flat
       floors do not trigger false alarms, while close walls cannot be subtracted away.
    2. Vertical depth-drop gradient scan in the lower walking strip to catch curbs/steps.
    3. Rolling EMA + multi-frame confirmation debounce to suppress single-frame flicker.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._ema_scores: dict[str, float] = {"left": 0.0, "center": 0.0, "right": 0.0}
        self._ema_curb: float = 0.0
        self._danger_streak: dict[str, int] = {"left": 0, "center": 0, "right": 0}
        self._curb_streak: int = 0

    def update_config(self, config: CalibrationConfig) -> None:
        self.config = config

    def reset(self) -> None:
        self._ema_scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        self._ema_curb = 0.0
        self._danger_streak = {"left": 0, "center": 0, "right": 0}
        self._curb_streak = 0

    def analyze(self, closeness: np.ndarray, stateful: bool = True) -> HazardResult:
        """Runs ground-aware hazard + curb analysis on a [0, 1] closeness map (1=closest)."""
        t0 = time.perf_counter()
        h, w = closeness.shape

        # Walking ROI: skip ceiling/sky (top 22%) and bottom shoe edge (bottom 6%)
        y_start = int(h * 0.22)
        y_end = int(h * 0.94)
        roi = closeness[y_start:y_end, :]
        roi_h = roi.shape[0]

        # 1. Expected flat-floor closeness profile (increases linearly from horizon to feet)
        y_norm = np.linspace(0.0, 1.0, roi_h, dtype=np.float32)
        physical_floor_prior = 0.08 + 0.62 * y_norm  # Horizon ~0.08 -> Feet ~0.70

        # Blend conservatively with observed 20th-percentile row floor (capped so full walls stay caught)
        observed_floor = np.percentile(roi, 20.0, axis=1).astype(np.float32)
        expected_floor = np.minimum(
            np.maximum(physical_floor_prior, observed_floor * 0.85),
            physical_floor_prior + 0.08,
        )

        # Positive obstacle excess map: pixels sticking closer than expected flat floor
        excess = np.maximum(
            0.0, roi - (expected_floor[:, None] + self.config.floor_margin)
        )

        # Near-field safety floor override: anything extremely close (>0.84) in the upper/mid ROI
        # guarantees walls filling 100% of frame still trigger DANGER
        mid_wall_boost = np.maximum(0.0, roi - 0.80) * 1.4
        hazard_map = np.clip(np.maximum(excess * 1.65, mid_wall_boost), 0.0, 1.0)

        # 2. Split into Left / Center / Right corridors
        x1 = w // 3
        x2 = (2 * w) // 3
        corridor_slices = {
            "left": (roi[:, :x1], hazard_map[:, :x1]),
            "center": (roi[:, x1:x2], hazard_map[:, x1:x2]),
            "right": (roi[:, x2:], hazard_map[:, x2:]),
        }

        zone_results: dict[str, ZoneMetrics] = {}
        for name, (raw_zone, haz_zone) in corridor_slices.items():
            # Focus obstacle scoring on upper/middle 80% of ROI where upright objects protrude
            upper_h = max(1, int(haz_zone.shape[0] * 0.82))
            haz_focus = haz_zone[:upper_h, :]
            raw_focus = raw_zone[:upper_h, :]

            # Robust top-15% excess score + area coverage
            p88 = float(np.percentile(haz_focus, 88.0))
            coverage = float(np.mean(haz_focus > 0.16))
            raw_score = float(np.clip(0.70 * p88 + 0.60 * coverage, 0.0, 1.0))

            if stateful:
                alpha = self.config.ema_alpha
                score = alpha * raw_score + (1.0 - alpha) * self._ema_scores[name]
                self._ema_scores[name] = score
            else:
                score = raw_score

            # Convert closeness to intuitive relative distance in meters (0.4m .. 5.0m)
            peak_closeness = float(np.percentile(raw_focus, 90.0))
            # Use excess-adjusted distance so empty floor reads > 2.5m
            effective_close = max(score, max(0.0, peak_closeness - 0.35))
            est_dist = float(np.clip(4.6 * (1.0 - effective_close) + 0.35, 0.35, 5.0))

            severity = self._classify_zone(name, score, stateful)
            zone_results[name] = ZoneMetrics(
                severity=severity,
                obstacle_score=round(score, 4),
                min_distance_m=round(est_dist, 2),
                coverage_ratio=round(coverage, 4),
            )

        # 3. Curb / downward step detection in lower-center walking path
        curb = self._detect_curb(closeness, stateful)

        # 4. Compute high-level steering guidance
        overall, steering, spoken = self._decide_guidance(
            zone_results["left"],
            zone_results["center"],
            zone_results["right"],
            curb,
        )

        hazard_ms = (time.perf_counter() - t0) * 1000.0
        return HazardResult(
            overall_severity=overall,
            steering=steering,
            spoken_cue=spoken,
            left=zone_results["left"],
            center=zone_results["center"],
            right=zone_results["right"],
            curb=curb,
            hazard_ms=round(hazard_ms, 3),
        )

    def _classify_zone(self, name: str, score: float, stateful: bool) -> Severity:
        if score >= self.config.danger_threshold:
            if not stateful:
                return "danger"
            self._danger_streak[name] += 1
            if self._danger_streak[name] >= self.config.confirm_frames:
                return "danger"
            return "caution"

        if stateful:
            self._danger_streak[name] = 0

        if score >= self.config.caution_threshold:
            return "caution"
        return "clear"

    def _detect_curb(self, closeness: np.ndarray, stateful: bool) -> CurbMetrics:
        h, w = closeness.shape
        y0 = int(h * 0.52)
        y1 = int(h * 0.90)
        x0 = int(w * 0.22)
        x1 = int(w * 0.78)

        patch = closeness[y0:y1, x0:x1]
        if patch.shape[0] < 12 or patch.shape[1] < 8:
            return CurbMetrics(
                detected=False, confidence=0.0, drop_row_ratio=0.72, gradient_peak=0.0
            )

        # Light blur suppresses sensor speckle while preserving sharp horizontal curb edges
        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        row_profile = np.median(blurred, axis=1)

        # Moving upward (ahead along the path), a curb drop causes closeness to plunge abruptly
        step = max(2, patch.shape[0] // 14)
        drops = row_profile[step:] - row_profile[:-step]
        baseline_slope = float(np.median(drops))
        excess_drops = np.maximum(0.0, drops - max(0.0, baseline_slope))

        peak_idx = int(np.argmax(excess_drops))
        peak_val = float(excess_drops[peak_idx])

        # Verify horizontal consistency along the suspected curb band (real curbs span across the path)
        band_y = peak_idx + step // 2
        row_std = float(np.std(blurred[min(band_y, blurred.shape[0] - 1), :]))
        consistency_weight = float(np.clip(1.0 - row_std * 2.2, 0.25, 1.0))

        # Ensure the lower foreground before the drop actually looks like a walking surface (not sky)
        foreground_floor = float(np.median(row_profile[-step:]))
        floor_gate = 1.0 if foreground_floor >= 0.35 else 0.3

        adjusted_peak = peak_val * consistency_weight * floor_gate
        raw_conf = float(
            np.clip(
                adjusted_peak / max(self.config.curb_gradient_threshold * 1.4, 1e-4),
                0.0,
                1.0,
            )
        )

        if stateful:
            alpha = self.config.ema_alpha
            conf = alpha * raw_conf + (1.0 - alpha) * self._ema_curb
            self._ema_curb = conf
        else:
            conf = raw_conf

        triggered = adjusted_peak >= self.config.curb_gradient_threshold
        if stateful:
            if triggered:
                self._curb_streak += 1
            else:
                self._curb_streak = 0
            detected = self._curb_streak >= self.config.confirm_frames
        else:
            detected = triggered

        drop_row_ratio = float((y0 + peak_idx) / max(h, 1))
        return CurbMetrics(
            detected=detected,
            confidence=round(conf, 4),
            drop_row_ratio=round(drop_row_ratio, 3),
            gradient_peak=round(adjusted_peak, 4),
        )

    def _decide_guidance(
        self,
        left: ZoneMetrics,
        center: ZoneMetrics,
        right: ZoneMetrics,
        curb: CurbMetrics,
    ) -> tuple[Severity, SteeringAction, str]:
        if curb.detected:
            return "danger", "CURB_DROP", "Curb or step down ahead. Slow down."

        if center.severity == "danger":
            if left.severity == "clear" and left.obstacle_score <= right.obstacle_score:
                return "danger", "VEER_LEFT", "Obstacle ahead. Step left."
            if right.severity == "clear":
                return "danger", "VEER_RIGHT", "Obstacle ahead. Step right."
            return "danger", "STOP", "Path blocked ahead. Stop."

        if center.severity == "caution":
            if left.obstacle_score + 0.05 < right.obstacle_score:
                return "caution", "VEER_LEFT", "Object ahead. Slight left."
            return "caution", "VEER_RIGHT", "Object ahead. Slight right."

        if left.severity == "danger" and right.severity != "danger":
            return "danger", "VEER_RIGHT", "Close obstacle on left. Move right."
        if right.severity == "danger" and left.severity != "danger":
            return "danger", "VEER_LEFT", "Close obstacle on right. Move left."
        if left.severity == "danger" and right.severity == "danger":
            return "danger", "STOP", "Narrow gap. Stop and check."

        if left.severity == "caution" and right.severity == "clear":
            return "caution", "VEER_RIGHT", "Obstacle on left."
        if right.severity == "caution" and left.severity == "clear":
            return "caution", "VEER_LEFT", "Obstacle on right."
        if left.severity == "caution" and right.severity == "caution":
            return "caution", "CLEAR", "Objects on both sides. Stay centered."

        return "clear", "CLEAR", "Path clear."
