from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class NaiveResult:
    obstacle_detected: bool
    center_score: float
    elapsed_ms: float


class NaiveCenterThresholdDetector:
    """Baseline detector: checks raw center-box depth without floor-slope or curb math.

    Included for midterm benchmarking to demonstrate why naive proximity thresholds
    suffer high false alarms on tilted floors and 0% recall on downward curbs/stairs.
    """

    def __init__(self, threshold: float = 0.56) -> None:
        self.threshold = threshold

    def analyze(self, closeness: np.ndarray) -> NaiveResult:
        t0 = time.perf_counter()
        h, w = closeness.shape
        patch = closeness[int(h * 0.35) : int(h * 0.88), int(w * 0.22) : int(w * 0.78)]
        score = float(np.percentile(patch, 85.0))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return NaiveResult(
            obstacle_detected=score >= self.threshold,
            center_score=round(score, 4),
            elapsed_ms=round(elapsed_ms, 3),
        )
