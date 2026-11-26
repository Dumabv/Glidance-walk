"""Self-diagnostic evaluator for recorded Glidance walking videos.

Inspects per-frame guidance timelines, alert segments, latency stats, and YOLO/Depth
agreement to automatically surface actionable bugs and calibration recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DiagnosticFinding:
    severity: str  # "CRITICAL" | "WARN" | "INFO" | "OK"
    code: str
    title: str
    evidence: str
    fix_hint: str
    timestamps: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "evidence": self.evidence,
            "fix_hint": self.fix_hint,
            "timestamps": self.timestamps,
        }


_CLUTTER_CLASSES = {"laptop", "bottle", "tv", "cup", "cell phone", "keyboard", "mouse"}
_VERTICAL_OBSTACLE_CLASSES = {"person", "chair", "couch", "dining table", "backpack", "suitcase", "bench"}


def diagnose_video_run(summary: dict[str, Any]) -> dict[str, Any]:
    """Runs automated health & calibration checks over an `analyze_video` summary."""
    timeline: list[dict[str, Any]] = summary.get("timeline", [])
    segments: list[dict[str, Any]] = summary.get("alert_segments", [])
    n = max(1, len(timeline))
    duration_sec = timeline[-1]["time_sec"] if timeline else 0.0
    minutes = max(duration_sec / 60.0, 0.25)

    findings: list[DiagnosticFinding] = []

    # 1. Safety cue contradiction: non-CLEAR steering speaking "Path clear."
    cue_mismatches = [
        f["time_sec"]
        for f in timeline
        if f["steering"] != "CLEAR" and "path clear" in str(f.get("spoken_cue", "")).lower()
    ]
    if cue_mismatches:
        findings.append(
            DiagnosticFinding(
                severity="CRITICAL",
                code="CUE_CONTRADICTION",
                title="Non-CLEAR steering still spoke 'Path clear.'",
                evidence=f"{len(cue_mismatches)} frame(s) contradicted visual/haptic steering.",
                fix_hint="Update `fuse_yolo_with_verified_hazard()` in `backend/vision/hazard_detector.py` so every STOP/VEER branch writes a matching spoken_cue.",
                timestamps=cue_mismatches[:10],
            )
        )

    # 2. Single-frame flicker rate (temporal instability)
    one_frame_segs = [s for s in segments if s.get("frames", 0) == 1]
    flicker_ratio = len(one_frame_segs) / max(1, len(segments))
    if len(one_frame_segs) >= 5 and flicker_ratio >= 0.25:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="SINGLE_FRAME_FLICKER",
                title="High single-frame alert flicker",
                evidence=f"{len(one_frame_segs)}/{len(segments)} alert segments ({flicker_ratio*100:.0f}%) lasted only 1 frame.",
                fix_hint="Increase `confirm_frames` (e.g. 2 -> 3) or lower `ema_alpha` in `CalibrationConfig` / verify `detector.fuse_objects(..., stateful=True)` is used.",
                timestamps=[round(s["start_sec"], 2) for s in one_frame_segs[:10]],
            )
        )

    # 3. Adjacent VEER_LEFT <-> VEER_RIGHT ping-pong oscillation
    reversals = [
        b["time_sec"]
        for a, b in zip(timeline, timeline[1:])
        if {a["steering"], b["steering"]} == {"VEER_LEFT", "VEER_RIGHT"}
    ]
    reversals_per_min = len(reversals) / minutes
    if len(reversals) >= 4 and reversals_per_min >= 4.0:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="STEERING_PING_PONG",
                title="Rapid LEFT <-> RIGHT steering oscillation",
                evidence=f"{len(reversals)} instant direction flips ({reversals_per_min:.1f}/min).",
                fix_hint="Increase directional deadband in `GroundAwareHazardDetector.fuse_objects()` (e.g. 0.08 -> 0.12) or smooth `ema_alpha`.",
                timestamps=reversals[:10],
            )
        )

    # 4. Suspicious CURB_DROP overlapping foreground vertical YOLO objects
    suspicious_curbs = []
    for f in timeline:
        if f["steering"] == "CURB_DROP" or f.get("curb_detected"):
            fg_objs = [
                o for o in f.get("objects", [])
                if o.get("label") in _VERTICAL_OBSTACLE_CLASSES and float(o.get("distance_m", 9.0)) <= 2.2
            ]
            if fg_objs:
                suspicious_curbs.append(f["time_sec"])
    if suspicious_curbs:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="CURB_OBJECT_COLLISION",
                title="CURB_DROP fired while foreground chair/person was detected",
                evidence=f"{len(suspicious_curbs)} curb frame(s) overlapped close vertical obstacles (likely chair/jacket edge).",
                fix_hint="Raise `curb_gradient_threshold` (e.g. +0.02) or widen `_has_foreground_object_on_curb()` suppression in `backend/vision/hazard_detector.py`.",
                timestamps=suspicious_curbs[:10],
            )
        )

    # 5. Overhead hazard vs close full-body person conflict
    overhead_conflicts = [
        f["time_sec"]
        for f in timeline
        if "overhead" in str(f.get("spoken_cue", "")).lower()
        and any(o.get("label") == "person" and float(o.get("distance_m", 9.0)) <= 1.2 for o in f.get("objects", []))
    ]
    if overhead_conflicts:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="OVERHEAD_VS_PERSON",
                title="'Low overhead obstacle' triggered on close full-body person",
                evidence=f"{len(overhead_conflicts)} frame(s) labeled a close person as an overhead overhang.",
                fix_hint="Suppress `overhead_hazard` in `fuse_yolo_with_verified_hazard()` when a close foreground person fills center.",
                timestamps=overhead_conflicts[:10],
            )
        )

    # 6. Tabletop/wall clutter triggering alerts
    clutter_hits = [
        f["time_sec"]
        for f in timeline
        if f["steering"] != "CLEAR"
        and any(o.get("label") in _CLUTTER_CLASSES for o in f.get("objects", []))
    ]
    if clutter_hits:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="CLUTTER_FALSE_ALARM",
                title="Non-floor clutter (laptop/bottle/tv) triggered walking alerts",
                evidence=f"{len(clutter_hits)} alert frame(s) involved tabletop/wall clutter classes.",
                fix_hint="Remove non-floor classes from `WALKING_OBSTACLE_CLASSES` in `backend/vision/object_detector.py`.",
                timestamps=clutter_hits[:10],
            )
        )

    # 7. Danger / STOP fatigue check
    danger_ratio = summary.get("severity_counts", {}).get("danger", 0) / n
    if danger_ratio > 0.55:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="HIGH_DANGER_FATIGUE",
                title="Excessive DANGER alert rate (possible over-sensitivity)",
                evidence=f"{danger_ratio*100:.0f}% of analyzed frames were rated DANGER.",
                fix_hint="Raise `danger_threshold` by +0.05 in `CalibrationConfig` or increase `DANGER_OBJECT_M` tightness in `object_detector.py`.",
                timestamps=[],
            )
        )

    # 8. Latency budget & native Rust kernel check
    p95_ms = float(summary.get("p95_ms", 0.0))
    if p95_ms > 90.0:
        findings.append(
            DiagnosticFinding(
                severity="WARN",
                code="LATENCY_OVER_BUDGET",
                title="P95 latency exceeded 90 ms real-time budget",
                evidence=f"P95={p95_ms:.1f}ms on {summary.get('device', 'unknown')}.",
                fix_hint="Lower `--max-side` (e.g. 280), ensure CUDA/MPS FP16 is active, or build native `rust_core`.",
                timestamps=[],
            )
        )
    if summary.get("kernel_backend") != "rust-simd":
        findings.append(
            DiagnosticFinding(
                severity="INFO",
                code="NUMPY_FALLBACK_ACTIVE",
                title="Running verified NumPy fallback (native Rust library not built on this host)",
                evidence=f"kernel_backend={summary.get('kernel_backend')}. Math is 1:1 verified identical, ~1ms vs ~0.05ms.",
                fix_hint="Run `scripts/build_rust.sh` (macOS) or `scripts/build_rust.ps1` (Windows) for native SIMD speed.",
                timestamps=[],
            )
        )

    # Compute overall health score (0-100)
    score = 100
    for item in findings:
        if item.severity == "CRITICAL":
            score -= 30
        elif item.severity == "WARN":
            score -= 12
        elif item.severity == "INFO":
            score -= 2
    score = max(0, score)

    if not any(f.severity in ("CRITICAL", "WARN") for f in findings):
        findings.insert(
            0,
            DiagnosticFinding(
                severity="OK",
                code="HEALTHY_PIPELINE",
                title="No critical or warning anomalies detected!",
                evidence=f"P95={p95_ms:.1f}ms, flicker={flicker_ratio*100:.0f}%, reversals={len(reversals)}.",
                fix_hint="Ready for BenchmarkLab keyframe extraction and live stage demo.",
                timestamps=[],
            ),
        )

    return {
        "health_score": score,
        "metrics": {
            "flicker_segments": len(one_frame_segs),
            "flicker_ratio": round(flicker_ratio, 3),
            "adjacent_reversals": len(reversals),
            "cue_contradictions": len(cue_mismatches),
            "suspicious_curbs": len(suspicious_curbs),
            "overhead_person_conflicts": len(overhead_conflicts),
            "danger_ratio": round(danger_ratio, 3),
        },
        "findings": [f.to_dict() for f in findings],
    }
