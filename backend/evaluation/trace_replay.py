"""Fast deterministic CI replay evaluator for pre-baked Glidance depth + YOLO telemetry traces.

Replays a quantized `.npz` + `.json` walking trace through the real Rust SIMD (or NumPy fallback)
`GroundAwareHazardDetector`, `fuse_close_range_metric_depth`, `fuse_objects`, Lean 4 safety
invariant checks, Golden Timestamp Scenarios, and `diagnose_video_run` — with zero PyTorch/GPU
requirements (~0.2s for a 71-second walk).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from backend.evaluation.benchmark_runner import (
    _compute_metrics,
    _is_smart_direction_correct,
)
from backend.evaluation.video_diagnostics import diagnose_video_run
from backend.schemas import CalibrationConfig, DetectedObject, SampleLabel
from backend.vision.baseline_naive import NaiveCenterThresholdDetector
from backend.vision.hazard_detector import (
    GroundAwareHazardDetector,
    fuse_close_range_metric_depth,
    fuse_yolo_with_verified_hazard,
)

_SEV_RANK = {"clear": 0, "caution": 1, "danger": 2}


def format_mmss(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:04.1f}"


def _evaluate_scenario(
    scenario: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    start_s = float(scenario["start_sec"])
    end_s = float(scenario["end_sec"])
    window = [f for f in timeline if start_s <= f["time_sec"] <= end_s]
    time_range = f"{format_mmss(start_s)}–{format_mmss(end_s)}"

    if not window:
        return {
            "id": scenario["id"],
            "time_range": time_range,
            "title": scenario["title"],
            "description": scenario.get("description", ""),
            "passed": False,
            "actual_summary": "No frames in window",
            "reason": f"Trace contains 0 frames between {time_range}.",
        }

    steerings = [f["steering"] for f in window]
    severities = [f["severity"] for f in window]
    cues_lower = [str(f["spoken_cue"]).lower() for f in window]
    steer_counts = dict(Counter(steerings))
    failures: list[str] = []

    req_any_steer = scenario.get("require_any_steering")
    if req_any_steer and not any(s in req_any_steer for s in steerings):
        failures.append(
            f"Expected at least one of {req_any_steer}, but got {steer_counts} "
            f"(sample cue at {format_mmss(window[0]['time_sec'])}: '{window[0]['spoken_cue']}')."
        )

    forbid_steer = scenario.get("forbid_steering")
    if forbid_steer:
        bad = [f for f in window if f["steering"] in forbid_steer]
        if bad:
            first = bad[0]
            failures.append(
                f"Forbidden steering `{first['steering']}` triggered at {format_mmss(first['time_sec'])} "
                f"({len(bad)}/{len(window)} frames; cue: '{first['spoken_cue']}')."
            )

    forbid_cues = scenario.get("forbid_cue_substrings")
    if forbid_cues:
        for phrase in forbid_cues:
            pl = phrase.lower()
            bad = [f for f in window if pl in str(f["spoken_cue"]).lower()]
            if bad:
                first = bad[0]
                failures.append(
                    f"Forbidden cue '{phrase}' spoken at {format_mmss(first['time_sec'])} "
                    f"(steering=`{first['steering']}`, cue='{first['spoken_cue']}')."
                )

    req_cues = scenario.get("require_cue_substrings_any")
    if req_cues:
        if not any(any(p.lower() in c for p in req_cues) for c in cues_lower):
            sample_cues = sorted({f["spoken_cue"] for f in window})[:2]
            failures.append(
                f"Expected cue containing one of {req_cues}, but got {sample_cues}."
            )

    req_min_sev = scenario.get("require_min_severity")
    if req_min_sev:
        need_rank = _SEV_RANK[req_min_sev]
        max_rank = max(_SEV_RANK[s] for s in severities)
        if max_rank < need_rank:
            failures.append(
                f"Expected severity >= `{req_min_sev}`, but peak was `{max(severities, key=lambda s: _SEV_RANK[s])}`."
            )

    max_danger_ratio = scenario.get("max_danger_ratio")
    if max_danger_ratio is not None:
        danger_frames = [f for f in window if f["severity"] == "danger"]
        ratio = len(danger_frames) / len(window)
        if ratio > float(max_danger_ratio):
            first = danger_frames[0]
            failures.append(
                f"Too many DANGER alerts: {len(danger_frames)}/{len(window)} ({ratio*100:.0f}% > {float(max_danger_ratio)*100:.0f}%); "
                f"first at {format_mmss(first['time_sec'])} (`{first['steering']}`: '{first['spoken_cue']}')."
            )

    actual_str = ", ".join(f"{k}:{v}f" for k, v in steer_counts.items())
    return {
        "id": scenario["id"],
        "time_range": time_range,
        "title": scenario["title"],
        "description": scenario.get("description", ""),
        "passed": len(failures) == 0,
        "actual_summary": actual_str,
        "reason": " ".join(failures) if failures else "Matched expected behavior.",
    }


def replay_trace(
    npz_path: Path,
    meta_path: Path,
    *,
    config: CalibrationConfig | None = None,
    use_close_range_metric: bool = True,
) -> dict[str, Any]:
    """Replays a baked depth+YOLO trace through the real hazard detector and evaluates invariants."""
    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    closeness_u8: np.ndarray = data["closeness_u8"]
    depth_m_u8: np.ndarray = data["depth_m_u8"]
    raw_spreads: np.ndarray = data["raw_spread_milli"]
    luma_means: np.ndarray = data["luma_mean"]
    luma_stds: np.ndarray = data["luma_std"]
    times_sec: np.ndarray = data["time_sec"]
    objects_per_frame: list[list[dict[str, Any]]] = meta.get("objects_per_frame", [])
    is_metric = bool(meta.get("is_metric_model", True))

    cfg = config or CalibrationConfig()
    detector = GroundAwareHazardDetector(config=cfg)

    steering_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    latencies_ms: list[float] = []
    timeline: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    invariant_violations: list[dict[str, Any]] = []
    active_seg: dict[str, Any] | None = None

    t_wall_start = time.perf_counter()
    n_frames = int(closeness_u8.shape[0])

    for i in range(n_frames):
        closeness = closeness_u8[i].astype(np.float32) * (1.0 / 255.0)
        depth_m = (0.25 + depth_m_u8[i].astype(np.float32) * (5.75 / 255.0)).astype(np.float32)
        spread = int(raw_spreads[i])
        luma_m = float(luma_means[i])
        luma_s = float(luma_stds[i])
        ts_sec = float(times_sec[i])

        raw_objs = objects_per_frame[i] if i < len(objects_per_frame) else []
        objs = [DetectedObject.model_validate(o) for o in raw_objs]

        t0 = time.perf_counter()
        haz = detector.analyze(
            closeness,
            stateful=True,
            raw_spread_milli=spread,
            luma_mean=luma_m,
            luma_std=luma_s,
            frame_fresh=True,
        )
        if use_close_range_metric and is_metric:
            haz = fuse_close_range_metric_depth(haz, depth_m, pitch_deg=0.0, config=cfg)
        haz = detector.fuse_objects(haz, objs, stateful=True)
        frame_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(frame_ms)

        # Verify Lean 4 runtime safety invariants on every replayed frame
        expected_max = max(
            _SEV_RANK[haz.left.severity],
            _SEV_RANK[haz.center.severity],
            _SEV_RANK[haz.right.severity],
            2 if haz.curb.detected else 0,
        )
        if _SEV_RANK[haz.overall_severity] != expected_max:
            invariant_violations.append(
                {
                    "time_sec": round(ts_sec, 2),
                    "rule": "SEVERITY_DOMINANCE",
                    "detail": f"overall={haz.overall_severity} != max(L={haz.left.severity}, C={haz.center.severity}, R={haz.right.severity}, curb={haz.curb.detected})",
                }
            )
        if haz.steering == "VEER_LEFT" and haz.left.severity == "danger":
            invariant_violations.append(
                {
                    "time_sec": round(ts_sec, 2),
                    "rule": "NEVER_STEER_INTO_DANGER",
                    "detail": "Steered VEER_LEFT while left corridor is danger!",
                }
            )
        if haz.steering == "VEER_RIGHT" and haz.right.severity == "danger":
            invariant_violations.append(
                {
                    "time_sec": round(ts_sec, 2),
                    "rule": "NEVER_STEER_INTO_DANGER",
                    "detail": "Steered VEER_RIGHT while right corridor is danger!",
                }
            )
        if haz.steering != "CLEAR" and "path clear" in haz.spoken_cue.lower():
            invariant_violations.append(
                {
                    "time_sec": round(ts_sec, 2),
                    "rule": "CUE_CONTRADICTION",
                    "detail": f"steering={haz.steering} spoke '{haz.spoken_cue}'",
                }
            )

        steering_counts[haz.steering] += 1
        severity_counts[haz.overall_severity] += 1
        timeline.append(
            {
                "frame": i,
                "time_sec": round(ts_sec, 2),
                "severity": haz.overall_severity,
                "steering": haz.steering,
                "spoken_cue": haz.spoken_cue,
                "min_dist_m": {
                    "left": haz.left.min_distance_m,
                    "center": haz.center.min_distance_m,
                    "right": haz.right.min_distance_m,
                },
                "curb_detected": haz.curb.detected,
                "objects": raw_objs,
                "total_ms": round(frame_ms, 3),
            }
        )

        if haz.steering != "CLEAR":
            if active_seg is not None and active_seg["steering"] == haz.steering:
                active_seg["end_sec"] = ts_sec
                active_seg["frames"] += 1
            else:
                if active_seg is not None:
                    segments.append(active_seg)
                active_seg = {
                    "start_sec": ts_sec,
                    "end_sec": ts_sec,
                    "steering": haz.steering,
                    "severity": haz.overall_severity,
                    "cue": haz.spoken_cue,
                    "frames": 1,
                }
        elif active_seg is not None:
            segments.append(active_seg)
            active_seg = None

    if active_seg is not None:
        segments.append(active_seg)

    wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
    p50_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    summary: dict[str, Any] = {
        "trace_name": meta.get("trace_name", npz_path.stem),
        "source_video": meta.get("source_video", "unknown"),
        "duration_sec": float(meta.get("duration_sec", 0.0)),
        "processed_frames": n_frames,
        "wall_ms": round(wall_ms, 1),
        "kernel_backend": detector.kernel_backend,
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "steering_counts": dict(steering_counts),
        "severity_counts": dict(severity_counts),
        "alert_segments": segments,
        "timeline": timeline,
    }

    diagnostics = diagnose_video_run(summary)
    scenario_results = [
        _evaluate_scenario(sc, timeline) for sc in meta.get("scenarios", [])
    ]
    failed_scenarios = [r for r in scenario_results if not r["passed"]]
    critical_findings = [
        f for f in diagnostics["findings"] if f["severity"] == "CRITICAL"
    ]

    passed = (
        len(failed_scenarios) == 0
        and len(invariant_violations) == 0
        and len(critical_findings) == 0
    )

    return {
        "passed": passed,
        "trace_name": summary["trace_name"],
        "source_video": summary["source_video"],
        "duration_sec": summary["duration_sec"],
        "processed_frames": n_frames,
        "wall_ms": summary["wall_ms"],
        "kernel_backend": detector.kernel_backend,
        "p50_ms": summary["p50_ms"],
        "p95_ms": summary["p95_ms"],
        "steering_counts": summary["steering_counts"],
        "severity_counts": summary["severity_counts"],
        "invariant_violations": invariant_violations,
        "scenarios": scenario_results,
        "diagnostics": diagnostics,
    }


def replay_real_photos_trace(
    npz_path: Path,
    meta_path: Path,
    *,
    config: CalibrationConfig | None = None,
    use_close_range_metric: bool = True,
) -> dict[str, Any]:
    """Replays baked real classroom/hallway photos (stateless) for Naive vs Glidance F1 CI check."""
    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = meta.get("samples", [])
    is_metric = bool(meta.get("is_metric_model", True))

    cfg = config or CalibrationConfig()
    naive_detector = NaiveCenterThresholdDetector()
    smart_detector = GroundAwareHazardDetector(config=cfg)

    labels: list[SampleLabel] = []
    truth_hazard: list[bool] = []
    naive_pred: list[bool] = []
    naive_curb: list[bool] = []
    naive_dir_ok: list[bool] = []
    naive_lat: list[float] = []
    smart_pred: list[bool] = []
    smart_curb: list[bool] = []
    smart_dir_ok: list[bool] = []
    smart_lat: list[float] = []
    rows: list[dict[str, Any]] = []

    t_wall0 = time.perf_counter()
    for entry in samples:
        idx = int(entry["index"])
        label: SampleLabel = entry["label"]
        note: str = entry.get("note", entry["sample_id"])
        closeness = data[f"cl_{idx}"].astype(np.float32) * (1.0 / 255.0)
        depth_m = (0.25 + data[f"dm_{idx}"].astype(np.float32) * (5.75 / 255.0)).astype(np.float32)
        objs = [DetectedObject.model_validate(o) for o in entry.get("objects", [])]

        n_res = naive_detector.analyze(closeness)
        t0 = time.perf_counter()
        s_res = smart_detector.analyze(
            closeness,
            stateful=False,
            raw_spread_milli=int(entry.get("raw_spread_milli", 600)),
            luma_mean=float(entry.get("luma_mean", 120.0)),
            luma_std=float(entry.get("luma_std", 35.0)),
            frame_fresh=True,
        )
        if use_close_range_metric and is_metric:
            s_res = fuse_close_range_metric_depth(s_res, depth_m, pitch_deg=0.0, config=cfg)
        if objs:
            s_res = fuse_yolo_with_verified_hazard(s_res, objs, cfg)
        s_ms = (time.perf_counter() - t0) * 1000.0

        is_haz = label != "clear"
        n_ok = (n_res.obstacle_detected == is_haz) and (label != "curb_drop")
        s_detected = (s_res.overall_severity != "clear") or s_res.curb.detected
        s_ok = _is_smart_direction_correct(label, s_res)

        labels.append(label)
        truth_hazard.append(is_haz)
        naive_pred.append(n_res.obstacle_detected)
        naive_curb.append(False)
        naive_dir_ok.append(n_ok)
        naive_lat.append(n_res.elapsed_ms)

        smart_pred.append(s_detected)
        smart_curb.append(s_res.curb.detected)
        smart_dir_ok.append(s_ok)
        smart_lat.append(s_ms)

        rows.append(
            {
                "sample_id": entry["sample_id"],
                "label": label,
                "note": note,
                "naive_correct": n_ok,
                "smart_steering": s_res.steering,
                "smart_severity": s_res.overall_severity,
                "smart_cue": s_res.spoken_cue,
                "smart_correct": s_ok,
                "min_dist_m": {
                    "L": s_res.left.min_distance_m,
                    "C": s_res.center.min_distance_m,
                    "R": s_res.right.min_distance_m,
                },
            }
        )

    wall_ms = (time.perf_counter() - t_wall0) * 1000.0
    naive_m = _compute_metrics("Naive Center Threshold", truth_hazard, naive_pred, labels, naive_curb, naive_dir_ok, naive_lat)
    smart_m = _compute_metrics(f"Glidance ({smart_detector.kernel_backend})", truth_hazard, smart_pred, labels, smart_curb, smart_dir_ok, smart_lat)

    return {
        "sample_count": len(rows),
        "wall_ms": round(wall_ms, 1),
        "kernel_backend": smart_detector.kernel_backend,
        "naive": naive_m.model_dump(),
        "smart": smart_m.model_dump(),
        "passed_count": sum(1 for r in rows if r["smart_correct"]),
        "samples": rows,
    }


def format_markdown_summary(
    report: dict[str, Any],
    photo_report: dict[str, Any] | None = None,
) -> str:
    """Formats a human-readable Markdown report for GitHub Actions `$GITHUB_STEP_SUMMARY`."""
    status_badge = "PASS" if report["passed"] else "FAIL"
    diag = report["diagnostics"]
    lines: list[str] = [
        f"## Glidance Classroom Evaluation CI — `{status_badge}`",
        "",
        f"- **71s Walk Trace:** `{report['source_video']}` ({report['duration_sec']:.1f}s, {report['processed_frames']} frames replayed in **{report['wall_ms']:.0f} ms**, P95={report['p95_ms']:.2f} ms/frame)",
        f"- **Kernel Backend:** `{report['kernel_backend']}`",
        f"- **Self-Diagnostic Health Score:** **{diag['health_score']}/100**",
        f"- **Lean 4 Runtime Safety Invariants:** {'0 violations' if not report['invariant_violations'] else str(len(report['invariant_violations'])) + ' VIOLATION(S)'}",
        f"- **Steering Distribution:** `{report['steering_counts']}`",
    ]

    if photo_report is not None:
        nm = photo_report["naive"]
        sm = photo_report["smart"]
        lines.extend(
            [
                "",
                f"### 1. Real Labeled Classroom Photos ({photo_report['sample_count']} scenes, {photo_report['wall_ms']:.0f} ms)",
                "",
                "| Metric | Naive Baseline | Glidance v2 (`Depth + YOLO + Rust`) |",
                "| :--- | :---: | :---: |",
                f"| **Overall F1 Score** | {nm['f1']*100:.1f}% | **{sm['f1']*100:.1f}%** |",
                f"| **Directional Accuracy** | {nm['direction_accuracy']*100:.1f}% | **{sm['direction_accuracy']*100:.1f}%** ({photo_report['passed_count']}/{photo_report['sample_count']}) |",
                f"| **Clear-Floor False Alarms** | {nm['false_alarm_rate_clear']*100:.1f}% | **{sm['false_alarm_rate_clear']*100:.1f}%** |",
            ]
        )
        missed = [r for r in photo_report["samples"] if not r["smart_correct"]]
        if missed:
            lines.extend(["", "**Photos to inspect / tune:**", ""])
            for m in missed:
                lines.append(
                    f"- `{m['label']}` — {m['note']} → got **`{m['smart_steering']}`** (`{m['smart_severity']}`, `{m['smart_cue']}`, dist `{m['min_dist_m']}`)"
                )

    lines.extend(
        [
            "",
            "### 2. Golden Classroom Walk Scenarios (by Timestamp)",
            "",
            "| Timestamp | Situation | Result | Observed Steering | Notes / Failure Reason |",
            "| :--- | :--- | :---: | :--- | :--- |",
        ]
    )

    for sc in report["scenarios"]:
        badge = "PASS" if sc["passed"] else "**FAIL**"
        reason = sc["reason"].replace("|", "/")
        title = sc["title"].replace("|", "/")
        lines.append(
            f"| `{sc['time_range']}` | {title} | {badge} | `{sc['actual_summary']}` | {reason} |"
        )

    if report["invariant_violations"]:
        lines.extend(["", "### Safety Invariant Violations (Blocking)", ""])
        for v in report["invariant_violations"][:10]:
            lines.append(
                f"- **`{format_mmss(v['time_sec'])}`** (`{v['rule']}`): {v['detail']}"
            )

    lines.extend(["", "### 3. Automated Self-Diagnostics Advisor", ""])
    for item in diag["findings"]:
        ts_str = (
            f" (at {', '.join('`' + format_mmss(t) + '`' for t in item['timestamps'][:5])})"
            if item["timestamps"]
            else ""
        )
        lines.append(
            f"- **`[{item['severity']}]` {item['title']}**{ts_str}: {item['evidence']}  \n"
            f"  *Suggested fix:* {item['fix_hint']}"
        )

    return "\n".join(lines) + "\n"
