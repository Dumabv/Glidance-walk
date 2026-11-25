from __future__ import annotations

import numpy as np

from backend.evaluation.dataset_store import (
    DatasetStore,
    build_synthetic_calibration_suite,
)
from backend.schemas import (
    BenchmarkReport,
    CalibrationConfig,
    EngineBenchmarkMetrics,
    PerSampleComparison,
    SampleLabel,
)
from backend.vision.baseline_naive import NaiveCenterThresholdDetector
from backend.vision.depth_engine import DepthEngine
from backend.vision.hazard_detector import GroundAwareHazardDetector, HazardResult


def _is_smart_direction_correct(label: SampleLabel, res: HazardResult) -> bool:
    """Strict directional & safety evaluation (verifies we never steer into an obstacle or confuse boxes with curbs)."""
    if label == "clear":
        return res.overall_severity == "clear" and res.steering == "CLEAR"
    if label == "curb_drop":
        return res.curb.detected and res.steering == "CURB_DROP"
    if label == "obstacle_left":
        return res.left.severity in ("caution", "danger") and res.steering not in (
            "VEER_LEFT",
            "CURB_DROP",
        )
    if label == "obstacle_center":
        return (
            res.center.severity in ("caution", "danger") or res.steering == "STOP"
        ) and res.steering != "CURB_DROP"
    if label == "obstacle_right":
        return res.right.severity in ("caution", "danger") and res.steering not in (
            "VEER_RIGHT",
            "CURB_DROP",
        )
    return False


def _compute_metrics(
    engine_name: str,
    truth_hazard: list[bool],
    pred_hazard: list[bool],
    labels: list[SampleLabel],
    curb_pred: list[bool],
    direction_correct: list[bool],
    latencies_ms: list[float],
) -> EngineBenchmarkMetrics:
    tp = sum(1 for t, p in zip(truth_hazard, pred_hazard) if t and p)
    fp = sum(1 for t, p in zip(truth_hazard, pred_hazard) if (not t) and p)
    fn = sum(1 for t, p in zip(truth_hazard, pred_hazard) if t and (not p))
    tn = sum(1 for t, p in zip(truth_hazard, pred_hazard) if (not t) and (not p))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    clear_total = fp + tn
    false_alarm_clear = fp / clear_total if clear_total > 0 else 0.0

    curb_indices = [i for i, lbl in enumerate(labels) if lbl == "curb_drop"]
    curb_hits = sum(1 for i in curb_indices if curb_pred[i])
    curb_recall = curb_hits / len(curb_indices) if curb_indices else 0.0

    dir_acc = (
        sum(1 for ok in direction_correct if ok) / len(direction_correct)
        if direction_correct
        else 0.0
    )

    p50 = float(np.percentile(latencies_ms, 50.0)) if latencies_ms else 0.0
    p95 = float(np.percentile(latencies_ms, 95.0)) if latencies_ms else 0.0

    return EngineBenchmarkMetrics(
        engine_name=engine_name,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_alarm_rate_clear=round(false_alarm_clear, 4),
        curb_recall=round(curb_recall, 4),
        direction_accuracy=round(dir_acc, 4),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
    )


def run_benchmark(
    depth_engine: DepthEngine,
    dataset_store: DatasetStore,
    config: CalibrationConfig,
    include_synthetic: bool = True,
) -> BenchmarkReport:
    """Benchmarks Naive Center Threshold vs Verified Glidance Rust/V-Disparity Engine."""
    naive_detector = NaiveCenterThresholdDetector()
    smart_detector = GroundAwareHazardDetector(config=config)

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

    comparisons: list[PerSampleComparison] = []

    real_records = dataset_store.list_samples(include_thumbnails=False)
    for rec in real_records:
        raw_bytes = dataset_store.get_image_bytes(rec.sample_id)
        if raw_bytes is None:
            continue
        dec = depth_engine.decode_frame(raw_bytes)
        pred = depth_engine.predict_rgb(dec.rgb, dec.luma_mean, dec.luma_std)

        n_res = naive_detector.analyze(pred.closeness)
        s_res = smart_detector.analyze(
            pred.closeness,
            stateful=False,
            raw_spread_milli=pred.raw_spread_milli,
            luma_mean=pred.luma_mean,
            luma_std=pred.luma_std,
        )

        is_haz = rec.label != "clear"
        n_total_ms = dec.decode_ms + pred.inference_ms + n_res.elapsed_ms
        s_total_ms = dec.decode_ms + pred.inference_ms + s_res.hazard_ms

        n_ok = (n_res.obstacle_detected == is_haz) and (rec.label != "curb_drop")
        s_detected = (s_res.overall_severity != "clear") or s_res.curb.detected
        s_ok = _is_smart_direction_correct(rec.label, s_res)

        labels.append(rec.label)
        truth_hazard.append(is_haz)

        naive_pred.append(n_res.obstacle_detected)
        naive_curb.append(False)
        naive_dir_ok.append(n_ok)
        naive_lat.append(n_total_ms)

        smart_pred.append(s_detected)
        smart_curb.append(s_res.curb.detected)
        smart_dir_ok.append(s_ok)
        smart_lat.append(s_total_ms)

        comparisons.append(
            PerSampleComparison(
                sample_id=rec.sample_id,
                label=rec.label,
                note=rec.note,
                source="real",
                naive_detected=n_res.obstacle_detected,
                naive_correct=n_ok,
                smart_steering=s_res.steering,
                smart_severity=s_res.overall_severity,
                smart_curb=s_res.curb.detected,
                smart_correct=s_ok,
                smart_ms=round(s_total_ms, 3),
            )
        )

    synthetic_count = 0
    if include_synthetic:
        for scene in build_synthetic_calibration_suite():
            synthetic_count += 1
            n_res = naive_detector.analyze(scene.closeness_map)
            s_res = smart_detector.analyze(
                scene.closeness_map,
                stateful=False,
                raw_spread_milli=scene.raw_spread_milli,
            )

            is_haz = scene.label != "clear"
            n_ok = (n_res.obstacle_detected == is_haz) and (scene.label != "curb_drop")
            s_detected = (s_res.overall_severity != "clear") or s_res.curb.detected
            s_ok = _is_smart_direction_correct(scene.label, s_res)

            labels.append(scene.label)
            truth_hazard.append(is_haz)

            naive_pred.append(n_res.obstacle_detected)
            naive_curb.append(False)
            naive_dir_ok.append(n_ok)
            naive_lat.append(n_res.elapsed_ms)

            smart_pred.append(s_detected)
            smart_curb.append(s_res.curb.detected)
            smart_dir_ok.append(s_ok)
            smart_lat.append(s_res.hazard_ms)

            comparisons.append(
                PerSampleComparison(
                    sample_id=scene.sample_id,
                    label=scene.label,
                    note=scene.note,
                    source="synthetic",
                    naive_detected=n_res.obstacle_detected,
                    naive_correct=n_ok,
                    smart_steering=s_res.steering,
                    smart_severity=s_res.overall_severity,
                    smart_curb=s_res.curb.detected,
                    smart_correct=s_ok,
                    smart_ms=round(s_res.hazard_ms, 3),
                )
            )

    naive_metrics = _compute_metrics(
        "Naive Center Threshold",
        truth_hazard,
        naive_pred,
        labels,
        naive_curb,
        naive_dir_ok,
        naive_lat,
    )
    smart_metrics = _compute_metrics(
        f"Glidance Verified ({smart_detector.kernel_backend})",
        truth_hazard,
        smart_pred,
        labels,
        smart_curb,
        smart_dir_ok,
        smart_lat,
    )

    return BenchmarkReport(
        total_samples=len(comparisons),
        real_samples=len(comparisons) - synthetic_count,
        synthetic_samples=synthetic_count,
        device=depth_engine.device,
        model_name=depth_engine.model_id,
        naive=naive_metrics,
        smart=smart_metrics,
        samples=comparisons,
    )
