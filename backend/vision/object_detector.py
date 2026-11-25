from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.schemas import Corridor, DetectedObject, Severity

# Walkable indoor & sidewalk obstacle classes from COCO (ignores wall clocks, ceiling lights, etc.)
WALKING_OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "dog",
    "cat",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "bottle",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "refrigerator",
}

DEFAULT_YOLO_WEIGHTS = os.getenv("GLIDANCE_YOLO_MODEL", "yolov8n.pt")
DANGER_OBJECT_M = 1.25
CAUTION_OBJECT_M = 2.25


def estimate_box_distance_m(
    depth_meters: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> float:
    """Computes median metric depth inside the inner 50% core of a bounding box.

    Using the center core instead of the full box prevents background walls/floors
    at the edges of the box from corrupting the object's true distance.
    """
    h, w = depth_meters.shape
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx1 = max(0, min(w - 1, int(round(x1 + 0.25 * bw))))
    cx2 = max(cx1 + 1, min(w, int(round(x2 - 0.25 * bw))))
    cy1 = max(0, min(h - 1, int(round(y1 + 0.25 * bh))))
    cy2 = max(cy1 + 1, min(h, int(round(y2 - 0.25 * bh))))

    core = depth_meters[cy1:cy2, cx1:cx2]
    if core.size == 0:
        return float(np.median(depth_meters))
    return float(np.clip(np.median(core), 0.25, 15.0))


def classify_object_severity(distance_m: float) -> Severity:
    if distance_m <= DANGER_OBJECT_M:
        return "danger"
    if distance_m <= CAUTION_OBJECT_M:
        return "caution"
    return "clear"


def corridor_from_center_x(cx_norm: float) -> Corridor:
    if cx_norm < 1.0 / 3.0:
        return "left"
    if cx_norm < 2.0 / 3.0:
        return "center"
    return "right"


@dataclass
class ObjectDetectionResult:
    objects: list[DetectedObject]
    yolo_ms: float


class SemanticObjectDetector:
    """Lightweight YOLOv8n obstacle detector fused with metric depth maps."""

    def __init__(
        self,
        weights_name: str = DEFAULT_YOLO_WEIGHTS,
        conf_threshold: float = 0.38,
        use_mock: bool = False,
    ) -> None:
        self.weights_name = weights_name
        self.conf_threshold = conf_threshold
        self.use_mock = use_mock
        self.device: str = "cpu"
        self._model = None
        self._lock = threading.Lock()

        if not use_mock:
            self._load_model()

    @property
    def is_active(self) -> bool:
        return (not self.use_mock) and (self._model is not None)

    def _load_model(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:
            self.use_mock = True
            self._model = None
            return

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        weight_path = models_dir / self.weights_name

        self._model = YOLO(str(weight_path))
        # Warm up on a blank frame so the first live stream frame stays fast
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            imgsz=320,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

    def detect(
        self,
        image: Image.Image | np.ndarray,
        depth_meters: np.ndarray,
        max_objects: int = 6,
    ) -> ObjectDetectionResult:
        t0 = time.perf_counter()
        if self.use_mock or self._model is None:
            return ObjectDetectionResult(objects=[], yolo_ms=0.0)

        rgb = image if isinstance(image, np.ndarray) else np.asarray(image)
        h, w = depth_meters.shape
        with self._lock:
            results = self._model.predict(
                rgb,
                imgsz=320,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

        detections: list[DetectedObject] = []
        if results:
            res = results[0]
            names = res.names or {}
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.detach().cpu().numpy()
                conf_list = boxes.conf.detach().cpu().numpy()
                cls_list = boxes.cls.detach().cpu().numpy()

                for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
                    label = str(names.get(int(cls_id), "object")).lower()
                    if label not in WALKING_OBSTACLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    # Skip tiny distant boxes (< 1.5% of image area)
                    area_ratio = ((x2 - x1) * (y2 - y1)) / max(float(w * h), 1.0)
                    if area_ratio < 0.015:
                        continue

                    dist_m = estimate_box_distance_m(depth_meters, x1, y1, x2, y2)
                    cx_norm = ((x1 + x2) * 0.5) / max(float(w), 1.0)
                    corridor = corridor_from_center_x(cx_norm)
                    severity = classify_object_severity(dist_m)

                    detections.append(
                        DetectedObject(
                            label=label,
                            confidence=round(float(conf), 3),
                            distance_m=round(dist_m, 2),
                            corridor=corridor,
                            bbox_norm=[
                                round(max(0.0, x1 / max(w, 1)), 4),
                                round(max(0.0, y1 / max(h, 1)), 4),
                                round(min(1.0, x2 / max(w, 1)), 4),
                                round(min(1.0, y2 / max(h, 1)), 4),
                            ],
                            severity=severity,
                        )
                    )

        detections.sort(key=lambda d: d.distance_m)
        yolo_ms = (time.perf_counter() - t0) * 1000.0
        return ObjectDetectionResult(
            objects=detections[:max_objects],
            yolo_ms=round(yolo_ms, 2),
        )
