from __future__ import annotations

import ast
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.schemas import Corridor, DetectedObject, Severity
from backend.vision.depth_engine import _register_nvidia_cuda_dlls

# Open-vocabulary walking obstacle classes for YOLOv8s-Worldv2 (includes stairs, step, curb, door, wall, pillar, box!)
WALKING_WORLD_CLASSES: list[str] = [
    "person",
    "stairs",
    "step",
    "curb",
    "door",
    "glass door",
    "wall",
    "pillar",
    "railing",
    "chair",
    "table",
    "desk",
    "bench",
    "couch",
    "box",
    "backpack",
    "suitcase",
    "trash can",
    "bicycle",
    "car",
    "dog",
    "pole",
]

STAIRS_CURB_LABELS = {"stairs", "step", "curb"}

# Superset of Open-Vocabulary + fine-tuned stairs/obstacles + COCO classes
WALKING_OBSTACLE_CLASSES = set(WALKING_WORLD_CLASSES) | {
    "manhole",
    "guard rail",
    "dustbin",
    "motorcycle",
    "bus",
    "truck",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "cat",
    "umbrella",
    "handbag",
    "potted plant",
    "bed",
    "dining table",
    "refrigerator",
}

DEFAULT_YOLO_WEIGHTS = os.getenv("GLIDANCE_YOLO_MODEL", "glidance_stairs_v1.onnx")
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
    """Obstacle & stairs detector supporting native ONNX CUDA (RTX), YOLO-World, and YOLOv8n."""

    def __init__(
        self,
        weights_name: str = DEFAULT_YOLO_WEIGHTS,
        conf_threshold: float = 0.32,
        use_mock: bool = False,
    ) -> None:
        self.weights_name = weights_name
        self.conf_threshold = conf_threshold
        self.use_mock = use_mock
        self.device: str = "cpu"
        self._model = None
        self._ort_session = None
        self._ort_input_name: str = ""
        self._ort_imgsz: int = 416
        self._ort_names: dict[int, str] = {}
        self._lock = threading.Lock()

        if not use_mock:
            self._load_model()

    @property
    def is_active(self) -> bool:
        return (not self.use_mock) and (self._model is not None or self._ort_session is not None)

    def _try_load_onnx(self, onnx_path: Path) -> bool:
        if not onnx_path.exists():
            return False
        try:
            _register_nvidia_cuda_dlls()
            import onnxruntime as ort

            avail = ort.get_available_providers()
            providers: list[str] = []
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            inp = sess.get_inputs()[0]
            shape = inp.shape
            imgsz = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) else 416
            meta = sess.get_modelmeta().custom_metadata_map or {}
            names: dict[int, str] = {}
            if "names" in meta:
                parsed = ast.literal_eval(meta["names"])
                if isinstance(parsed, dict):
                    names = {int(k): str(v).lower() for k, v in parsed.items()}
                elif isinstance(parsed, list):
                    names = {i: str(v).lower() for i, v in enumerate(parsed)}

            # Warmup
            dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
            sess.run(None, {inp.name: dummy})

            self._ort_session = sess
            self._ort_input_name = inp.name
            self._ort_imgsz = imgsz
            self._ort_names = names
            self.weights_name = onnx_path.name
            self.conf_threshold = max(self.conf_threshold, 0.45)
            active_p = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"
            self.device = "cuda-onnx" if "CUDA" in active_p else "cpu-onnx"
            return True
        except Exception:
            self._ort_session = None
            return False

    def _load_model(self) -> None:
        models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prefer fine-tuned ONNX model on RTX CUDA (fast ~4-6ms, high stairs recall)
        onnx_candidates: list[Path] = []
        if self.weights_name.lower().endswith(".onnx"):
            cand = Path(self.weights_name)
            onnx_candidates.append(cand if cand.is_absolute() else (models_dir / cand))
            onnx_candidates.append(Path(r"D:\glidance_models\runs\glidance_stairs_v1\weights\best.onnx"))
        for cand in onnx_candidates:
            if self._try_load_onnx(cand):
                return

        # 2. Fallback to Ultralytics PyTorch (YOLOv8s-Worldv2 open-vocab -> YOLOv8n)
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

        pt_name = (
            "yolov8s-worldv2.pt"
            if self.weights_name.lower().endswith(".onnx")
            else self.weights_name
        )
        weight_path = models_dir / pt_name
        self.weights_name = pt_name

        if "world" in pt_name.lower():
            try:
                from ultralytics import YOLOWorld

                world_model = YOLOWorld(str(weight_path))
                world_model.set_classes(WALKING_WORLD_CLASSES)
                self._model = world_model
                self.conf_threshold = min(self.conf_threshold, 0.28)
            except Exception:
                self.weights_name = "yolov8n.pt"
                self._model = YOLO(str(models_dir / self.weights_name))
        else:
            self._model = YOLO(str(weight_path))

        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            imgsz=320,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

    def _detect_onnx(
        self, rgb: np.ndarray, depth_meters: np.ndarray
    ) -> list[DetectedObject]:
        assert self._ort_session is not None
        h, w = rgb.shape[:2]
        dh, dw = depth_meters.shape[:2]
        sz = self._ort_imgsz

        # Letterbox resize preserving aspect ratio
        scale = min(sz / max(w, 1), sz / max(h, 1))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (sz - nw) // 2
        pad_y = (sz - nh) // 2
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) * (1.0 / 255.0)
        blob = blob[None, ...]

        with self._lock:
            out = self._ort_session.run(None, {self._ort_input_name: blob})[0]

        # Ultralytics YOLO11/YOLOv8 export shape: (1, 4 + nc, N) -> transpose to (N, 4 + nc)
        preds = out[0].T if out.ndim == 3 and out.shape[1] < out.shape[2] else out[0]
        if preds.shape[1] < 5:
            return []

        scores_mat = preds[:, 4:]
        cls_ids = np.argmax(scores_mat, axis=1)
        confs = scores_mat[np.arange(len(preds)), cls_ids]
        keep = confs >= self.conf_threshold
        if not np.any(keep):
            return []

        boxes_xywh = preds[keep, :4]
        confs_k = confs[keep]
        cls_k = cls_ids[keep]

        # Convert center xywh in letterbox coords -> xyxy in depth_meters coords
        x1s = (boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5 - pad_x) / scale
        y1s = (boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5 - pad_y) / scale
        x2s = (boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5 - pad_x) / scale
        y2s = (boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5 - pad_y) / scale

        nms_boxes = [
            [float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))]
            for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)
        ]
        idxs = cv2.dnn.NMSBoxes(nms_boxes, [float(c) for c in confs_k], self.conf_threshold, 0.45)
        if len(idxs) == 0:
            return []

        detections: list[DetectedObject] = []
        sx = dw / max(float(w), 1.0)
        sy = dh / max(float(h), 1.0)
        for idx in np.asarray(idxs).flatten():
            label = self._ort_names.get(int(cls_k[idx]), "object").lower()
            if label not in WALKING_OBSTACLE_CLASSES:
                continue
            bx1 = int(round(np.clip(x1s[idx] * sx, 0, dw - 1)))
            by1 = int(round(np.clip(y1s[idx] * sy, 0, dh - 1)))
            bx2 = int(round(np.clip(x2s[idx] * sx, bx1 + 1, dw)))
            by2 = int(round(np.clip(y2s[idx] * sy, by1 + 1, dh)))

            area_ratio = ((bx2 - bx1) * (by2 - by1)) / max(float(dw * dh), 1.0)
            if area_ratio < 0.015:
                continue

            dist_m = estimate_box_distance_m(depth_meters, bx1, by1, bx2, by2)
            cx_norm = ((bx1 + bx2) * 0.5) / max(float(dw), 1.0)
            detections.append(
                DetectedObject(
                    label=label,
                    confidence=round(float(confs_k[idx]), 3),
                    distance_m=round(dist_m, 2),
                    corridor=corridor_from_center_x(cx_norm),
                    bbox_norm=[
                        round(max(0.0, bx1 / max(dw, 1)), 4),
                        round(max(0.0, by1 / max(dh, 1)), 4),
                        round(min(1.0, bx2 / max(dw, 1)), 4),
                        round(min(1.0, by2 / max(dh, 1)), 4),
                    ],
                    severity=classify_object_severity(dist_m),
                )
            )
        return detections

    def detect(
        self,
        image: Image.Image | np.ndarray,
        depth_meters: np.ndarray,
        max_objects: int = 6,
    ) -> ObjectDetectionResult:
        t0 = time.perf_counter()
        if self.use_mock or (self._model is None and self._ort_session is None):
            return ObjectDetectionResult(objects=[], yolo_ms=0.0)

        rgb = image if isinstance(image, np.ndarray) else np.asarray(image)
        if self._ort_session is not None:
            detections = self._detect_onnx(rgb, depth_meters)
            detections.sort(key=lambda d: d.distance_m)
            yolo_ms = (time.perf_counter() - t0) * 1000.0
            return ObjectDetectionResult(
                objects=detections[:max_objects],
                yolo_ms=round(yolo_ms, 2),
            )

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
