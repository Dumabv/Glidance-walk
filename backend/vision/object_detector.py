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

DEFAULT_YOLO_WEIGHTS = os.getenv("GLIDANCE_YOLO_MODEL", "glidance_indoor_v2.onnx")
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
        self._ort_general_session = None
        self._ort_general_input_name: str = ""
        self._ort_general_names: dict[int, str] = {}
        self._general_conf_threshold: float = 0.38
        self._lock = threading.Lock()

        if not use_mock:
            self._load_model()

    @property
    def is_active(self) -> bool:
        return (not self.use_mock) and (self._model is not None or self._ort_session is not None)

    @staticmethod
    def _open_ort_session(onnx_path: Path):
        _register_nvidia_cuda_dlls()
        import onnxruntime as ort

        avail = ort.get_available_providers()
        providers: list[str] = []
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        elif "CoreMLExecutionProvider" in avail:
            # Apple Silicon (M1/M2/M3/M4) Neural Engine + Metal GPU acceleration on macOS
            providers.append("CoreMLExecutionProvider")
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

        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        sess.run(None, {inp.name: dummy})
        return sess, inp.name, imgsz, names

    def _try_load_onnx(self, onnx_path: Path) -> bool:
        if not onnx_path.exists():
            return False
        try:
            sess, inp_name, imgsz, names = self._open_ort_session(onnx_path)
            self._ort_session = sess
            self._ort_input_name = inp_name
            self._ort_imgsz = imgsz
            self._ort_names = names
            self.weights_name = onnx_path.name
            self.conf_threshold = max(self.conf_threshold, 0.50)
            active_p = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"
            if "CUDA" in active_p:
                self.device = "cuda-onnx"
            elif "CoreML" in active_p:
                self.device = "coreml-onnx"
            else:
                self.device = "cpu-onnx"

            # Optional companion general-obstacle ONNX head (e.g. yolo11n.onnx for chair/table/person/bench/car)
            general_candidates = [
                onnx_path.parent / "yolo11n.onnx",
                Path(r"D:\glidance_models\yolo11n.onnx"),
            ]
            for gc in general_candidates:
                if gc.exists() and gc.resolve() != onnx_path.resolve():
                    try:
                        gsess, ginp, gsz, gnames = self._open_ort_session(gc)
                        if gsz == imgsz:
                            self._ort_general_session = gsess
                            self._ort_general_input_name = ginp
                            self._ort_general_names = gnames
                            self.weights_name = f"{onnx_path.name}+{gc.name}"
                            break
                    except Exception:
                        self._ort_general_session = None
            return True
        except Exception:
            self._ort_session = None
            return False

    def _load_model(self) -> None:
        models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prefer fine-tuned ONNX model on RTX CUDA (fast ~20ms dual-head, high stairs+indoor recall)
        onnx_candidates: list[Path] = []
        if self.weights_name.lower().endswith(".onnx"):
            cand = Path(self.weights_name)
            onnx_candidates.append(cand if cand.is_absolute() else (models_dir / cand))
            onnx_candidates.append(models_dir / "glidance_indoor_v2.onnx")
            onnx_candidates.append(Path(r"D:\glidance_models\runs\glidance_indoor_v2\weights\best.onnx"))
            onnx_candidates.append(models_dir / "glidance_stairs_v1.onnx")
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

    @staticmethod
    def _decode_yolo_raw(
        out: np.ndarray,
        names: dict[int, str],
        conf_th: float,
        scale: float,
        pad_x: int,
        pad_y: int,
        w: int,
        h: int,
        allowed_labels: set[str] | None = None,
    ) -> list[tuple[str, float, list[float]]]:
        """Decodes YOLO ONNX output into (label, conf, bbox_norm) without needing depth_meters yet."""
        preds = out[0].T if out.ndim == 3 and out.shape[1] < out.shape[2] else out[0]
        if preds.shape[1] < 5:
            return []

        scores_mat = preds[:, 4:]
        cls_ids = np.argmax(scores_mat, axis=1)
        confs = scores_mat[np.arange(len(preds)), cls_ids]
        keep = confs >= conf_th
        if not np.any(keep):
            return []

        boxes_xywh = preds[keep, :4]
        confs_k = confs[keep]
        cls_k = cls_ids[keep]

        x1s = (boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5 - pad_x) / scale
        y1s = (boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5 - pad_y) / scale
        x2s = (boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5 - pad_x) / scale
        y2s = (boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5 - pad_y) / scale

        nms_boxes = [
            [float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))]
            for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)
        ]
        idxs = cv2.dnn.NMSBoxes(nms_boxes, [float(c) for c in confs_k], conf_th, 0.45)
        if len(idxs) == 0:
            return []

        fw, fh = max(float(w), 1.0), max(float(h), 1.0)
        raw_out: list[tuple[str, float, list[float]]] = []
        for idx in np.asarray(idxs).flatten():
            label = names.get(int(cls_k[idx]), "object").lower()
            if label not in WALKING_OBSTACLE_CLASSES:
                continue
            if allowed_labels is not None and label not in allowed_labels:
                continue
            nx1 = float(np.clip(x1s[idx] / fw, 0.0, 1.0))
            ny1 = float(np.clip(y1s[idx] / fh, 0.0, 1.0))
            nx2 = float(np.clip(x2s[idx] / fw, nx1, 1.0))
            ny2 = float(np.clip(y2s[idx] / fh, ny1, 1.0))
            if (nx2 - nx1) * (ny2 - ny1) < 0.015:
                continue
            raw_out.append((label, round(float(confs_k[idx]), 3), [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)]))
        return raw_out

    def detect_boxes_raw(
        self, image: Image.Image | np.ndarray
    ) -> tuple[list[tuple[str, float, list[float]]], float]:
        """Runs YOLO inference without needing depth_meters — safe to run in parallel with DepthEngine!"""
        t0 = time.perf_counter()
        if self.use_mock or (self._model is None and self._ort_session is None):
            return [], 0.0

        rgb = image if isinstance(image, np.ndarray) else np.asarray(image)
        if self._ort_session is not None:
            h, w = rgb.shape[:2]
            sz = self._ort_imgsz
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
                out_primary = self._ort_session.run(None, {self._ort_input_name: blob})[0]
                out_general = (
                    self._ort_general_session.run(None, {self._ort_general_input_name: blob})[0]
                    if self._ort_general_session is not None
                    else None
                )

            primary_allowed = (
                STAIRS_CURB_LABELS | {"chair", "table", "door", "wall", "couch"}
                if out_general is not None
                else None
            )
            decode_floor = min(self.conf_threshold, 0.30)
            raw_primary = self._decode_yolo_raw(
                out_primary, self._ort_names, decode_floor, scale, pad_x, pad_y, w, h, allowed_labels=primary_allowed
            )
            boxes = [
                b for b in raw_primary
                if (b[1] >= self.conf_threshold if b[0] in STAIRS_CURB_LABELS else b[1] >= 0.32)
            ]
            if out_general is not None:
                general_boxes = self._decode_yolo_raw(
                    out_general, self._ort_general_names, self._general_conf_threshold, scale, pad_x, pad_y, w, h
                )
                vehicle_boxes = [g[2] for g in general_boxes if g[0] in {"car", "truck", "bus"} and g[1] >= 0.40]
                if vehicle_boxes:
                    filtered: list[tuple[str, float, list[float]]] = []
                    for b in boxes:
                        if b[0] not in STAIRS_CURB_LABELS:
                            filtered.append(b)
                            continue
                        sx1, sy1, sx2, sy2 = b[2]
                        s_area = max(1e-6, (sx2 - sx1) * (sy2 - sy1))
                        if not any(
                            (max(0.0, min(sx2, vx2) - max(sx1, vx1)) * max(0.0, min(sy2, vy2) - max(sy1, vy1))) / s_area >= 0.35
                            for vx1, vy1, vx2, vy2 in vehicle_boxes
                        ):
                            filtered.append(b)
                    boxes = filtered
                for g in general_boxes:
                    gx1, gy1, gx2, gy2 = g[2]
                    g_area = max(1e-6, (gx2 - gx1) * (gy2 - gy1))
                    is_dup = False
                    for b in boxes:
                        dx1, dy1, dx2, dy2 = b[2]
                        d_area = max(1e-6, (dx2 - dx1) * (dy2 - dy1))
                        inter = max(0.0, min(gx2, dx2) - max(gx1, dx1)) * max(0.0, min(gy2, dy2) - max(gy1, dy1))
                        if (inter / max(1e-6, g_area + d_area - inter)) >= 0.45 and (
                            b[0] == g[0] or {b[0], g[0]} == {"table", "dining table"}
                        ):
                            is_dup = True
                            break
                    if not is_dup:
                        boxes.append(g)
            return boxes, round((time.perf_counter() - t0) * 1000.0, 2)

        # PyTorch fallback path
        h, w = rgb.shape[:2]
        with self._lock:
            results = self._model.predict(rgb, imgsz=320, conf=self.conf_threshold, device=self.device, verbose=False)
        boxes_pt: list[tuple[str, float, list[float]]] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            res = results[0]
            names = res.names or {}
            fw, fh = max(float(w), 1.0), max(float(h), 1.0)
            for xyxy, conf, cls_id in zip(
                res.boxes.xyxy.detach().cpu().numpy(),
                res.boxes.conf.detach().cpu().numpy(),
                res.boxes.cls.detach().cpu().numpy(),
            ):
                label = str(names.get(int(cls_id), "object")).lower()
                if label not in WALKING_OBSTACLE_CLASSES:
                    continue
                nx1, ny1, nx2, ny2 = [float(np.clip(v, 0.0, 1.0)) for v in (xyxy[0] / fw, xyxy[1] / fh, xyxy[2] / fw, xyxy[3] / fh)]
                if (nx2 - nx1) * (ny2 - ny1) < 0.015:
                    continue
                boxes_pt.append((label, round(float(conf), 3), [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)]))
        return boxes_pt, round((time.perf_counter() - t0) * 1000.0, 2)

    @staticmethod
    def attach_depth(
        raw_boxes: list[tuple[str, float, list[float]]],
        depth_meters: np.ndarray,
        max_objects: int = 6,
    ) -> list[DetectedObject]:
        """Attaches median metric depth, corridor, and severity to raw boxes in <0.05ms."""
        dh, dw = depth_meters.shape[:2]
        detections: list[DetectedObject] = []
        for label, conf, bbox_norm in raw_boxes:
            bx1 = int(round(np.clip(bbox_norm[0] * dw, 0, dw - 1)))
            by1 = int(round(np.clip(bbox_norm[1] * dh, 0, dh - 1)))
            bx2 = int(round(np.clip(bbox_norm[2] * dw, bx1 + 1, dw)))
            by2 = int(round(np.clip(bbox_norm[3] * dh, by1 + 1, dh)))
            dist_m = estimate_box_distance_m(depth_meters, bx1, by1, bx2, by2)
            cx_norm = 0.5 * (bbox_norm[0] + bbox_norm[2])
            detections.append(
                DetectedObject(
                    label=label,
                    confidence=conf,
                    distance_m=round(dist_m, 2),
                    corridor=corridor_from_center_x(cx_norm),
                    bbox_norm=bbox_norm,
                    severity=classify_object_severity(dist_m),
                )
            )
        detections.sort(key=lambda d: d.distance_m)
        return detections[:max_objects]

    def detect(
        self,
        image: Image.Image | np.ndarray,
        depth_meters: np.ndarray,
        max_objects: int = 6,
    ) -> ObjectDetectionResult:
        """Single source of truth for both live streaming and BenchmarkLab."""
        raw_boxes, yolo_ms = self.detect_boxes_raw(image)
        if not raw_boxes:
            return ObjectDetectionResult(objects=[], yolo_ms=yolo_ms)
        return ObjectDetectionResult(
            objects=self.attach_depth(raw_boxes, depth_meters, max_objects=max_objects),
            yolo_ms=yolo_ms,
        )
