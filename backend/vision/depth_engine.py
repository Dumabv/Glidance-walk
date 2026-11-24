from __future__ import annotations

import io
import os
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.schemas import GuidancePayload

DEFAULT_MODEL_ID = os.getenv(
    "GLIDANCE_DEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Small-hf",
)


class DepthEngine:
    """Loads Depth-Anything-V2-Small with automatic MPS -> CUDA -> CPU selection."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_input_width: int = 320,
        use_mock: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_input_width = max_input_width
        self.use_mock = use_mock
        self.device: str = "synthetic"
        self._processor = None
        self._model = None
        self._torch = None

        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            # Smaller resize on older CPU-only Windows machines for smooth dev FPS
            self.max_input_width = min(self.max_input_width, 256)

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

        # Warm up accelerator so first live frame is fast
        dummy = Image.fromarray(np.zeros((180, 240, 3), dtype=np.uint8))
        self.predict_pil(dummy)

    def decode_jpeg(self, jpeg_bytes: bytes) -> tuple[Image.Image, float]:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        if img.width > self.max_input_width:
            ratio = self.max_input_width / float(img.width)
            new_h = max(64, int(round(img.height * ratio)))
            img = img.resize((self.max_input_width, new_h), Image.Resampling.BILINEAR)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        return img, decode_ms

    def predict_pil(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """Returns normalized inverse-depth closeness map [0, 1] (1=closest) and inference ms."""
        t0 = time.perf_counter()

        if self.use_mock or self._model is None or self._processor is None:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            closeness = np.clip(gray, 0.0, 1.0).astype(np.float32)
            return closeness, (time.perf_counter() - t0) * 1000.0

        torch = self._torch
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate back to the resized frame shape
        h, w = image.height, image.width
        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth_np = depth_resized.detach().float().cpu().numpy()

        # Robust percentile normalization to [0, 1] where 1.0 = closest
        p2, p98 = np.percentile(depth_np, [2.0, 98.0])
        denom = max(float(p98 - p2), 1e-5)
        closeness = np.clip((depth_np - p2) / denom, 0.0, 1.0).astype(np.float32)

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return closeness, inference_ms


def render_depth_overlay_jpeg(
    closeness_map: np.ndarray,
    guidance: Optional[GuidancePayload] = None,
    jpeg_quality: int = 72,
) -> bytes:
    """Colorizes the depth map and overlays L/C/R corridor status & curb line for Mission Control."""
    h, w = closeness_map.shape
    u8 = np.clip(closeness_map * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

    # Corridor boundaries (Left 0..33%, Center 33..67%, Right 67..100%)
    x1 = w // 3
    x2 = (2 * w) // 3
    y_top = int(h * 0.22)
    y_bot = int(h * 0.94)

    cv2.line(colored, (x1, y_top), (x1, y_bot), (210, 210, 210), 1)
    cv2.line(colored, (x2, y_top), (x2, y_bot), (210, 210, 210), 1)
    cv2.rectangle(colored, (2, y_top), (w - 3, y_bot), (160, 160, 160), 1)

    severity_colors = {
        "clear": (80, 220, 120),    # BGR green
        "caution": (0, 200, 255),   # BGR amber
        "danger": (60, 60, 255),    # BGR red
    }

    if guidance is not None:
        zones = [
            ("L", guidance.left, 0, x1),
            ("C", guidance.center, x1, x2),
            ("R", guidance.right, x2, w),
        ]
        for label, z, sx, ex in zones:
            color = severity_colors[z.severity]
            if z.severity != "clear":
                overlay = colored.copy()
                cv2.rectangle(overlay, (sx + 2, y_top + 2), (ex - 2, y_bot - 2), color, -1)
                alpha = 0.28 if z.severity == "danger" else 0.14
                cv2.addWeighted(overlay, alpha, colored, 1.0 - alpha, 0, colored)

            text = f"{label}:{z.min_distance_m:.1f}m"
            cv2.putText(
                colored,
                text,
                (sx + 8, y_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        if guidance.curb.detected:
            curb_y = int(h * guidance.curb.drop_row_ratio)
            cv2.line(colored, (int(w * 0.15), curb_y), (int(w * 0.85), curb_y), (255, 255, 0), 2)
            cv2.putText(
                colored,
                "CURB / STEP DROP",
                (int(w * 0.18), max(16, curb_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    ok, encoded = cv2.imencode(
        ".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return encoded.tobytes() if ok else b""
