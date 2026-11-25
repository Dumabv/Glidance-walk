from __future__ import annotations

import ctypes
from dataclasses import dataclass
import io
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional
import urllib.request

import cv2
import numpy as np
from PIL import Image

from backend.schemas import GuidancePayload

DEFAULT_MODEL_ID = os.getenv(
    "GLIDANCE_DEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Small-hf",
)
ONNX_QUANTIZED_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx"
)
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _CDepthNormState(ctypes.Structure):
    _fields_ = [
        ("ema_p2", ctypes.c_float),
        ("ema_p98", ctypes.c_float),
        ("ema_spread", ctypes.c_float),
        ("initialized", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
    ]


def _load_rust_preproc():
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
                rgb_fn = lib.glidance_rgb_to_chw_norm
                rgb_fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_float),
                ]
                rgb_fn.restype = ctypes.c_int32

                norm_fn = lib.glidance_normalize_depth_inplace
                norm_fn.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.POINTER(_CDepthNormState),
                ]
                norm_fn.restype = ctypes.c_uint32
                return rgb_fn, norm_fn
            except (OSError, AttributeError):
                continue
    return None, None


_RUST_RGB_TO_CHW, _RUST_NORM_DEPTH = _load_rust_preproc()


@dataclass
class DecodedFrame:
    rgb: np.ndarray
    luma_mean: float
    luma_std: float
    decode_ms: float


@dataclass
class DepthPrediction:
    closeness: np.ndarray
    raw_spread_milli: int
    luma_mean: float
    luma_std: float
    inference_ms: float


def _align_to_patch14(h: int, w: int, max_side: int) -> tuple[int, int]:
    longest = max(h, w, 1)
    scale = min(1.0, float(max_side) / float(longest))
    nh = max(56, int(round((h * scale) / 14.0)) * 14)
    nw = max(56, int(round((w * scale) / 14.0)) * 14)
    return nh, nw


class DepthEngine:
    """Ultra-low-latency Depth-Anything-V2 engine with zero-alloc Rust LUT & SIMD normalization."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_input_width: int = 336,
        use_mock: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_input_width = max_input_width
        self.use_mock = use_mock
        self.device: str = "synthetic"
        self.runtime: str = "mock"
        self._lock = threading.Lock()

        self._torch = None
        self._model = None
        self._use_fp16 = False
        self._is_metric_model = "metric" in model_id.lower()
        self._ort_session = None
        self._ort_input_name: str = "pixel_values"

        # Preallocated reusable CHW input tensor buffer (zero per-frame 5MB allocations!)
        self._chw_buf: np.ndarray | None = None
        self._norm_state = _CDepthNormState()

        # Fallback Python EMA anchors
        self._ema_p2: float | None = None
        self._ema_p98: float | None = None
        self._ema_spread: float | None = None

        if not use_mock:
            self._load_best_backend()

    def _load_best_backend(self) -> None:
        preferred = os.getenv("GLIDANCE_RUNTIME", "auto").lower()

        if preferred in ("auto", "torch"):
            try:
                import torch
                from transformers import AutoModelForDepthEstimation

                mps_backend = getattr(torch.backends, "mps", None)
                if mps_backend is not None and mps_backend.is_available():
                    self.device = "mps"
                    self._use_fp16 = True
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    self._use_fp16 = True
                    torch.backends.cudnn.benchmark = True
                elif preferred == "torch":
                    self.device = "cpu"
                    self.max_input_width = min(self.max_input_width, 280)
                else:
                    raise RuntimeError("Prefer ONNX Runtime over CPU PyTorch")

                dtype = torch.float16 if self._use_fp16 else torch.float32
                self._torch = torch
                self._model = AutoModelForDepthEstimation.from_pretrained(
                    self.model_id, torch_dtype=dtype
                )
                self._model.to(device=self.device, dtype=dtype)
                self._model.eval()
                self.runtime = "torch-fp16" if self._use_fp16 else "torch-fp32"

                dummy = np.full((196, 252, 3), 120, dtype=np.uint8)
                self.predict_rgb(dummy)
                return
            except Exception:
                self._model = None
                self._torch = None

        if preferred in ("auto", "onnx"):
            try:
                import onnxruntime as ort

                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                onnx_path = CACHE_DIR / "depth_anything_v2_small_q8.onnx"
                if not onnx_path.exists():
                    urllib.request.urlretrieve(ONNX_QUANTIZED_URL, str(onnx_path))

                avail = ort.get_available_providers()
                providers: list[str] = []
                if "CUDAExecutionProvider" in avail:
                    providers.append("CUDAExecutionProvider")
                    self.device = "cuda-ort"
                elif "DmlExecutionProvider" in avail:
                    providers.append("DmlExecutionProvider")
                    self.device = "dml-ort"
                else:
                    self.device = "cpu-ort"
                providers.append("CPUExecutionProvider")

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_opts.enable_cpu_mem_arena = True
                sess_opts.enable_mem_pattern = True
                # Match physical Zen/P-cores (avoid hyperthread AVX2 thrashing)
                sess_opts.intra_op_num_threads = min(8, max(2, (os.cpu_count() or 4) // 2))
                sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")

                self._ort_session = ort.InferenceSession(
                    str(onnx_path), sess_options=sess_opts, providers=providers
                )
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                self.runtime = "onnxruntime"
                self.max_input_width = min(self.max_input_width, 280)

                dummy = np.full((196, 252, 3), 120, dtype=np.uint8)
                self.predict_rgb(dummy)
                return
            except Exception:
                self._ort_session = None

        try:
            import torch
            from transformers import AutoModelForDepthEstimation

            self._torch = torch
            self.device = "cpu"
            self.max_input_width = min(self.max_input_width, 252)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self._model.to("cpu")
            self._model.eval()
            self.runtime = "torch-cpu"
            dummy = np.full((196, 252, 3), 120, dtype=np.uint8)
            self.predict_rgb(dummy)
        except Exception:
            self.use_mock = True
            self.device = "synthetic"
            self.runtime = "mock"

    def decode_frame(self, jpeg_bytes: bytes) -> DecodedFrame:
        """Fast libjpeg-turbo decode (with 1/2 IDCT downscale for large uploads) + strided luma stats."""
        t0 = time.perf_counter()
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        # Large JPEG photos (>64KB) decode 2.5x faster via libjpeg-turbo hardware 1/2 IDCT scaling
        decode_flag = (
            cv2.IMREAD_REDUCED_COLOR_2
            if len(jpeg_bytes) > 65536 and jpeg_bytes[:2] == b"\xff\xd8"
            else cv2.IMREAD_COLOR
        )
        bgr = cv2.imdecode(buf, decode_flag)
        if bgr is None:
            pil_img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            rgb = np.asarray(pil_img, dtype=np.uint8)
            h, w = rgb.shape[:2]
            nh, nw = _align_to_patch14(h, w, self.max_input_width)
            if (nh, nw) != (h, w):
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            h0, w0 = bgr.shape[:2]
            nh, nw = _align_to_patch14(h0, w0, self.max_input_width)
            if (nh, nw) != (h0, w0):
                bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 4x4 strided green-weighted luma check takes ~0.015ms instead of full-frame cvtColor!
        sub = rgb[::4, ::4, :]
        luma_approx = (
            0.299 * sub[:, :, 0].astype(np.float32)
            + 0.587 * sub[:, :, 1].astype(np.float32)
            + 0.114 * sub[:, :, 2].astype(np.float32)
        )
        luma_mean = float(np.mean(luma_approx))
        luma_std = float(np.std(luma_approx))

        decode_ms = (time.perf_counter() - t0) * 1000.0
        return DecodedFrame(
            rgb=np.ascontiguousarray(rgb),
            luma_mean=luma_mean,
            luma_std=luma_std,
            decode_ms=decode_ms,
        )

    def decode_jpeg(self, jpeg_bytes: bytes) -> tuple[Image.Image, float]:
        dec = self.decode_frame(jpeg_bytes)
        return Image.fromarray(dec.rgb), dec.decode_ms

    def _rgb_to_chw_fast(self, rgb_in: np.ndarray) -> np.ndarray:
        """Zero-allocation Rust 256-entry LUT conversion `(H, W, 3) uint8 -> (1, 3, H, W) float32`."""
        nh, nw = rgb_in.shape[:2]
        if self._chw_buf is None or self._chw_buf.shape != (1, 3, nh, nw):
            self._chw_buf = np.empty((1, 3, nh, nw), dtype=np.float32)

        if _RUST_RGB_TO_CHW is not None:
            rgb_c = np.ascontiguousarray(rgb_in, dtype=np.uint8)
            rc = _RUST_RGB_TO_CHW(
                rgb_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                nh,
                nw,
                self._chw_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if rc == 0:
                return self._chw_buf

        norm = (rgb_in.astype(np.float32) * (1.0 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        np.copyto(self._chw_buf[0], norm.transpose(2, 0, 1))
        return self._chw_buf

    def _normalize_depth_fast(self, depth_np: np.ndarray) -> tuple[np.ndarray, int]:
        """O(N) Rust SIMD 256-bin histogram `[p2, p98]` normalizer (eliminates O(N log N) sort!)."""
        if self._is_metric_model:
            safe_d = np.nan_to_num(depth_np, nan=0.35, posinf=6.0, neginf=0.35)
            z = np.clip(safe_d, 0.35, 6.0)
            inv_z = 1.0 / z
            inv_near, inv_far = 1.0 / 0.35, 1.0 / 5.0
            closeness = np.clip((inv_z - inv_far) / (inv_near - inv_far), 0.0, 1.0).astype(
                np.float32
            )
            return closeness, 600

        arr = np.ascontiguousarray(depth_np, dtype=np.float32)
        if _RUST_NORM_DEPTH is not None:
            spread_milli = int(
                _RUST_NORM_DEPTH(
                    arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    arr.size,
                    ctypes.byref(self._norm_state),
                )
            )
            return arr, spread_milli

        sub = arr[::2, ::2]
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return np.ones_like(arr, dtype=np.float32), 0
        p2, p98 = [float(x) for x in np.percentile(finite, [2.0, 98.0])]
        spread = max(0.0, p98 - p2)
        if self._ema_spread is None or self._ema_p2 is None or self._ema_p98 is None:
            self._ema_p2, self._ema_p98, self._ema_spread = p2, p98, max(spread, 1e-4)
            spread_milli = 600 if spread > 1e-4 else 0
        else:
            spread_milli = int(np.clip((spread / max(self._ema_spread, 1e-4)) * 550.0, 0.0, 1000.0))
            if spread_milli > 120:
                self._ema_p2 = 0.85 * self._ema_p2 + 0.15 * p2
                self._ema_p98 = 0.85 * self._ema_p98 + 0.15 * p98
                self._ema_spread = max(1e-4, self._ema_p98 - self._ema_p2)
            else:
                p2, p98 = self._ema_p2, self._ema_p98
        safe_arr = np.nan_to_num(arr, nan=p2, posinf=p98, neginf=p2)
        closeness = np.clip((safe_arr - p2) / max(p98 - p2, 1e-5), 0.0, 1.0).astype(np.float32)
        return closeness, spread_milli

    def predict_rgb(
        self,
        rgb: np.ndarray,
        luma_mean: float = 120.0,
        luma_std: float = 35.0,
    ) -> DepthPrediction:
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]

        if self.use_mock or (self._model is None and self._ort_session is None):
            arr = rgb.astype(np.float32) * (1.0 / 255.0)
            gray = np.ascontiguousarray(
                0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2],
                dtype=np.float32,
            )
            p2, p98 = [float(x) for x in np.percentile(gray[::2, ::2], [2.0, 98.0])]
            spread_milli = int(np.clip((p98 - p2) * 1000.0, 0.0, 1000.0))
            closeness = np.clip(gray, 0.0, 1.0)
            return DepthPrediction(
                closeness=closeness,
                raw_spread_milli=max(spread_milli, 200) if spread_milli > 5 else 200,
                luma_mean=luma_mean,
                luma_std=luma_std,
                inference_ms=(time.perf_counter() - t0) * 1000.0,
            )

        nh, nw = _align_to_patch14(h, w, self.max_input_width)
        rgb_in = (
            cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            if (nh, nw) != (h, w)
            else rgb
        )

        with self._lock:
            chw = self._rgb_to_chw_fast(rgb_in)
            if self._ort_session is not None:
                ort_out = self._ort_session.run(None, {self._ort_input_name: chw})[0]
                raw_depth = np.squeeze(ort_out).astype(np.float32, copy=False)
            else:
                torch = self._torch
                dtype = torch.float16 if self._use_fp16 else torch.float32
                tensor = torch.from_numpy(chw).to(device=self.device, dtype=dtype)
                with torch.inference_mode():
                    outputs = self._model(pixel_values=tensor)
                    pred = outputs.predicted_depth
                    if pred.shape[-2:] != (h, w):
                        pred = torch.nn.functional.interpolate(
                            pred.unsqueeze(1).float(),
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    raw_depth = pred.squeeze(0).detach().float().cpu().numpy()

            if raw_depth.shape != (h, w):
                raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)

            closeness, spread_milli = self._normalize_depth_fast(raw_depth)

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return DepthPrediction(
            closeness=closeness,
            raw_spread_milli=spread_milli,
            luma_mean=luma_mean,
            luma_std=luma_std,
            inference_ms=inference_ms,
        )

    def predict_pil(self, image: Image.Image) -> tuple[np.ndarray, float]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        sub = rgb[::4, ::4, :].astype(np.float32)
        luma = 0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1] + 0.114 * sub[:, :, 2]
        pred = self.predict_rgb(rgb, float(np.mean(luma)), float(np.std(luma)))
        return pred.closeness, pred.inference_ms


def render_depth_overlay_jpeg(
    closeness_map: np.ndarray,
    guidance: Optional[GuidancePayload] = None,
    jpeg_quality: int = 70,
) -> bytes:
    """Colorizes the depth map and overlays L/C/R corridor status & curb line for Mission Control."""
    h, w = closeness_map.shape
    u8 = np.clip(closeness_map * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

    x1 = w // 3
    x2 = (2 * w) // 3
    y_top = int(h * 0.20)
    y_bot = int(h * 0.96)

    cv2.line(colored, (x1, y_top), (x1, y_bot), (210, 210, 210), 1)
    cv2.line(colored, (x2, y_top), (x2, y_bot), (210, 210, 210), 1)
    cv2.rectangle(colored, (2, y_top), (w - 3, y_bot), (160, 160, 160), 1)

    severity_colors = {
        "clear": (80, 220, 120),
        "caution": (0, 200, 255),
        "danger": (60, 60, 255),
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
                alpha = 0.28 if z.severity == "danger" else 0.14
                roi_patch = colored[y_top + 2 : y_bot - 2, sx + 2 : ex - 2]
                tint = np.empty_like(roi_patch)
                tint[:] = color
                cv2.addWeighted(tint, alpha, roi_patch, 1.0 - alpha, 0, roi_patch)

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

        if guidance.overhead_hazard:
            cv2.rectangle(
                colored, (int(w * 0.24), 4), (int(w * 0.76), y_top - 2), (60, 60, 255), 2
            )
            cv2.putText(
                colored,
                "LOW OVERHANG",
                (int(w * 0.26), max(14, y_top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (60, 60, 255),
                1,
                cv2.LINE_AA,
            )

        if guidance.curb.detected:
            curb_y = int(h * guidance.curb.drop_row_ratio)
            cv2.line(
                colored, (int(w * 0.15), curb_y), (int(w * 0.85), curb_y), (255, 255, 0), 2
            )
            curb_lbl = "STAIRS DOWN" if guidance.curb.stairs_likely else "CURB / STEP DROP"
            cv2.putText(
                colored,
                curb_lbl,
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
