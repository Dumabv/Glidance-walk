from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.schemas import SampleLabel, SampleRecord

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
META_FILE = DATA_DIR / "metadata.json"


@dataclass
class SyntheticCalibrationScene:
    sample_id: str
    label: SampleLabel
    note: str
    closeness_map: np.ndarray


def build_synthetic_calibration_suite(
    h: int = 180, w: int = 240
) -> list[SyntheticCalibrationScene]:
    """Generates deterministic depth maps representing common hallway & sidewalk edge cases."""
    rng = np.random.default_rng(42)

    def flat_floor(bottom_val: float = 0.68, top_val: float = 0.08) -> np.ndarray:
        y = np.linspace(top_val, bottom_val, h, dtype=np.float32)[:, None]
        base = np.repeat(y, w, axis=1)
        noise = rng.normal(0.0, 0.012, size=(h, w)).astype(np.float32)
        return np.clip(base + noise, 0.0, 1.0)

    scenes: list[SyntheticCalibrationScene] = []

    # 1. Clear tilted hallways (these often break naive proximity baselines)
    for i, (bot, note) in enumerate(
        [
            (0.64, "Flat hallway floor (level camera)"),
            (0.71, "Slightly downward-tilted hallway floor"),
            (0.69, "Wide corridor floor with smooth lighting"),
            (0.66, "Sidewalk straight ahead, no obstacles"),
        ],
        start=1,
    ):
        scenes.append(
            SyntheticCalibrationScene(
                sample_id=f"syn-clear-{i}",
                label="clear",
                note=note,
                closeness_map=flat_floor(bottom_val=bot, top_val=0.07),
            )
        )

    # 2. Left corridor obstacles
    for i, (cx1, cx2, val, note) in enumerate(
        [
            (0.02, 0.30, 0.85, "Chair protruding on left side"),
            (0.05, 0.28, 0.90, "Person walking close on left"),
            (0.00, 0.29, 0.78, "Trash bin along left wall"),
        ],
        start=1,
    ):
        m = flat_floor(0.66)
        m[int(h * 0.26) : int(h * 0.82), int(w * cx1) : int(w * cx2)] = val
        scenes.append(
            SyntheticCalibrationScene(
                sample_id=f"syn-left-{i}",
                label="obstacle_left",
                note=note,
                closeness_map=m,
            )
        )

    # 3. Center corridor obstacles (including full close wall)
    for i, (cx1, cx2, y1, y2, val, note) in enumerate(
        [
            (0.35, 0.65, 0.28, 0.80, 0.88, "Pillar / bollard directly ahead"),
            (0.30, 0.70, 0.30, 0.84, 0.92, "Office chair blocking center hallway"),
            (0.00, 1.00, 0.20, 0.94, 0.94, "Closed door / wall 0.6m ahead (full frame)"),
        ],
        start=1,
    ):
        m = flat_floor(0.65)
        m[int(h * y1) : int(h * y2), int(w * cx1) : int(w * cx2)] = val
        scenes.append(
            SyntheticCalibrationScene(
                sample_id=f"syn-center-{i}",
                label="obstacle_center",
                note=note,
                closeness_map=m,
            )
        )

    # 4. Right corridor obstacles
    for i, (cx1, cx2, val, note) in enumerate(
        [
            (0.70, 0.98, 0.86, "Desk corner protruding on right"),
            (0.72, 0.96, 0.91, "Standing person on right"),
        ],
        start=1,
    ):
        m = flat_floor(0.66)
        m[int(h * 0.28) : int(h * 0.82), int(w * cx1) : int(w * cx2)] = val
        scenes.append(
            SyntheticCalibrationScene(
                sample_id=f"syn-right-{i}",
                label="obstacle_right",
                note=note,
                closeness_map=m,
            )
        )

    # 5. Downward curbs / staircases (sudden depth plunge across lower walking band)
    for i, (drop_y_ratio, drop_amount, note) in enumerate(
        [
            (0.68, 0.26, "Street curb drop-off 1.2m ahead"),
            (0.73, 0.32, "Staircase leading down (sharp ledge)"),
            (0.64, 0.24, "Platform / porch step down"),
        ],
        start=1,
    ):
        m = flat_floor(bottom_val=0.70, top_val=0.10)
        cut = int(h * drop_y_ratio)
        # Everything beyond the curb edge suddenly drops farther away (lower closeness)
        m[:cut, :] = np.clip(m[:cut, :] - drop_amount, 0.04, 1.0)
        scenes.append(
            SyntheticCalibrationScene(
                sample_id=f"syn-curb-{i}",
                label="curb_drop",
                note=note,
                closeness_map=m,
            )
        )

    return scenes


class DatasetStore:
    """Stores real hallway snapshots captured from phone/webcam for midterm benchmarking."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.meta_file = data_dir / "metadata.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _read_meta(self) -> list[dict]:
        if not self.meta_file.exists():
            return []
        try:
            return json.loads(self.meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write_meta(self, rows: list[dict]) -> None:
        self.meta_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def list_samples(self, include_thumbnails: bool = True) -> list[SampleRecord]:
        rows = self._read_meta()
        records: list[SampleRecord] = []
        for row in rows:
            img_path = self.data_dir / f"{row['sample_id']}.jpg"
            img_b64 = None
            if include_thumbnails and img_path.exists():
                img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            records.append(
                SampleRecord(
                    sample_id=row["sample_id"],
                    label=row["label"],
                    note=row.get("note", ""),
                    source="real",
                    created_at=row["created_at"],
                    image_b64=img_b64,
                )
            )
        return records

    def save_sample(
        self, jpeg_bytes: bytes, label: SampleLabel, note: str = ""
    ) -> SampleRecord:
        sample_id = f"real-{uuid.uuid4().hex[:8]}"
        img_path = self.data_dir / f"{sample_id}.jpg"
        img_path.write_bytes(jpeg_bytes)

        now = time.time()
        rows = self._read_meta()
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "note": note.strip() or f"Captured {label.replace('_', ' ')}",
                "created_at": now,
            }
        )
        self._write_meta(rows)
        return SampleRecord(
            sample_id=sample_id,
            label=label,
            note=rows[-1]["note"],
            source="real",
            created_at=now,
            image_b64=base64.b64encode(jpeg_bytes).decode("ascii"),
        )

    def get_image_bytes(self, sample_id: str) -> bytes | None:
        img_path = self.data_dir / f"{sample_id}.jpg"
        return img_path.read_bytes() if img_path.exists() else None

    def delete_sample(self, sample_id: str) -> bool:
        rows = self._read_meta()
        new_rows = [r for r in rows if r["sample_id"] != sample_id]
        if len(new_rows) == len(rows):
            return False
        self._write_meta(new_rows)
        img_path = self.data_dir / f"{sample_id}.jpg"
        if img_path.exists():
            img_path.unlink()
        return True
