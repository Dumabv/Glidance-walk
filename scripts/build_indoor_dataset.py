"""Build unified YOLO dataset for Glidance Indoor + Stairs:
Classes:
  0: stairs
  1: wall
  2: chair
  3: table   (SUNRGBD table + desk + counter)
  4: door
  5: couch   (SUNRGBD sofa)
  6: person
  7: box
  8: bed
Combines real SUNRGBD_seg indoor photos (mask -> tight connected-component YOLO boxes)
with our downloaded HuggingFace Stairs & Obstacles dataset.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import shutil
import urllib.request

import cv2
import numpy as np
from PIL import Image
import polars as pl

PARQUET_URL = (
    "https://huggingface.co/datasets/wyrx/SUNRGBD_seg/resolve/main/data/train-00001-of-00004.parquet"
)

NAMES = [
    "stairs",
    "wall",
    "chair",
    "table",
    "door",
    "couch",
    "person",
    "box",
    "bed",
]

# SUNRGBD_seg id -> unified Glidance class id
SUN_TO_DST: dict[int, int] = {
    0: 1,   # wall -> wall
    4: 2,   # chair -> chair
    6: 3,   # table -> table
    13: 3,  # desk -> table
    11: 3,  # counter -> table
    7: 4,   # door -> door
    5: 5,   # sofa -> couch
    30: 6,  # person -> person
    28: 7,  # box -> box
    3: 8,   # bed -> bed
}

# Minimum area fraction for a connected component to become a YOLO box
MIN_AREA_BY_CLS: dict[int, float] = {
    1: 0.06,   # wall: only significant wall surfaces (>= 6% of image)
    2: 0.012,  # chair
    3: 0.018,  # table / desk
    4: 0.020,  # door
    5: 0.025,  # couch
    6: 0.012,  # person
    7: 0.010,  # box
    8: 0.030,  # bed
}


def _extract_image_bytes(val) -> bytes | None:
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, dict) and val.get("bytes") is not None:
        return bytes(val["bytes"])
    return None


def _mask_to_yolo_lines(mask: np.ndarray) -> list[str]:
    h, w = mask.shape[:2]
    img_area = float(max(1, h * w))
    lines: list[str] = []

    # Group source ids that map to the same destination class (e.g. table/desk/counter -> 3)
    dst_to_srcs: dict[int, list[int]] = {}
    for s_id, d_id in SUN_TO_DST.items():
        dst_to_srcs.setdefault(d_id, []).append(s_id)

    for dst_cls, src_ids in dst_to_srcs.items():
        bin_mask = np.isin(mask, src_ids).astype(np.uint8)
        if not np.any(bin_mask):
            continue
        # Morphological open to remove thin noisy mask slivers
        kernel = np.ones((5, 5), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        min_area = MIN_AREA_BY_CLS.get(dst_cls, 0.015) * img_area
        for i in range(1, num_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = float(stats[i, cv2.CC_STAT_LEFT])
            y = float(stats[i, cv2.CC_STAT_TOP])
            bw = float(stats[i, cv2.CC_STAT_WIDTH])
            bh = float(stats[i, cv2.CC_STAT_HEIGHT])
            xc = np.clip((x + bw * 0.5) / w, 0.0, 1.0)
            yc = np.clip((y + bh * 0.5) / h, 0.0, 1.0)
            nw = np.clip(bw / w, 0.001, 1.0)
            nh = np.clip(bh / h, 0.001, 1.0)
            lines.append(f"{dst_cls} {xc:.5f} {yc:.5f} {nw:.5f} {nh:.5f}")
    return lines


def _copy_stairs_split(src_root: Path, dst_root: Path, split: str, max_count: int) -> int:
    # In stairs_obstacles: 0=stairs, 1=person, 2=chair
    remap = {0: 0, 1: 6, 2: 2}
    src_imgs = sorted((src_root / split / "images").glob("*.jpg"))
    copied = 0
    for img_p in src_imgs:
        if copied >= max_count:
            break
        lbl_p = src_root / split / "labels" / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        out_lines: list[str] = []
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if cid in remap:
                out_lines.append(f"{remap[cid]} {' '.join(parts[1:5])}")
        if not out_lines:
            continue
        dst_img = dst_root / split / "images" / f"stairs_{img_p.name}"
        dst_lbl = dst_root / split / "labels" / f"stairs_{img_p.stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_p, dst_img)
        dst_lbl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=r"D:\glidance_datasets\sunrgbd_train1.parquet")
    parser.add_argument("--stairs-dir", default=r"D:\glidance_datasets\stairs_obstacles")
    parser.add_argument("--out", default=r"D:\glidance_datasets\glidance_indoor_v2")
    parser.add_argument("--sun-train", type=int, default=320)
    parser.add_argument("--sun-valid", type=int, default=60)
    parser.add_argument("--sun-test", type=int, default=30)
    args = parser.parse_args()

    pq_path = Path(args.parquet)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    if not pq_path.exists():
        print(f"Downloading SUNRGBD_seg shard (~227 MB) to {pq_path}...")
        urllib.request.urlretrieve(PARQUET_URL, pq_path)
        print("Download complete!")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading SUNRGBD_seg parquet...")
    df = pl.read_parquet(pq_path)
    print("Columns:", df.columns, "Rows:", len(df))

    # Auto-detect image & mask column names
    img_col = next(c for c in df.columns if "image" in c.lower() or "rgb" in c.lower())
    seg_col = next(c for c in df.columns if "label" in c.lower() or "seg" in c.lower() or "annot" in c.lower() or "mask" in c.lower())
    print(f"Using columns: image={img_col}, mask={seg_col}")

    total_needed = args.sun_train + args.sun_valid + args.sun_test
    saved = 0
    cls_counts = {n: 0 for n in NAMES}

    for idx in range(len(df)):
        if saved >= total_needed:
            break
        row = df.row(idx, named=True)
        img_bytes = _extract_image_bytes(row[img_col])
        seg_bytes = _extract_image_bytes(row[seg_col])
        if not img_bytes or not seg_bytes:
            continue

        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        seg = np.asarray(Image.open(io.BytesIO(seg_bytes)))
        if seg.ndim == 3:
            seg = seg[:, :, 0]

        lines = _mask_to_yolo_lines(seg)
        if not lines:
            continue

        if saved < args.sun_train:
            split = "train"
        elif saved < args.sun_train + args.sun_valid:
            split = "valid"
        else:
            split = "test"

        img_out = out_dir / split / "images" / f"sun_{idx:05d}.jpg"
        lbl_out = out_dir / split / "labels" / f"sun_{idx:05d}.txt"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        im.save(img_out, format="JPEG", quality=90)
        lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for l in lines:
            cls_counts[NAMES[int(l.split()[0])]] += 1
        saved += 1

    print(f"Extracted {saved} SUNRGBD indoor photos. Box counts: {cls_counts}")

    stairs_root = Path(args.stairs_dir)
    if stairs_root.exists():
        st_tr = _copy_stairs_split(stairs_root, out_dir, "train", 260)
        st_va = _copy_stairs_split(stairs_root, out_dir, "valid", 55)
        st_te = _copy_stairs_split(stairs_root, out_dir, "test", 35)
        print(f"Merged stairs photos: train={st_tr}, valid={st_va}, test={st_te}")

    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Unified dataset ready at {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
