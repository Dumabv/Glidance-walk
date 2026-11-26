"""Download real YOLO-annotated Stairs & Walking Obstacles dataset from HuggingFace to D:\\glidance_datasets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
from pathlib import Path
import urllib.request

BASE_URL = "https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO/resolve/main"

# Map selected source class IDs -> compact 11-class Glidance navigation taxonomy
# Source: 0:Bike, 2:Car, 3:Person, 4:Stairs, 6:Electrical Pole, 9:Dustbin, 10:Dog,
#         11:Manhole, 13:Guard rail, 17:Bench, 23:Chair
SRC_TO_DST: dict[int, int] = {
    4: 0,   # stairs
    3: 1,   # person
    23: 2,  # chair
    17: 3,  # bench
    13: 4,  # railing (guard rail)
    6: 5,   # pole
    9: 6,   # trash can (dustbin)
    11: 7,  # manhole
    2: 8,   # car
    0: 9,   # bicycle
    10: 10, # dog
}

DST_NAMES = [
    "stairs",
    "person",
    "chair",
    "bench",
    "railing",
    "pole",
    "trash can",
    "manhole",
    "car",
    "bicycle",
    "dog",
]


def _download_one(split: str, img_name: str, lbl_name: str, out_dir: Path) -> bool:
    img_out = out_dir / split / "images" / img_name
    lbl_out = out_dir / split / "labels" / lbl_name
    if img_out.exists() and lbl_out.exists():
        return True

    img_url = f"{BASE_URL}/{split}/images/{img_name}"
    lbl_url = f"{BASE_URL}/{split}/labels/{lbl_name}"
    try:
        img_bytes = urllib.request.urlopen(img_url, timeout=20).read()
        lbl_text = urllib.request.urlopen(lbl_url, timeout=20).read().decode("utf-8", errors="replace")
    except Exception:
        return False

    remapped_lines: list[str] = []
    for line in lbl_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(float(parts[0]))
        except ValueError:
            continue
        if src_cls in SRC_TO_DST:
            remapped_lines.append(f"{SRC_TO_DST[src_cls]} {' '.join(parts[1:5])}")

    if not remapped_lines:
        return False

    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)
    img_out.write_bytes(img_bytes)
    lbl_out.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=r"D:\glidance_datasets\stairs_obstacles", help="Target directory")
    parser.add_argument("--train-stairs", type=int, default=260)
    parser.add_argument("--train-other", type=int, default=60)
    parser.add_argument("--valid-stairs", type=int, default=55)
    parser.add_argument("--valid-other", type=int, default=15)
    parser.add_argument("--test-stairs", type=int, default=35)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching train/metadata.csv (sharded train split)...")
    train_meta_raw = urllib.request.urlopen(f"{BASE_URL}/train/metadata.csv", timeout=30).read().decode("utf-8")
    train_reader = csv.DictReader(io.StringIO(train_meta_raw))
    train_selected: list[tuple[str, str, str]] = []
    t_stairs = 0
    t_other = 0
    for row in train_reader:
        cnames = {x.strip() for x in row["class_names"].replace(";", ",").split(",") if x.strip()}
        cids = {x.strip() for x in row["class_ids"].replace(";", ",").split(",") if x.strip()}
        has_stairs = "Stairs" in cnames or "4" in cids
        if has_stairs:
            if t_stairs >= args.train_stairs:
                continue
            t_stairs += 1
        else:
            if t_other >= args.train_other:
                continue
            if not (cnames & {"Chair", "Bench", "Guard rail", "Person", "Dustbin", "Electrical Pole"}):
                continue
            t_other += 1
        train_selected.append(("train", row["file_name"].strip(), row["label_file"].strip()))
        if t_stairs >= args.train_stairs and t_other >= args.train_other:
            break

    def _download_rel(split: str, rel_img: str, rel_lbl: str) -> bool:
        img_name = Path(rel_img).name
        lbl_name = Path(rel_lbl).name
        img_out = out_dir / split / "images" / img_name
        lbl_out = out_dir / split / "labels" / lbl_name
        if img_out.exists() and lbl_out.exists():
            return True
        try:
            img_bytes = urllib.request.urlopen(f"{BASE_URL}/{split}/{rel_img}", timeout=20).read()
            lbl_text = urllib.request.urlopen(f"{BASE_URL}/{split}/{rel_lbl}", timeout=20).read().decode("utf-8", errors="replace")
        except Exception:
            return False
        remapped: list[str] = []
        for line in lbl_text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                src_cls = int(float(parts[0]))
            except ValueError:
                continue
            if src_cls in SRC_TO_DST:
                remapped.append(f"{SRC_TO_DST[src_cls]} {' '.join(parts[1:5])}")
        if not remapped:
            return False
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        img_out.write_bytes(img_bytes)
        lbl_out.write_text("\n".join(remapped) + "\n", encoding="utf-8")
        return True

    print(f"Downloading {len(train_selected)} train images (stairs={t_stairs}, other={t_other})...")
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_download_rel, s, im, lb) for s, im, lb in train_selected]
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
                if ok % 50 == 0:
                    print(f"  train downloaded {ok}/{len(train_selected)}...")

    print("Fetching images_manifest.csv (valid & test splits)...")
    manifest_raw = urllib.request.urlopen(f"{BASE_URL}/images_manifest.csv", timeout=30).read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(manifest_raw))

    selected: list[tuple[str, str, str]] = []
    counts = {
        ("valid", True): 0,
        ("valid", False): 0,
        ("test", True): 0,
    }
    limits = {
        ("valid", True): args.valid_stairs,
        ("valid", False): args.valid_other,
        ("test", True): args.test_stairs,
    }

    for row in reader:
        split = row["split"].strip()
        if split == "train":
            continue
        names = {x.strip() for x in row["class_names_present"].split(";") if x.strip()}
        if not names:
            names = {x.strip() for x in row["class_names_present"].split(",") if x.strip()}
        has_stairs = "Stairs" in names or "4" in row["classes_present"].replace(";", ",").split(",")
        key = (split, has_stairs)
        if key not in limits or counts[key] >= limits[key]:
            continue
        if not has_stairs:
            if not (names & {"Chair", "Bench", "Guard rail", "Person", "Dustbin", "Electrical Pole"}):
                continue
        counts[key] += 1
        selected.append((split, row["image_filename"].strip(), row["label_filename"].strip()))

    print(f"Downloading {len(selected)} valid/test labeled images to {out_dir}...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_download_one, s, im, lb, out_dir) for s, im, lb in selected]
        for fut in as_completed(futs):
            if fut.result():
                ok += 1

    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(DST_NAMES)}\n"
        f"names: {DST_NAMES}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Done! Saved {ok} images + YOLO labels and data.yaml at {out_dir}")


if __name__ == "__main__":
    main()
