"""Fine-tune YOLO11s on the downloaded Stairs & Walking Obstacles dataset and export ONNX for RTX 3050."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"D:\glidance_datasets\stairs_obstacles\data.yaml")
    parser.add_argument("--model", default=r"D:\glidance_models\yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=r"D:\glidance_models\runs")
    parser.add_argument("--name", default="glidance_stairs_v1")
    args = parser.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=8,
        workers=4,
    )
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"Exporting best weights to ONNX: {best_pt}")
        best = YOLO(str(best_pt))
        onnx_path = best.export(format="onnx", imgsz=args.imgsz, simplify=False)
        print(f"Ready for RTX 3050 CUDA inference: {onnx_path}")


if __name__ == "__main__":
    main()
