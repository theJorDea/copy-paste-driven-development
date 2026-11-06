
import argparse
import os
from pathlib import Path

from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "visdrone.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO model on VisDrone dataset"
    )

    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--name", default="baseline-yolo11n")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(DATASET_CONFIG)

    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "visdrone-training"
    os.environ["MLFLOW_RUN"] = args.name

    settings.update({"mlflow": True})

    model = YOLO(args.model)

    model.train(
        data=str(DATASET_CONFIG),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=True,
        fraction=args.fraction,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=False,
        plots=True,
        save=True,
        val=True,
        patience=10,
    )


if __name__ == "__main__":
    main()
