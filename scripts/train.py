
import argparse
import hashlib
import os
from pathlib import Path
from time import perf_counter

import mlflow
import pandas as pd
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "visdrone.yaml"

DATASETS = {
    "training": ("visdrone-train", "visdrone2019-det-train"),
    "validation": ("visdrone", "visdrone2019-det-val"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on VisDrone with MLflow tracking"
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    return parser.parse_args()


def log_dataset(split: str) -> None:
    folder, name = DATASETS[split]
    manifest_path = ROOT / "data" / folder / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    df = pd.read_csv(manifest_path)

    full_digest = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    dataset = mlflow.data.from_pandas(
        df,
        source=manifest_path.resolve().as_uri(),
        name=name,
        digest=full_digest[:32],
    )

    mlflow.log_input(dataset, context=split)
    mlflow.set_tag(f"{split}_manifest_sha256", full_digest)


def main() -> None:
    args = parse_args()

    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(DATASET_CONFIG)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "visdrone-training"
    os.environ["MLFLOW_RUN"] = args.name

    settings.update({"mlflow": True})
    mlflow.set_experiment("visdrone-training")

    with mlflow.start_run(run_name=args.name):
        # Связываем обучение с версиями данных.
        log_dataset("training")
        log_dataset("validation")

        mlflow.log_artifact(
            str(DATASET_CONFIG),
            artifact_path="config",
        )

        # Встроенная интеграция Ultralytics записывает
        # параметры, loss и метрики по эпохам.
        model = YOLO(args.model)

        start = perf_counter()

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
            patience=0,
        )

        duration = perf_counter() - start
        mlflow.log_metric("train_seconds", duration)

        # Берём лучшую модель по результатам обучения.
        best_path = Path(model.trainer.best)

        if not best_path.is_file():
            raise FileNotFoundError(best_path)

        # Явное сохранение весов с предсказуемым путём.
        mlflow.log_artifact(
            str(best_path),
            artifact_path="model_weights",
        )

        # Отдельная итоговая проверка best.pt.
        best_model = YOLO(str(best_path))

        metrics = best_model.val(
            data=str(DATASET_CONFIG),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            project=str(ROOT / "runs" / "validation"),
            name=args.name,
            exist_ok=False,
        )

        mlflow.log_metrics(
            {
                "final/map50_95": float(metrics.box.map),
                "final/map50": float(metrics.box.map50),
                "final/precision": float(metrics.box.mp),
                "final/recall": float(metrics.box.mr),
            }
        )

        # Создаём таблицу метрик по отдельным классам.
        rows = []

        for index, class_id in enumerate(metrics.box.ap_class_index):
            precision, recall, map50, map50_95 = (
                metrics.box.class_result(index)
            )

            rows.append(
                {
                    "class": metrics.names[int(class_id)],
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                }
            )

        class_metrics_path = (
            Path(metrics.save_dir) / "per_class_metrics.csv"
        )

        pd.DataFrame(rows).to_csv(
            class_metrics_path,
            index=False,
        )

        # Сохраняем графики и таблицу итоговой валидации.
        mlflow.log_artifacts(
            str(metrics.save_dir),
            artifact_path="validation",
        )

        print(f"Run: {args.name}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"Best weights: {best_path}")


if __name__ == "__main__":
    main()
