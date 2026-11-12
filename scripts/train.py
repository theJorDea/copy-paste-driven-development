"""Обучение и итоговая валидация YOLO с явным логированием в MLflow."""

import argparse
import hashlib
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
from ultralytics import YOLO, settings
from ultralytics.engine.trainer import BaseTrainer

import mlflow

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "visdrone.yaml"
DATASETS = {
    "training": ("visdrone-train", "visdrone2019-det-train"),
    "validation": ("visdrone", "visdrone2019-det-val"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Оценить готовые веса без обучения"
    )
    return parser.parse_args()


def log_dataset(split: str) -> None:
    folder, name = DATASETS[split]
    manifest_path = ROOT / "data" / folder / "manifest.csv"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    dataset = mlflow.data.from_pandas(
        pd.read_csv(manifest_path),
        source=manifest_path.resolve().as_uri(),
        name=name,
        digest=digest[:32],
    )
    mlflow.log_input(dataset, context=split)
    mlflow.set_tag(f"{split}_manifest_sha256", digest)
    mlflow.log_artifact(str(manifest_path), artifact_path=f"dataset/{split}")


def log_epoch(trainer: BaseTrainer) -> None:
    metrics = {
        **trainer.metrics,
        **trainer.lr,
        **trainer.label_loss_items(trainer.tloss, prefix="train"),
    }
    mlflow.log_metrics(
        {key.replace("(", "").replace(")", ""): float(value) for key, value in metrics.items()},
        step=trainer.epoch,
    )


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("visdrone-training")
    # Run открывает и закрывает только этот скрипт, а не интеграция Ultralytics.
    settings.update({"mlflow": False})
    with mlflow.start_run(run_name=args.name) as run:
        log_dataset("training")
        log_dataset("validation")
        mlflow.log_artifact(str(DATASET_CONFIG), artifact_path="config")
        mlflow.log_params({key: value for key, value in vars(args).items() if value is not None})
        model = YOLO(str(args.resume_from or args.model))
        if not args.evaluate_only:
            model.add_callback("on_fit_epoch_end", log_epoch)
            start = perf_counter()
            if args.resume_from:
                model.train(resume=True)
            else:
                model.train(
                    data=str(DATASET_CONFIG),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                    fraction=args.fraction,
                    optimizer="AdamW",
                    lr0=0.000909,
                    project=str(ROOT / "runs" / "train"),
                    name=args.name,
                    plots=True,
                    patience=0,
                )
            mlflow.log_metric("train_seconds", perf_counter() - start)
            best_path = Path(model.trainer.best)
        else:
            best_path = Path(args.model)
        mlflow.log_artifact(str(best_path), artifact_path="model_weights")
        # При восстановлении baseline сохраняем также историю завершённых эпох.
        for filename in ("args.yaml", "results.csv"):
            path = best_path.parent.parent / filename
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="training")
        metrics = YOLO(str(best_path)).val(
            data=str(DATASET_CONFIG),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            project=str(ROOT / "runs" / "validation"),
            name=args.name,
        )
        mlflow.log_metrics(
            {
                "final/map50_95": float(metrics.box.map),
                "final/map50": float(metrics.box.map50),
                "final/precision": float(metrics.box.mp),
                "final/recall": float(metrics.box.mr),
            }
        )
        rows = []
        for index, class_id in enumerate(metrics.box.ap_class_index):
            precision, recall, map50, map50_95 = metrics.box.class_result(index)
            rows.append(
                {
                    "class": metrics.names[int(class_id)],
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                }
            )
        pd.DataFrame(rows).to_csv(Path(metrics.save_dir) / "per_class_metrics.csv", index=False)
        mlflow.log_artifacts(str(metrics.save_dir), artifact_path="validation")
        print(f"Run ID: {run.info.run_id}")
        print(f"mAP50-95: {metrics.box.map:.4f}; best weights: {best_path}")


if __name__ == "__main__":
    main()
