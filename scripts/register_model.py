"""Опубликовать веса из завершённого Run в Registry и назначить alias."""

import argparse
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from health.inference import YOLOModel
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="visdrone-yolo")
    parser.add_argument("--alias", help="Например champion; без него только создаётся версия")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    run = client.get_run(args.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Register a successfully completed run")
    with TemporaryDirectory() as temporary:
        weights = mlflow.artifacts.download_artifacts(
            f"runs:/{args.run_id}/model_weights/best.pt", dst_path=temporary
        )
        model_path = Path(temporary) / "model"
        mlflow.pyfunc.save_model(
            path=str(model_path),
            python_model=YOLOModel(),
            artifacts={"weights": weights},
            model_config={"imgsz": args.imgsz, "device": "cpu", "confidence": 0.25},
            pip_requirements=[
                f"{package}=={version(package)}" for package in ("mlflow", "ultralytics", "pillow")
            ],
        )
        client.log_artifacts(args.run_id, str(model_path), artifact_path="model")
    model_version = mlflow.register_model(f"runs:/{args.run_id}/model", args.name)
    if args.alias:
        client.set_registered_model_alias(args.name, args.alias, model_version.version)
    print(f"Registered {args.name} version {model_version.version}; alias: {args.alias}")


if __name__ == "__main__":
    main()
