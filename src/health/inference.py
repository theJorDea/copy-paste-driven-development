from pathlib import Path
from typing import Any

import mlflow.pyfunc
from PIL import Image
from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def predict(self, image: Image.Image) -> dict[str, Any]:
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                coordinates = box.xyxy[0].cpu().tolist()

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "confidence": float(box.conf[0].item()),
                        "bbox": {
                            "x1": float(coordinates[0]),
                            "y1": float(coordinates[1]),
                            "x2": float(coordinates[2]),
                            "y2": float(coordinates[3]),
                        },
                    }
                )

        return {
            "image_width": image.width,
            "image_height": image.height,
            "detections": detections,
        }


class YOLOModel(mlflow.pyfunc.PythonModel):
    """MLflow хранит веса и параметры, а Detector выполняет предсказание."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        config = context.model_config
        self.detector = Detector(
            Path(context.artifacts["weights"]),
            confidence=config.get("confidence", 0.25),
            imgsz=config.get("imgsz", 640),
            device=config.get("device", "cpu"),
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Image.Image,
        params: dict | None = None,
    ) -> dict[str, Any]:
        return self.detector.predict(model_input)
