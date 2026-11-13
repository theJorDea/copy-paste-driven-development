"""Контракт инференса и загрузка закреплённой версии модели при старте."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from health import main
from health.config import Settings
from health.inference import Detector, YOLOModel


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (16, 12)).save(stream, format="PNG")
    return stream.getvalue()


def test_process_uses_loaded_model_and_returns_version(client):
    detector = main.app.state.detector
    detector.predict.return_value = {"image_width": 16, "image_height": 12, "detections": []}
    for _ in range(2):
        response = client.post(
            "/api/v1/infer", files={"file": ("frame.png", image_bytes(), "image/png")}
        )
        assert response.status_code == 200
        assert response.json() == {
            "model_version": "1",
            "filename": "frame.png",
            "count": 0,
            "image_width": 16,
            "image_height": 12,
            "detections": [],
        }
    main.load_model.assert_called_once()
    assert detector.predict.call_count == 2
    assert detector.predict.call_args.args[0].mode == "RGB"


@pytest.mark.parametrize(
    ("content", "code"),
    [(b"", 400), (b"not an image", 400), (b"x" * (10 * 1024 * 1024 + 1), 413)],
    ids=["empty", "not-image", "too-large"],
)
def test_process_rejects_invalid_uploads(client, content, code):
    response = client.post("/api/v1/infer", files={"file": ("bad.png", content, "image/png")})
    assert response.status_code == code
    main.app.state.detector.predict.assert_not_called()


def test_model_startup_failure_closes_database_pool(database):
    main.load_model.side_effect = RuntimeError("registry unavailable")
    with pytest.raises(RuntimeError, match="registry unavailable"), TestClient(main.app):
        pass
    database[0].close.assert_awaited_once()


def test_alias_is_resolved_before_loading(monkeypatch):
    registry = MagicMock()
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    model = MagicMock()
    loader = MagicMock(return_value=model)
    monkeypatch.setattr(main, "MlflowClient", lambda: registry)
    monkeypatch.setattr(main.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(main.mlflow.pyfunc, "load_model", loader)
    settings = Settings(_env_file=None, database={"database_url": "postgresql://test/test"})
    assert main.load_model(settings) == (model, "7")
    registry.get_model_version_by_alias.assert_called_once_with("visdrone-yolo", "champion")
    assert loader.call_args.args == ("models:/visdrone-yolo/7",)


def test_detector_converts_boxes(monkeypatch, tmp_path):
    box = MagicMock()
    box.cls.__getitem__.return_value.item.return_value = 0
    box.conf.__getitem__.return_value.item.return_value = 0.9
    box.xyxy.__getitem__.return_value.cpu.return_value.tolist.return_value = [1, 2, 10, 11]
    yolo = MagicMock()
    yolo.names = {0: "car"}
    yolo.predict.return_value = [SimpleNamespace(boxes=[box])]
    monkeypatch.setattr("health.inference.YOLO", MagicMock(return_value=yolo))
    wrapper = YOLOModel()
    wrapper.load_context(
        SimpleNamespace(
            model_config={"imgsz": 480},
            artifacts={"weights": str(tmp_path / "best.pt")},
        )
    )
    assert isinstance(wrapper.detector, Detector)
    result = wrapper.predict(None, Image.new("RGB", (16, 12)))
    assert result["detections"] == [
        {
            "class_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": {"x1": 1.0, "y1": 2.0, "x2": 10.0, "y2": 11.0},
        }
    ]
    assert yolo.predict.call_args.kwargs["imgsz"] == 480
