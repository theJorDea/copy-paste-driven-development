from pydantic import BaseModel, Field


class HealthStatusResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str


class DataBaseResponse(BaseModel):
    status: str
    database: str
    response_time_ms: float
    db_version: str | None


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class InferenceResponse(BaseModel):
    model_version: str
    filename: str
    image_width: int
    image_height: int
    count: int
    detections: list[Detection]


class TaskRequest(BaseModel):
    """Изображение в base64 для асинхронной пакетной обработки."""

    image_base64: str = Field(min_length=1, max_length=14_000_000)


class TaskAccepted(BaseModel):
    task_id: str
    status: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
    error: str | None = None
