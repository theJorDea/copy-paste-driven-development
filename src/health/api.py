import asyncio
import base64
import binascii
import json
import logging
import uuid
from importlib.metadata import version
from io import BytesIO
from time import perf_counter

import asyncpg
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status
from opentelemetry import propagate, trace
from PIL import Image, UnidentifiedImageError
from PIL.Image import open as open_image

from health.schemas import (
    DataBaseResponse,
    HealthStatusResponse,
    InferenceResponse,
    TaskAccepted,
    TaskRequest,
    TaskStatus,
    VersionResponse,
)
from health.telemetry import task_id_context

router = APIRouter()
APP_VERSION = version("mlops-health-service")
MAX_IMAGE_SIZE = 10 * 1024 * 1024
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class QueuePublishRejectedError(Exception):
    """RabbitMQ negatively confirmed a publish because bounded capacity was full."""


@router.get("/healthz")
async def health_check() -> HealthStatusResponse:
    return HealthStatusResponse(status="healthy")


@router.get("/api/v1/ping")
async def ping() -> dict[str, str]:
    return {"message": "pong"}


@router.get("/api/v1/version")
async def get_version() -> VersionResponse:
    return VersionResponse(version=APP_VERSION)


@router.get("/api/v1/health")
async def health_check_v1(
    request: Request,
    response: Response,
) -> DataBaseResponse:
    start_time = perf_counter()
    pool: asyncpg.Pool = request.app.state.db_pool

    try:
        async with asyncio.timeout(5):
            async with pool.acquire() as connection:
                db_version = await connection.fetchval("SHOW server_version")

        elapsed = (perf_counter() - start_time) * 1000

        return DataBaseResponse(
            status="healthy",
            database="connected",
            db_version=db_version,
            response_time_ms=round(elapsed, 2),
        )

    except Exception:
        logging.getLogger(__name__).exception("PostgreSQL health check failed")

        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        elapsed = (perf_counter() - start_time) * 1000

        return DataBaseResponse(
            status="unhealthy",
            database="disconnected",
            db_version=None,
            response_time_ms=round(elapsed, 2),
        )


@router.post(
    "/api/v1/infer",
    response_model=InferenceResponse,
)
async def process_image(
    request: Request,
    file: UploadFile = File(...),
) -> InferenceResponse:

    content = await file.read(MAX_IMAGE_SIZE + 1)

    if not content:
        raise HTTPException(400, "Empty image")

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(413, "Image is too large")

    try:
        with open_image(BytesIO(content)) as source:
            source.load()
            image = source.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(400, "Invalid image file") from None

    detector = request.app.state.detector

    predictions = await asyncio.to_thread(
        detector.predict,
        image,
    )

    return InferenceResponse(
        filename=file.filename or "upload",
        count=len(predictions["detections"]),
        model_version=request.app.state.model_version,
        **predictions,
    )


@router.post("/api/v1/tasks", response_model=TaskAccepted, status_code=202)
async def submit_task(payload: TaskRequest, request: Request, response: Response) -> TaskAccepted:
    """Принять задачу только после publisher confirm; очередь сама сигнализирует перегрузку."""
    try:
        image = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "image_base64 must contain valid base64") from None
    if not image or len(image) > MAX_IMAGE_SIZE:
        raise HTTPException(413 if image else 400, "Image is empty or exceeds 10 MiB")
    try:
        with open_image(BytesIO(image)) as source:
            source.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid image file") from None

    task_id = str(uuid.uuid4())
    task_token = task_id_context.set(task_id)
    pool: asyncpg.Pool = request.app.state.db_pool
    broker = request.app.state.rabbit_broker
    if broker is None:
        task_id_context.reset(task_token)
        retry_after = str(request.app.state.retry_after_seconds)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Asynchronous task processing is disabled",
            headers={"Retry-After": retry_after},
        )
    # Запись и публикация находятся в одной DB-транзакции. При nack транзакция
    # откатывается, поэтому 503 не оставляет застрявшую PENDING-задачу.
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO tasks(task_id,status) VALUES($1,'PENDING')", task_id
                )
                try:
                    with tracer.start_as_current_span("publish task"):
                        headers: dict[str, str] = {"task_id": task_id}
                        propagate.inject(headers)
                        confirmation = await broker.publish(
                            {
                                "task_id": task_id,
                                "image_base64": payload.image_base64,
                                "attempt": 0,
                            },
                            queue=request.app.state.task_queue,
                            correlation_id=task_id,
                            headers=headers,
                            persistent=True,
                            mandatory=True,
                        )
                        if confirmation is False or type(confirmation).__name__ in {
                            "Nack",
                            "Reject",
                        }:
                            raise RuntimeError("RabbitMQ negatively confirmed the publish")
                except Exception as exc:
                    raise QueuePublishRejectedError from exc
    except QueuePublishRejectedError as exc:
        # A negative publisher confirm is the queue's reject-publish backpressure signal.
        logger.warning(
            "task_publish_rejected", extra={"task_id": task_id, "error_type": type(exc).__name__}
        )
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        response.headers["Retry-After"] = str(request.app.state.retry_after_seconds)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Task queue is full; retry later",
            headers={"Retry-After": str(request.app.state.retry_after_seconds)},
        ) from exc
    finally:
        task_id_context.reset(task_token)
    return TaskAccepted(task_id=task_id, status="PENDING")


@router.post("/process", response_model=TaskAccepted, status_code=202)
async def process_upload(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
) -> TaskAccepted:
    """Compatibility route for the original multipart /process client; now it queues work."""
    content = await file.read(MAX_IMAGE_SIZE + 1)
    if not content:
        raise HTTPException(400, "Empty image")
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(413, "Image is too large")
    try:
        with open_image(BytesIO(content)) as source:
            source.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid image file") from None
    payload = TaskRequest(image_base64=base64.b64encode(content).decode("ascii"))
    return await submit_task(payload, request, response)


@router.get("/api/v1/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str, request: Request) -> TaskStatus:
    pool: asyncpg.Pool = request.app.state.db_pool
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT status,result,error FROM tasks WHERE task_id=$1", task_id
        )
    if row is None:
        raise HTTPException(404, "Task not found")
    result = row["result"]
    if isinstance(result, str):
        result = json.loads(result)
    return TaskStatus(task_id=task_id, status=row["status"], result=result, error=row["error"])
