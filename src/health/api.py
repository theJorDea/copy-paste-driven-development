import asyncio
import logging
from importlib.metadata import version
from io import BytesIO
from time import perf_counter

import asyncpg
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status
from PIL import Image, UnidentifiedImageError
from PIL.Image import open as open_image

from health.schemas import (
    DataBaseResponse,
    HealthStatusResponse,
    InferenceResponse,
    VersionResponse,
)

router = APIRouter()
APP_VERSION = version("mlops-health-service")
MAX_IMAGE_SIZE = 10 * 1024 * 1024


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
    "/process",
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
