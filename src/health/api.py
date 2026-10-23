"""Эндпоинты проверки процесса, версии приложения и PostgreSQL."""

import asyncio
import logging
from importlib.metadata import version
from time import perf_counter

import asyncpg
from fastapi import APIRouter, Request, Response, status

from health.schemas import DataBaseResponse, HealthStatusResponse, VersionResponse

router = APIRouter()
# Версия установленного проекта из метаданных pyproject.toml.
APP_VERSION = version("mlops-health-service")


@router.get("/healthz")
async def health_check() -> HealthStatusResponse:
    """Подтвердить, что приложение отвечает, без проверки БД."""
    return HealthStatusResponse(status="healthy")


@router.get("/api/v1/ping")
async def ping() -> dict[str, str]:
    return {"message": "pong"}


@router.get("/api/v1/version")
async def get_version() -> VersionResponse:
    """Вернуть версию пакета mlops-health-service."""
    return VersionResponse(version=APP_VERSION)


@router.get("/api/v1/health")
async def health_check_v1(
    request: Request,
    response: Response,
) -> DataBaseResponse:
    """Проверить PostgreSQL: вернуть 200 при успехе или 503 при ошибке."""
    # Время включает ожидание соединения из пула и выполнение запроса.
    start_time = perf_counter()
    pool: asyncpg.Pool = request.app.state.db_pool

    try:
        # Общий лимит охватывает как получение соединения, так и SQL-запрос.
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
        # Ошибка зависимости превращается в health-отчёт и записывается со стеком.
        logging.getLogger(__name__).exception("PostgreSQL health check failed")

        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        elapsed = (perf_counter() - start_time) * 1000

        return DataBaseResponse(
            status="unhealthy",
            database="disconnected",
            db_version=None,
            response_time_ms=round(elapsed, 2),
        )
