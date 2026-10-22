"""Асинхронный сервис проверки состояния приложения и PostgreSQL.

`/healthz` проверяет, что процесс отвечает, без обращения к внешним системам.
`/api/v1/version` возвращает версию установленного пакета приложения.
`/api/v1/health` проверяет БД и возвращает её версию и время ответа.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from time import perf_counter

import asyncpg
from fastapi import FastAPI, Request, Response, status
from pydantic import BaseModel

from mlops_health_service.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить настройки и управлять пулом соединений на время работы приложения."""

    app_settings = Settings()
    app.state.settings = app_settings

    pool = await asyncpg.create_pool(
        dsn=app_settings.database.database_url,
        min_size=app_settings.database.db_pool_min_size,
        max_size=app_settings.database.db_pool_max_size,
        command_timeout=app_settings.database.db_command_timeout,
    )

    # Один пул используется всеми запросами к БД в рамках процесса приложения.
    app.state.db_pool = pool

    try:
        yield

    finally:
        # Освобождаем соединения при штатном завершении приложения.
        await pool.close()


app = FastAPI(lifespan=lifespan)
# Источник версии — метаданные пакета, установленные из pyproject.toml.
APP_VERSION = version("mlops-health-service")


class HealthStatusResponse(BaseModel):
    """Ответ быстрой проверки доступности процесса."""

    status: str


class VersionResponse(BaseModel):
    """Версия установленного приложения."""

    version: str


class DataBaseResponse(BaseModel):
    """Результат проверки PostgreSQL; время указано в миллисекундах."""

    status: str
    database: str
    response_time_ms: float
    db_version: str | None


@app.get("/healthz")
async def health_check() -> HealthStatusResponse:
    """Подтвердить, что приложение отвечает, без проверки БД."""
    return HealthStatusResponse(status="healthy")


@app.get("/api/v1/version")
async def get_version() -> VersionResponse:
    """Вернуть версию пакета mlops-health-service."""
    return VersionResponse(version=APP_VERSION)


@app.get("/api/v1/health")
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
