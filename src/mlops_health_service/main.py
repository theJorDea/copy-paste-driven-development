from _collections_abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

import asyncpg

from mlops_health_service.config import Settings
from fastapi import FastAPI, Request
from pydantic import BaseModel
from time import perf_counter

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:

    app_settings = Settings()
    app.state.settings = app_settings

    print('Старт приложения. Создание пула соединения с БД...')


    pool = await asyncpg.create_pool(
        dsn=app_settings.database.database_url,
        min_size=app_settings.database.db_pool_min_size,
        max_size=app_settings.database.db_pool_max_size,
        command_timeout=app_settings.database.db_command_timeout
    )

    app.state.db_pool = pool

    try:
        yield

    finally:
        print("Остановка приложения")
        await pool.close()
app = FastAPI(lifespan=lifespan)
APP_VERSION = version("mlops-health-service")


class HealthStatusResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str

class DataBaseResponse(BaseModel):
    status: str
    database: str
    response_time_ms : float
    db_version : str | None
@app.get("/healthz")
async def health_check() -> HealthStatusResponse:
    return HealthStatusResponse(status="healthy")


@app.get("/api/v1/version")
async def get_version() -> VersionResponse:
    return VersionResponse(version=APP_VERSION)


@app.get("/api/v1/health")
async def health_check_v1(request: Request) -> DataBaseResponse:
    start_time = perf_counter()
    try:
        poll: asyncpg.pool = request.app.state.db_pool
        db_version = await poll.fetchval('SELECT version()')
        time = perf_counter() - start_time * 1000
        return {"status": "healthy", "database": "connected", "db_version": db_version, "response_time_ms": time}
    except Exception as e:
        time = perf_counter() - start_time * 1000
        return {"status": "unhealthy", "database": "unconnected", "db_version": None, "response_time_ms": time , "error": str(e)}