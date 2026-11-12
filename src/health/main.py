import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

import mlflow
from health.api import router
from health.config import Settings
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


def load_model(settings: Settings) -> tuple[mlflow.pyfunc.PyFuncModel, str]:
    # RustFS доступен серверу MLflow, поэтому веса скачиваем через Tracking Server.
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_version = MlflowClient().get_model_version_by_alias(
        settings.model_name, settings.model_alias
    )
    # Фиксируем версию: переключение alias подействует после перезапуска.
    model = mlflow.pyfunc.load_model(
        f"models:/{settings.model_name}/{model_version.version}",
        model_config={"device": settings.model_device, "confidence": settings.model_confidence},
    )
    return model, model_version.version


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()

    app.state.settings = settings

    pool = await asyncpg.create_pool(
        dsn=settings.database.database_url,
        min_size=settings.database.db_pool_min_size,
        max_size=settings.database.db_pool_max_size,
        command_timeout=settings.database.db_command_timeout,
    )

    try:
        app.state.db_pool = pool

        app.state.detector, app.state.model_version = await asyncio.to_thread(load_model, settings)
        logger.info("Loaded %s version %s", settings.model_name, app.state.model_version)

        yield

    finally:
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router)
