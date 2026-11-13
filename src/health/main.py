import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi_instrumentator import Instrumentator
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

import mlflow
from health.api import router
from health.config import Settings
from health.messaging import DLQ_QUEUE, RETRY_QUEUE, TASK_QUEUE
from health.messaging import broker as rabbit_broker
from health.telemetry import configure_telemetry
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
    configure_telemetry("api")
    AsyncPGInstrumentor().instrument()

    app.state.settings = settings

    pool = await asyncpg.create_pool(
        dsn=settings.database.database_url,
        min_size=settings.database.db_pool_min_size,
        max_size=settings.database.db_pool_max_size,
        command_timeout=settings.database.db_command_timeout,
    )

    try:
        app.state.db_pool = pool
        tasks_enabled = os.getenv("TASKS_ENABLED", "false").lower() == "true"
        app.state.rabbit_broker = None
        if tasks_enabled:
            async with pool.acquire() as connection:
                await connection.execute(
                    """CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL CHECK (
                            status IN ('PENDING','RETRYING','COMPLETED','DLQ')
                        ),
                        result JSONB,
                        error TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )"""
                )
            await rabbit_broker.start()
            for queue in (TASK_QUEUE, RETRY_QUEUE, DLQ_QUEUE):
                await rabbit_broker.declare_queue(queue)
            app.state.rabbit_broker = rabbit_broker
        app.state.task_queue = TASK_QUEUE
        app.state.retry_after_seconds = int(os.getenv("RETRY_AFTER_SECONDS", "2"))

        app.state.detector, app.state.model_version = await asyncio.to_thread(load_model, settings)
        logger.info("Loaded %s version %s", settings.model_name, app.state.model_version)

        yield

    finally:
        if app.state.rabbit_broker is not None:
            await rabbit_broker.stop()
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
FastAPIInstrumentor.instrument_app(app)
