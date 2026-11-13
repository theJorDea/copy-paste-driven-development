"""FastStream inference worker; ack выполняется только после транзакции результата."""

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter
from typing import Protocol

import asyncpg
from faststream import AckPolicy, Context, ContextRepo, FastStream
from faststream.rabbit import RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from faststream.rabbit.schemas import Channel
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.trace import Status, StatusCode
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from health.config import Settings
from health.messaging import (
    BACKPRESSURE_ENABLED,
    DLQ_QUEUE,
    PREFETCH_COUNT,
    QUEUE_LIMIT,
    QUEUE_NAME,
    RABBIT_URL,
    RETRY_QUEUE,
    TASK_QUEUE,
)
from health.telemetry import configure_telemetry, task_id_context

configure_telemetry(os.getenv("OTEL_SERVICE_NAME", "worker"))
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
broker = RabbitBroker(
    RABBIT_URL, graceful_timeout=float(os.getenv("WORKER_GRACEFUL_TIMEOUT", "300"))
)
MAX_ATTEMPTS = int(os.getenv("TASK_MAX_ATTEMPTS", "3"))
TASKS_TOTAL = Counter("pipeline_worker_tasks_total", "Worker tasks by outcome", ["outcome"])
INFERENCE_SECONDS_METRIC = Histogram(
    "pipeline_inference_seconds",
    "Inference duration",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
IN_PROGRESS = Gauge("pipeline_worker_in_progress", "Tasks currently handled")
RETRIES_TOTAL = Counter("pipeline_worker_retries_total", "Scheduled task retries")
QUEUE_LIMIT_METRIC = Gauge(
    "pipeline_queue_limit", "Maximum ready messages in work queue", ["queue"]
)


class InferenceModel(Protocol):
    def predict(self, image: Image.Image) -> dict[str, object]: ...


def delivery_headers(task_id: str, attempt: int) -> dict[str, str]:
    headers = {"task_id": task_id, "attempt": str(attempt)}
    propagate.inject(headers)
    return headers


async def health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Minimal HTTP liveness endpoint on the worker event loop."""
    try:
        await reader.readline()
        body = b'{"status":"healthy"}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


@asynccontextmanager
async def lifespan(context: ContextRepo) -> AsyncIterator[None]:
    settings = Settings()
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    version = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)
    model = mlflow.pyfunc.load_model(
        f"models:/{settings.model_name}/{version.version}",
        model_config={"device": settings.model_device, "confidence": settings.model_confidence},
    )
    AsyncPGInstrumentor().instrument()
    pool = await asyncpg.create_pool(dsn=settings.database.database_url)
    await broker.connect()
    for queue in (TASK_QUEUE, RETRY_QUEUE, DLQ_QUEUE):
        await broker.declare_queue(queue)
    start_http_server(int(os.getenv("WORKER_METRICS_PORT", "8001")))
    health_server = await asyncio.start_server(
        health_handler, "0.0.0.0", int(os.getenv("WORKER_HEALTH_PORT", "8081"))
    )
    context.set_global("pool", pool)
    context.set_global("model", model)
    context.set_global("model_version", str(version.version))
    QUEUE_LIMIT_METRIC.labels(queue=QUEUE_NAME).set(QUEUE_LIMIT if BACKPRESSURE_ENABLED else 0)
    logger.info(
        "model_loaded",
        extra={"model_alias": settings.model_alias, "model_version": str(version.version)},
    )
    try:
        yield
    finally:
        health_server.close()
        await health_server.wait_closed()
        await pool.close()


app = FastStream(broker, lifespan=lifespan)


@broker.subscriber(
    TASK_QUEUE, channel=Channel(prefetch_count=PREFETCH_COUNT), ack_policy=AckPolicy.MANUAL
)
async def handle_task(
    message: dict,
    msg: RabbitMessage,
    pool: asyncpg.Pool = Context(),
    model: InferenceModel = Context(),
    model_version: str = Context(),
) -> None:
    task_id = str(message.get("task_id", "unknown"))
    attempt = int(message.get("attempt", 0))
    task_token = task_id_context.set(task_id)
    IN_PROGRESS.inc()
    try:
        parent_context = propagate.extract(msg.headers or {})
        with tracer.start_as_current_span("process", context=parent_context) as process_span:
            process_span.set_attribute("messaging.delivery.attempt", attempt)
            try:
                async with pool.acquire() as connection:
                    state = await connection.fetchval(
                        "SELECT status FROM tasks WHERE task_id=$1", task_id
                    )
                if state in ("COMPLETED", "DLQ"):
                    await msg.ack()
                    return
                if state is None:
                    # Publisher confirm can precede the API transaction commit by a few ms.
                    failed = {**message, "attempt": attempt + 1}
                    if attempt + 1 < MAX_ATTEMPTS:
                        await broker.publish(
                            failed,
                            queue=RETRY_QUEUE,
                            correlation_id=task_id,
                            headers=delivery_headers(task_id, attempt + 1),
                        )
                        RETRIES_TOTAL.inc()
                    else:
                        await broker.publish(
                            failed,
                            queue=DLQ_QUEUE,
                            correlation_id=task_id,
                            headers=delivery_headers(task_id, attempt + 1),
                        )
                        TASKS_TOTAL.labels(outcome="dlq").inc()
                        logger.error("task_without_database_record_sent_to_dlq")
                    await msg.ack()
                    return

                raw = base64.b64decode(message["image_base64"], validate=True)
                with Image.open(BytesIO(raw)) as source:
                    image = source.convert("RGB")
                with tracer.start_as_current_span("inference") as span:
                    span.set_attribute("task.id", task_id)
                    span.set_attribute("model.alias", Settings().model_alias)
                    span.set_attribute("model.version", model_version)
                    inference_started = perf_counter()
                    await asyncio.sleep(float(os.getenv("INFERENCE_SECONDS", "0")))
                    predictions = await asyncio.to_thread(model.predict, image)
                    INFERENCE_SECONDS_METRIC.observe(perf_counter() - inference_started)
                output = {
                    "model_version": model_version,
                    "filename": "batch-upload",
                    "image_width": predictions["image_width"],
                    "image_height": predictions["image_height"],
                    "count": len(predictions["detections"]),
                    **predictions,
                }
                with tracer.start_as_current_span("db.write") as db_span:
                    db_span.set_attribute("task.id", task_id)
                    async with pool.acquire() as connection:
                        async with connection.transaction():
                            await connection.execute(
                                """UPDATE tasks SET status='COMPLETED', result=$2::jsonb,
                                error=NULL, updated_at=now() WHERE task_id=$1""",
                                task_id,
                                json.dumps(output),
                            )
                logger.info("task_completed", extra={"task_id": task_id, "attempt": attempt})
                TASKS_TOTAL.labels(outcome="ok").inc()
                await msg.ack()
            except Exception as exc:
                process_span.record_exception(exc)
                process_span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                TASKS_TOTAL.labels(outcome="error").inc()
                logger.exception(
                    "task_processing_failed", extra={"task_id": task_id, "attempt": attempt}
                )
                failed = {**message, "attempt": attempt + 1, "last_error": type(exc).__name__}
                if attempt + 1 < MAX_ATTEMPTS:
                    await broker.publish(
                        failed,
                        queue=RETRY_QUEUE,
                        correlation_id=task_id,
                        headers=delivery_headers(task_id, attempt + 1),
                    )
                    async with pool.acquire() as connection:
                        await connection.execute(
                            "UPDATE tasks SET status='RETRYING', updated_at=now() WHERE task_id=$1",
                            task_id,
                        )
                    RETRIES_TOTAL.inc()
                else:
                    await broker.publish(
                        failed,
                        queue=DLQ_QUEUE,
                        correlation_id=task_id,
                        headers=delivery_headers(task_id, attempt + 1),
                    )
                    async with pool.acquire() as connection:
                        await connection.execute(
                            """UPDATE tasks SET status='DLQ', error=$2,
                            updated_at=now() WHERE task_id=$1""",
                            task_id,
                            type(exc).__name__,
                        )
                    logger.error(
                        "task_sent_to_dlq", extra={"task_id": task_id, "attempt": attempt + 1}
                    )
                    TASKS_TOTAL.labels(outcome="dlq").inc()
                await msg.ack()
    finally:
        IN_PROGRESS.dec()
        task_id_context.reset(task_token)


if __name__ == "__main__":
    app.run()
