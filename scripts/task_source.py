"""Пакетный источник: держит в памяти только одно изображение и уважает 503/Retry-After."""

import argparse
import asyncio
import base64
import logging
import os
from pathlib import Path
from time import monotonic

import httpx
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway


async def run(directory: Path, endpoint: str) -> None:
    registry = CollectorRegistry()
    submitted = Counter("pipeline_source_tasks_total", "Tasks accepted by API", registry=registry)
    rejected = Counter(
        "pipeline_source_overload_total", "Queue overload responses", registry=registry
    )
    wait = Histogram(
        "pipeline_source_wait_seconds",
        "Time spent waiting on overload",
        buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
        registry=registry,
    )
    timeout = httpx.Timeout(30.0, connect=5.0)
    tracer = trace.get_tracer(__name__)
    pushgateway = os.getenv("PUSHGATEWAY_URL")
    last_metrics_push = monotonic()

    async def push_metrics(force: bool = False) -> None:
        nonlocal last_metrics_push
        if pushgateway and (force or monotonic() - last_metrics_push >= 5):
            await asyncio.to_thread(
                push_to_gateway, pushgateway, job="batch-source", registry=registry
            )
            last_metrics_push = monotonic()

    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in sorted(directory.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            delay = 0.25
            with tracer.start_as_current_span("source.wait_for_capacity") as source_span:
                source_span.set_attribute("source.file_name", path.name)
                attempts = 0
                while True:
                    attempts += 1
                    response = await client.post(
                        f"{endpoint.rstrip('/')}/api/v1/tasks", json={"image_base64": encoded}
                    )
                    if response.status_code == 202:
                        task_id = response.json()["task_id"]
                        source_span.set_attribute("task.id", task_id)
                        submitted.inc()
                        logging.getLogger("source").info(
                            "task_submitted", extra={"task_id": task_id}
                        )
                        await push_metrics()
                        break
                    if response.status_code != 503:
                        response.raise_for_status()
                    rejected.inc()
                    retry_after = float(response.headers.get("Retry-After", delay))
                    pause = max(retry_after, delay)
                    wait.observe(pause)
                    source_span.add_event(
                        "queue_overloaded",
                        {"attempt": attempts, "retry_after_seconds": retry_after},
                    )
                    logging.getLogger("source").warning(
                        "source_backpressure", extra={"error_type": "queue_full"}
                    )
                    await push_metrics()
                    await asyncio.sleep(pause)
                    delay = min(delay * 2, 10.0)
    await push_metrics(force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Папка с JPG/PNG изображениями")
    parser.add_argument("--endpoint", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    source_otlp = os.getenv("SOURCE_OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if source_otlp:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = source_otlp
    else:
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    from health.telemetry import configure_telemetry

    configure_telemetry("source")
    HTTPXClientInstrumentor().instrument()
    asyncio.run(run(args.directory, args.endpoint))
    trace.get_tracer_provider().force_flush()


if __name__ == "__main__":
    main()
