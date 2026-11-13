"""Инициализация логов, OpenTelemetry и низкокардинальных метрик."""

import json
import logging
import os
import re
from contextvars import ContextVar
from datetime import UTC, datetime

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False
task_id_context: ContextVar[str | None] = ContextVar("task_id", default=None)
_URL_CREDENTIALS = re.compile(r"(amqps?|postgres(?:ql)?://[^:/\s]+:)[^@/\s]+@", re.IGNORECASE)


def _redact_secrets(value: str) -> str:
    return _URL_CREDENTIALS.sub(r"\1***@", value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_secrets(record.getMessage()),
            "service.name": os.getenv("OTEL_SERVICE_NAME", "api"),
        }
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            payload["trace_id"] = format(span_context.trace_id, "032x")
            payload["span_id"] = format(span_context.span_id, "016x")
        task_id = task_id_context.get()
        if task_id:
            payload["task_id"] = task_id
        for key in ("task_id", "trace_id", "span_id", "error_type", "attempt"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = _redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_telemetry(service_name: str | None = None) -> None:
    global _configured
    if _configured:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "faststream", "asyncpg", "mlflow"):
        library_logger = logging.getLogger(name)
        library_logger.setLevel(level)
        for library_handler in library_logger.handlers:
            library_handler.setFormatter(JsonFormatter())
    resource = Resource.create(
        {
            "service.name": service_name or os.getenv("OTEL_SERVICE_NAME", "api"),
            "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "deployment.environment.name": os.getenv("ENVIRONMENT", "development"),
            "service.instance.id": os.getenv("HOSTNAME", "local"),
        }
    )
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True
