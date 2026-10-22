FROM python:3.12-slim

WORKDIR /app

# копирование UV из образа Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# настройки Python и UV
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group web

COPY src/ ./src/

RUN useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD uv run --no-sync python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uv", "run", "uvicorn", "mlops_health_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
