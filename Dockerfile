FROM python:3.12-slim

WORKDIR /app

# копирование UV из образа Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group web --group train --no-install-project

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN uv sync --frozen --no-default-groups --group web --group train \
    && useradd --create-home appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD uv run --no-sync python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uv", "run", "--no-sync", "uvicorn", "health.main:app", "--host", "0.0.0.0", "--port", "8000"]
