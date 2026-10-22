"""Проверки настоящего образа приложения и PostgreSQL через Docker Compose."""

import asyncio
import json
import subprocess
import uuid
from importlib.metadata import version
from pathlib import Path

import asyncpg
import httpx
import pytest
from fastapi.testclient import TestClient

from health.main import app

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stack(tmp_path_factory, pytestconfig):
    """Собрать изолированный стек; удалить только его контейнеры и том после тестов."""
    override = tmp_path_factory.mktemp("compose") / "ports.yml"
    # Убираем фиксированные порты, чтобы не затронуть локальный запущенный проект.
    build_network = pytestconfig.getoption("--build-network")
    override.write_text(
        "services:\n"
        "  app:\n"
        '    ports: !override ["127.0.0.1::8000"]\n'
        "  postgres:\n"
        '    ports: !override ["127.0.0.1::5432"]\n',
        encoding="utf-8",
    )
    project = f"health-test-{uuid.uuid4().hex[:10]}"
    command = [
        "docker",
        "compose",
        "--progress",
        "plain",
        "--project-name",
        project,
        "--file",
        str(ROOT / "docker-compose.yml"),
        "--file",
        str(override),
    ]

    def compose(*args, timeout=120):
        result = subprocess.run(
            [*command, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
        return result.stdout.strip()

    try:
        if build_network:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--network",
                    build_network,
                    "--tag",
                    f"{project}-app",
                    str(ROOT),
                ],
                check=True,
                timeout=600,
            )
        else:
            compose("build", timeout=600)
        compose("up", "--no-build", "--detach", "--wait", "--wait-timeout", "120", timeout=180)
        app_address = compose("port", "app", "8000")
        db_address = compose("port", "postgres", "5432")
        yield compose, f"http://{app_address}", f"postgresql://postgres:postgres@{db_address}/mlops"
    finally:
        # Логи сохраняются в выводе pytest при падении; пользовательский стек не затрагивается.
        logs = subprocess.run(
            [*command, "logs", "--no-color"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        print(logs.stdout + logs.stderr)
        compose("down", "--volumes", "--remove-orphans")


async def database_version(dsn):
    """Прочитать версию напрямую, независимо от HTTP-эндпоинта."""
    connection = await asyncpg.connect(dsn, timeout=5)
    try:
        return await connection.fetchval("SHOW server_version")
    finally:
        await connection.close()


def assert_healthy(body, expected_version):
    assert body["status"] == "healthy"
    assert body["database"] == "connected"
    assert body["db_version"] == expected_version
    assert isinstance(body["response_time_ms"], (int, float))
    assert body["response_time_ms"] >= 0


def test_container_endpoints_and_healthcheck(stack):
    compose, url, dsn = stack
    expected_version = asyncio.run(database_version(dsn))

    with httpx.Client(base_url=url, timeout=10) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        assert response.json() == {"version": version("mlops-health-service")}
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert_healthy(response.json(), expected_version)

    container = compose("ps", "--quiet", "app")
    result = subprocess.run(
        ["docker", "inspect", container],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    details = json.loads(result.stdout)[0]
    assert details["State"]["Health"]["Status"] == "healthy"
    assert details["Config"]["User"] == "appuser"


def test_local_application_with_real_postgres(stack, monkeypatch):
    """Выполнить код под coverage с настоящим asyncpg, без подмены пула."""
    _, _, dsn = stack
    monkeypatch.setenv("DATABASE__DATABASE_URL", dsn)
    monkeypatch.setenv("DATABASE__DB_POOL_MIN_SIZE", "0")
    monkeypatch.setenv("DATABASE__DB_POOL_MAX_SIZE", "3")
    monkeypatch.setenv("DATABASE__DB_COMMAND_TIMEOUT", "5")
    expected_version = asyncio.run(database_version(dsn))

    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            assert_healthy(response.json(), expected_version)
        pool = app.state.db_pool

    assert pool.is_closing()


def test_container_reports_database_outage_and_recovery(stack):
    compose, url, _ = stack
    compose("stop", "postgres")
    try:
        with httpx.Client(base_url=url, timeout=10) as client:
            response = client.get("/api/v1/health")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "unhealthy"
            assert body["database"] == "disconnected"
            assert body["db_version"] is None
            assert body["response_time_ms"] >= 0
            assert client.get("/healthz").status_code == 200
    finally:
        compose("up", "--detach", "--wait", "--wait-timeout", "120", "postgres", timeout=150)

    response = httpx.get(f"{url}/api/v1/health", timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
