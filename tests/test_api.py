"""Проверки публичных HTTP-контрактов и отказов зависимости."""

import asyncio
import logging
from importlib.metadata import version

import pytest

from health import api


def test_ping(client):
    response = client.get("/api/v1/ping")

    assert response.status_code == 200
    assert response.json() == {"message": "pong"}


def test_liveness_does_not_access_database(client, database):
    pool, connection, _ = database
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    pool.acquire.assert_not_called()
    connection.fetchval.assert_not_awaited()


def test_version_comes_from_installed_package(client, database):
    response = client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json() == {"version": version("mlops-health-service")}
    database[0].acquire.assert_not_called()


def test_health_success(client, database, monkeypatch):
    pool, connection, _ = database
    ticks = iter([10.0, 10.012346])
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks))

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "database": "connected",
        "response_time_ms": 12.35,
        "db_version": "17.0",
    }
    connection.fetchval.assert_awaited_once_with("SHOW server_version")
    pool.acquire.return_value.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("failure_stage", ["acquire", "query"])
def test_health_database_failure(client, database, monkeypatch, caplog, failure_stage):
    pool, connection, _ = database
    failure = ConnectionError("database unavailable")
    if failure_stage == "acquire":
        pool.acquire.return_value.__aenter__.side_effect = failure
    else:
        connection.fetchval.side_effect = failure
    ticks = iter([20.0, 20.025])
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks))

    with caplog.at_level(logging.ERROR, logger=api.__name__):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "database": "disconnected",
        "response_time_ms": 25.0,
        "db_version": None,
    }
    assert any(
        record.message == "PostgreSQL health check failed" and record.exc_info
        for record in caplog.records
    )
    if failure_stage == "query":
        pool.acquire.return_value.__aexit__.assert_awaited_once()
    else:
        connection.fetchval.assert_not_awaited()
    # Сбой зависимости не мешает быстрой проверке процесса.
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize("failure_stage", ["acquire", "query"])
def test_health_timeout(client, database, monkeypatch, failure_stage):
    pool, connection, _ = database
    real_timeout = asyncio.timeout
    deadlines = []

    def short_timeout(delay):
        deadlines.append(delay)
        return real_timeout(0.01)

    async def wait_forever(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(api.asyncio, "timeout", short_timeout)
    if failure_stage == "acquire":
        pool.acquire.return_value.__aenter__.side_effect = wait_forever
    else:
        connection.fetchval.side_effect = wait_forever

    response = client.get("/api/v1/health")

    assert deadlines == [5]
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["database"] == "disconnected"
    assert body["db_version"] is None
    assert body["response_time_ms"] >= 0
    if failure_stage == "query":
        pool.acquire.return_value.__aexit__.assert_awaited_once()
