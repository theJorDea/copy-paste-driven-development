"""Пул создаётся один раз и освобождается даже при ошибке внутри lifespan."""

import pytest
from fastapi.testclient import TestClient

from health import main


def test_pool_lifecycle(database):
    pool, _, create_pool = database

    with TestClient(main.app) as client:
        create_pool.assert_awaited_once_with(
            dsn="postgresql://test:test@localhost/test",
            min_size=0,
            max_size=3,
            command_timeout=2.0,
        )
        assert main.app.state.db_pool is pool
        assert main.app.state.settings.environment == "development"
        pool.close.assert_not_awaited()
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200
        create_pool.assert_awaited_once()

    pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_pool_closed_on_lifespan_body_failure(database):
    pool, _, _ = database

    with pytest.raises(RuntimeError, match="application failure"):
        async with main.lifespan(main.app):
            raise RuntimeError("application failure")

    pool.close.assert_awaited_once()


def test_pool_startup_failure_is_propagated(database):
    pool, _, create_pool = database
    create_pool.side_effect = ConnectionError("cannot create pool")

    with pytest.raises(ConnectionError, match="cannot create pool"), TestClient(main.app):
        pass

    pool.close.assert_not_awaited()
