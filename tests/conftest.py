"""Изолированное приложение: настоящий lifespan, но без внешнего PostgreSQL."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from health import main


def pytest_addoption(parser):
    parser.addoption(
        "--build-network",
        default=None,
        help="Сеть сборки Docker; --build-network=host при проблемах сети Docker на Linux",
    )


@pytest.fixture
def database(monkeypatch):
    """Подменить границу доступа к БД, сохранив асинхронные контекстные менеджеры."""
    monkeypatch.setenv("DATABASE__DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("DATABASE__DB_POOL_MIN_SIZE", "0")
    monkeypatch.setenv("DATABASE__DB_POOL_MAX_SIZE", "3")
    monkeypatch.setenv("DATABASE__DB_COMMAND_TIMEOUT", "2")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    # Локальный .env разработчика не должен влиять на тесты.
    monkeypatch.setitem(main.Settings.model_config, "env_file", None)

    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value="17.0")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.close = AsyncMock()
    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(main.asyncpg, "create_pool", create_pool)
    return pool, connection, create_pool


@pytest.fixture
def client(database):
    """Вход в TestClient запускает lifespan, выход закрывает пул."""
    with TestClient(main.app) as test_client:
        yield test_client
