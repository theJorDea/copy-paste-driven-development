"""Создание FastAPI и управление пулом PostgreSQL."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from health.api import router
from health.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить настройки и управлять пулом соединений на время работы приложения."""

    app_settings = Settings()
    app.state.settings = app_settings

    pool = await asyncpg.create_pool(
        dsn=app_settings.database.database_url,
        min_size=app_settings.database.db_pool_min_size,
        max_size=app_settings.database.db_pool_max_size,
        command_timeout=app_settings.database.db_command_timeout,
    )

    # Один пул используется всеми запросами к БД в рамках процесса приложения.
    app.state.db_pool = pool

    try:
        yield

    finally:
        # Освобождаем соединения при штатном завершении приложения.
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router)
