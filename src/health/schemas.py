"""Модели HTTP-ответов сервиса."""

from pydantic import BaseModel


class HealthStatusResponse(BaseModel):
    """Ответ быстрой проверки доступности процесса."""

    status: str


class VersionResponse(BaseModel):
    """Версия установленного приложения."""

    version: str


class DataBaseResponse(BaseModel):
    """Результат проверки PostgreSQL; время указано в миллисекундах."""

    status: str
    database: str
    response_time_ms: float
    db_version: str | None
