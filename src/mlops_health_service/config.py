"""Настройки приложения, загружаемые из окружения и локального файла .env."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type Environment = Literal["development", "production", "staging"]


class DataBaseSettings(BaseModel):
    """Параметры подключения к PostgreSQL и ограничения пула соединений."""

    database_url: str

    db_pool_min_size: int = Field(default=0, ge=0)
    db_pool_max_size: int = Field(default=10, gt=0)
    db_command_timeout: float = Field(default=5.0, gt=0)


class Settings(BaseSettings):
    """Конфигурация сервиса; вложенные поля задаются через двойное подчёркивание."""

    # Например, DATABASE__DATABASE_URL задаёт database.database_url.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    database: DataBaseSettings = Field(default_factory=DataBaseSettings)

    environment: Environment = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
