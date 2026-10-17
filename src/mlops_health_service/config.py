from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type Environment = Literal["development", "production"]


class DataBaseSettings(BaseModel):
    database_url: str = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
    db_pool_min_size: int = Field(default=1, ge=0)
    db_pool_max_size: int = Field(default=10, gt=0)
    db_command_timeout: float = Field(default=5, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    database: DataBaseSettings = Field(default_factory=DataBaseSettings)

    environment: Environment = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
