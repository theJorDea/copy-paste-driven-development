from os import getenv
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

type Environment = Literal["development", "production", 'staging']


class DataBaseSettings(BaseModel):
    database_url: str | None = getenv("DATABASE__URL")
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
