"""Проверки загрузки настроек, обязательных полей и ограничений значений."""

import os

import pytest
from pydantic import ValidationError

from health.config import DataBaseSettings, Settings


def test_database_defaults():
    settings = DataBaseSettings(database_url="postgresql://localhost/test")

    assert settings.db_pool_min_size == 0
    assert settings.db_pool_max_size == 10
    assert settings.db_command_timeout == 5.0


def test_database_url_is_required():
    with pytest.raises(ValidationError, match="database_url"):
        DataBaseSettings()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("db_pool_min_size", -1),
        ("db_pool_max_size", 0),
        ("db_command_timeout", 0),
        ("db_command_timeout", -1),
    ],
)
def test_database_rejects_invalid_limits(field, value):
    with pytest.raises(ValidationError, match=field):
        DataBaseSettings(database_url="postgresql://localhost/test", **{field: value})


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "DATABASE__DATABASE_URL=postgresql://file/test\n"
        "DATABASE__DB_POOL_MAX_SIZE=4\n"
        "ENVIRONMENT=staging\nLOG_LEVEL=WARNING\nUNUSED_SETTING=ignored\n",
        encoding="utf-8",
    )
    # Удаляем только переменные приложения, чтобы окружение машины не меняло результат.
    for name in list(os.environ):
        if name.upper().startswith("DATABASE") or name.upper() in {"ENVIRONMENT", "LOG_LEVEL"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE__DB_POOL_MAX_SIZE", "7")

    settings = Settings(_env_file=dotenv)

    assert settings.database.database_url == "postgresql://file/test"
    assert settings.database.db_pool_max_size == 7
    assert settings.environment == "staging"
    assert settings.log_level == "WARNING"


@pytest.mark.parametrize(("field", "value"), [("environment", "unknown"), ("log_level", "TRACE")])
def test_settings_reject_invalid_choices(field, value):
    with pytest.raises(ValidationError, match=field):
        Settings(
            _env_file=None,
            database={"database_url": "postgresql://localhost/test"},
            **{field: value},
        )
