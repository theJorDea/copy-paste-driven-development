# Загруженность дороги по аэроснимкам

Учебный проект по MLOps. Цель — находить транспорт на снимках с дрона
и по его количеству и плотности оценивать загруженность дороги.

Сейчас есть FastAPI-сервис с проверкой PostgreSQL, анализ разметки VisDrone
и регистрация версии датасета в MLflow. Обучение модели, инференс и оценка
загруженности ещё не реализованы.

## Что используется

| Задача | Инструменты |
| --- | --- |
| Окружение и зависимости | Python 3.12, uv |
| HTTP API | FastAPI, Uvicorn, Pydantic Settings |
| Подключение к БД | PostgreSQL 16, asyncpg |
| Анализ данных | NumPy, pandas, Matplotlib, Pillow |
| Метрики и артефакты | MLflow, PostgreSQL, S3-совместимое хранилище RustFS |
| Проверки | Ruff, pytest, coverage, pre-commit, GitHub Actions |
| Контейнеры | Docker Compose, публикация образа в GHCR |

Зависимости разделены на группы: `web` — сервис, `dev` — проверки,
`eda` — анализ данных и клиент MLflow. Команда `uv sync` по умолчанию
устанавливает `web` и `dev`; группа `eda` подключается отдельно.

## Структура

```text
src/health/                  # API, конфигурация, модели ответов и пул PostgreSQL
tests/                       # Модульные и Docker-интеграционные тесты
scripts/
    eda_visdrone.py           # Статистика разметки и графики в MLflow
    register_dataset.py      # Манифест, SHA-256 и регистрация датасета в MLflow
mlflow/
    compose.yml              # MLflow, его PostgreSQL и RustFS
    Dockerfile               # Образ сервера MLflow с драйверами PostgreSQL и S3
.github/workflows/ci.yml     # Проверки и публикация образа
Dockerfile                   # Образ FastAPI-сервиса
docker-compose.yml           # Приложение и его PostgreSQL
pyproject.toml               # Зависимости, настройки Ruff и pytest
uv.lock                      # Зафиксированные версии зависимостей
.pre-commit-config.yaml      # Git-хуки для проверки оформления и файлов
```

Локальный датасет хранится в `data/visdrone/`. Каталог `data/`, файлы `.env`
и виртуальное окружение исключены из Git.

## Запуск API

Все команды выполняются из корня репозитория. Для запуска в контейнерах
нужны работающий Docker Engine и Docker Compose.

```bash
docker compose up --build -d
```

Compose поднимает приложение на порту `8000` и PostgreSQL на порту `5433`
хоста. Параметры подключения приложения уже заданы в `docker-compose.yml`.

- [Swagger UI](http://127.0.0.1:8000/docs) — описание и вызов эндпоинтов.
- [Проверка процесса](http://127.0.0.1:8000/healthz).
- [Проверка PostgreSQL](http://127.0.0.1:8000/api/v1/health).

Остановить стек: `docker compose down`. Данные БД сохраняются в томе
`postgres-data`.

### Локальная разработка

Понадобятся Python 3.12 и установленный uv. Установите зависимости
из `uv.lock` и запустите только БД:

```bash
uv sync --locked
docker compose up -d postgres
```

Создайте в корне файл `.env` с подключением к этой БД:

```dotenv
DATABASE__DATABASE_URL=postgresql://postgres:postgres@localhost:5433/mlops
```

Если используете свой PostgreSQL, укажите его адрес и заранее создайте БД.
Затем запустите приложение:

```bash
uv run --locked uvicorn health.main:app --host 127.0.0.1 --port 8000 --reload
```

| Переменная окружения | По умолчанию | Назначение |
| --- | --- | --- |
| `DATABASE__DATABASE_URL` | Обязательная | URL подключения к PostgreSQL |
| `DATABASE__DB_POOL_MIN_SIZE` | `0` | Минимальный размер пула |
| `DATABASE__DB_POOL_MAX_SIZE` | `10` | Максимальный размер пула |
| `DATABASE__DB_COMMAND_TIMEOUT` | `5.0` | Таймаут SQL-команд, секунды |
| `ENVIRONMENT` | `development` | `development`, `staging` или `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` или `ERROR` |

Переменные окружения имеют приоритет над `.env`. `ENVIRONMENT` и `LOG_LEVEL`
пока только загружаются в настройки; настройка логирования через них не реализована.

### Эндпоинты

| Метод и путь | Результат |
| --- | --- |
| `GET /healthz` | `200`, `{"status": "healthy"}`; без запроса к БД |
| `GET /api/v1/ping` | `200`, `{"message": "pong"}` |
| `GET /api/v1/version` | `200`, версия установленного пакета `mlops-health-service` |
| `GET /api/v1/health` | `200` при доступной БД, `503` при ошибке или таймауте |

Проверка БД получает соединение из общего пула и выполняет
`SHOW server_version`. Общий таймаут получения соединения и запроса — 5 секунд.
В ответе есть `status`, `database`, `response_time_ms` и `db_version`;
при ошибке `db_version` равен `null`.

## MLflow и данные

MLflow запускается отдельным Compose-стеком: PostgreSQL хранит метаданные,
RustFS — артефакты. При старте создаётся бакет `mlflow-artifacts`.

Создайте `mlflow/.env` и задайте значения для локального стенда:

```dotenv
MLFLOW_DB_PASSWORD=local-mlflow-db
S3_ACCESS_KEY=local-mlflow-access-key
S3_SECRET_KEY=local-mlflow-secret-key
```

```bash
docker compose --env-file mlflow/.env -f mlflow/compose.yml up --build -d
uv sync --locked --group eda
```

Интерфейс [MLflow](http://localhost:5000) доступен на порту `5000`,
консоль [RustFS](http://localhost:9001) — на `9001`. Скрипты обращаются
к `http://localhost:5000` и используют эксперимент `visdrone-eda`.

Для остановки:

```bash
docker compose --env-file mlflow/.env -f mlflow/compose.yml down
```

### Подготовка VisDrone

Скачайте `valset` из раздела **Task 1: Object Detection in Images**
в [репозитории VisDrone](https://github.com/VisDrone/VisDrone-Dataset#download).
Разместите содержимое `VisDrone2019-DET-val` так:

```text
data/visdrone/
    images/                  # Изображения .jpg
    annotations/             # Файлы .txt с совпадающими именами
```

Оба скрипта пропускают записи с `score == 0`, неизвестными категориями
и неположительными размерами рамки. Транспортом считаются классы
`4–10`: car, van, truck, tricycle, awning-tricycle, bus и motor.

### Анализ разметки

```bash
uv run --locked --group eda python scripts/eda_visdrone.py
```

Запуск `validation-eda-v2` сохраняет число изображений и объектов, статистику
транспорта, относительных площадей рамок и число некорректных рамок.
В артефактах `plots/` появляются четыре графика: распределение классов,
площадей рамок, транспорта по кадрам и примеры минимальной, медианной
и максимальной плотности.

В текущем EDA повторяется блок накопления площадей и подсчёта объектов
и транспорта. Из-за этого количество объектов и транспорта на кадр
завышено вдвое; распределение классов считается отдельно. Этот участок
требует исправления перед интерпретацией результатов.

### Регистрация датасета

```bash
uv run --locked --group eda python scripts/register_dataset.py
```

Скрипт создаёт `data/visdrone/manifest.csv`: одна строка на изображение,
имена изображения и аннотации, их SHA-256, количество объектов и транспорта.
Порядок строк, кодировка и переводы строк фиксированы.

В запуске `dataset-tracking-validation` регистрируется входной датасет
`visdrone2019-det-val` с контекстом `validation`. Манифест сохраняется
как артефакт, его полный SHA-256 — как тег `manifest_sha256`, первые
32 символа хеша — как `digest` датасета. Сами изображения не загружаются
в MLflow: источник датасета указывает на локальный CSV.

## Проверки

Проверки оформления охватывают весь репозиторий, включая `scripts/`:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Для применения форматирования: `uv run --locked ruff format .`.
Ruff настроен на Python 3.12 и длину строки 100 символов.

Модульные тесты работают без PostgreSQL и Docker:

```bash
uv run --locked pytest -m "not integration"
```

Интеграционные тесты требуют Docker Engine и Docker Compose 2.24.4+
из-за использования `!override`. Они собирают образ и поднимают отдельный
стек со случайными портами, проверяют HTTP, реальную БД, отказ и восстановление
PostgreSQL. После тестов временные контейнеры и том удаляются.

```bash
uv run --locked pytest -m integration
```

Весь набор: `uv run --locked pytest`. Покрытие пакета `health` включено
в настройках pytest; EDA-скрипты в него не входят. Запросы к приложению
в контейнере проверяются по HTTP, но не учитываются в покрытии Python-процесса pytest.

Для установки Git-хуков и проверки всех отслеживаемых файлов:

```bash
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

## CI/CD

Workflow `.github/workflows/ci.yml` запускается при push в `main`,
при pull request в `main`, при push тега `v*` и вручную.
Он проверяет Ruff, форматирование, модульные и Docker-интеграционные тесты.
После успешных проверок push тега `v*` также публикует образ
в `ghcr.io/<владелец>/<репозиторий>` с соответствующим тегом версии.
