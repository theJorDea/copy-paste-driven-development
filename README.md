# Детекция транспорта на аэроснимках

Учебный проект по MLOps: YOLO11n находит транспорт на кадрах VisDrone,
MLflow хранит эксперименты и версии модели, FastAPI принимает изображение
и возвращает рамки объектов. Отдельный показатель загруженности дороги
пока не вычисляется.

## Как устроен проект

| Файл | Что делает |
| --- | --- |
| `scripts/eda_visdrone.py` | Анализ разметки, распределения классов и размеров объектов |
| `scripts/register_dataset.py` | Манифест изображений и аннотаций, SHA-256, dataset tracking |
| `scripts/prepare_yolo_dataset.py` | Перевод исходных рамок VisDrone в формат YOLO |
| `scripts/train.py` | Обучение, валидация, метрики и артефакты MLflow |
| `scripts/register_model.py` | Упаковка весов в MLflow-модель, версия Registry и alias |
| `src/health/inference.py` | Загрузка YOLO и преобразование предсказаний в JSON |
| `src/health/main.py` | Загрузка модели из Registry один раз при старте API |
| `src/health/api.py` | HTTP-эндпоинты, включая `POST /process` |
| `mlflow/compose.yml` | MLflow + PostgreSQL для метаданных + RustFS для артефактов |
| `docker-compose.yml` | API и его PostgreSQL |
| `tests/` | Проверки API, загрузки модели и Docker-стека |

`data/`, `runs/`, `outputs/`, веса `.pt`, `.env` и `.venv/` не входят в Git.
Для работы нужны Python 3.12, uv, Docker Engine и Docker Compose 2.24.4+.
Обучение выполняется локально на NVIDIA GPU; API в Docker использует CPU.

## 1. Поднять MLflow

Создать `mlflow/.env` с настройками локального стенда:

```dotenv
MLFLOW_DB_PASSWORD=local-mlflow-db
S3_ACCESS_KEY=local-mlflow-access-key
S3_SECRET_KEY=local-mlflow-secret-key
```

```bash
docker compose --env-file mlflow/.env -f mlflow/compose.yml up --build -d
uv sync --locked --group eda --group train
```

[MLflow UI](http://127.0.0.1:5000), [консоль RustFS](http://127.0.0.1:9001).
PostgreSQL хранит параметры, метрики и связи; RustFS — файлы весов и графики.
Клиентам не нужны ключи S3: загрузка артефактов идёт через MLflow.

## 2. Подготовить данные

Скачать trainset и valset задачи **Object Detection in Images** из
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset#download).

```text
data/
    visdrone-train/
        images/             # 6471 JPG обучающей выборки
        annotations/        # исходные TXT VisDrone
    visdrone/
        images/             # 548 JPG валидационной выборки
        annotations/
```

```bash
uv run --locked --group eda python -X utf8 -m scripts.eda_visdrone
uv run --locked --group eda python -X utf8 -m scripts.register_dataset --split training
uv run --locked --group eda python -X utf8 -m scripts.register_dataset --split validation
uv run --locked python -X utf8 -m scripts.prepare_yolo_dataset --split training
uv run --locked python -X utf8 -m scripts.prepare_yolo_dataset --split validation
```

В `labels/` появятся рамки YOLO: класс, центр X/Y, ширина и высота,
нормированные на размер изображения. Используются семь классов транспорта:
car, van, truck, tricycle, awning-tricycle, bus, motor.
Игнорируемые объекты и рамки с неположительным размером пропускаются.
Пути и классы обучения заданы в `visdrone.yaml`.

EDA сохраняет распределение классов, площадей рамок, числа машин на кадр
и примеры кадров разной плотности. Повторный подсчёт объектов исправлен.
Мелкие объекты важны для выбора разрешения обучения.

`manifest.csv` содержит имена файлов, их SHA-256 и число объектов.
Полный хеш манифеста записывается в тег MLflow, первые 32 символа — в digest.
Каждый запуск обучения связан с train и validation через `mlflow.log_input`;
копии обоих манифестов и конфигурация данных сохраняются в артефактах.

## 3. Обучить и сравнить модели

Три эксперимента используют YOLO11n, 30 эпох, batch=4, seed=42,
AdamW и одинаковые выборки. Меняется разрешение входа:

```bash
uv run --locked --group train python -X utf8 -m scripts.train --name baseline-n640 --imgsz 640
uv run --locked --group train python -X utf8 -m scripts.train --name resolution-n320 --imgsz 320
uv run --locked --group train python -X utf8 -m scripts.train --name resolution-n480 --imgsz 480
```

Существующий baseline повторно обучать не нужно. Его готовые веса можно
оценить и сохранить отдельным завершённым запуском:

```bash
uv run --locked --group train python -X utf8 -m scripts.train --name baseline-n640-recovered --evaluate-only --model runs/train/baseline-n640/weights/best.pt --imgsz 640
```

Для продолжения прерванного обучения есть `--resume-from PATH/last.pt`.
В этом случае YOLO восстанавливает настройки из checkpoint.

Сейчас дополнительные обучения остановлены по просьбе автора. Baseline уже
обучен и зарегистрирован как `visdrone-yolo`, версия 1, alias `champion`;
его `final/map50_95` = 0.20349. Два дополнительных результата и итоговое
сравнение ещё нужны для полной сдачи второй лабораторной.
Ночью в PowerShell из корня проекта выполнить последовательно:

```powershell
uv run --locked --group train python -X utf8 -m scripts.train --name resolution-n320-resumed --imgsz 320 --resume-from runs/train/resolution-n320/weights/last.pt
uv run --locked --group train python -X utf8 -m scripts.train --name resolution-n480 --imgsz 480
```

Первый запуск продолжит сохранённое обучение до 30 эпох, второй выполнит
30 эпох с разрешением 480. После завершения сравнить три модели в MLflow
и зарегистрировать новые веса командами из следующего раздела.

В эксперименте **visdrone-training** выбрать три завершённых запуска
и нажать **Compare**. Основная метрика — `final/map50_95`: она учитывает
качество рамок при разных порогах IoU. Дополнительные — `final/map50`,
`final/precision`, `final/recall`, время обучения `train_seconds`.
У восстановленного baseline время исходного обучения смотреть в `training/results.csv`.

В артефактах каждого запуска находятся `model_weights/best.pt`, история
обучения, `validation/per_class_metrics.csv`, confusion matrix, PR-кривая
и примеры предсказаний. Они помогают увидеть, какие классы модель путает
и какие мелкие объекты пропускает.

## 4. Зарегистрировать модель

Для каждого завершённого запуска взять Run ID из MLflow или вывода скрипта.
Передать разрешение, с которым его обучали:

```bash
uv run --locked --group train python -X utf8 -m scripts.register_model --run-id RUN_ID --imgsz 640 --alias champion
```

Скрипт берёт веса из артефактов запуска, упаковывает их вместе с параметрами
инференса в `mlflow.pyfunc` и создаёт версию модели **visdrone-yolo**.
Без `--alias` версия создаётся, но текущий champion не меняется.
Назначить `champion` следует версии с лучшим `final/map50_95`.

## 5. Запустить API

Перед запуском MLflow должен работать, а alias `champion` — существовать.

```bash
docker compose up --build -d
```

[Swagger UI](http://127.0.0.1:8000/docs) позволяет отправить изображение
на `POST /process`. Имя multipart-поля — `file`, лимит — 10 МиБ.
Ответ содержит размеры изображения, версию модели и список объектов:
класс, уверенность и координаты рамки `x1, y1, x2, y2` в пикселях.

Модель загружается при старте приложения, а не на каждый запрос.
Тяжёлое предсказание выполняется через `asyncio.to_thread` внутри async-эндпоинта.
После смены alias в MLflow нужен `docker compose restart app`.
Код API при этом не меняется; поле `model_version` показывает новую версию.

| Настройка | Значение по умолчанию |
| --- | --- |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000`; в Compose — `http://host.docker.internal:5000` |
| `MODEL_NAME` | `visdrone-yolo` |
| `MODEL_ALIAS` | `champion` |
| `MODEL_DEVICE` | `cpu` |
| `MODEL_CONFIDENCE` | `0.25` |
| `DATABASE__DATABASE_URL` | Обязательный URL PostgreSQL; уже задан в Compose |

Прямые multipart-скачивания MLflow отключены при загрузке модели:
внутренний адрес `storage:9000` недоступен клиенту на Windows,
поэтому артефакты скачиваются через сервер MLflow.

Для локального API поднять `docker compose up -d postgres`, создать `.env`:

```dotenv
DATABASE__DATABASE_URL=postgresql://postgres:postgres@localhost:5433/mlops
```

```bash
uv run --locked uvicorn health.main:app --host 127.0.0.1 --port 8000
```

Эндпоинты первой лабораторной сохранены: `/healthz`, `/api/v1/ping`,
`/api/v1/version`, `/api/v1/health`. Последний проверяет PostgreSQL
через общий пул asyncpg и возвращает 503 при недоступной БД.

## Проверки

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -X utf8 -m pytest -m "not integration"
uv run --locked python -X utf8 -m pytest -m integration
uv run --locked pre-commit run --all-files
```

Модульные тесты работают без MLflow, PostgreSQL и GPU.
Интеграционные собирают настоящий образ API и поднимают отдельные
PostgreSQL и MLflow со случайными портами. Проверяют HTTP, отказ и
восстановление БД, предсказание и смену версии модели после перезапуска.
Временный стек удаляется после тестов, основной стенд не затрагивается.

CI выполняет Ruff и оба набора тестов. При push тега `v*` после успешных
проверок публикуется Docker-образ в GHCR.
