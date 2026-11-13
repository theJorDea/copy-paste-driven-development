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
| `src/health/api.py` | HTTP-эндпоинты, асинхронная постановка задач и просмотр статуса |
| `src/health/worker.py` | FastStream consumer, inference, повторы и DLQ |
| `src/health/messaging.py` | Ограниченная RabbitMQ-очередь, очередь повторов и DLQ |
| `scripts/task_source.py` | Потоковый пакетный источник изображений с адаптивным ожиданием |
| `observability/` | OTEL Collector, Prometheus, Loki, Tempo, Alloy, Grafana и алерты |
| `mlflow/compose.yml` | MLflow + PostgreSQL для метаданных + RustFS для артефактов |
| `docker-compose.yml` | API, worker, RabbitMQ и PostgreSQL |
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

## 5. Запустить API (лабы 1–3)

Перед запуском MLflow должен работать, а alias `champion` — существовать.

```bash
docker compose up --build -d
```

[Swagger UI](http://127.0.0.1:8000/docs) позволяет отправить изображение
на `POST /api/v1/infer`. Имя multipart-поля — `file`, лимит — 10 МиБ.
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

## 6. Лаба 4 — очередь, backpressure и надёжность

Скопировать `.env.example` в `.env`, запустить MLflow и убедиться, что alias
`champion` доступен. Затем поднять приложение:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
```

`POST /process` принимает прежний multipart-запрос с полем `file`, а
`POST /api/v1/tasks` принимает JSON `{"image_base64":"..."}`. Оба маршрута
ставят задачу в RabbitMQ и сразу возвращают `202` с `task_id`. Статус и
результат доступны через `GET /api/v1/tasks/{task_id}`. Для прямого синхронного
инференса оставлен `POST /api/v1/infer`.

Очередь `inference.tasks` по умолчанию ограничена 100 готовыми сообщениями
(`TASK_QUEUE_LIMIT`) и настроена с `x-overflow=reject-publish`. Producer
использует publisher confirms. При отрицательном подтверждении DB-транзакция
откатывается, задача не остаётся `PENDING`, а API отвечает `503` и
`Retry-After: 2`. Значение равно двум задержкам очереди повторов по 1 секунде:
оно дольше одной retry-задержки, чтобы клиент не повторил публикацию сразу
после одного короткого сбоя; значение настраивается и уточняется по замерам
скорости освобождения очереди. Встроенный
batch source удваивает локальную задержку при повторных `503` и держит в
памяти только одно изображение:

```powershell
docker compose --profile batch-source up --build source
```

`WORKER_PREFETCH=1` совпадает с конкурентностью по умолчанию: один FastStream
consumer и один inference в пуле потоков на процесс. Добавление второго
worker удваивает число слотов обработки, не меняя исходный код:

```powershell
docker compose up -d --scale worker=2 worker
```

Синхронный `model.predict` выполняется через `asyncio.to_thread`; конфигурация
`INFERENCE_SECONDS` добавляет воспроизводимую задержку эксперимента. Worker
отвечает на `GET /healthz` на порту 8081 на своём event loop; проверяющий может
проверить endpoint из контейнера во время работы inference:

```powershell
docker compose exec worker uv run --no-sync python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8081/healthz').read().decode())"
```

Worker
загружает MLflow alias при старте и закрепляет конкретную версию на весь срок
жизни процесса. `AckPolicy.MANUAL` подтверждает исходное сообщение после
транзакционной записи результата или после подтверждённой публикации в retry
или DLQ. Ack до записи мог бы потерять результат при сбое; бесконечный nack
зациклил бы poison-сообщение. Повторная доставка после записи результата увидит `COMPLETED` и
завершится без повторной записи. После трёх попыток сообщение попадает в
`inference.tasks.dlq`. Очередь retry ждёт `TASK_RETRY_DELAY_MS`, затем RabbitMQ
возвращает сообщение в рабочую очередь. FastStream ждёт активные обработчики
до `WORKER_GRACEFUL_TIMEOUT=300`; для контейнера установлен stop grace period
5 минут.

Для сравнения с режимом без ограничения `TASKS_BACKPRESSURE_ENABLED=false`.
Перед переключением удалите прежние очереди вместе с данными RabbitMQ, иначе
RabbitMQ отклонит повторное объявление очереди с другими аргументами:

```powershell
$env:TASKS_BACKPRESSURE_ENABLED = "false"
docker compose down -v
docker compose up --build -d
```

Снять показатели на одинаковом наборе изображений, одинаковом
`INFERENCE_SECONDS` и одном числе воркеров. Повторить с ограничением очереди,
затем подключить второй worker и повторить. В Grafana/Prometheus фиксировать
глубину очереди, память, скорость приёма, скорость завершения и p95 inference.
README не содержит выдуманных цифр: таблицу заполняют значениями с конкретного
прогона перед защитой. Проверка доступна в RabbitMQ Management UI на порту
15672. Ограничение длины относится к ready сообщениям; `unacked` показаны
отдельно, и при `prefetch=1` их число ограничено одним на worker.

| Режим | Воркеры | Пик ready очереди | Память источника | Задач/с от API | Задач/с успешно | p95 inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Без backpressure | 1 | заполнить | заполнить | заполнить | заполнить | заполнить |
| Backpressure | 1 | заполнить | заполнить | заполнить | заполнить | заполнить |
| Backpressure | 2 | заполнить | заполнить | заполнить | заполнить | заполнить |

В режиме без ограничения ожидается рост очереди при генераторе быстрее worker.
В ограниченном режиме ready очередь удерживается на `TASK_QUEUE_LIMIT`, источник
получает `503` и возобновляет отправку с паузой. Со вторым worker растёт
совокупная пропускная способность, а источник реже получает `503`.

## 7. Лаба 5 — логи, трейсы и метрики

Наблюдаемость отделена профилем Compose. Приложение запускается без Grafana и
хранилищ. Для полного стенда:

```powershell
$env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://otel-collector:4317"
docker compose -f docker-compose.yml -f observability/compose.yaml --profile observability up --build -d
```

Grafana: [http://127.0.0.1:3000](http://127.0.0.1:3000), локальный пользователь
`admin`, пароль задаётся `GRAFANA_PASSWORD`. Provisioning создаёт Prometheus,
Loki, Tempo, дашборд «Конвейер» и Prometheus/Alertmanager rules. Все сервисы
наблюдаемости имеют ограничение памяти и версии образов зафиксированы.

Чтобы batch source опубликовал счётчики короткоживущего процесса в Prometheus,
запустить его при активном профиле:

```powershell
$env:PUSHGATEWAY_URL = "http://pushgateway:9091"
$env:SOURCE_OTEL_EXPORTER_OTLP_ENDPOINT = "http://otel-collector:4317"
docker compose -f docker-compose.yml -f observability/compose.yaml --profile observability --profile batch-source run --rm source
```

Источник из Docker Compose пишет structured logs в stdout, Alloy доставляет их
в Loki; Prometheus забирает его короткоживущие метрики из Pushgateway.

Логи всех контейнеров идут в stdout как JSON; Alloy читает Docker logs и
передаёт их в Loki. В labels остаются только `service` и `level`; `task_id` и
`trace_id` хранятся как поля записи. Из `trace_id` можно перейти к трейсу, а
из Tempo — открыть логи по trace ID. Один запрос имеет контекст API → publish →
worker process → inference → SQL update. W3C `traceparent` переносится в
RabbitMQ headers; worker восстанавливает контекст перед созданием спана. В
batch source спан `source.wait_for_capacity` содержит события `queue_overloaded`
и время пауз при 503. Два worker имеют общий `service.name=worker` и уникальный
`service.instance.id` по hostname контейнера.
Пул потоков запускается через `asyncio.to_thread`, поэтому спан inference
создаётся вокруг await и остаётся дочерним к span обработки сообщения.

OpenTelemetry Collector применяет `memory_limiter` перед `batch`, затем пишет
OTLP traces в Tempo. При недоступном Collector модель и очередь продолжают
работать; батч exporter ограничен внутренней очередью SDK, но временно теряет
телеметрию после исчерпания повторов/буфера. Выбор стенда — Prometheus для
метрик, Loki для структурированных логов и Tempo для трейсов: каждый сигнал
остаётся в своём хранилище с простой локальной конфигурацией. Для production
следует задать retention, аутентификацию, TLS, внешнее хранилище и tail-based
sampling; локальный стенд использует `always_on`, чтобы защита была повторяема.

Prometheus получает метрики API с `/metrics`, метрики worker с порта 8001,
глубину очередей из RabbitMQ Prometheus plugin на 15692 и память контейнеров
из cAdvisor. Batch source отправляет краткоживущие счётчики в Pushgateway.
Гистограмма inference использует границы `0.1, 0.25, 0.5, 1, 2, 5, 10, 30,
60` секунд; они охватывают конфигурируемую задержку и длительные CPU-запросы.
На дашборде есть p50/p95/p99 API и inference, глубина очереди с лимитом,
скорость source и worker по экземплярам, память, 503, DLQ, повторы и ошибки.
Алерт очереди срабатывает выше 90% лимита в течение минуты; алерт DLQ — сразу,
поскольку любое poison-сообщение требует разбора.

Лейблы метрик ограничены route/method/status, worker instance, outcome,
queue и конечным набором bucket. `task_id` и `trace_id` не используются как
лейблы. При 10 API routes, 5 методах и 6 кодах ответов верхняя оценка API —
300 counter series и около 900 series histogram (10×5×18). Для двух worker
histogram inference — 24 series (12 на экземпляр), outcome — 6, плюс около 20
серий retry/in-flight/queue limit. Очереди RabbitMQ дают около 9–12 series для
трёх очередей. К этому добавляются source (около 12) и меняющиеся cAdvisor
series; ожидаемый порядок величины — около 1,300 рядов до cAdvisor, он уточняется через
Prometheus `TSDB Status` после прогона. Добавление task_id как label создало бы
несколько новых рядов на каждую задачу и быстро росло бы без верхней границы.
