"""RabbitMQ топология, общая для API-producer и FastStream consumer."""

import os

from faststream.rabbit import RabbitBroker, RabbitQueue
from faststream.rabbit.schemas import Channel

RABBIT_URL = os.getenv("RABBITMQ_URL", "amqp://app:app-local@localhost:5672/")
QUEUE_NAME = os.getenv("TASK_QUEUE", "inference.tasks")
DLQ_NAME = os.getenv("TASK_DLQ", "inference.tasks.dlq")
RETRY_NAME = os.getenv("TASK_RETRY_QUEUE", "inference.tasks.retry")
QUEUE_LIMIT = int(os.getenv("TASK_QUEUE_LIMIT", "100"))
BACKPRESSURE_ENABLED = os.getenv("TASKS_BACKPRESSURE_ENABLED", "true").lower() == "true"
PREFETCH_COUNT = int(os.getenv("WORKER_PREFETCH", "1"))

_queue_arguments = {
    "x-dead-letter-exchange": "",
    "x-dead-letter-routing-key": DLQ_NAME,
}
if BACKPRESSURE_ENABLED:
    _queue_arguments.update({"x-max-length": QUEUE_LIMIT, "x-overflow": "reject-publish"})

TASK_QUEUE = RabbitQueue(QUEUE_NAME, durable=True, arguments=_queue_arguments)
DLQ_QUEUE = RabbitQueue(DLQ_NAME, durable=True)
RETRY_QUEUE = RabbitQueue(
    RETRY_NAME,
    durable=True,
    arguments={
        "x-message-ttl": int(os.getenv("TASK_RETRY_DELAY_MS", "1000")),
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": QUEUE_NAME,
    },
)
broker = RabbitBroker(
    RABBIT_URL,
    default_channel=Channel(publisher_confirms=True, on_return_raises=True),
    graceful_timeout=float(os.getenv("WORKER_GRACEFUL_TIMEOUT", "300")),
)
