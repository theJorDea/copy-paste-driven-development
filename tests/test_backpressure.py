"""Проверка контракта consumer и overflow confirm на реальном RabbitMQ."""

import pytest
from faststream import AckPolicy
from faststream.rabbit import RabbitBroker, RabbitQueue, TestRabbitBroker


@pytest.mark.asyncio
async def test_handler_receives_task_and_acks_on_success() -> None:
    broker = RabbitBroker()
    queue = RabbitQueue("test.inference.tasks", durable=False, auto_delete=True)

    @broker.subscriber(queue, ack_policy=AckPolicy.ACK)
    async def handle_task(message: dict) -> None:
        assert message["task_id"] == "task-1"

    async with TestRabbitBroker(broker) as test_broker:
        await test_broker.publish({"task_id": "task-1"}, queue=queue)
        handle_task.mock.assert_called_once_with({"task_id": "task-1"})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bounded_rabbit_queue_negative_confirms_overflow() -> None:
    from testcontainers.rabbitmq import RabbitMqContainer

    with RabbitMqContainer("rabbitmq:4.1-management") as rabbitmq:
        broker = RabbitBroker(rabbitmq.get_connection_url())
        queue = RabbitQueue(
            "test.bounded.tasks",
            durable=False,
            auto_delete=True,
            arguments={"x-max-length": 1, "x-overflow": "reject-publish"},
        )
        try:
            await broker.connect()
            await broker.declare_queue(queue)
            await broker.publish("first", queue=queue, persistent=True)
            confirmation = await broker.publish("second", queue=queue, persistent=True)
            assert type(confirmation).__name__ in {"Nack", "Basic.Nack"}
        finally:
            await broker.stop()
