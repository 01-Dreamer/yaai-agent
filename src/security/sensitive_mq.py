from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import aio_pika

from src.config import settings
from src.security.chat_buffer import chat_redis_buffer
from src.security.sensitive_matcher import sensitive_matcher


@dataclass(frozen=True)
class ChatAuditMessage:
    check_id: str
    connection_id: str
    request_id: str
    message_id: str
    role: str
    stage: str
    content: str
    session_id: int | None = None
    user_id: int | None = None
    attachments: list[dict[str, Any]] | None = None
    sub_agent: str | None = None
    assistant_message_id: str | None = None


SensitiveHitHandler = Callable[[ChatAuditMessage, str], Awaitable[None]]
FinalAcceptedHandler = Callable[[str], Awaitable[None]]


class ChatAuditMqService:
    def __init__(self) -> None:
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None
        self._queue: aio_pika.abc.AbstractQueue | None = None

    def _url(self) -> str:
        username = quote(settings.rabbitmq_username, safe="")
        password = quote(settings.rabbitmq_password, safe="")
        vhost = quote(settings.rabbitmq_vhost, safe="")
        return f"amqp://{username}:{password}@{settings.rabbitmq_host}:{settings.rabbitmq_port}/{vhost}"

    async def connect(self) -> None:
        if self._connection is not None:
            return
        self._connection = await aio_pika.connect_robust(self._url())
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)

        dlx = await self._channel.declare_exchange(
            settings.chat_dead_letter_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dlq = await self._channel.declare_queue(settings.chat_dead_letter_queue, durable=True)
        await dlq.bind(dlx, routing_key=settings.chat_dead_letter_queue)

        self._exchange = await self._channel.declare_exchange(
            settings.chat_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        self._queue = await self._channel.declare_queue(
            settings.chat_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": settings.chat_dead_letter_exchange,
                "x-dead-letter-routing-key": settings.chat_dead_letter_queue,
            },
        )
        await self._queue.bind(self._exchange, routing_key=settings.chat_routing_key)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
        self._connection = None
        self._channel = None
        self._exchange = None
        self._queue = None

    async def publish(self, message: ChatAuditMessage) -> None:
        await chat_redis_buffer.append_event(message)
        await self.connect()
        assert self._exchange is not None
        body = json.dumps(asdict(message), ensure_ascii=False).encode("utf-8")
        await self._exchange.publish(
            aio_pika.Message(body=body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
            routing_key=settings.chat_routing_key,
        )

    async def consume(self, on_hit: SensitiveHitHandler, on_final_accepted: FinalAcceptedHandler) -> None:
        await self.connect()
        assert self._queue is not None
        async with self._queue.iterator() as queue_iter:
            async for mq_message in queue_iter:
                async with mq_message.process(requeue=False):
                    payload: dict[str, Any] = json.loads(mq_message.body.decode("utf-8"))
                    check_message = ChatAuditMessage(**payload)
                    if await chat_redis_buffer.is_rejected(check_message.request_id):
                        continue

                    hit = sensitive_matcher.find_first(check_message.content)
                    if hit is not None:
                        await chat_redis_buffer.mark_rejected(check_message.request_id, "命中敏感词")
                        await on_hit(check_message, hit.word)
                        continue

                    if check_message.stage == "final_output":
                        await on_final_accepted(check_message.request_id)


chat_audit_mq_service = ChatAuditMqService()
