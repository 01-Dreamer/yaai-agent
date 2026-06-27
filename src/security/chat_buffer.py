from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import redis.asyncio as redis

from src.config import settings


class ChatRedisBuffer:
    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_database,
            password=settings.redis_password or None,
            decode_responses=True,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    def _key(self, request_id: str, suffix: str) -> str:
        return f"{settings.redis_prefix}:chat:{request_id}:{suffix}"

    async def append_event(self, event: Any) -> None:
        await self.connect()
        assert self._client is not None
        request_id = event.request_id
        pipe = self._client.pipeline()
        pipe.rpush(self._key(request_id, "events"), json.dumps(asdict(event), ensure_ascii=False))
        pipe.hset(
            self._key(request_id, "meta"),
            mapping={
                "connection_id": event.connection_id,
                "session_id": "" if event.session_id is None else str(event.session_id),
                "user_id": "" if event.user_id is None else str(event.user_id),
                "assistant_message_id": event.assistant_message_id or "",
            },
        )
        pipe.expire(self._key(request_id, "events"), settings.chat_buffer_ttl_seconds)
        pipe.expire(self._key(request_id, "meta"), settings.chat_buffer_ttl_seconds)
        pipe.expire(self._key(request_id, "status"), settings.chat_buffer_ttl_seconds)
        await pipe.execute()

    async def mark_rejected(self, request_id: str, reason: str) -> None:
        await self.connect()
        assert self._client is not None
        await self._client.hset(self._key(request_id, "status"), mapping={"status": "rejected", "reason": reason})
        await self._client.expire(self._key(request_id, "status"), settings.chat_buffer_ttl_seconds)

    async def is_rejected(self, request_id: str) -> bool:
        await self.connect()
        assert self._client is not None
        return await self._client.hget(self._key(request_id, "status"), "status") == "rejected"

    async def load_events(self, request_id: str) -> list[dict[str, Any]]:
        await self.connect()
        assert self._client is not None
        values = await self._client.lrange(self._key(request_id, "events"), 0, -1)
        return [json.loads(value) for value in values]

    async def clear(self, request_id: str) -> None:
        await self.connect()
        assert self._client is not None
        await self._client.delete(
            self._key(request_id, "events"),
            self._key(request_id, "meta"),
            self._key(request_id, "status"),
        )


chat_redis_buffer = ChatRedisBuffer()
