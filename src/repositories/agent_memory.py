from __future__ import annotations

import json
import base64
from datetime import datetime
from typing import Any

import aiomysql

from src.config import settings


def _normalize_limit(limit: int, *, default: int, maximum: int) -> int:
    if limit <= 0:
        return default
    return min(limit, maximum)


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    padding = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid cursor")
    return data


class AgentMemoryRepository:
    def __init__(self) -> None:
        self._pool: aiomysql.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_username,
            password=settings.mysql_password,
            db=settings.mysql_database,
            charset=settings.mysql_charset,
            minsize=1,
            maxsize=settings.mysql_pool_size,
            autocommit=True,
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None

    async def create_session(self, *, user_id: int | None, title: str) -> int:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO agent_session (user_id, title) VALUES (%s, %s)",
                    (user_id, title[:200] or "新会话"),
                )
                return int(cursor.lastrowid)

    async def list_sessions(
        self,
        *,
        user_id: int | None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        limit = _normalize_limit(limit, default=20, maximum=50)
        cursor_updated_at: datetime | None = None
        cursor_id: int | None = None
        if cursor:
            try:
                decoded = _decode_cursor(cursor)
                cursor_updated_at = datetime.fromisoformat(str(decoded["updatedAt"]))
                cursor_id = int(decoded["id"])
            except Exception as exc:
                raise ValueError("invalid session cursor") from exc

        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT id, title, created_at, updated_at
                    FROM agent_session
                    WHERE user_id <=> %s
                      AND (
                        %s IS NULL
                        OR updated_at < %s
                        OR (updated_at = %s AND id < %s)
                      )
                    ORDER BY updated_at DESC, id DESC
                    LIMIT %s
                    """,
                    (user_id, cursor_updated_at, cursor_updated_at, cursor_updated_at, cursor_id, limit + 1),
                )
                rows = await cursor.fetchall()
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        sessions = [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
                "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
            }
            for row in page_rows
        ]
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = _encode_cursor(
                {
                    "updatedAt": last["updated_at"].isoformat() if last.get("updated_at") else "",
                    "id": int(last["id"]),
                }
            )
        return {"sessions": sessions, "nextCursor": next_cursor, "hasMore": has_more}

    async def list_messages(
        self,
        *,
        session_id: int,
        user_id: int | None,
        cursor: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        limit = _normalize_limit(limit, default=30, maximum=100)
        cursor_id: int | None = None
        if cursor:
            try:
                cursor_id = int(cursor)
            except Exception as exc:
                raise ValueError("invalid message cursor") from exc

        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT id, role, content, attachments, created_at
                    FROM agent_memory
                    WHERE session_id = %s
                      AND user_id <=> %s
                      AND role IN ('user', 'assistant')
                      AND (%s IS NULL OR id < %s)
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (session_id, user_id, cursor_id, cursor_id, limit + 1),
                )
                rows = await cursor.fetchall()
        has_more = len(rows) > limit
        page_rows = list(reversed(rows[:limit]))
        messages: list[dict[str, Any]] = []
        for row in page_rows:
            attachments = None
            if row.get("attachments"):
                try:
                    attachments = json.loads(row["attachments"])
                except Exception:
                    attachments = None
            messages.append(
                {
                    "id": f"memory_{row['id']}",
                    "role": row["role"],
                    "content": row["content"],
                    "attachments": attachments or [],
                    "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
                }
            )
        next_cursor = str(page_rows[0]["id"]) if has_more and page_rows else None
        return {"messages": messages, "nextCursor": next_cursor, "hasMore": has_more}

    async def insert_memory(
        self,
        *,
        session_id: int | None,
        user_id: int | None,
        role: str,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> int | None:
        if session_id is None:
            return None
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO agent_memory (session_id, user_id, role, content, attachments)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        user_id,
                        role,
                        content,
                        json.dumps(attachments or [], ensure_ascii=False) if attachments else None,
                    ),
                )
                return int(cursor.lastrowid)

    async def update_session_title(self, *, session_id: int, title: str) -> bool:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "UPDATE agent_session SET title = %s WHERE id = %s",
                    (title[:200] or "新会话", session_id),
                )
                return cursor.rowcount > 0

    async def clear_session_memory(self, *, session_id: int) -> bool:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM agent_memory WHERE session_id = %s", (session_id,))
                await cursor.execute(
                    "UPDATE agent_session SET memory_content = NULL, memory_updated_at = NULL WHERE id = %s",
                    (session_id,),
                )
                return cursor.rowcount > 0

    async def delete_session(self, *, session_id: int) -> bool:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM agent_memory WHERE session_id = %s", (session_id,))
                await cursor.execute("DELETE FROM agent_session WHERE id = %s", (session_id,))
                return cursor.rowcount > 0


agent_memory_repository = AgentMemoryRepository()
