from __future__ import annotations

from typing import Any

import httpx

from src.config import settings


async def resolve_user_context(token: str | None) -> dict[str, Any]:
    if not token:
        return {"authenticated": False, "userId": None, "role": None, "roles": []}

    url = f"{settings.backend_base_url}/agent/user-context"
    headers = {"X-AGENT-TOKEN": settings.agent_token}
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(url, json={"token": token}, headers=headers)
        response.raise_for_status()
        payload = response.json()

    if payload.get("success") is not True:
        return {"authenticated": False, "userId": None, "role": None, "roles": []}
    data = payload.get("data") or {}
    return {
        "authenticated": bool(data.get("authenticated")),
        "userId": data.get("userId"),
        "role": data.get("role"),
        "roles": data.get("roles") or [],
    }
