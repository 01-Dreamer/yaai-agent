from __future__ import annotations

from typing import Any

import httpx

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


class BackendTool:
    def __init__(self, name: str, path: str, method: str = "POST", expose_to_llm: bool = True) -> None:
        self.name = name
        self.path = path
        self.method = method.upper()
        self.description = f"Call yaai-backend AgentController: {path}"
        self.expose_to_llm = expose_to_llm
        self.spec = ToolSpec(
            name=name,
            description=self.description,
            namespace="backend",
            risk_level="medium",
            expose_to_llm=expose_to_llm,
            tags=("backend", "business"),
            capabilities=("business_api",),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        url = f"{settings.backend_base_url}{self.path}"
        headers = {"X-AGENT-TOKEN": settings.agent_token}
        async with httpx.AsyncClient(timeout=10.0) as client:
            if self.method == "GET":
                response = await client.get(url, params=kwargs, headers=headers)
            else:
                response = await client.request(self.method, url, json=kwargs, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") is not True:
            return ToolResult(False, data=payload, error=payload.get("message") or "backend request failed")
        return ToolResult(True, data=payload.get("data") or {})
