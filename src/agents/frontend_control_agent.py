from __future__ import annotations

from typing import Any

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.context import RuntimeContext
from src.core.registry import tool_registry


class FrontendControlAgent:
    spec = AgentSpec(
        name="frontend_control",
        description="生成并请求前端白名单 action 的 Agent",
        tools=("frontend.navigate", "frontend.fill", "frontend.highlight"),
        tags=("frontend", "action"),
        capabilities=("navigate", "fill", "highlight"),
    )

    async def request_action(self, context: RuntimeContext, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = f"frontend.{action}"
        if tool_name not in set(self.spec.tools):
            return {"success": False, "error": f"frontend action is not allowed: {action}"}
        item = tool_registry.get(tool_name)
        result = await item.handler.run(context, **payload)
        return {"success": result.success, "data": result.data, "error": result.error}

    async def run(self, request: AgentRequest) -> AgentResponse:
        action = request.intent or (request.payload or {}).get("action")
        if not action:
            return AgentResponse(False, error="missing frontend action")
        payload = dict(request.payload or {})
        payload.pop("action", None)
        result = await self.request_action(request.context, action, payload)
        return AgentResponse(bool(result.get("success")), data=result, error=result.get("error"))
