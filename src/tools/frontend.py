from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec

ActionSender = Callable[[RuntimeContext, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class FrontendActionTool:
    def __init__(self, action: str, sender: ActionSender | None = None) -> None:
        self.action = action
        self.name = f"frontend.{action}"
        self.description = f"Request frontend action: {action}"
        self.expose_to_llm = True
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="frontend",
            risk_level="medium",
            platforms=("frontend", "lowcode"),
            tags=("frontend", "action"),
            capabilities=(action,),
        )
        self._sender = sender

    def bind_sender(self, sender: ActionSender) -> None:
        self._sender = sender

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        if self._sender is None:
            return ToolResult(False, error="frontend action sender is not bound")
        result = await self._sender(context, self.action, kwargs)
        return ToolResult(bool(result.get("success")), data=result, error=result.get("error"))
