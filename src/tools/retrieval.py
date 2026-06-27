from __future__ import annotations

from typing import Any

from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


class RetrievalTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.expose_to_llm = True
        namespace = name.split(".", 1)[0]
        self.spec = ToolSpec(
            name=name,
            description=name,
            namespace=namespace,
            risk_level="low",
            tags=(namespace,),
            capabilities=(name.split(".", 1)[-1],),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        return ToolResult(True, data={"message": f"{self.name} scaffold", "args": kwargs})
