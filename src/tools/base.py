from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.core.context import RuntimeContext


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    namespace: str
    platforms: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    summary: str = ""


class Tool(Protocol):
    name: str
    description: str
    spec: ToolSpec | None

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        ...
