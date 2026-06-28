from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.core.context import RuntimeContext


@dataclass(frozen=True)
class AgentRequest:
    context: RuntimeContext
    user_input: str = ""
    intent: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentResponse:
    success: bool
    content: str = ""
    data: dict[str, Any] | None = None
    error: str | None = None
    used_tools: tuple[str, ...] = ()
    next_agents: tuple[str, ...] = ()

@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    tools: tuple[str, ...]
    model_tier: str = "small"
    skills: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    version: str = "0.1.0"


class Agent(Protocol):
    spec: AgentSpec

    async def run(self, request: AgentRequest) -> AgentResponse:
        ...
