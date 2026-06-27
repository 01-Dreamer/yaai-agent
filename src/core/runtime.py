from __future__ import annotations

from typing import Any

from src.agents.base import AgentRequest, AgentResponse
from src.core.context import RuntimeContext
from src.core.registry import agent_registry, skill_registry, tool_registry
from src.tools.base import ToolResult


class AgentRuntime:
    """Thin runtime facade around registries.

    The main agent can start with direct calls and gradually migrate to this
    facade as more sub-agents/tools become real implementations.
    """

    def list_capabilities(self, context: RuntimeContext) -> dict[str, list[dict[str, Any]]]:
        return {
            "agents": agent_registry.describe(),
            "tools": tool_registry.describe(),
            "skills": skill_registry.describe(),
        }

    async def run_tool(
        self,
        context: RuntimeContext,
        name: str,
        *,
        caller_agent: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if caller_agent:
            agent_item = agent_registry.get(caller_agent)
            spec = getattr(agent_item.handler, "spec", None)
            allowed_tools = set(getattr(spec, "tools", ()) or ())
            if name not in allowed_tools:
                return ToolResult(False, error=f"tool {name} is not allowed for agent {caller_agent}")
        item = tool_registry.get(name)
        return await item.handler.run(context, **kwargs)

    async def run_agent(
        self,
        context: RuntimeContext,
        name: str,
        *,
        allowed_agents: tuple[str, ...] | list[str] | set[str] | None = None,
        user_input: str = "",
        intent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentResponse:
        if allowed_agents is not None and name not in set(allowed_agents):
            return AgentResponse(False, error=f"agent {name} is not allowed by current skill")
        item = agent_registry.get(name)
        handler = item.handler
        if not hasattr(handler, "run"):
            return AgentResponse(False, error=f"agent does not implement run: {name}")
        request = AgentRequest(context=context, user_input=user_input, intent=intent, payload=payload or {})
        return await handler.run(request)


runtime = AgentRuntime()
