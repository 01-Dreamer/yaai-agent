from __future__ import annotations

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.context import RuntimeContext


class MemoryCompressionAgent:
    spec = AgentSpec(
        name="memory_compression",
        description="压缩 session 记忆并更新 agent_session.memory_content",
        tools=("memory.load_recent", "memory.update_session_summary"),
        tags=("memory", "compression"),
        capabilities=("compress", "summarize"),
    )

    async def compress(self, context: RuntimeContext, messages: list[str]) -> str:
        return "\n".join(messages)[-2000:]

    async def run(self, request: AgentRequest) -> AgentResponse:
        payload = request.payload or {}
        messages = payload.get("messages") or [request.user_input]
        content = await self.compress(request.context, [str(message) for message in messages])
        return AgentResponse(True, content=content)
