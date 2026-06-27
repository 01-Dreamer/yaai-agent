from __future__ import annotations

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.context import RuntimeContext


class RetrievalAgent:
    spec = AgentSpec(
        name="retrieval",
        description="文档RAG、Neo4j、Java 后端业务信息检索 Agent",
        tools=("memory.load_recent", "rag.search_documents", "graph.query", "backend.user_context"),
        tags=("retrieval", "rag", "graph"),
        capabilities=("search", "deduplicate", "evidence"),
    )

    async def search(self, context: RuntimeContext, query: str) -> dict[str, object]:
        return {"summary": "retrieval scaffold", "query": query, "evidence": []}

    async def run(self, request: AgentRequest) -> AgentResponse:
        result = await self.search(request.context, request.user_input)
        return AgentResponse(True, content=str(result.get("summary") or ""), data=result)
