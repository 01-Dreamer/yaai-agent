from __future__ import annotations

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.context import RuntimeContext
from src.core.runtime import runtime


class RetrievalAgent:
    spec = AgentSpec(
        name="retrieval_agent",
        description="文档 RAG 和 Neo4j 知识图谱信息检索 Agent",
        tools=(
            "rag.search_documents_tool",
            "rag.keyword_search_tool",
            "graph.query_tool",
        ),
        capabilities=("search", "deduplicate", "evidence"),
    )

    graph_triggers = (
        "专家",
        "老师",
        "教授",
        "导师",
        "研究方向",
        "项目",
        "政策",
        "活动",
        "会议",
        "论坛",
        "研讨会",
        "组织",
        "单位",
        "学院",
        "企业",
        "协会",
        "知识图谱",
        "是谁",
        "任职",
        "介绍",
    )

    async def search(self, context: RuntimeContext, query: str) -> dict[str, object]:
        graph = None
        if any(trigger in query for trigger in self.graph_triggers):
            graph = await runtime.run_tool(
                context,
                "graph.query_tool",
                caller_agent=self.spec.name,
                query=query,
                limit=8,
            )

        semantic = await runtime.run_tool(
            context,
            "rag.search_documents_tool",
            caller_agent=self.spec.name,
            query=query,
            limit=5,
        )
        keyword = await runtime.run_tool(
            context,
            "rag.keyword_search_tool",
            caller_agent=self.spec.name,
            keyword=query,
            limit=5,
        )
        evidence = []
        if semantic.success:
            evidence.extend((semantic.data.get("results") or [])[:5])
        if keyword.success:
            seen = {item.get("chunkId") for item in evidence if isinstance(item, dict)}
            evidence.extend(
                item for item in (keyword.data.get("results") or [])[:5]
                if isinstance(item, dict) and item.get("chunkId") not in seen
            )
        doc_summary = "\n\n".join(
            f"- {item.get('title')} / {item.get('heading')}：{str(item.get('content') or '')[:300]}"
            for item in evidence[:6]
            if isinstance(item, dict)
        )
        summary_parts = []
        if graph and graph.success and graph.summary:
            summary_parts.append(graph.summary)
        if doc_summary:
            summary_parts.append(f"知识文档检索结果：\n{doc_summary}")
        summary = "\n\n".join(summary_parts)
        return {
            "summary": summary or "未检索到相关知识",
            "query": query,
            "evidence": evidence[:8],
            "graph": graph.data if graph and graph.success else None,
            "graphError": graph.error if graph else None,
            "semanticError": semantic.error,
            "keywordError": keyword.error,
        }

    async def run(self, request: AgentRequest) -> AgentResponse:
        result = await self.search(request.context, request.user_input)
        return AgentResponse(True, content=str(result.get("summary") or ""), data=result)
