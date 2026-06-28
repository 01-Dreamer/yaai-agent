from __future__ import annotations

from typing import Any

from src.core.context import RuntimeContext
from src.core.rag_index import markdown_knowledge_index
from src.tools.base import ToolResult, ToolSpec


class RagSearchTool:
    def __init__(self, mode: str) -> None:
        if mode not in {"semantic", "keyword"}:
            raise ValueError(f"unsupported rag mode: {mode}")
        self.mode = mode
        self.name = "rag.search_documents_tool" if mode == "semantic" else "rag.keyword_search_tool"
        self.description = (
            "语义检索本地 Markdown 知识库。参数：查询内容必填，返回条数可选，默认 5 条，最多 20 条。"
            "返回：结果列表，包含文章标题、段落标题、来源文件、正文内容、相似度分数等。"
            "示例：{\"查询内容\":\"会员申请流程\",\"返回条数\":5}。适合概念解释、流程说明、配置指南等语义相近问题。"
        ) if mode == "semantic" else (
            "关键词检索本地 Markdown 知识库。参数：关键词或查询内容必填，返回条数可选，默认 8 条，最多 20 条。"
            "返回：结果列表，包含文章标题、段落标题、来源文件、正文内容等。"
            "示例：{\"关键词\":\"会员审核\",\"返回条数\":8}。适合用户给出明确词语、标题、字段名时使用。"
        )
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="rag",

            capabilities=("semantic_search" if mode == "semantic" else "keyword_search",),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query") or kwargs.get("keyword") or "").strip()
        if not query:
            return ToolResult(False, error="missing query")
        limit = int(kwargs.get("limit") or kwargs.get("top_k") or kwargs.get("topK") or (5 if self.mode == "semantic" else 8))
        limit = max(1, min(limit, 20))
        if self.mode == "semantic":
            data = await markdown_knowledge_index.semantic_search(query, limit=limit)
        else:
            data = await markdown_knowledge_index.keyword_search(query, limit=limit)
        results = data.get("results") or []
        summary = "\n\n".join(
            f"[{index}] {item.get('title')} / {item.get('heading')} / {item.get('source')}\n{item.get('content')}"
            for index, item in enumerate(results, start=1)
        )
        return ToolResult(True, data=data, summary=summary)
