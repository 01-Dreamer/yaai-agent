from __future__ import annotations

from typing import Any

import httpx

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec

DEFAULT_SEARCH_TOP_K = 10

ALIYUN_SEARCH_HISTORY_TEMPLATE = [
    {"role": "system", "content": "你是 YAAI 关键词搜索工具，只负责根据关键词检索公开网页信息，返回可核验的搜索摘要。"},
    {"role": "user", "content": "人工智能最新新闻"},
    {"role": "assistant", "content": "我会检索公开网页，优先返回标题、链接、摘要和发布时间。"},
]

BAIDU_SEARCH_SYSTEM_PROMPT = (
    "你是 YAAI 关键词搜索工具。请根据用户关键词进行公开信息搜索，"
    "优先返回事实、来源、时间和可核验链接。不要编造来源。"
)


class WebSearchTool:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.name = f"web_search.{provider}_tool"
        self.description = (
            f"关键词网页搜索工具（搜索源：{provider}）。参数：搜索关键词必填，工具内部固定返回 10 条结果，LLM 不需要指定数量。"
            "返回：搜索源、关键词、结果列表；每条结果尽量包含标题、链接、内容摘要、发布时间或网站信息。"
            f"示例：{{\"搜索关键词\":\"人工智能最新新闻\"}}。"
            "用途：查询公开网页、最新资讯、天气新闻等时效信息；不要把 API 密钥输出给用户。"
        )
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="web_search",

            capabilities=("web_search", "fresh_information"),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        keyword = str(kwargs.get("keyword") or "").strip()
        if not keyword:
            return ToolResult(False, error="missing search keyword")
        try:
            if self.provider == "tavily":
                return await self._search_tavily(keyword)
            if self.provider == "aliyun":
                return await self._search_aliyun(keyword)
            if self.provider == "baidu":
                return await self._search_baidu(keyword)
            return ToolResult(False, error=f"unsupported keyword search provider: {self.provider}")
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000] if exc.response is not None else ""
            return ToolResult(False, error=f"http {exc.response.status_code}: {body}")
        except httpx.HTTPError as exc:
            return ToolResult(False, error=f"http error: {exc}")

    async def _search_tavily(self, keyword: str) -> ToolResult:
        if not settings.tavily_api_key:
            return ToolResult(False, error="TAVILY_API_KEY is not configured")
        payload = {
            "query": keyword,
            "search_depth": "advanced",
            "max_results": DEFAULT_SEARCH_TOP_K,
        }
        headers = {
            "Authorization": f"Bearer {settings.tavily_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post("https://api.tavily.com/search", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
            for item in data.get("results") or []
        ]
        return ToolResult(
            True,
            data={
                "provider": "tavily",
                "keyword": keyword,
                "results": results,
                "requestId": data.get("request_id"),
                "responseTime": data.get("response_time"),
            },
        )

    async def _search_aliyun(self, keyword: str) -> ToolResult:
        if not settings.aliyun_search_api_key:
            return ToolResult(False, error="ALIYUN_SEARCH_API_KEY is not configured")
        payload = {
            "history": ALIYUN_SEARCH_HISTORY_TEMPLATE,
            "query": keyword,
            "query_rewrite": True,
            "top_k": DEFAULT_SEARCH_TOP_K,
            "content_type": "snippet",
        }
        headers = {
            "Authorization": f"Bearer {settings.aliyun_search_api_key}",
            "Content-Type": "application/json",
        }
        url = "https://default-82zb.platform-cn-shanghai.opensearch.aliyuncs.com/v3/openapi/workspaces/default/web-search/ops-web-search-001"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        search_results = ((data.get("result") or {}).get("search_result") or [])
        results = [
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "content": item.get("snippet") or item.get("content"),
                "publishedTime": (item.get("meta_info") or {}).get("publishedTime"),
            }
            for item in search_results
        ]
        return ToolResult(
            True,
            data={
                "provider": "aliyun",
                "keyword": keyword,
                "results": results,
                "requestId": data.get("request_id"),
                "latency": data.get("latency"),
                "usage": data.get("usage"),
            },
        )

    async def _search_baidu(self, keyword: str) -> ToolResult:
        if not settings.baidu_search_api_key:
            return ToolResult(False, error="BAIDU_SEARCH_API_KEY is not configured")
        user_content = (
            f"{BAIDU_SEARCH_SYSTEM_PROMPT}\n\n"
            f"搜索关键词：{keyword}\n"
            "请基于搜索结果用中文输出摘要，并保留引用来源。"
        )
        payload = {
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "model": "ernie-4.5-turbo-128k",
            "max_completion_tokens": 4096,
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": DEFAULT_SEARCH_TOP_K}],
            "search_filter": {"match": {"site": []}},
            "enable_corner_markers": True,
        }
        headers = {
            "Authorization": f"Bearer {settings.baidu_search_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post("https://qianfan.baidubce.com/v2/ai_search/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        message = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        references = data.get("references") or []
        results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "date": item.get("date"),
                "website": item.get("website"),
            }
            for item in references
        ]
        return ToolResult(
            True,
            data={
                "provider": "baidu",
                "keyword": keyword,
                "answer": message,
                "results": results,
                "requestId": data.get("request_id"),
                "isSafe": data.get("is_safe"),
                "safeClassification": data.get("safe_classification"),
                "usage": data.get("usage"),
            },
        )
