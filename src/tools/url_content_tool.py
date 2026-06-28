from __future__ import annotations

from typing import Any

import httpx

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


class UrlContentTool:
    def __init__(self, action: str) -> None:
        self.action = action
        self.name = f"url_content.{action}_tool"
        self.description = (
            "抽取指定网页链接的正文内容。参数：网页链接必填。返回：操作类型、链接、成功结果列表、失败结果列表；"
            "每条成功结果包含链接、标题、正文。示例：{\"网页链接\":\"https://example.com/article\"}。"
            "用途：用户给出具体网页并需要理解正文时优先使用。"
        ) if action == "extract" else (
            "深度爬取指定网页链接。参数：网页链接必填。返回：操作类型、链接、基础网址、成功结果列表、失败结果列表；"
            "每条成功结果包含多个子页面的链接、标题、正文。示例：{\"网页链接\":\"https://example.com\"}。"
            "限制：开销较大，只有用户明确要求全文爬取、整站爬取、深度爬取时才使用。"
        )
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="url_content",

            capabilities=("extract_url_content", action),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url") or "").strip()
        if not url:
            return ToolResult(False, error="missing url")
        if not settings.tavily_api_key:
            return ToolResult(False, error="TAVILY_API_KEY is not configured")
        try:
            if self.action == "extract":
                return await self._extract(url)
            if self.action == "crawl":
                return await self._crawl(url)
            return ToolResult(False, error=f"unsupported url content action: {self.action}")
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000] if exc.response is not None else ""
            return ToolResult(False, error=f"http {exc.response.status_code}: {body}")
        except httpx.HTTPError as exc:
            return ToolResult(False, error=f"http error: {exc}")

    async def _extract(self, url: str) -> ToolResult:
        payload = {"urls": [url]}
        headers = {
            "Authorization": f"Bearer {settings.tavily_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.tavily.com/extract", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        results = [
            {
                "url": item.get("url"),
                "title": item.get("title"),
                "content": item.get("raw_content"),
            }
            for item in data.get("results") or []
        ]
        return ToolResult(
            True,
            data={
                "action": "extract",
                "url": url,
                "results": results,
                "failedResults": data.get("failed_results") or [],
                "requestId": data.get("request_id"),
                "responseTime": data.get("response_time"),
            },
        )

    async def _crawl(self, url: str) -> ToolResult:
        payload = {"url": url, "extract_depth": "advanced"}
        headers = {
            "Authorization": f"Bearer {settings.tavily_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post("https://api.tavily.com/crawl", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        results = [
            {
                "url": item.get("url"),
                "title": item.get("title"),
                "content": item.get("raw_content"),
            }
            for item in data.get("results") or []
        ]
        return ToolResult(
            True,
            data={
                "action": "crawl",
                "url": url,
                "baseUrl": data.get("base_url"),
                "results": results,
                "failedResults": data.get("failed_results") or [],
                "requestId": data.get("request_id"),
            },
        )
