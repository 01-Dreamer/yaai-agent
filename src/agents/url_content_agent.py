from __future__ import annotations

import json
import re

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.runtime import runtime
from src.core.tool_prompt import render_agent_tool_prompt
from src.models.llm import chat_complete
from src.prompts.system import URL_CONTENT_AGENT_PROMPT

URL_PATTERN = re.compile(r"(?:(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s，。；、]*)?)")
CRAWL_KEYWORDS = ("全文爬取", "整站爬取", "深度爬取", "crawl", "爬取整个", "全站")


class UrlContentAgent:
    spec = AgentSpec(
        name="url_content_agent",
        description="根据 URL 抽取正文或按明确需求深度爬取内容的 Agent",
        tools=("url_content.extract_tool", "url_content.crawl_tool"),
        model_tier="large",
        capabilities=("extract_url_content", "crawl_url", "summarize_url"),
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        urls = self._urls(request)
        if not urls:
            return AgentResponse(False, error="missing url")
        use_crawl = self._should_crawl(request)
        tool_name = "url_content.crawl_tool" if use_crawl else "url_content.extract_tool"

        tool_results: list[dict[str, object]] = []
        for url in urls[:3]:
            result = await runtime.run_tool(
                request.context,
                tool_name,
                caller_agent=self.spec.name,
                url=url,
            )
            tool_results.append(
                {
                    "tool": tool_name,
                    "url": url,
                    "success": result.success,
                    "error": result.error,
                    "data": result.data,
                }
            )

        summary = await self._summarize(request.user_input, tool_results, use_crawl)
        return AgentResponse(
            True,
            content=summary,
            data={"urls": urls, "mode": "crawl" if use_crawl else "extract", "results": tool_results},
            used_tools=(tool_name,),
        )

    def _urls(self, request: AgentRequest) -> list[str]:
        payload = request.payload or {}
        raw_urls = payload.get("urls")
        if isinstance(raw_urls, list):
            urls = [str(item).strip() for item in raw_urls if str(item).strip()]
        else:
            url = str(payload.get("url") or "").strip()
            urls = [url] if url else []
        if not urls:
            urls = [match.group(0).strip() for match in URL_PATTERN.finditer(request.user_input)]
        normalized: list[str] = []
        for url in urls:
            normalized.append(url if url.startswith(("http://", "https://")) else f"https://{url}")
        return normalized

    def _should_crawl(self, request: AgentRequest) -> bool:
        payload = request.payload or {}
        if payload.get("mode") == "crawl":
            return True
        content = request.user_input.lower()
        return any(keyword in content for keyword in CRAWL_KEYWORDS)

    async def _summarize(self, user_input: str, tool_results: list[dict[str, object]], use_crawl: bool) -> str:
        compact_results = json.dumps(tool_results, ensure_ascii=False, default=str)[:24000]
        mode = "crawl 深度爬取" if use_crawl else "extract 正文抽取"
        user_prompt = (
            f"用户请求：{user_input}\n\n"
            f"本轮模式：{mode}\n\n"
            f"URL 内容工具返回结果如下：\n{compact_results}\n\n"
            "请汇总为给主 Agent 使用的中文结论，保留页面标题、URL、关键事实、可引用内容和缺失信息。"
        )
        try:
            system_prompt = f"{URL_CONTENT_AGENT_PROMPT}\n\n{render_agent_tool_prompt(self.spec)}"
            return await chat_complete(system_prompt, user_prompt, tier="large")
        except Exception:
            return self._fallback_summary(tool_results, use_crawl)

    def _fallback_summary(self, tool_results: list[dict[str, object]], use_crawl: bool) -> str:
        mode = "crawl" if use_crawl else "extract"
        lines = [f"URL 内容 Agent 结果：mode={mode}"]
        for item in tool_results:
            if not item.get("success"):
                lines.append(f"- {item.get('url')}：失败，原因={item.get('error')}")
                continue
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            results = data.get("results") if isinstance(data, dict) else []
            if not isinstance(results, list):
                continue
            for result in results[:3]:
                if not isinstance(result, dict):
                    continue
                title = result.get("title") or "无标题"
                url = result.get("url") or item.get("url")
                content = str(result.get("content") or "")[:1200]
                lines.append(f"- {title} {url}\n{content}")
        return "\n\n".join(lines)
