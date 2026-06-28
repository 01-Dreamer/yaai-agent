from __future__ import annotations

import json

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.tool_prompt import render_agent_tool_prompt
from src.core.runtime import runtime
from src.models.llm import chat_complete
from src.prompts.system import WEB_SEARCH_AGENT_PROMPT


class WebSearchAgent:
    spec = AgentSpec(
        name="web_search_agent",
        description="根据关键词调用 Tavily、阿里云 OpenSearch、百度千帆 AI Search 并汇总信息的 Agent",
        tools=("web_search.tavily_tool", "web_search.aliyun_tool", "web_search.baidu_tool"),
        model_tier="large",
        capabilities=("web_search_agent", "summarize_sources", "fresh_information"),
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        keyword = self._keyword(request)
        if not keyword:
            return AgentResponse(False, error="missing search keyword")

        provider_results: list[dict[str, object]] = []
        used_tools: list[str] = []
        for tool_name in self.spec.tools:
            result = await runtime.run_tool(
                request.context,
                tool_name,
                caller_agent=self.spec.name,
                keyword=keyword,
            )
            used_tools.append(tool_name)
            provider_results.append(
                {
                    "tool": tool_name,
                    "success": result.success,
                    "error": result.error,
                    "data": result.data,
                }
            )

        summary = await self._summarize(keyword, provider_results)
        return AgentResponse(
            True,
            content=summary,
            data={"keyword": keyword, "providers": provider_results},
            used_tools=tuple(used_tools),
        )

    def _keyword(self, request: AgentRequest) -> str:
        payload = request.payload or {}
        keyword = str(payload.get("keyword") or request.user_input or "").strip()
        return keyword[:200]

    async def _summarize(self, keyword: str, provider_results: list[dict[str, object]]) -> str:
        compact_results = json.dumps(provider_results, ensure_ascii=False, default=str)[:24000]
        user_prompt = (
            f"搜索关键词：{keyword}\n\n"
            f"三个搜索工具返回结果如下：\n{compact_results}\n\n"
            "请汇总为给主 Agent 使用的中文结论。要求：\n"
            "1. 优先列出一致事实和最新信息；\n"
            "2. 标明来源标题、链接、发布时间或网站；\n"
            "3. 区分事实、推断和不确定信息；\n"
            "4. 如果某个搜索工具失败，说明失败但继续使用其他来源；\n"
            "5. 不要编造没有来源的信息。"
        )
        try:
            system_prompt = f"{WEB_SEARCH_AGENT_PROMPT}\n\n{render_agent_tool_prompt(self.spec)}"
            return await chat_complete(system_prompt, user_prompt, tier="large")
        except Exception:
            return self._fallback_summary(keyword, provider_results)

    def _fallback_summary(self, keyword: str, provider_results: list[dict[str, object]]) -> str:
        lines = [f"关键词搜索 Agent 结果：关键词={keyword}"]
        for provider in provider_results:
            tool_name = provider.get("tool")
            if not provider.get("success"):
                lines.append(f"- {tool_name}：失败，原因={provider.get('error')}")
                continue
            data = provider.get("data") if isinstance(provider.get("data"), dict) else {}
            results = data.get("results") if isinstance(data, dict) else []
            lines.append(f"- {tool_name}：返回 {len(results) if isinstance(results, list) else 0} 条结果")
            if isinstance(results, list):
                for item in results[:3]:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title") or "无标题"
                    url = item.get("url") or ""
                    content = str(item.get("content") or "")[:200]
                    lines.append(f"  - {title} {url}\n    {content}")
            answer = data.get("answer") if isinstance(data, dict) else None
            if answer:
                lines.append(f"  AI 搜索摘要：{str(answer)[:1000]}")
        return "\n".join(lines)
