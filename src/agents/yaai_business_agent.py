from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.runtime import runtime
from src.core.tool_prompt import render_agent_tool_prompt
from src.models.llm import chat_complete
from src.prompts.system import YAAI_BUSINESS_AGENT_PROMPT


class YaaiBusinessAgent:
    spec = AgentSpec(
        name="yaai_business_agent",
        description="YAAI 后端业务 Agent，负责会员、委员会、新闻、审核、日志、缴费等业务接口编排",
        tools=(
            "backend.search_news_tool",
            "backend.get_member_profile_tool",
            "backend.list_committees_tool",
            "backend.get_committee_detail_tool",
            "backend.list_member_audits_tool",
            "backend.get_operation_logs_tool",
            "backend.create_payment_url_tool",
        ),
        model_tier="large",
        capabilities=("yaai_business", "member_service", "committee_service", "payment", "admin_query"),
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        plan = await self._plan(request)
        calls = plan.get("calls") if isinstance(plan, dict) else None
        if not isinstance(calls, list) or not calls:
            return AgentResponse(False, error="业务 Agent 未能选择合适的后端业务工具")

        tool_results: list[dict[str, Any]] = []
        used_tools: list[str] = []
        for call in calls[:3]:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool") or "").strip()
            if tool_name not in self.spec.tools:
                tool_results.append({"tool": tool_name, "success": False, "error": "tool is not allowed"})
                continue
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            result = await runtime.run_tool(
                request.context,
                tool_name,
                caller_agent=self.spec.name,
                **arguments,
            )
            used_tools.append(tool_name)
            tool_results.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "success": result.success,
                    "error": result.error,
                    "summary": result.summary,
                    "data": result.data,
                }
            )

        summary = await self._summarize(request.user_input, tool_results)
        return AgentResponse(
            True,
            content=summary,
            data={"plan": plan, "toolResults": tool_results},
            used_tools=tuple(used_tools),
        )

    async def _plan(self, request: AgentRequest) -> dict[str, Any]:
        payload = request.payload or {}
        user_prompt = (
            f"用户请求：{request.user_input}\n"
            f"运行时信息：platform={request.context.platform}, role={request.context.role}, "
            f"userId={request.context.user_id}\n"
            f"额外上下文：{json.dumps(payload, ensure_ascii=False, default=str)[:6000]}\n\n"
            f"{render_agent_tool_prompt(self.spec)}\n\n"
            "请根据用户请求选择 1 到 3 个最合适的后端业务工具，并只输出 JSON object。\n"
            "格式：{\"calls\":[{\"tool\":\"backend.xxx_tool\",\"arguments\":{...}}],\"reason\":\"...\"}\n"
            "要求：\n"
            "1. 只使用绑定 Tool 列表中的工具；\n"
            "2. 不要自行编造 memberId、committeeId、categoryId；用户未提供且工具可默认当前用户时，可以不传；\n"
            "3. 普通用户和管理员不在这里硬拆，后端会基于 token 做最终权限校验；\n"
            "4. 如果请求需要缴费/支付链接，选择 backend.create_payment_url_tool；\n"
            "5. 如果信息不足以调用任何工具，返回 {\"calls\":[],\"reason\":\"缺少...\"}。"
        )
        raw = await chat_complete(YAAI_BUSINESS_AGENT_PROMPT, user_prompt, tier="large")
        return self._extract_json_object(raw)

    async def _summarize(self, user_input: str, tool_results: list[dict[str, Any]]) -> str:
        compact = json.dumps(tool_results, ensure_ascii=False, default=str)[:24000]
        user_prompt = (
            f"用户请求：{user_input}\n\n"
            f"后端业务工具结果：\n{compact}\n\n"
            "请生成给用户看的中文回复。要求：\n"
            "1. 只说用户需要知道的信息；\n"
            "2. 如果后端因权限拒绝或参数不足失败，直接说明原因和下一步需要的信息；\n"
            "3. 不暴露内部 tool 名称、token、接口路径和原始大 JSON；\n"
            "4. 金额、链接、会员状态、审核状态、日志时间等关键信息要准确保留。"
        )
        try:
            return await chat_complete(YAAI_BUSINESS_AGENT_PROMPT, user_prompt, tier="large")
        except Exception:
            lines = ["业务查询结果："]
            for item in tool_results:
                if not item.get("success"):
                    lines.append(f"- 查询失败：{item.get('error')}")
                else:
                    lines.append(f"- {item.get('summary') or '查询成功'}")
            return "\n".join(lines)

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("business plan is not a JSON object")
        return data
