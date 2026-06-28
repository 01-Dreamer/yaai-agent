from __future__ import annotations

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.models.llm import chat_complete
from src.prompts.system import RESPONSE_AGENT_PROMPT


class ResponseAgent:
    spec = AgentSpec(
        name="response_agent",
        description="最终输出 Agent，负责汇总所有子 Agent 结果并生成简洁、安全的用户回复",
        tools=(),
        model_tier="small",
        capabilities=("final_response",),
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        user_input = request.user_input or ""
        observations = str(request.payload.get("observations") or "").strip()
        intent = str(request.payload.get("intent") or "chat").strip()
        role = str(request.payload.get("role") or "")
        platform = str(request.payload.get("platform") or "")

        user_prompt = (
            f"用户原始请求：{user_input}\n"
            f"用户角色：{role or '未知'}\n"
            f"当前平台：{platform}\n"
            f"本轮意图：{intent}\n\n"
            f"子 Agent 执行结果汇总：\n{observations or '无'}\n\n"
            "请基于以上信息，生成一段简洁、自然、面向用户的最终回复。"
        )
        try:
            reply = await chat_complete(RESPONSE_AGENT_PROMPT, user_prompt, tier="small")
        except Exception:
            reply = observations or user_input or "已处理你的请求。"

        return AgentResponse(True, content=reply, data={"reply": reply})
