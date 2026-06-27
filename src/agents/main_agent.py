from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict
import uuid

from langgraph.graph import END, StateGraph

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.attachments import build_attachment_prompt_context
from src.core.context import RuntimeContext
from src.core.registry import skill_registry
from src.core.runtime import runtime
from src.models.llm import chat_complete_stream
from src.prompts.system import MAIN_AGENT_PROMPT, YAAI_SYSTEM_PROMPT
from src.security.sensitive_mq import ChatAuditMessage, chat_audit_mq_service
from src.skills.base import SkillSpec


class MainAgentState(TypedDict, total=False):
    user_input: str
    role: str | None
    platform: str
    current_page: str | None
    page_type: str | None
    reply: str
    intent: str
    action_payload: dict[str, Any]
    model_tier: str


class MainAgent:
    spec = AgentSpec(
        name="main",
        description="Supervisor 主 Agent，负责选择 Skill 并调度子 Agent",
        tools=(),
        model_tier="small",
        tags=("supervisor", "router", "orchestrator"),
        capabilities=("route", "skill_select", "dispatch", "answer"),
    )

    def __init__(self) -> None:
        graph = StateGraph(MainAgentState)
        graph.add_node("route", self._route)
        graph.set_entry_point("route")
        graph.add_edge("route", END)
        self._graph = graph.compile()

    def _route(self, state: MainAgentState) -> MainAgentState:
        content = state.get("user_input", "")
        platform = state.get("platform", "frontend")
        intent = "chat"
        model_tier = "small"
        action_payload: dict[str, Any] = {}

        if "高亮" in content:
            intent = "highlight"
            action_payload = {"selector": "body", "durationMs": 3000}
        elif "跳转" in content or "打开" in content:
            intent = "navigate"
            action_payload = {"path": "/"}
        elif "填" in content or "表单" in content or "申请" in content:
            intent = "fill"
            model_tier = "large"
            action_payload = {"values": {}, "diff": []}

        if platform == "lowcode" or any(word in content for word in ["低代码", "节点", "组件", "props_json", "页面配置"]):
            model_tier = "large"

        return {**state, "intent": intent, "action_payload": action_payload, "model_tier": model_tier}

    def _select_skill(self, context: RuntimeContext, content: str) -> SkillSpec:
        candidates = skill_registry.briefs_for_context(
            platform=context.platform,
            role=context.role,
            current_page=context.current_page,
            page_type=context.page_type,
        )
        if not candidates:
            return skill_registry.get("knowledge_qa_skill").handler

        lowered = content.lower()

        def score(brief: dict[str, Any]) -> int:
            name = str(brief.get("name") or "")
            text = f"{name} {brief.get('description') or ''} {brief.get('summary') or ''}".lower()
            value = 0
            if name == "knowledge_qa_skill":
                value += 1
            if context.current_page and context.current_page in tuple(brief.get("currentPages") or ()):
                value += 8
            if context.page_type and context.page_type in tuple(brief.get("pageTypes") or ()):
                value += 8
            if context.platform in {"lowcode", "yaai_admin"} and "lowcode" in name:
                value += 4
            if context.platform in {"frontend", "yaai_portal"} and "frontend" in name:
                value += 3
            if any(word in lowered for word in ["填", "表单", "申请", "跳转", "打开", "高亮"]) and "frontend_control" in tuple(brief.get("allowedAgents") or ()):
                value += 3
            if any(word in lowered for word in ["活动", "发布"]) and "activity" in text:
                value += 5
            if any(word in lowered for word in ["低代码", "节点", "组件", "props_json", "页面配置"]) and "lowcode" in text:
                value += 5
            if any(word in lowered for word in ["知识", "解释", "介绍", "总结", "文件", "附件"]) and "qa" in text:
                value += 2
            return value

        selected = max(candidates, key=score)
        return skill_registry.get(str(selected["name"])).handler

    async def _run_sub_agent(
        self,
        context: RuntimeContext,
        skill: SkillSpec,
        agent_name: str,
        *,
        user_input: str,
        intent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentResponse:
        return await runtime.run_agent(
            context,
            agent_name,
            allowed_agents=skill.allowed_agents,
            user_input=user_input,
            intent=intent,
            payload=payload,
        )

    async def _maybe_request_frontend_action(self, context: RuntimeContext, skill: SkillSpec, state: MainAgentState) -> str:
        intent = state.get("intent")
        if intent not in {"navigate", "fill", "highlight"}:
            return ""
        action_payload = state.get("action_payload") or {}
        response = await self._run_sub_agent(
            context,
            skill,
            "frontend_control",
            user_input=f"用户请求触发前端 action：{intent}",
            intent=intent,
            payload=action_payload,
        )
        result = response.data or {"success": response.success, "error": response.error}
        await self._publish_sub_agent_result(
            context,
            "frontend_control",
            input_content=f"action={intent}, payload={action_payload}",
            output_content=str(result),
        )
        if not response.success:
            return f"前端动作未执行：action={intent}，error={response.error or result.get('error') or 'not allowed'}。"

        return f"前端动作已执行：action={intent}，结果摘要={result.get('data')}"

    async def _publish_sub_agent_result(
        self,
        context: RuntimeContext,
        sub_agent: str,
        *,
        input_content: str,
        output_content: str,
    ) -> None:
        if not context.request_id or not output_content:
            return
        memory_content = (
            f"子Agent：{sub_agent}\n"
            f"输入：{input_content.strip() or '无'}\n"
            f"最终输出：{output_content.strip()}"
        )
        try:
            await chat_audit_mq_service.publish(
                ChatAuditMessage(
                    check_id=f"chk_{uuid.uuid4().hex}",
                    connection_id=context.connection_id,
                    request_id=context.request_id,
                    message_id=f"sub_{uuid.uuid4().hex}",
                    role="sub_agent",
                    stage="sub_agent_result",
                    content=memory_content,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    sub_agent=sub_agent,
                    assistant_message_id=context.assistant_message_id,
                )
            )
        except Exception:
            pass

    async def _analyze_files(self, context: RuntimeContext, skill: SkillSpec, content: str) -> str:
        has_file_attachment = any(
            not (
                attachment.get("type") == "image"
                or str(attachment.get("mime") or "").startswith("image/")
            )
            for attachment in context.attachments
        )
        if not has_file_attachment:
            return ""
        response = await self._run_sub_agent(
            context,
            skill,
            "file_analysis",
            user_input=content,
            intent="analyze_file",
            payload={"attachments": context.attachments},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "file_analysis",
            input_content=f"用户输入：{content}\n附件：{context.attachments}",
            output_content=result,
        )
        return result

    async def _maybe_retrieve(self, context: RuntimeContext, skill: SkillSpec, content: str) -> str:
        should_retrieve = any(
            word in content
            for word in ["检索", "查询", "查找", "重复", "是否存在", "资料", "知识", "历史", "相关"]
        )
        if not should_retrieve:
            return ""
        response = await self._run_sub_agent(
            context,
            skill,
            "retrieval",
            user_input=content,
            intent="search",
            payload={"page": context.page},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "retrieval",
            input_content=content,
            output_content=result,
        )
        return result

    async def _run_controlled_react(
        self,
        context: RuntimeContext,
        skill: SkillSpec,
        content: str,
        state: MainAgentState,
    ) -> str:
        observations: list[str] = []
        for step in range(1, 4):
            if step == 1:
                file_analysis = await self._analyze_files(context, skill, content)
                if file_analysis:
                    observations.append(file_analysis)
                continue
            if step == 2:
                retrieval = await self._maybe_retrieve(context, skill, content)
                if retrieval:
                    observations.append(f"检索 Agent 结果：{retrieval}")
                continue
            if step == 3:
                frontend_action_context = await self._maybe_request_frontend_action(context, skill, state)
                if frontend_action_context:
                    observations.append(frontend_action_context)
                break
        return "\n\n".join(observations)

    async def _stream_answer(self, context: RuntimeContext, content: str, state: MainAgentState) -> AsyncIterator[str]:
        skill = self._select_skill(context, content)
        skill_prompt = skill.load_prompt()
        system_prompt = (
            f"{YAAI_SYSTEM_PROMPT}\n\n"
            f"{MAIN_AGENT_PROMPT}\n\n"
            "当前主 Agent 是 Supervisor，只能按 Supervisor -> Skill -> Sub Agent -> Tool 的链路调度；"
            "主 Agent 不直接调用底层 Tool。\n\n"
            f"已命中 Skill：{skill.name}\n"
            f"Skill 摘要：{skill.summary}\n"
            f"Skill 允许的子 Agent：{', '.join(skill.allowed_agents) or '无'}\n"
            f"Skill 详细 Prompt：{skill_prompt}"
        )
        attachment_context = await build_attachment_prompt_context(context.attachments)
        observations = await self._run_controlled_react(context, skill, content, state)
        user_prompt = (
            f"运行时信息：platform={context.platform}，userId={context.user_id}，role={context.role}，"
            f"currentPage={context.current_page}，pageType={context.page_type}，"
            f"pageDescription={context.page_description}，intent={state.get('intent')}。\n\n"
            f"{attachment_context}\n\n"
            f"Supervisor 受控 ReAct 观察结果：\n{observations}\n\n"
            f"用户输入：{content}\n\n"
            "请给出面向用户的自然回复，不要输出内部 action result、JSON、OSS URL、actionId 或调试字段。\n"
            "如果已经有前端动作上下文，直接基于该上下文回答用户，不要再说需要授权。\n"
            "如果用户要求截图、看页面或介绍当前页面，但本轮没有图片附件，请提示用户点击聊天输入框旁的截图按钮，"
            "截图会作为图片附件上传，随后你可以基于视觉模型识别结果回答。\n"
            "如果本轮包含文件附件，该文件已经由用户主动上传，并已交给文件分析 Agent 处理；不要再请求用户授权解析附件，"
            "也不要回复“只能看到文件元信息”。你必须优先基于文件分析 Agent 结果回答。"
        )
        tier = "large" if observations or state.get("model_tier") == "large" else "small"
        async for delta in chat_complete_stream(system_prompt, user_prompt, tier=tier):
            yield delta

    def _fallback_reply(self, context: RuntimeContext, content: str, error: Exception | None = None) -> str:
        has_file_attachment = any(
            not (
                attachment.get("type") == "image"
                or str(attachment.get("mime") or "").startswith("image/")
            )
            for attachment in context.attachments
        )
        if has_file_attachment and error is not None:
            return (
                "我收到了文件附件，但文件分析流程执行失败，暂时无法读取 PDF/Word 等文件内容。\n\n"
                f"失败原因：{error}"
            )

        prefix = "已收到你的消息"
        if context.role:
            prefix += f"（{context.role}）"
        return (
            f"{prefix}：{content}\n\n"
            "当前已接入 YAAI 中文提示词、LangGraph 主流程、子 Agent、ToolRegistry 和 SkillRegistry。"
            "模型调用暂时不可用时，会先使用这条框架回复兜底。"
        )

    async def stream_reply(self, context: RuntimeContext, content: str) -> AsyncIterator[str]:
        state = await self._graph.ainvoke(
            {
                "user_input": content,
                "role": context.role,
                "platform": context.platform,
                "current_page": context.current_page,
                "page_type": context.page_type,
            }
        )
        try:
            async for delta in self._stream_answer(context, content, state):
                yield delta
        except Exception as exc:
            for chunk in self._fallback_reply(context, content, exc):
                yield chunk

    async def run(self, request: AgentRequest) -> AgentResponse:
        chunks: list[str] = []
        async for chunk in self.stream_reply(request.context, request.user_input):
            chunks.append(chunk)
        return AgentResponse(True, content="".join(chunks))
