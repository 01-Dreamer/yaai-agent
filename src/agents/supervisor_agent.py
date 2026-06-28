from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict
import uuid
import re
import json

from langgraph.graph import END, StateGraph

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.attachments import build_attachment_prompt_context
from src.core.context import RuntimeContext
from src.core.registry import skill_registry
from src.core.runtime import runtime
from src.core.tool_prompt import render_allowed_agents_tool_prompt
from src.models.llm import chat_complete, chat_complete_stream
from src.repositories.agent_memory import agent_memory_repository
from src.prompts.system import SUPERVISOR_AGENT_PROMPT, YAAI_SYSTEM_PROMPT
from src.security.sensitive_mq import ChatAuditMessage, chat_audit_mq_service


URL_PATTERN = re.compile(r"(?:(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s，。；、]*)?)")

class SupervisorAgentState(TypedDict, total=False):
    user_input: str
    role: str | None
    platform: str
    current_page: str | None
    page_type: str | None
    reply: str
    intent: str
    action_payload: dict[str, Any]
    model_tier: str


class SupervisorAgent:
    spec = AgentSpec(
        name="supervisor_agent",
        description="Supervisor 主 Agent，负责选择 Skill 并调度子 Agent",
        tools=("memory.import_full_memory_tool", "skill.activate_skill_tool"),
        model_tier="small",
        capabilities=("route", "skill_select", "dispatch", "answer"),
    )

    def __init__(self) -> None:
        graph = StateGraph(SupervisorAgentState)
        graph.add_node("route", self._route)
        graph.set_entry_point("route")
        graph.add_edge("route", END)
        self._graph = graph.compile()

    def _route(self, state: SupervisorAgentState) -> SupervisorAgentState:
        content = state.get("user_input", "")
        lowered = content.lower()
        platform = state.get("platform", "frontend")
        intent = "chat"
        model_tier = "small"
        action_payload: dict[str, Any] = {}

        if any(
            word in lowered
            for word in [
                "html",
                "页面结构",
                "表单字段",
                "字段格式",
                "填表格式",
                "页面信息",
                "选项",
                "下拉",
                "有哪些可以选",
                "有哪些可以选择",
                "可以选择的",
                "可选项",
                "所属委员会",
            ]
        ):
            intent = "inspect_html"
            action_payload = {
                "maxHtmlLength": 20000,
                "maxFields": 120,
                "targetField": content,
                "requiresConfirm": False,
            }
        elif any(word in content for word in ["高亮", "标红", "标注", "标记", "红色", "变红", "突出显示"]):
            intent = "highlight"
            model_tier = "large"
            action_payload = {
                "maxHtmlLength": 20000,
                "maxFields": 120,
                "targetField": content,
                "requiresConfirm": False,
            }
        elif any(word in content for word in ["跳转", "打开", "进入", "前往", "去到", "切换到"]):
            intent = "navigate"
            action_payload = {"target": content, "requiresConfirm": True}
        elif "填" in content or "表单" in content or "申请" in content:
            intent = "fill"
            model_tier = "large"
            action_payload = {
                "maxHtmlLength": 20000,
                "maxFields": 120,
                "targetField": content,
            }

        if platform == "lowcode" or any(word in content for word in ["低代码", "节点", "组件", "props_json", "页面配置"]):
            model_tier = "large"

        return {**state, "intent": intent, "action_payload": action_payload, "model_tier": model_tier}

    def _select_skill_name(self, context: RuntimeContext, content: str) -> str:
        candidates = skill_registry.briefs_for_context(
            authenticated=context.authenticated,
            platform=context.platform,
            role=context.role,
            current_page=context.current_page,
            page_type=context.page_type,
        )
        if not candidates:
            return "browser_operation_skill"

        lowered = content.lower()

        def score(brief: dict[str, Any]) -> int:
            name = str(brief.get("name") or "")
            text = f"{name} {brief.get('description') or ''}".lower()
            value = 0
            if name == "browser_operation_skill":
                value += 1
            if context.platform in {"lowcode", "yaai_admin"} and "lowcode" in name:
                value += 4
            if context.platform in {"frontend", "yaai_portal"} and "frontend" in name:
                value += 3
            if any(word in lowered for word in ["填", "表单", "申请", "跳转", "打开", "进入", "前往", "去到", "切换到", "高亮", "标红", "标注", "标记", "红色", "变红", "突出显示", "html", "页面结构", "表单字段", "字段格式", "填表格式"]) and any(word in text for word in ["browser", "浏览器", "前端", "页面"]):
                value += 3
            if any(word in lowered for word in ["活动", "发布"]) and "activity" in text:
                value += 5
            if any(word in lowered for word in ["低代码", "节点", "组件", "props_json", "页面配置"]) and "lowcode" in text:
                value += 5
            if any(word in lowered for word in ["知识", "解释", "介绍", "总结", "文件", "附件"]) and "qa" in text:
                value += 2
            if any(word in lowered for word in ["最新", "今日", "今天", "现在", "实时", "新闻", "联网", "网页", "搜索", "查一下"]) and any(word in text for word in ["search", "搜索", "网页", "新闻"]):
                value += 6
            if URL_PATTERN.search(content) and any(word in text for word in ["url", "链接", "网页"]):
                value += 6
            if any(word in content for word in ["会员", "委员会", "订单", "缴费", "支付", "审核", "日志", "操作日志", "后端业务"]) and any(word in text for word in ["业务", "会员", "委员会", "审核", "日志", "订单"]):
                value += 8
            if any(word in content for word in ["全部记忆", "完整记忆", "历史记忆", "之前说过", "之前的上下文"]) and any(word in text for word in ["记忆", "历史", "上下文"]):
                value += 8
            if any(word in content for word in ["写", "生成", "文案", "新闻稿", "通知", "邮件", "图片", "海报", "画"]) and any(word in text for word in ["内容", "创作", "邮件", "图片", "文案"]):
                value += 6
            return value

        selected = max(candidates, key=score)
        return str(selected["name"])

    async def _activate_skill(self, context: RuntimeContext, skill_name: str) -> dict[str, Any]:
        result = await runtime.run_tool(
            context,
            "skill.activate_skill_tool",
            caller_agent=self.spec.name,
            skill_name=skill_name,
        )
        if not result.success:
            return {
                "name": skill_name,
                "description": "",
                "summary": "",
                "allowedAgents": (),
                "prompt": "",
                "error": result.error,
            }
        return result.data

    async def _load_recent_memory_context(self, context: RuntimeContext) -> str:
        messages = await agent_memory_repository.load_recent_context(
            session_id=context.session_id,
            user_id=context.user_id,
            limit=20,
        )
        if not messages:
            return "无"
        lines: list[str] = []
        for message in messages:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            attachments = message.get("attachments") or []
            attachment_text = f" 附件={attachments}" if attachments else ""
            lines.append(
                f"[{message.get('id')}] {message.get('createdAt') or ''} "
                f"{message.get('role')}: {content}{attachment_text}"
            )
        return "\n".join(lines) or "无"

    async def _maybe_import_full_memory(self, context: RuntimeContext, content: str) -> str:
        if not any(word in content for word in ["全部记忆", "完整记忆", "所有记忆", "历史记忆", "全部上下文", "完整上下文", "之前说过"]):
            return ""
        result = await runtime.run_tool(
            context,
            "memory.import_full_memory_tool",
            caller_agent=self.spec.name,
            session_id=context.session_id,
        )
        if not result.success:
            return f"完整记忆导入失败：{result.error or 'unknown error'}"
        return str(result.data.get("memoryText") or result.summary or "")

    async def _run_sub_agent(
        self,
        context: RuntimeContext,
        allowed_agents: tuple[str, ...],
        agent_name: str,
        *,
        user_input: str,
        intent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentResponse:
        return await runtime.run_agent(
            context,
            agent_name,
            allowed_agents=allowed_agents,
            user_input=user_input,
            intent=intent,
            payload=payload,
        )

    async def _maybe_request_frontend_action(self, context: RuntimeContext, allowed_agents: tuple[str, ...], state: SupervisorAgentState) -> str:
        intent = state.get("intent")
        if intent not in {"navigate", "fill", "highlight", "inspect_html"}:
            return ""
        action_payload = state.get("action_payload") or {}
        observation, _ = await self._request_frontend_action(context, allowed_agents, intent, action_payload)
        return observation

    async def _request_frontend_action(
        self,
        context: RuntimeContext,
        allowed_agents: tuple[str, ...],
        action: str,
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        response = await self._run_sub_agent(
            context,
            allowed_agents,
            "browser_agent",
            user_input=f"用户请求触发前端 action：{action}",
            intent=action,
            payload=payload,
        )
        result = response.data or {"success": response.success, "error": response.error}
        await self._publish_sub_agent_result(
            context,
            "browser_agent",
            input_content=f"action={action}, payload={payload}",
            output_content=str(result),
        )
        if not response.success:
            return (
                f"前端动作未执行：action={action}，error={response.error or result.get('error') or 'not allowed'}。",
                result,
            )

        return f"前端动作已执行：action={action}，结果摘要={result.get('data')}", result

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
            raise ValueError("LLM fill payload is not a JSON object")
        return data

    async def _generate_fill_payload_with_llm(
        self,
        content: str,
        inspect_data: dict[str, Any],
    ) -> dict[str, Any]:
        fields = inspect_data.get("fields") if isinstance(inspect_data, dict) else []
        compact_fields = []
        for field in fields or []:
            if not isinstance(field, dict):
                continue
            compact_fields.append(
                {
                    "label": field.get("label"),
                    "name": field.get("name"),
                    "id": field.get("id"),
                    "selector": field.get("selector"),
                    "tag": field.get("tag"),
                    "type": field.get("type"),
                    "placeholder": field.get("placeholder"),
                    "required": field.get("required"),
                    "options": field.get("options"),
                }
            )
        system_prompt = (
            "你是前端填表 JSON 生成器。只能根据用户要求和页面字段生成 JSON，不要输出解释。\n"
            "输出格式必须是：{\"values\": {\"字段label或name\": \"要填写的值\"}, "
            "\"diff\": [{\"field\": \"字段label或name\", \"to\": \"要填写的值\"}]}\n"
            "规则：\n"
            "1. 只填写页面 fields 中存在的可编辑字段。\n"
            "2. 如果字段有 options，必须从 options 的 label/value 中选择一个已有选项。\n"
            "3. 用户要求“随便填写/示例/参考”时，生成合理但明显为示例的数据。\n"
            "4. 不填写提交按钮，不提交表单，不输出 markdown。"
        )
        user_prompt = (
            f"用户要求：{content}\n\n"
            f"页面字段 JSON：{json.dumps(compact_fields, ensure_ascii=False)[:24000]}\n\n"
            "请只返回一个 JSON object。"
        )
        raw = await chat_complete(system_prompt, user_prompt, tier="large")
        payload = self._extract_json_object(raw)
        values = payload.get("values")
        diff = payload.get("diff")
        if not isinstance(values, dict):
            values = {}
        if not isinstance(diff, list):
            diff = [{"field": key, "to": value} for key, value in values.items()]
        normalized_values: dict[str, Any] = {}
        normalized_diff: list[dict[str, Any]] = []
        for item in diff:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or item.get("name") or "").strip()
            value = item.get("to", item.get("value"))
            if not field:
                continue
            normalized_values[field] = value
            normalized_diff.append({"field": field, "to": value})
        for field, value in values.items():
            field_text = str(field).strip()
            if field_text and field_text not in normalized_values:
                normalized_values[field_text] = value
                normalized_diff.append({"field": field_text, "to": value})
        return {"values": normalized_values, "diff": normalized_diff}

    async def _generate_navigate_payload(self, context: RuntimeContext, content: str) -> dict[str, Any]:
        from src.agents.browser_agent import BrowserAgent
        agent = BrowserAgent()
        return await agent.generate_navigate_payload(context, content)

    async def _generate_highlight_payload_with_llm(
        self,
        content: str,
        inspect_data: dict[str, Any],
    ) -> dict[str, Any]:
        html = str(inspect_data.get("html") or "")[:16000] if isinstance(inspect_data, dict) else ""
        text = str(inspect_data.get("text") or "")[:24000] if isinstance(inspect_data, dict) else ""
        fields = inspect_data.get("fields") if isinstance(inspect_data, dict) else []
        buttons = inspect_data.get("buttons") if isinstance(inspect_data, dict) else []
        compact_page = {
            "title": inspect_data.get("title") if isinstance(inspect_data, dict) else None,
            "url": inspect_data.get("url") if isinstance(inspect_data, dict) else None,
            "path": inspect_data.get("path") if isinstance(inspect_data, dict) else None,
            "fields": fields,
            "buttons": buttons,
            "text": text,
            "html": html,
        }
        system_prompt = (
            "你是前端 highlight JSON 生成器。只能根据用户要求和页面结构生成 JSON，不要输出解释。\n"
            "如果用户要把页面中的某类文本标红/标注/高亮，输出："
            "{\"mode\":\"text_mark\",\"marks\":[{\"context\":\"包含目标文本的一段较长原文上下文\","
            "\"target\":\"需要标红的精确原文\"}],"
            "\"color\":\"#dc2626\",\"backgroundColor\":\"rgba(220, 38, 38, 0.10)\"}。\n"
            "如果用户要高亮某个按钮或元素，输出：{\"selector\":\"CSS选择器\",\"durationMs\":3000}。\n"
            "规则：\n"
            "1. 文本标红必须使用 marks，不要使用正则表达式，不要输出 patterns。\n"
            "2. 每个 mark 的 context 必须来自页面原文，长度建议 20-80 字，且包含 target。\n"
            "3. target 必须是页面中的精确原文。比如标红“开展活动”，context 可以是"
            "“竞赛组织和技术服务开展活动。学会致力于建设开”，target 是“开展活动”。\n"
            "4. 如果用户只要求标红价格，就只返回价格文本的 marks，不要把日期、编号、普通数字放入 target。\n"
            "5. 如果用户要求标红数字相关信息，必须先理解语义：只返回页面中真实存在、且符合用户语义的数字文本。"
            "比如“价格”只标价格，“日期”只标日期，“编号”只标编号；如果用户宽泛说“数字有关”，可以标页面中所有真实数字文本，但仍必须逐个给出 context/target。\n"
            "6. 如果用户要求标红时间信息，就只返回页面中真实存在的时间/日期/期限文本，不要用泛化规则猜测。\n"
            "7. selector 必须来自页面字段、按钮或 HTML 中可以定位的元素。\n"
            "8. 不输出 markdown，不解释。"
        )
        user_prompt = (
            f"用户要求：{content}\n\n"
            f"页面结构 JSON：{json.dumps(compact_page, ensure_ascii=False)[:28000]}\n\n"
            "请只返回一个 JSON object。"
        )
        try:
            payload = self._extract_json_object(await chat_complete(system_prompt, user_prompt, tier="large"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        if payload.get("mode") == "text_mark" or isinstance(payload.get("marks"), list):
            marks: list[dict[str, str]] = []
            for item in payload.get("marks") or []:
                if not isinstance(item, dict):
                    continue
                context_text = str(item.get("context") or "").strip()
                target_text = str(item.get("target") or "").strip()
                if not target_text:
                    continue
                marks.append({"context": context_text, "target": target_text})
            return {
                "mode": "text_mark",
                "marks": marks,
                "color": str(payload.get("color") or "#dc2626"),
                "backgroundColor": str(payload.get("backgroundColor") or "rgba(220, 38, 38, 0.10)"),
                "requiresConfirm": False,
            }

        selector = str(payload.get("selector") or "").strip()
        if selector:
            return {
                "selector": selector,
                "durationMs": int(payload.get("durationMs") or 3000),
                "requiresConfirm": False,
            }

        return {
            "mode": "text_mark",
            "marks": [],
            "color": "#dc2626",
            "backgroundColor": "rgba(220, 38, 38, 0.10)",
            "requiresConfirm": False,
        }

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

    async def _analyze_files(self, context: RuntimeContext, allowed_agents: tuple[str, ...], content: str) -> str:
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
            allowed_agents,
            "file_analysis_agent",
            user_input=content,
            intent="analyze_file",
            payload={"attachments": context.attachments},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "file_analysis_agent",
            input_content=f"用户输入：{content}\n附件：{context.attachments}",
            output_content=result,
        )
        return result

    async def _maybe_retrieve(self, context: RuntimeContext, allowed_agents: tuple[str, ...], content: str) -> str:
        should_retrieve = any(
            word in content
            for word in ["检索", "查询", "查找", "重复", "是否存在", "资料", "知识", "历史", "相关"]
        )
        if not should_retrieve:
            return ""
        response = await self._run_sub_agent(
            context,
            allowed_agents,
            "retrieval_agent",
            user_input=content,
            intent="search",
            payload={"page": context.page},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "retrieval_agent",
            input_content=content,
            output_content=result,
        )
        return result

    async def _maybe_web_search(self, context: RuntimeContext, allowed_agents: tuple[str, ...], content: str) -> str:
        should_search = any(
            word in content
            for word in ["最新", "今日", "今天", "现在", "实时", "新闻", "联网", "网页", "搜索", "查一下", "天气"]
        )
        if not should_search:
            return ""
        response = await self._run_sub_agent(
            context,
            allowed_agents,
            "web_search_agent",
            user_input=content,
            intent="web_search_agent",
            payload={"keyword": content},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "web_search_agent",
            input_content=content,
            output_content=result,
        )
        return result

    async def _maybe_url_content(self, context: RuntimeContext, allowed_agents: tuple[str, ...], content: str) -> str:
        urls = [match.group(0).strip() for match in URL_PATTERN.finditer(content)]
        if not urls:
            return ""
        mode = "crawl" if any(word in content.lower() for word in ["全文爬取", "整站爬取", "深度爬取", "crawl", "爬取整个", "全站"]) else "extract"
        response = await self._run_sub_agent(
            context,
            allowed_agents,
            "url_content_agent",
            user_input=content,
            intent="url_content_agent",
            payload={"urls": urls, "mode": mode},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "url_content_agent",
            input_content=content,
            output_content=result,
        )
        return result

    async def _maybe_yaai_business(self, context: RuntimeContext, allowed_agents: tuple[str, ...], content: str) -> str:
        response = await self._run_sub_agent(
            context,
            allowed_agents,
            "yaai_business_agent",
            user_input=content,
            intent="yaai_business",
            payload={"page": context.page},
        )
        result = response.content or response.error or ""
        await self._publish_sub_agent_result(
            context,
            "yaai_business_agent",
            input_content=content,
            output_content=result,
        )
        return result

    async def _run_controlled_react(
        self,
        context: RuntimeContext,
        allowed_agents: tuple[str, ...],
        content: str,
        state: SupervisorAgentState,
    ) -> str:
        observations: list[str] = []
        completed_actions: set[str] = set()
        inspected_page: dict[str, Any] | None = None
        for _ in range(6):
            next_action = self._select_next_react_action(context, allowed_agents, content, state, observations, completed_actions)
            if next_action is None:
                break
            completed_actions.add(next_action)

            if next_action == "full_memory":
                full_memory = await self._maybe_import_full_memory(context, content)
                if full_memory:
                    observations.append(f"完整记忆 Tool 结果：\n{full_memory}")
                continue
            if next_action == "file_analysis":
                if "file_analysis_agent" not in allowed_agents:
                    continue
                file_analysis = await self._analyze_files(context, allowed_agents, content)
                if file_analysis:
                    observations.append(file_analysis)
                continue
            if next_action == "url_content":
                if "url_content_agent" not in allowed_agents:
                    continue
                url_content = await self._maybe_url_content(context, allowed_agents, content)
                if url_content:
                    observations.append(f"URL 内容 Agent 结果：{url_content}")
                continue
            if next_action == "web_search":
                if "web_search_agent" not in allowed_agents:
                    continue
                web_search = await self._maybe_web_search(context, allowed_agents, content)
                if web_search:
                    observations.append(f"网页搜索 Agent 结果：{web_search}")
                continue
            if next_action == "retrieval":
                if "retrieval_agent" not in allowed_agents:
                    continue
                retrieval = await self._maybe_retrieve(context, allowed_agents, content)
                if retrieval:
                    observations.append(f"检索 Agent 结果：{retrieval}")
                continue
            if next_action == "yaai_business":
                if "yaai_business_agent" not in allowed_agents:
                    continue
                business_result = await self._maybe_yaai_business(context, allowed_agents, content)
                if business_result:
                    observations.append(f"YAAI 业务 Agent 结果：{business_result}")
                continue
            if next_action == "frontend_action":
                if "browser_agent" not in allowed_agents:
                    continue
                frontend_action_context = await self._maybe_request_frontend_action(context, allowed_agents, state)
                if frontend_action_context:
                    observations.append(frontend_action_context)
                continue
            if next_action == "generate_and_navigate":
                if "browser_agent" not in allowed_agents:
                    continue
                try:
                    navigate_payload = await self._generate_navigate_payload(context, content)
                except Exception as exc:
                    observations.append(f"LLM 生成跳转 JSON 失败：{exc}")
                    continue
                if not navigate_payload.get("path"):
                    observations.append(f"没有匹配到可跳转页面：{navigate_payload.get('reason') or '目标不在当前平台页面目录中'}")
                    continue
                frontend_action_context, _ = await self._request_frontend_action(
                    context,
                    allowed_agents,
                    "navigate",
                    navigate_payload,
                )
                observations.append(f"LLM 跳转 JSON：{navigate_payload}\n{frontend_action_context}")
                continue
            if next_action == "inspect_for_fill":
                if "browser_agent" not in allowed_agents:
                    continue
                inspect_payload = {
                    "maxHtmlLength": 20000,
                    "maxFields": 160,
                    "targetField": content,
                    "requiresConfirm": False,
                }
                frontend_action_context, result = await self._request_frontend_action(
                    context,
                    allowed_agents,
                    "inspect_html",
                    inspect_payload,
                )
                inspected_page = (result.get("data") or {}).get("data") if isinstance(result.get("data"), dict) else result.get("data")
                if isinstance(inspected_page, dict):
                    observations.append(frontend_action_context)
                else:
                    observations.append(f"页面结构读取失败，无法生成填表 JSON：{frontend_action_context}")
                continue
            if next_action == "generate_and_fill":
                if "browser_agent" not in allowed_agents:
                    continue
                if not isinstance(inspected_page, dict):
                    observations.append("缺少页面字段观察结果，无法生成填表 JSON。")
                    continue
                try:
                    fill_payload = await self._generate_fill_payload_with_llm(content, inspected_page)
                except Exception as exc:
                    observations.append(f"LLM 生成填表 JSON 失败：{exc}")
                    continue
                if not fill_payload.get("values"):
                    observations.append("LLM 没有生成可填写字段，未触发填表。")
                    continue
                frontend_action_context, _ = await self._request_frontend_action(
                    context,
                    allowed_agents,
                    "fill",
                    fill_payload,
                )
                observations.append(f"LLM 填表 JSON：{fill_payload}\n{frontend_action_context}")
                continue
            if next_action == "inspect_for_highlight":
                if "browser_agent" not in allowed_agents:
                    continue
                inspect_payload = {
                    "maxHtmlLength": 20000,
                    "maxFields": 160,
                    "targetField": content,
                    "requiresConfirm": False,
                }
                frontend_action_context, result = await self._request_frontend_action(
                    context,
                    allowed_agents,
                    "inspect_html",
                    inspect_payload,
                )
                inspected_page = (result.get("data") or {}).get("data") if isinstance(result.get("data"), dict) else result.get("data")
                if isinstance(inspected_page, dict):
                    observations.append(frontend_action_context)
                else:
                    observations.append(f"页面结构读取失败，无法生成高亮 JSON：{frontend_action_context}")
                continue
            if next_action == "generate_and_highlight":
                if "browser_agent" not in allowed_agents:
                    continue
                if not isinstance(inspected_page, dict):
                    observations.append("缺少页面结构观察结果，无法生成高亮 JSON。")
                    continue
                try:
                    highlight_payload = await self._generate_highlight_payload_with_llm(content, inspected_page)
                except Exception as exc:
                    observations.append(f"LLM 生成高亮 JSON 失败：{exc}")
                    continue
                frontend_action_context, _ = await self._request_frontend_action(
                    context,
                    allowed_agents,
                    "highlight",
                    highlight_payload,
                )
                observations.append(f"LLM 高亮 JSON：{highlight_payload}\n{frontend_action_context}")
                continue
            if next_action == "image_generation":
                if "image_generation_agent" not in allowed_agents:
                    continue
                response = await runtime.run_agent(context, "image_generation_agent", allowed_agents=allowed_agents, user_input=f"请生成图片：{content}")
                observations.append(response.content or response.error or "")
                continue
            if next_action == "email_sender":
                if "email_sender_agent" not in allowed_agents:
                    continue
                response = await runtime.run_agent(context, "email_sender_agent", allowed_agents=allowed_agents, user_input=f"请发送邮件：{content}")
                observations.append(response.content or response.error or "")
                continue
        return "\n\n".join(observations)

    def _select_next_react_action(
        self,
        context: RuntimeContext,
        allowed_agents: tuple[str, ...],
        content: str,
        state: SupervisorAgentState,
        observations: list[str],
        completed_actions: set[str],
    ) -> str | None:
        intent = state.get("intent")
        if intent == "navigate":
            if "generate_and_navigate" not in completed_actions:
                return "generate_and_navigate"
            return None
        if intent == "fill":
            if "inspect_for_fill" not in completed_actions:
                return "inspect_for_fill"
            if "generate_and_fill" not in completed_actions:
                return "generate_and_fill"
            return None
        if intent == "highlight":
            if "inspect_for_highlight" not in completed_actions:
                return "inspect_for_highlight"
            if "generate_and_highlight" not in completed_actions:
                return "generate_and_highlight"
            return None
        if intent in {"navigate", "inspect_html"} and "frontend_action" not in completed_actions:
            return "frontend_action"

        has_file_attachment = any(
            not (
                attachment.get("type") == "image"
                or str(attachment.get("mime") or "").startswith("image/")
            )
            for attachment in context.attachments
        )
        if has_file_attachment and "file_analysis" not in completed_actions:
            return "file_analysis"

        if URL_PATTERN.search(content) and "url_content" not in completed_actions:
            return "url_content"

        if any(word in content for word in ["全部记忆", "完整记忆", "所有记忆", "历史记忆", "全部上下文", "完整上下文", "之前说过"]) and "full_memory" not in completed_actions:
            return "full_memory"

        if "yaai_business_agent" in allowed_agents and "yaai_business" not in completed_actions:
            return "yaai_business"

        if any(word in content for word in ["最新", "今日", "今天", "现在", "实时", "新闻", "联网", "网页", "搜索", "查一下", "天气"]) and "web_search" not in completed_actions:
            return "web_search"

        needs_retrieval = any(
            word in content
            for word in ["检索", "查询", "查找", "重复", "是否存在", "资料", "知识", "历史", "相关"]
        )
        if needs_retrieval and "retrieval" not in completed_actions:
            return "retrieval"

        if any(word in content for word in ["图片", "生成", "海报", "画", "设计图", "插图", "配图"]) and "image_generation" not in completed_actions:
            return "image_generation"
        if any(word in content for word in ["邮件", "发送", "发邮件", "邮箱", "发到", "发至"]) and "email_sender" not in completed_actions:
            return "email_sender"
        if observations and intent == "inspect_html":
            return None
        return None

    async def _stream_answer(self, context: RuntimeContext, content: str, state: SupervisorAgentState) -> AsyncIterator[str]:
        skill_name = self._select_skill_name(context, content)
        skill_activation = await self._activate_skill(context, skill_name)
        skill_prompt = str(skill_activation.get("prompt") or "")
        skill_allowed_agents = tuple(skill_activation.get("allowedAgents") or ())
        allowed_agent_tool_prompt = render_allowed_agents_tool_prompt(skill_allowed_agents)
        recent_memory_context = await self._load_recent_memory_context(context)
        system_prompt = (
            f"{YAAI_SYSTEM_PROMPT}\n\n"
            f"{SUPERVISOR_AGENT_PROMPT}\n\n"
            "当前主 Agent 是 Supervisor，只能按 Supervisor -> Skill -> Sub Agent -> Tool 的链路调度；"
            "Supervisor 只持有两个工具：memory.import_full_memory_tool 和 skill.activate_skill_tool；"
            "其他底层 Tool 必须由 Skill 允许的子 Agent 调用。\n\n"
            f"最近 20 条 agent_memory 强制记忆：\n{recent_memory_context}\n\n"
            f"已命中 Skill：{skill_activation.get('name') or skill_name}\n"
            f"Skill 描述：{skill_activation.get('description') or ''}\n"
            f"Skill 摘要：{skill_activation.get('summary') or ''}\n"
            f"Skill 激活状态：{'成功' if skill_prompt else '失败'}\n"
            f"Skill 允许的子 Agent：{', '.join(skill_allowed_agents) or '无'}\n"
            f"Skill 详细 Prompt：{skill_prompt}\n\n"
            f"{allowed_agent_tool_prompt}"
        )
        attachment_context = await build_attachment_prompt_context(context.attachments)
        observations = await self._run_controlled_react(context, skill_allowed_agents, content, state)
        response = await runtime.run_agent(
            context,
            "response_agent",
            allowed_agents=skill_allowed_agents,
            user_input=content,
            payload={
                "observations": observations,
                "intent": state.get("intent"),
                "role": context.role,
                "platform": context.platform,
            },
        )
        reply = response.content or observations or content or "已处理你的请求。"
        for chunk in reply:
            yield chunk

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
