from __future__ import annotations

from typing import Any

from src.core.context import RuntimeContext
from src.core.registry import skill_registry
from src.tools.base import ToolResult, ToolSpec


class SkillActivationTool:
    name = "skill.activate_skill_tool"
    description = (
        "激活一个已通过 login/platform/role/current_page/page_type 作用域过滤的 Skill。"
        "参数：skill_name/skillName 必填，必须是 Skill brief 中出现的 name。"
        "返回：name、description、summary、allowedAgents、prompt、scopes、version。"
        "用途：Supervisor 默认只能看到 Skill 的 name/description；命中具体 Skill 后必须调用本工具加载详细 Prompt 和 allowedAgents，"
        "之后只能调度 allowedAgents 中的子 Agent。"
        "示例：{\"skill_name\":\"browser_assistant_skill\"}。"
    )
    spec = ToolSpec(
        name=name,
        description=description,
        namespace="skill",

        capabilities=("activate_skill",),
    )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        skill_name = str(kwargs.get("skill_name") or kwargs.get("skillName") or "").strip()
        if not skill_name:
            return ToolResult(False, error="missing skill_name")
        try:
            skill = skill_registry.get(skill_name).handler
        except KeyError:
            return ToolResult(False, error=f"skill not found: {skill_name}")
        if not skill.matches(
            authenticated=context.authenticated,
            platform=context.platform,
            role=context.role,
            current_page=context.current_page,
            page_type=context.page_type,
        ):
            return ToolResult(False, error=f"skill is not available in current context: {skill_name}")
        prompt = skill.load_prompt()
        data = {
            "name": skill.name,
            "description": skill.description,
            "summary": skill.summary,
            "allowedAgents": skill.allowed_agents,
            "prompt": prompt,
            "scopes": {
                "login": skill.login,
                "platforms": skill.platforms,
                "roles": skill.roles,
                "currentPages": skill.current_pages,
                "pageTypes": skill.page_types,
            },
            "version": skill.version,
        }
        return ToolResult(
            True,
            data=data,
            summary=prompt,
        )
