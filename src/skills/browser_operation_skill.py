from __future__ import annotations

from src.skills.base import SkillSpec


def _load_prompt() -> str:
    return (
        "你是 Browser Operation Skill，负责当前浏览器宿主中的受控前端操作。"
        "适用任务：页面跳转、读取当前页面 HTML/字段/按钮、填表、元素高亮、页面文本标红/标注。"
        "必须通过 Browser Agent 执行白名单 action：navigate、fill、highlight、inspect_html。"
        "填表前必须 inspect_html，再由 LLM 生成 values/diff JSON；"
        "文本标红必须 inspect_html，再由 LLM 生成 mode=text_mark、marks=[{context,target}]。"
        "禁止生成任意 JS、读取 cookie/localStorage、静默提交或保存表单。"
    )


browser_operation_skill = SkillSpec(
    name="browser_operation_skill",
    description="前端页面操作：跳转、读取页面结构、填表、高亮和文本标红",
    summary="用于 yaai-frontend / yaai-lowcode 的受控浏览器操作。",
    allowed_agents=("browser_agent", "response_agent"),
    login=("login",),
    platforms=("*",),
    roles=("*",),
    current_pages=("*",),
    page_types=("*",),
    prompt_loader=_load_prompt,
)
