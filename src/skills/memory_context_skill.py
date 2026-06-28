from __future__ import annotations

from src.skills.base import SkillSpec


def _load_prompt() -> str:
    return (
        "你是 Memory Context Skill，负责处理用户明确要求基于历史上下文、完整记忆或之前偏好继续的请求。"
        "完整记忆由 Supervisor 自己通过 memory.import_full_memory_tool 导入；"
        "本 Skill 只允许 Response Agent 汇总和面向用户表达，不允许子 Agent 直接读取数据库记忆。"
        "回答时不要暴露原始记忆表结构或内部消息 ID。"
    )


memory_context_skill = SkillSpec(
    name="memory_context_skill",
    description="记忆上下文：回忆历史、基于之前内容继续、总结会话偏好",
    summary="用于用户明确要求使用完整历史记忆或之前上下文的场景。",
    allowed_agents=("response_agent",),
    login=("login",),
    platforms=("*",),
    roles=("*",),
    current_pages=("*",),
    page_types=("*",),
    prompt_loader=_load_prompt,
)
