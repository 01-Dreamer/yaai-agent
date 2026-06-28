from __future__ import annotations

from src.skills.base import SkillSpec


def _load_prompt() -> str:
    return (
        "你是 Content Creation Skill，负责内容生成、图片生成、邮件发送以及基于资料的写作。"
        "适用任务：新闻稿、活动通知、总结文案、邮件正文、宣传图/海报图片生成。"
        "需要资料支撑时可调度 Retrieval/WebSearch/FileAnalysis；最终发送邮件或生成图片前，"
        "必须确保收件人/标题/正文或图片描述等关键参数齐全。"
        "输出要面向用户，不暴露内部 Agent 或 Tool 名称。"
    )


content_creation_skill = SkillSpec(
    name="content_creation_skill",
    description="内容创作：文案、邮件、图片生成和基于资料的写作",
    summary="用于生成新闻稿、通知、总结、邮件和图片。",
    allowed_agents=(
        "retrieval_agent",
        "file_analysis_agent",
        "web_search_agent",
        "email_sender_agent",
        "image_generation_agent",
        "response_agent",
    ),
    login=("login",),
    platforms=("*",),
    roles=("*",),
    current_pages=("*",),
    page_types=("*",),
    prompt_loader=_load_prompt,
)
