from __future__ import annotations

from src.skills.base import SkillSpec


def _load_prompt() -> str:
    return (
        "你是 Fresh Web Search Skill，负责最新、实时、公开网页信息搜索和 URL 内容读取。"
        "适用任务：最新新闻、今天/当前/实时信息、联网搜索、公开网页资料核验、指定 URL 正文抽取。"
        "优先调度 Web Search Agent；用户提供 URL 时可调度 URL Content Agent。"
        "必须保留来源、链接、发布时间或网站信息；不编造来源。"
    )


fresh_web_search_skill = SkillSpec(
    name="fresh_web_search_skill",
    description="公网搜索：最新信息、实时资讯、联网搜索和 URL 读取",
    summary="用于时效性强的公开网页搜索和来源汇总。",
    allowed_agents=("web_search_agent", "url_content_agent", "response_agent"),
    login=("login",),
    platforms=("*",),
    roles=("*",),
    current_pages=("*",),
    page_types=("*",),
    prompt_loader=_load_prompt,
)
