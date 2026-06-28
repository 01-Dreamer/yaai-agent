from __future__ import annotations

from src.skills.base import SkillSpec


def _load_prompt() -> str:
    return (
        "你是 Knowledge QA Skill，负责内部知识问答、附件理解、RAG 文档检索、Neo4j 关系检索和 URL 内容读取。"
        "适用任务：政策/文档解释、专家/单位/项目/活动关系查询、附件总结、URL 页面总结、内部知识库问答。"
        "优先使用内部知识和用户附件，不要把时效性公网新闻问题误判为内部知识问答。"
        "输出必须区分事实、推断和缺失信息；不能编造未检索到的依据。"
    )


knowledge_qa_skill = SkillSpec(
    name="knowledge_qa_skill",
    description="知识问答：RAG、Neo4j、附件分析和 URL 内容理解",
    summary="用于内部知识、知识图谱、文件附件和网页内容的问答与总结。",
    allowed_agents=("retrieval_agent", "file_analysis_agent", "url_content_agent", "response_agent"),
    login=("login",),
    platforms=("*",),
    roles=("*",),
    current_pages=("*",),
    page_types=("*",),
    prompt_loader=_load_prompt,
)
