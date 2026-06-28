from __future__ import annotations

from src.agents.browser_agent import BrowserAgent
from src.agents.email_sender_agent import EmailSenderAgent
from src.agents.file_analysis_agent import FileAnalysisAgent
from src.agents.image_generation_agent import ImageGenerationAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.memory_compression_agent import MemoryCompressionAgent
from src.agents.response_agent import ResponseAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.web_search_agent import WebSearchAgent
from src.agents.url_content_agent import UrlContentAgent
from src.agents.yaai_business_agent import YaaiBusinessAgent
from src.core.registry import agent_registry, skill_registry, tool_registry
from src.skills.browser_operation_skill import browser_operation_skill
from src.skills.content_creation_skill import content_creation_skill
from src.skills.fresh_web_search_skill import fresh_web_search_skill
from src.skills.knowledge_qa_skill import knowledge_qa_skill
from src.skills.memory_context_skill import memory_context_skill
from src.skills.yaai_business_skill import yaai_business_skill
from src.tools.yaai_business_tool import YaaiBusinessTool
from src.tools.browser_action_tool import BrowserActionTool
from src.tools.email_sender_tool import EmailSenderTool
from src.tools.file_info_tool import FileInfoTool
from src.tools.graph_query_tool import GraphQueryTool
from src.tools.image_generation_tool import ImageGenerationTool
from src.tools.memory_tool import MemoryTool
from src.tools.rag_search_tool import RagSearchTool
from src.tools.web_search_tool import WebSearchTool
from src.tools.skill_activation_tool import SkillActivationTool
from src.tools.url_content_tool import UrlContentTool


def _register_agent(handler) -> None:
    spec = handler.spec
    if agent_registry.has(spec.name):
        return
    agent_registry.register(
        spec.name,
        handler,
        description=spec.description,
        platforms=set(spec.platforms),
        roles=set(spec.roles),
        capabilities=set(spec.capabilities),
        metadata={"tools": spec.tools, "skills": spec.skills, "modelTier": spec.model_tier},
        version=spec.version,
    )


def _register_tool(name: str, handler, *, description: str = "", **metadata) -> None:
    if tool_registry.has(name):
        return
    spec = getattr(handler, "spec", None)
    tool_registry.register(
        name,
        handler,
        description=description or getattr(spec, "description", "") or getattr(handler, "description", ""),
        platforms=set(getattr(spec, "platforms", ())),
        roles=set(getattr(spec, "roles", ())),
        capabilities=set(getattr(spec, "capabilities", ())),
        metadata=metadata or {"namespace": getattr(spec, "namespace", name.split(".", 1)[0])},
    )


def _register_skill(skill) -> None:
    if skill_registry.has(skill.name):
        return
    skill_registry.register(
        skill.name,
        skill,
        description=skill.description,
        platforms=set(skill.platforms),
        roles=set(skill.roles),
        metadata={
            "summary": skill.summary,
            "allowedAgents": skill.allowed_agents,
            "login": skill.login,
            "currentPages": skill.current_pages,
            "pageTypes": skill.page_types,
        },
        version=skill.version,
    )


def bootstrap_registries() -> None:
    _register_agent(SupervisorAgent())
    _register_agent(RetrievalAgent())
    _register_agent(FileAnalysisAgent())
    _register_agent(BrowserAgent())
    _register_agent(MemoryCompressionAgent())
    _register_agent(WebSearchAgent())
    _register_agent(UrlContentAgent())
    _register_agent(EmailSenderAgent())
    _register_agent(ImageGenerationAgent())
    _register_agent(ResponseAgent())
    _register_agent(YaaiBusinessAgent())

    for action in ["navigate", "fill", "highlight", "inspect_html"]:
        _register_tool(f"browser.{action}_tool", BrowserActionTool(action))

    _register_tool("memory.import_full_memory_tool", MemoryTool(), description=MemoryTool.description)
    _register_tool("skill.activate_skill_tool", SkillActivationTool(), description=SkillActivationTool.description)

    _register_tool("rag.search_documents_tool", RagSearchTool("semantic"))
    _register_tool("rag.keyword_search_tool", RagSearchTool("keyword"))
    _register_tool("graph.query_tool", GraphQueryTool(), description=GraphQueryTool.description)

    for kind in ["file", "image"]:
        tool = FileInfoTool(kind)
        _register_tool(tool.name, tool, description=tool.description)

    for provider in ["tavily", "aliyun", "baidu"]:
        _register_tool(f"web_search.{provider}_tool", WebSearchTool(provider))

    for action in ["extract", "crawl"]:
        _register_tool(f"url_content.{action}_tool", UrlContentTool(action))

    for action in [
        "search_news",
        "get_member_profile",
        "list_committees",
        "get_committee_detail",
        "list_member_audits",
        "get_operation_logs",
        "create_payment_url",
    ]:
        tool = YaaiBusinessTool(action)
        _register_tool(tool.name, tool, description=tool.description)

    _register_tool("email.send_tool", EmailSenderTool(), description=EmailSenderTool.description)
    _register_tool("image.generate_tool", ImageGenerationTool(), description=ImageGenerationTool.description)

    for skill in [
        browser_operation_skill,
        knowledge_qa_skill,
        fresh_web_search_skill,
        yaai_business_skill,
        content_creation_skill,
        memory_context_skill,
    ]:
        _register_skill(skill)
