from __future__ import annotations

from src.agents.file_analysis_agent import FileAnalysisAgent
from src.agents.frontend_control_agent import FrontendControlAgent
from src.agents.main_agent import MainAgent
from src.agents.memory_compression_agent import MemoryCompressionAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.core.registry import agent_registry, skill_registry, tool_registry
from src.skills.catalog import SKILLS
from src.tools.backend import BackendTool
from src.tools.frontend import FrontendActionTool
from src.tools.memory import MemoryTool
from src.tools.retrieval import RetrievalTool


def _register_agent(handler, *, expose_to_llm: bool = True) -> None:
    spec = handler.spec
    if agent_registry.has(spec.name):
        return
    agent_registry.register(
        spec.name,
        handler,
        description=spec.description,
        platforms=set(spec.platforms),
        roles=set(spec.roles),
        tags=set(spec.tags),
        capabilities=set(spec.capabilities),
        metadata={"tools": spec.tools, "skills": spec.skills, "modelTier": spec.model_tier},
        version=spec.version,
        expose_to_llm=expose_to_llm,
    )


def _register_tool(name: str, handler, *, description: str = "", expose_to_llm: bool | None = None, **metadata) -> None:
    if tool_registry.has(name):
        return
    spec = getattr(handler, "spec", None)
    tool_registry.register(
        name,
        handler,
        description=description or getattr(spec, "description", "") or getattr(handler, "description", ""),
        platforms=set(getattr(spec, "platforms", ())),
        roles=set(getattr(spec, "roles", ())),
        tags=set(getattr(spec, "tags", ())),
        capabilities=set(getattr(spec, "capabilities", ())),
        metadata=metadata or {"namespace": getattr(spec, "namespace", name.split(".", 1)[0])},
        risk_level=getattr(spec, "risk_level", "low"),
        expose_to_llm=getattr(spec, "expose_to_llm", True) if expose_to_llm is None else expose_to_llm,
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
        tags=set(skill.tags),
        metadata={
            "summary": skill.summary,
            "allowedAgents": skill.allowed_agents,
            "currentPages": skill.current_pages,
            "pageTypes": skill.page_types,
        },
        version=skill.version,
    )


def bootstrap_registries() -> None:
    _register_agent(MainAgent(), expose_to_llm=False)
    _register_agent(RetrievalAgent())
    _register_agent(FileAnalysisAgent())
    _register_agent(FrontendControlAgent())
    _register_agent(MemoryCompressionAgent())

    for action in ["navigate", "fill", "highlight"]:
        _register_tool(f"frontend.{action}", FrontendActionTool(action), description=f"Frontend action: {action}")

    _register_tool(
        "backend.user_context",
        BackendTool("backend.user_context", "/agent/user-context"),
        description="Resolve user context through yaai-backend AgentController",
    )

    for name in ["memory.load_session", "memory.load_recent", "memory.update_session_summary"]:
        _register_tool(name, MemoryTool(name), description=name)

    for name in [
        "rag.search_documents",
        "graph.query",
        "file.download",
        "file.extract_text",
        "file.extract_images",
        "file.summarize",
        "vision.describe_image",
    ]:
        _register_tool(name, RetrievalTool(name), description=name)

    for name in ["mq.sensitive_check", "system.revoke_message"]:
        _register_tool(name, RetrievalTool(name), description=name, expose_to_llm=False)

    for skill in SKILLS:
        _register_skill(skill)
