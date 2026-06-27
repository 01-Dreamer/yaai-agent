from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RegistryItem:
    name: str
    handler: Any
    description: str = ""
    platforms: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    version: str = "0.1.0"
    expose_to_llm: bool = True


class BaseRegistry:
    def __init__(self) -> None:
        self._items: dict[str, RegistryItem] = {}

    def register(
        self,
        name: str,
        handler: Any,
        *,
        description: str = "",
        platforms: set[str] | None = None,
        roles: set[str] | None = None,
        tags: set[str] | None = None,
        capabilities: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        risk_level: str = "low",
        version: str = "0.1.0",
        expose_to_llm: bool = True,
    ) -> None:
        if name in self._items:
            raise ValueError(f"registry item already exists: {name}")
        self._items[name] = RegistryItem(
            name=name,
            handler=handler,
            description=description,
            platforms=platforms or set(),
            roles=roles or set(),
            tags=tags or set(),
            capabilities=capabilities or set(),
            metadata=metadata or {},
            risk_level=risk_level,
            version=version,
            expose_to_llm=expose_to_llm,
        )

    def get(self, name: str) -> RegistryItem:
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def list_available(
        self,
        *,
        platform: str | None = None,
        role: str | None = None,
        llm_only: bool = False,
        capability: str | None = None,
        tag: str | None = None,
    ) -> list[RegistryItem]:
        result: list[RegistryItem] = []
        for item in self._items.values():
            if llm_only and not item.expose_to_llm:
                continue
            if platform and item.platforms and platform not in item.platforms:
                continue
            if role and item.roles and role not in item.roles:
                continue
            if capability and capability not in item.capabilities:
                continue
            if tag and tag not in item.tags:
                continue
            result.append(item)
        return result

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "platforms": sorted(item.platforms),
                "roles": sorted(item.roles),
                "tags": sorted(item.tags),
                "capabilities": sorted(item.capabilities),
                "riskLevel": item.risk_level,
                "version": item.version,
                "exposeToLlm": item.expose_to_llm,
                "metadata": item.metadata,
            }
            for item in self._items.values()
        ]


AgentRegistry = BaseRegistry
ToolRegistry = BaseRegistry


class SkillRegistry(BaseRegistry):
    def list_for_context(
        self,
        *,
        platform: str | None,
        role: str | None,
        current_page: str | None,
        page_type: str | None,
    ) -> list[RegistryItem]:
        result: list[RegistryItem] = []
        for item in self._items.values():
            skill = item.handler
            if hasattr(skill, "matches") and not skill.matches(
                platform=platform,
                role=role,
                current_page=current_page,
                page_type=page_type,
            ):
                continue
            result.append(item)
        return result

    def briefs_for_context(
        self,
        *,
        platform: str | None,
        role: str | None,
        current_page: str | None,
        page_type: str | None,
    ) -> list[dict[str, Any]]:
        briefs: list[dict[str, Any]] = []
        for item in self.list_for_context(
            platform=platform,
            role=role,
            current_page=current_page,
            page_type=page_type,
        ):
            skill = item.handler
            if hasattr(skill, "brief"):
                briefs.append(skill.brief())
            else:
                briefs.append(
                    {
                        "name": item.name,
                        "description": item.description,
                        "summary": item.description,
                        "allowedAgents": item.metadata.get("allowedAgents", ()),
                    }
                )
        return briefs


agent_registry = AgentRegistry()
tool_registry = ToolRegistry()
skill_registry = SkillRegistry()
