from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


WILDCARDS = {"", "*", "all", "any"}


def _normalize_scope(values: tuple[str, ...] | list[str] | set[str] | None) -> tuple[str, ...]:
    if not values:
        return ("*",)
    result = tuple(str(value).strip() for value in values if str(value).strip())
    return result or ("*",)


def _scope_matches(scope: tuple[str, ...], value: str | None) -> bool:
    normalized = {item.lower() for item in scope}
    if normalized & WILDCARDS:
        return True
    if value is None:
        return False
    return str(value).strip().lower() in normalized


def _login_value(authenticated: bool | None) -> str | None:
    if authenticated is None:
        return None
    return "login" if authenticated else "anonymous"


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    summary: str
    allowed_agents: tuple[str, ...]
    login: tuple[str, ...] = ("*",)
    platforms: tuple[str, ...] = ("*",)
    roles: tuple[str, ...] = ("*",)
    current_pages: tuple[str, ...] = ("*",)
    page_types: tuple[str, ...] = ("*",)
    version: str = "0.1.0"
    prompt_template: str = ""
    prompt_loader: Callable[[], str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "login", _normalize_scope(self.login))
        object.__setattr__(self, "platforms", _normalize_scope(self.platforms))
        object.__setattr__(self, "roles", _normalize_scope(self.roles))
        object.__setattr__(self, "current_pages", _normalize_scope(self.current_pages))
        object.__setattr__(self, "page_types", _normalize_scope(self.page_types))

    @property
    def recommended_agents(self) -> tuple[str, ...]:
        return self.allowed_agents

    @property
    def recommended_tools(self) -> tuple[str, ...]:
        return ()

    def matches(
        self,
        *,
        authenticated: bool | None,
        platform: str | None,
        role: str | None,
        current_page: str | None,
        page_type: str | None,
    ) -> bool:
        return (
            _scope_matches(self.login, _login_value(authenticated))
            and _scope_matches(self.platforms, platform)
            and _scope_matches(self.roles, role)
            and _scope_matches(self.current_pages, current_page)
            and _scope_matches(self.page_types, page_type)
        )

    def brief(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
        }

    def load_prompt(self) -> str:
        if self.prompt_loader is not None:
            return self.prompt_loader()
        return self.prompt_template
