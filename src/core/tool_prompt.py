from __future__ import annotations

from collections.abc import Iterable

from src.agents.base import AgentSpec
from src.core.registry import agent_registry, tool_registry


def render_tool_prompt(tool_names: Iterable[str], *, title: str = "当前 Agent 可用 Tool") -> str:
    lines = [f"{title}："]
    count = 0
    for tool_name in tool_names:
        try:
            item = tool_registry.get(tool_name)
        except KeyError:
            continue
        count += 1
        lines.append(f"- {item.name}\n  {item.description.strip() or '无说明'}")
    if count == 0:
        lines.append("- 无")
    return "\n".join(lines)


def render_agent_tool_prompt(spec: AgentSpec, *, title: str | None = None) -> str:
    return render_tool_prompt(spec.tools, title=title or f"{spec.name} 绑定的 Tool")


def render_allowed_agents_tool_prompt(agent_names: Iterable[str]) -> str:
    lines = ["当前 Skill 允许调度的子 Agent 及其绑定 Tool："]
    count = 0
    for agent_name in agent_names:
        try:
            item = agent_registry.get(agent_name)
        except KeyError:
            continue
        spec = getattr(item.handler, "spec", None)
        if spec is None:
            continue
        count += 1
        lines.append(f"\n[{spec.name}] {spec.description}")
        lines.append(render_agent_tool_prompt(spec, title="绑定 Tool"))
    if count == 0:
        lines.append("- 无")
    return "\n".join(lines)
