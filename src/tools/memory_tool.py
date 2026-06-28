from __future__ import annotations

from typing import Any

from src.core.context import RuntimeContext
from src.repositories.agent_memory import agent_memory_repository
from src.tools.base import ToolResult, ToolSpec


def _format_memory_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        created_at = message.get("createdAt") or ""
        role = message.get("role") or "unknown"
        content = str(message.get("content") or "").strip()
        attachments = message.get("attachments") or []
        attachment_text = f" 附件={attachments}" if attachments else ""
        lines.append(f"[{message.get('id')}] {created_at} {role}: {content}{attachment_text}")
    return "\n".join(lines)


class MemoryTool:
    name = "memory.import_full_memory_tool"
    description = (
        "导入完整会话记忆。参数：会话编号可选，默认使用当前会话。"
        "返回：会话编号、压缩摘要、最近压缩时间、消息列表、完整记忆文本。"
        "用途：当 Supervisor 判断最近 20 条消息不足，需要完整历史、全部上下文、之前偏好时调用。"
        "示例：{\"会话编号\":123}。限制：只读，不做向量检索，不直接展示给用户。"
    )

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="memory",

            capabilities=("import_full_memory",),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        session_id = int(kwargs.get("session_id") or kwargs.get("sessionId") or context.session_id or 0)
        if not session_id:
            return ToolResult(False, error="missing session_id")
        data = await agent_memory_repository.load_full_memory_context(
            session_id=session_id,
            user_id=context.user_id,
        )
        messages = data.get("messages") or []
        summary = str(data.get("summary") or "").strip()
        memory_text = (
            f"压缩记忆：\n{summary or '无'}\n\n"
            f"未压缩原始记忆：\n{_format_memory_messages(messages) or '无'}"
        )
        return ToolResult(
            True,
            data={
                "sessionId": session_id,
                "summary": summary,
                "memoryUpdatedAt": data.get("memoryUpdatedAt"),
                "messages": messages,
                "memoryText": memory_text,
            },
            summary=memory_text,
        )
