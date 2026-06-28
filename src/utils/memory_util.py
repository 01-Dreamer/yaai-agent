from __future__ import annotations

from typing import Any

from src.core.context import RuntimeContext
from src.models.llm import chat_complete
from src.prompts.system import MEMORY_COMPRESSION_AGENT_PROMPT
from src.repositories.agent_memory import agent_memory_repository
from src.utils.base import UtilResult


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


class MemoryCompressionUtil:
    name = "memory.compress_session_util"
    description = (
        "系统 util：压缩未压缩会话记忆到 agent_session.memory_content。参数：session_id/sessionId、user_id 可选。"
        "返回：sessionId、compressed、summary。用途：MQ 落库后累计 memory 数达到阈值时由后端代码调用；不注册为 Tool。"
    )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> UtilResult:
        session_id = int(kwargs.get("session_id") or kwargs.get("sessionId") or context.session_id or 0)
        user_id = kwargs.get("user_id", context.user_id)
        if not session_id:
            return UtilResult(False, error="missing session_id")
        data = await agent_memory_repository.load_full_memory_context(session_id=session_id, user_id=user_id)
        messages = data.get("messages") or []
        old_summary = str(data.get("summary") or "").strip()
        if not messages:
            return UtilResult(True, data={"sessionId": session_id, "compressed": 0, "summary": old_summary}, summary=old_summary)

        raw_memory = _format_memory_messages(messages)
        user_prompt = (
            "请把以下未压缩记忆合并到已有压缩记忆中，保留用户目标、偏好、重要事实、未完成事项和已执行动作；"
            "删除寒暄、重复内容和无价值中间过程。输出一段中文压缩记忆，不要超过 2000 字。\n\n"
            f"已有压缩记忆：\n{old_summary or '无'}\n\n"
            f"未压缩记忆：\n{raw_memory}"
        )
        try:
            summary = await chat_complete(MEMORY_COMPRESSION_AGENT_PROMPT, user_prompt, tier="large")
        except Exception:
            merged = f"{old_summary}\n\n{raw_memory}" if old_summary else raw_memory
            summary = merged[-4000:]

        summary = summary.strip()
        updated = await agent_memory_repository.update_session_summary(
            session_id=session_id,
            user_id=user_id,
            memory_content=summary,
        )
        return UtilResult(
            bool(updated),
            data={"sessionId": session_id, "compressed": len(messages), "summary": summary},
            error=None if updated else "session not found",
            summary=summary,
        )
