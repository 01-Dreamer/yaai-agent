from __future__ import annotations

from typing import Any

from src.models.llm import vision_describe_image


def _is_image(attachment: dict[str, Any]) -> bool:
    mime = str(attachment.get("mime") or "")
    return attachment.get("type") == "image" or mime.startswith("image/")


async def build_attachment_prompt_context(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return ""

    parts: list[str] = ["附件上下文："]
    for index, attachment in enumerate(attachments, start=1):
        name = str(attachment.get("name") or "未命名附件")
        url = str(attachment.get("url") or "")
        mime = str(attachment.get("mime") or "unknown")
        size = attachment.get("size")
        header = f"{index}. {name}（mime={mime}，size={size}，url={url}）"

        if _is_image(attachment):
            try:
                description = await vision_describe_image(
                    url,
                    "请用中文识别这张图片，重点描述图片中的文字、界面、表单字段、对象、状态和可能与用户任务有关的信息。",
                )
            except Exception as exc:
                description = f"图片识别失败：{exc}"
            parts.append(f"{header}\n   类型：图片；视觉模型识别结果：{description}")
            continue

        parts.append(f"{header}\n   类型：文件；文件内容由 FileAnalysisAgent 使用 Python 解析后总结。")

    return "\n".join(parts)
