from __future__ import annotations

from typing import Any

from src.core.context import RuntimeContext
from src.models.llm import vision_describe_image
from src.tools.base import ToolResult, ToolSpec


class FileInfoTool:
    def __init__(self, kind: str) -> None:
        if kind not in {"file", "image"}:
            raise ValueError(f"unsupported file info tool kind: {kind}")
        self.kind = kind
        if kind == "file":
            self.name = "file.extract_file_info_tool"
            self.description = (
                "提取并总结文件内容。参数：文件链接或附件必填；附件可包含文件名、类型、大小；用户问题可选，用于聚焦总结方向。"
                "返回：附件信息、内容摘要。"
                "行为：下载到临时文件，解析 PDF、Word 文档、Excel 表格、CSV 表格、TXT 文本、代码等格式；PDF 和 Word 会尝试提取内嵌图片并调用视觉模型识别；处理完成后自动删除临时文件。"
                "示例：{\"附件\":{\"链接\":\"https://oss/file.pdf\",\"文件名\":\"简历.pdf\",\"类型\":\"application/pdf\"},\"用户问题\":\"总结教育经历\"}。"
            )
            namespace = "file"
            capabilities = ("extract_file_info",)
        else:
            self.name = "vision.extract_image_info_tool"
            self.description = (
                "识别图片内容。参数：图片链接或附件必填；提示词可选。返回：图片链接、内容描述。"
                "用途：用户上传图片或截图附件时调用视觉模型，提取文字、界面布局、表单字段、物体和状态。"
                "示例：{\"图片链接\":\"https://oss/screenshot.png\",\"提示词\":\"请识别当前页面表单字段\"}。"
            )
            namespace = "vision"
            capabilities = ("extract_image_info",)

        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace=namespace,

            capabilities=capabilities,
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        if self.kind == "image":
            return await self._run_image(**kwargs)
        return await self._run_file(context, **kwargs)

    async def _run_image(self, **kwargs: Any) -> ToolResult:
        attachment = kwargs.get("attachment") if isinstance(kwargs.get("attachment"), dict) else {}
        url = str(kwargs.get("url") or attachment.get("url") or "").strip()
        if not url:
            return ToolResult(False, error="missing image url")
        prompt = str(
            kwargs.get("prompt")
            or "请用中文识别这张图片，重点描述图片中的文字、界面、表单字段、对象、状态和可能与用户任务有关的信息。"
        )
        description = await vision_describe_image(url, prompt)
        return ToolResult(
            True,
            data={"url": url, "description": description},
            summary=description,
        )

    async def _run_file(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        attachment = kwargs.get("attachment") if isinstance(kwargs.get("attachment"), dict) else {}
        url = str(kwargs.get("url") or attachment.get("url") or "").strip()
        if not url:
            return ToolResult(False, error="missing file url")
        if not attachment:
            attachment = {
                "url": url,
                "name": kwargs.get("name") or url.rsplit("/", 1)[-1] or "未命名文件",
                "mime": kwargs.get("mime") or "unknown",
                "size": kwargs.get("size"),
                "type": kwargs.get("type") or "file",
            }

        from src.agents.file_analysis_agent import FileAnalysisAgent

        user_input = str(kwargs.get("user_input") or kwargs.get("userInput") or "")
        summary = await FileAnalysisAgent()._analyze_one(context, attachment, user_input)
        return ToolResult(
            True,
            data={"attachment": attachment, "summary": summary},
            summary=summary,
        )
