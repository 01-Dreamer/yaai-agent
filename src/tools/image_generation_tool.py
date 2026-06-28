from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


class ImageGenerationTool:
    name = "image.generate_tool"
    description = (
        "根据文字描述生成图片，返回图片链接。参数：图片描述必填，图片尺寸可选（默认 1024x1024）。"
        "返回：图片链接列表。示例：{\"图片描述\":\"一只在草地上奔跑的金毛犬\"}。"
        "限制：生成的图片链接有效期有限，建议及时查看或转存到 OSS。"
    )
    spec = ToolSpec(
        name=name,
        description=description,
        namespace="image",
        capabilities=("generate_image",),
    )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        prompt = str(kwargs.get("图片描述") or kwargs.get("prompt") or kwargs.get("描述") or "").strip()
        size = str(kwargs.get("图片尺寸") or kwargs.get("size") or "1024x1024").strip()

        if not prompt:
            return ToolResult(False, error="缺少图片描述")

        if not settings.image_model_url or not settings.image_model_name or not settings.image_model_key:
            return ToolResult(False, error="图片生成模型未配置，请检查 IMAGE_MODEL_URL/NAME/KEY")

        try:
            api_key = settings.image_model_key
            if api_key.startswith("sk-ark-"):
                api_key = api_key[len("sk-"):]
            base_url = settings.image_model_url.rstrip("/")
            if base_url.endswith("/images/generations"):
                base_url = base_url[: -len("/images/generations")]
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = await client.images.generate(
                model=settings.image_model_name,
                prompt=prompt,
                n=1,
                size=size,
            )
            urls = [item.url for item in response.data if item.url]
            if not urls:
                return ToolResult(False, error="图片生成成功但未返回链接")

            return ToolResult(
                True,
                data={"urls": urls, "prompt": prompt, "size": size},
                summary=f"已生成 {len(urls)} 张图片，链接：{urls[0]}",
            )
        except Exception as exc:
            return ToolResult(False, error=f"图片生成失败：{exc}")
