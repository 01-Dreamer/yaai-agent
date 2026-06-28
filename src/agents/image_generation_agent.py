from __future__ import annotations

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.runtime import runtime


class ImageGenerationAgent:
    spec = AgentSpec(
        name="image_generation_agent",
        description="根据文字描述生成图片，返回图片链接。通常作为最终输出步骤，在与其他 Agent 充分讨论确定需求后调用。",
        tools=("image.generate_tool",),
        model_tier="large",
        capabilities=("generate_image",),
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        prompt = str((request.payload or {}).get("图片描述") or request.user_input or "").strip()
        if not prompt:
            return AgentResponse(False, error="缺少图片描述")

        result = await runtime.run_tool(
            request.context,
            "image.generate_tool",
            caller_agent=self.spec.name,
            prompt=prompt,
            size=(request.payload or {}).get("size", "1024x1024"),
        )
        if not result.success:
            return AgentResponse(False, error=result.error)

        urls = result.data.get("urls") or []
        return AgentResponse(
            True,
            content=f"已生成 {len(urls)} 张图片，链接：{urls[0] if urls else '无'}",
            data={"urls": urls, "prompt": prompt},
        )
