from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI

from src.config import settings

ModelTier = Literal["small", "large", "vision", "embedding"]


@dataclass(frozen=True)
class ChatModelConfig:
    base_url: str
    model: str
    api_key: str


def _model_config(tier: ModelTier) -> ChatModelConfig:
    if tier == "large":
        return ChatModelConfig(settings.large_model_url, settings.large_model_name, settings.large_model_key)
    if tier == "vision":
        return ChatModelConfig(settings.vision_model_url, settings.vision_model_name, settings.vision_model_key)
    return ChatModelConfig(settings.small_model_url, settings.small_model_name, settings.small_model_key)


async def chat_complete_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    tier: ModelTier = "small",
) -> AsyncIterator[str]:
    config = _model_config(tier)
    if not config.base_url or not config.model or not config.api_key:
        raise RuntimeError(f"{tier} model is not configured")

    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    stream = await client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def chat_complete(
    system_prompt: str,
    user_prompt: str,
    *,
    tier: ModelTier = "small",
) -> str:
    chunks: list[str] = []
    async for delta in chat_complete_stream(system_prompt, user_prompt, tier=tier):
        chunks.append(delta)
    return "".join(chunks)


async def vision_describe_image(image_url: str, prompt: str) -> str:
    config = _model_config("vision")
    if not config.base_url or not config.model or not config.api_key:
        raise RuntimeError("vision model is not configured")

    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    response = await client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""
