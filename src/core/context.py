from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeContext:
    connection_id: str
    platform: str
    session_id: int | None = None
    user_id: int | None = None
    role: str | None = None
    roles: list[str] = field(default_factory=list)
    authenticated: bool = False
    request_id: str | None = None
    assistant_message_id: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    page: dict[str, Any] = field(default_factory=dict)
    current_page: str | None = None
    page_type: str | None = None
    page_description: str | None = None
