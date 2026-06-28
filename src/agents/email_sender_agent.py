from __future__ import annotations

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.runtime import runtime


class EmailSenderAgent:
    spec = AgentSpec(
        name="email_sender_agent",
        description="使用 QQ 邮箱发送邮件。通常作为最终输出步骤，在与其他 Agent 充分讨论确定邮件内容后调用。",
        tools=("email.send_tool",),
        model_tier="small",
        capabilities=("send_email",),
    )

    async def run(self, request: AgentRequest) -> AgentResponse:
        payload = request.payload or {}
        to_addr = str(payload.get("to") or payload.get("收件人") or "").strip()
        subject = str(payload.get("subject") or payload.get("标题") or "").strip()
        body = str(payload.get("body") or payload.get("content") or payload.get("正文") or request.user_input or "").strip()

        if not to_addr:
            return AgentResponse(False, error="缺少收件人地址")
        if not subject:
            return AgentResponse(False, error="缺少邮件标题")
        if not body:
            return AgentResponse(False, error="缺少邮件正文")

        result = await runtime.run_tool(
            request.context,
            "email.send_tool",
            caller_agent=self.spec.name,
            to=to_addr,
            subject=subject,
            body=body,
        )
        if not result.success:
            return AgentResponse(False, error=result.error)

        return AgentResponse(
            True,
            content=f"邮件已发送至 {to_addr}，标题：{subject}",
            data={"to": to_addr, "subject": subject},
        )
