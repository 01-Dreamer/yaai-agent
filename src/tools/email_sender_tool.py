from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


class EmailSenderTool:
    name = "email.send_tool"
    description = (
        "使用 QQ 邮箱发送邮件。参数：收件人地址必填，邮件标题必填，邮件正文必填。"
        "返回：发送结果。示例：{\"收件人\":\"user@example.com\",\"标题\":\"学会通知\",\"正文\":\"您好，这是测试邮件。\"}。"
        "限制：仅支持 QQ 邮箱 SMTP 发送，发送人固定为系统配置的发件箱。"
    )
    spec = ToolSpec(
        name=name,
        description=description,
        namespace="email",
        capabilities=("send_email",),
    )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        to_addr = str(kwargs.get("收件人") or kwargs.get("to") or "").strip()
        subject = str(kwargs.get("标题") or kwargs.get("subject") or "").strip()
        body = str(kwargs.get("正文") or kwargs.get("body") or kwargs.get("content") or "").strip()

        if not to_addr:
            return ToolResult(False, error="缺少收件人地址")
        if not subject:
            return ToolResult(False, error="缺少邮件标题")
        if not body:
            return ToolResult(False, error="缺少邮件正文")

        if not settings.qq_email_host or not settings.qq_email_username or not settings.qq_email_auth:
            return ToolResult(False, error="QQ 邮箱 SMTP 未配置，请检查 QQ_EMAIL_HOST/USERNAME/AUTH")

        try:
            msg = MIMEMultipart()
            msg["From"] = settings.qq_email_username
            msg["To"] = to_addr
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            server = smtplib.SMTP(settings.qq_email_host, settings.qq_email_port, timeout=15)
            server.starttls()
            server.login(settings.qq_email_username, settings.qq_email_auth)
            server.send_message(msg)
            server.quit()

            return ToolResult(
                True,
                data={"to": to_addr, "subject": subject},
                summary=f"邮件已发送至 {to_addr}",
            )
        except smtplib.SMTPAuthenticationError:
            return ToolResult(False, error="QQ 邮箱认证失败，请检查 QQ_EMAIL_USERNAME/AUTH")
        except Exception as exc:
            return ToolResult(False, error=f"邮件发送失败：{exc}")
