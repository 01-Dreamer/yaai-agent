from __future__ import annotations

from typing import Any

import httpx

from src.config import settings
from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec


class YaaiBusinessTool:
    _ACTIONS: dict[str, dict[str, str]] = {
        "search_news": {
            "path": "/agent/business/news/search",
            "description": (
                "按关键词、分类搜索已发布新闻。参数：keyword/query 可选，categoryId 可选，isTop 可选，"
                "page 默认 1，size 默认 10、最大 50，sort 可选 publish_time_desc/publish_time_asc。"
                "返回：items、total、page、size、pages，items 为新闻列表。"
                "示例：{\"keyword\":\"学术会议\",\"categoryId\":1,\"page\":1,\"size\":10}。"
            ),
        },
        "get_member_profile": {
            "path": "/agent/business/member/profile",
            "description": (
                "查询会员全貌，包括会员基础信息、个人/单位资料、教育经历、工作经历、委员会信息和最近订单。"
                "参数：memberId 可选，userId 可选；不传时默认查询当前登录用户对应会员。"
                "权限：普通会员只能查自己，管理员可查任意会员。"
                "返回：memberId、memberType、profile、orders。"
                "示例：{\"memberId\":12}。"
            ),
        },
        "list_committees": {
            "path": "/agent/business/committees",
            "description": (
                "查询委员会列表并统计成员数。参数：无必填参数。"
                "权限：普通用户只看启用委员会，管理员可看全部。"
                "返回：items、total，items 包含 id、name、category、description、status、memberCount 等。"
                "示例：{}。"
            ),
        },
        "get_committee_detail": {
            "path": "/agent/business/committee/detail",
            "description": (
                "查询单个委员会详情和成员名单。参数：committeeId/id 必填。"
                "权限：普通用户只能查启用委员会及正常成员，管理员可查全部状态。"
                "返回：committee、members、memberCount。"
                "示例：{\"committeeId\":3}。"
            ),
        },
        "list_member_audits": {
            "path": "/agent/business/member/audits",
            "description": (
                "查询待审核会员列表，管理员专用。参数：memberType/type 可选，single/individual 表示个人会员，"
                "company/organization 表示单位会员；page 默认 1，size 默认 10、最大 50。"
                "返回：single 和/或 company 分页结果。"
                "示例：{\"memberType\":\"single\",\"page\":1,\"size\":10}。"
            ),
        },
        "get_operation_logs": {
            "path": "/agent/business/operation/logs",
            "description": (
                "查询系统操作日志，管理员专用。参数：tableName、operationType、operator、keyword 可选，"
                "page 默认 1，size 默认 10、最大 100。"
                "返回：items、total、page、size、pages。"
                "示例：{\"tableName\":\"member\",\"operationType\":\"UPDATE\",\"page\":1,\"size\":20}。"
            ),
        },
        "create_payment_url": {
            "path": "/agent/payment/create",
            "description": (
                "为当前登录用户生成支付宝沙箱测试缴费链接。无需参数，根据用户身份自动确定金额。"
                "返回：payUrl、payNo、amount。链接可直接在浏览器打开进行沙箱支付测试。"
                "限制：不写入数据库，仅生成支付链接，纯测试用途。"
                "示例：{}。"
            ),
        },
    }

    def __init__(self, action: str) -> None:
        if action not in self._ACTIONS:
            raise ValueError(f"unsupported backend tool action: {action}")
        self.action = action
        self.name = f"backend.{action}_tool"
        self.description = self._ACTIONS[action]["description"]
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="backend",
            capabilities=("backend_query", action),
        )

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        if not context.authenticated:
            return ToolResult(False, error="login required")
        if not context.auth_token:
            return ToolResult(False, error="missing yaai cookie token")

        payload = {
            **kwargs,
            "token": context.auth_token,
        }
        headers = {
            "X-AGENT-TOKEN": settings.agent_token,
            "Content-Type": "application/json",
        }
        url = f"{settings.backend_base_url}{self._ACTIONS[self.action]['path']}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000] if exc.response is not None else ""
            return ToolResult(False, error=f"backend http {exc.response.status_code}: {body}")
        except httpx.HTTPError as exc:
            return ToolResult(False, error=f"backend http error: {exc}")
        except ValueError as exc:
            return ToolResult(False, error=f"backend returned invalid json: {exc}")

        if not result.get("success"):
            return ToolResult(
                False,
                data={"response": result},
                error=str(result.get("message") or result.get("code") or "backend query failed"),
            )

        data = result.get("data") or {}
        return ToolResult(True, data=data, summary=self._summary(data))

    def _summary(self, data: dict[str, Any]) -> str:
        if "total" in data:
            return f"后端查询成功，共 {data.get('total')} 条结果。"
        if self.action == "get_member_profile":
            return f"已获取会员 {data.get('memberId')} 的完整资料。"
        if self.action == "get_committee_detail":
            return f"已获取委员会详情，成员数 {data.get('memberCount')}。"
        if self.action == "create_payment_url":
            return f"已生成沙箱支付链接，金额 {data.get('amount')} 元。"
        return "后端查询成功。"
