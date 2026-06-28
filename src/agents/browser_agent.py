from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base import AgentRequest, AgentResponse, AgentSpec
from src.core.context import RuntimeContext
from src.core.registry import tool_registry
from src.models.llm import chat_complete


FRONTEND_NAVIGATION_PAGES: tuple[dict[str, Any], ...] = (
    {"path": "/", "title": "首页", "aliases": ("官网首页", "门户首页", "主页", "home"), "description": "云南人工智能学会官网首页"},
    {"path": "/news", "title": "新闻动态", "aliases": ("新闻", "动态", "通知公告", "学术动态", "新闻列表"), "description": "官网新闻动态列表页"},
    {"path": "/conference", "title": "学术会议", "aliases": ("会议", "会议活动", "学术会议页面", "conference"), "description": "学术会议频道页"},
    {"path": "/services", "title": "服务频道", "aliases": ("服务", "资源", "业务入口", "服务页面"), "description": "学会服务、资源和业务入口"},
    {"path": "/about", "title": "学会介绍", "aliases": ("关于我们", "学会概况", "组织架构", "about"), "description": "学会介绍频道页"},
    {"path": "/about/introduction", "title": "学会简介", "aliases": ("简介", "学会介绍详情"), "description": "学会简介页面"},
    {"path": "/about/charter", "title": "学会章程", "aliases": ("章程",), "description": "学会章程页面"},
    {"path": "/about/regulations", "title": "规章制度", "aliases": ("制度", "规章"), "description": "学会规章制度页面"},
    {"path": "/about/leaders", "title": "组织领导", "aliases": ("领导", "理事会", "组织领导"), "description": "学会组织领导页面"},
    {"path": "/about/branches", "title": "分支机构", "aliases": ("分支", "专业委员会", "委员会"), "description": "学会分支机构页面"},
    {"path": "/about/local", "title": "地方组织", "aliases": ("地方", "地方组织"), "description": "学会地方组织页面"},
    {"path": "/apply", "title": "会员注册", "aliases": ("入会申请", "申请入会", "会员申请", "注册入口"), "description": "会员注册入口页"},
    {"path": "/apply/profile/personal", "title": "个人会员资料填写", "aliases": ("个人会员申请", "个人入会", "个人资料填写"), "description": "个人会员资料填写页"},
    {"path": "/apply/profile/company", "title": "单位会员资料填写", "aliases": ("单位会员申请", "单位入会", "企业会员申请"), "description": "单位会员资料填写页"},
    {"path": "/login", "title": "会员登录", "aliases": ("登录", "账号登录"), "description": "会员登录页"},
    {"path": "/user", "title": "会员个人中心", "aliases": ("个人中心", "用户中心", "会员中心"), "description": "会员个人中心"},
)

LOWCODE_NAVIGATION_PAGES: tuple[dict[str, Any], ...] = (
    {"path": "/workbench", "title": "工作台", "aliases": ("后台首页", "控制台", "dashboard"), "description": "低代码后台工作台"},
    {"path": "/pages", "title": "页面管理", "aliases": ("页面列表", "官网页面管理"), "description": "管理官网页面和页面版本"},
    {"path": "/pages/create", "title": "新增页面", "aliases": ("创建页面", "页面创建"), "description": "新增官网页面基础信息"},
    {"path": "/page-templates", "title": "页面模板管理", "aliases": ("页面模板", "模板列表"), "description": "管理整页模板"},
    {"path": "/page-templates/create", "title": "新增页面模板", "aliases": ("创建模板", "模板创建"), "description": "新增整页模板基础信息"},
    {"path": "/reusable-fragments", "title": "可复用片段管理", "aliases": ("片段管理", "可复用片段", "片段列表"), "description": "管理 reusable_fragment"},
    {"path": "/reusable-fragments/create", "title": "新增片段", "aliases": ("创建片段", "片段创建"), "description": "新增可复用片段基础信息"},
    {"path": "/component-defs", "title": "组件定义管理", "aliases": ("组件定义", "组件管理", "组件列表"), "description": "查看与维护组件定义"},
    {"path": "/component-defs/create", "title": "新增组件定义", "aliases": ("创建组件定义", "新增组件"), "description": "新增组件 schema 和默认配置"},
    {"path": "/data-bindings", "title": "数据绑定管理", "aliases": ("数据绑定", "数据源绑定"), "description": "管理低代码数据源绑定配置"},
    {"path": "/data-bindings/create", "title": "新增数据绑定", "aliases": ("创建数据绑定", "新增数据源绑定"), "description": "新增数据源绑定"},
    {"path": "/menus", "title": "菜单管理", "aliases": ("导航菜单", "菜单配置"), "description": "维护官网导航菜单与页面映射"},
    {"path": "/banner-management", "title": "轮播图管理", "aliases": ("轮播图", "首页轮播图", "banner"), "description": "维护官网首页轮播图数据"},
    {"path": "/news", "title": "新闻管理", "aliases": ("新闻后台", "新闻内容管理"), "description": "维护新闻内容数据"},
    {"path": "/news-categories", "title": "新闻分类管理", "aliases": ("新闻分类", "分类管理"), "description": "维护新闻分类数据"},
    {"path": "/member-audit", "title": "会员审核", "aliases": ("审核会员", "入会审核", "会员申请审核"), "description": "处理个人和单位会员审核"},
    {"path": "/member-orders", "title": "会员订单查看", "aliases": ("会员订单", "订单查看", "订单管理"), "description": "按会员查询订单记录"},
    {"path": "/log", "title": "操作日志管理", "aliases": ("操作日志", "日志管理", "系统日志"), "description": "查看系统操作日志"},
)


class BrowserAgent:
    spec = AgentSpec(
        name="browser_agent",
        description="通过浏览器宿主请求前端白名单 action 的 Agent",
        tools=("browser.navigate_tool", "browser.fill_tool", "browser.highlight_tool", "browser.inspect_html_tool"),
        capabilities=("navigate", "fill", "highlight", "inspect_html"),
    )

    async def request_action(self, context: RuntimeContext, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = f"browser.{action}_tool"
        if tool_name not in set(self.spec.tools):
            return {"success": False, "error": f"browser action is not allowed: {action}"}
        item = tool_registry.get(tool_name)
        result = await item.handler.run(context, **payload)
        return {"success": result.success, "data": result.data, "error": result.error}

    # ---------- 导航 ----------
    def _navigation_pages_for_platform(self, platform: str) -> tuple[dict[str, Any], ...]:
        if platform in {"lowcode", "yaai_admin"}:
            return LOWCODE_NAVIGATION_PAGES
        return FRONTEND_NAVIGATION_PAGES

    def _navigation_page_paths(self, platform: str) -> set[str]:
        return {str(item["path"]) for item in self._navigation_pages_for_platform(platform)}

    def _navigation_page_by_path(self, platform: str, path: str) -> dict[str, Any] | None:
        for page in self._navigation_pages_for_platform(platform):
            if str(page.get("path") or "") == path:
                return page
        return None

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start: end + 1]
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("LLM navigate payload is not a JSON object")
        return data

    async def generate_navigate_payload(self, context: RuntimeContext, content: str) -> dict[str, Any]:
        pages = self._navigation_pages_for_platform(context.platform)
        system_prompt = (
            "你是浏览器导航 JSON 生成器。只能根据用户请求和当前平台页面目录选择一个 path，不要输出解释。\n"
            "输出格式必须是：{\"path\":\"/xxx\",\"title\":\"页面标题\"}。如果没有明确匹配，输出 {\"path\":\"\",\"reason\":\"原因\"}。\n"
            "硬规则：\n"
            "1. path 必须严格来自页面目录，不允许编造路径。\n"
            "2. frontend 平台只能选择官网前台页面；lowcode 平台只能选择低代码后台页面。\n"
            "3. 不要默认选择 /；只有用户明确要求首页/主页时才选择 / 或 /workbench。\n"
            "4. 动态详情或编辑路由需要具体 ID，目录没有给出具体 path 时不要猜。"
            "5. 例子：用户说「学术会议页面」，frontend 平台应选择「学术会议」对应的 /conference；"
            "用户说「新闻管理」，lowcode 平台应选择「新闻管理」对应的 /news。"
        )
        user_prompt = (
            f"当前平台：{context.platform}\n"
            f"当前页面：currentPage={context.current_page}, pageType={context.page_type}, path={context.page.get('path') if isinstance(context.page, dict) else ''}\n"
            f"用户请求：{content}\n\n"
            f"页面目录：{json.dumps(pages, ensure_ascii=False)}\n\n"
            "请只返回 JSON object。"
        )
        try:
            payload = self._extract_json_object(await chat_complete(system_prompt, user_prompt, tier="small"))
        except Exception as exc:
            return {"path": "", "reason": f"导航决策 LLM 调用失败：{exc}"}
        path = str(payload.get("path") or "").strip()
        if path not in self._navigation_page_paths(context.platform):
            return {"path": "", "reason": f"LLM 返回的路径不在当前平台页面目录中：{path or '空'}"}
        page = self._navigation_page_by_path(context.platform, path)
        returned_title = str(payload.get("title") or "").strip()
        expected_title = str((page or {}).get("title") or "")
        if returned_title != expected_title:
            return {"path": "", "reason": f"LLM 返回的页面标题与路径不一致：title={returned_title or '空'}, path={path}"}
        return {"path": path, "requiresConfirm": True}

    async def run(self, request: AgentRequest) -> AgentResponse:
        action = request.intent or (request.payload or {}).get("action")
        if not action:
            return AgentResponse(False, error="missing frontend action")
        payload = dict(request.payload or {})
        payload.pop("action", None)
        result = await self.request_action(request.context, action, payload)
        return AgentResponse(bool(result.get("success")), data=result, error=result.get("error"))
