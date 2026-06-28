from __future__ import annotations

from src.skills.base import SkillSpec


def _load_prompt() -> str:
    return (
        "你是 YAAI Business Skill，负责 YAAI 后端业务查询和业务辅助。"
        "适用任务：新闻搜索、会员资料、委员会列表/详情、会员审核、操作日志、订单/缴费/支付链接。"
        "本 Skill 同时服务普通用户和管理员，不在 Skill 层拆权限；"
        "由 Yaai Business Agent 根据用户请求选择后端业务 Tool，Java 后端根据 token 做最终权限校验。"
        "如果工具返回权限不足或参数缺失，要向用户说明需要登录、权限或补充的业务 ID。"
    )


yaai_business_skill = SkillSpec(
    name="yaai_business_skill",
    description="YAAI 业务：新闻、会员、委员会、审核、日志、订单和缴费",
    summary="用于调用 Java AgentController 后端业务接口，普通用户和管理员统一入口。",
    allowed_agents=("yaai_business_agent", "response_agent"),
    login=("login",),
    platforms=("*",),
    roles=("*",),
    current_pages=("*",),
    page_types=("*",),
    prompt_loader=_load_prompt,
)
