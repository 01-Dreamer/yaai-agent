from __future__ import annotations

from src.skills.base import SkillSpec


def _knowledge_qa_prompt() -> str:
    return (
        "你是 YAAI 通用知识问答 Skill。优先让检索类子 Agent 提供证据；"
        "如果用户上传文件，则允许文件分析子 Agent 先提取附件内容。"
    )


def _frontend_assistant_prompt() -> str:
    return (
        "你是 YAAI 官网前台辅助 Skill。适用于官网浏览、会员注册、资料填写、"
        "页面跳转与字段解释。需要操作前端时只能调度前端控制子 Agent。"
    )


def _lowcode_assistant_prompt() -> str:
    return (
        "你是 YAAI 低代码后台辅助 Skill。适用于页面、模板、片段、组件、"
        "数据绑定、菜单、新闻、会员审核等后台管理场景。"
    )


def _activity_publish_prompt() -> str:
    return (
        "你是活动发布 Skill。只有管理员在活动创建页可用。先检索是否存在重复活动，"
        "再按需调度前端控制子 Agent 生成填表建议。"
    )


SKILLS = [
    SkillSpec(
        name="knowledge_qa_skill",
        description="通用知识问答与附件理解",
        summary="所有平台、所有角色、所有页面可用；用于普通问答、文档知识问答和附件总结。",
        allowed_agents=("retrieval", "file_analysis"),
        platforms=("*",),
        roles=("*",),
        current_pages=("*",),
        page_types=("*",),
        tags=("qa", "default"),
        prompt_loader=_knowledge_qa_prompt,
    ),
    SkillSpec(
        name="frontend_apply_assistant",
        description="官网前台浏览、会员申请与资料填写",
        summary="官网前台可用；用于导航、入会申请、用户中心、内容浏览和前台表单辅助。",
        allowed_agents=("retrieval", "file_analysis", "frontend_control"),
        platforms=("frontend", "yaai_portal"),
        roles=("*",),
        current_pages=("*",),
        page_types=("*",),
        tags=("frontend", "member"),
        prompt_loader=_frontend_assistant_prompt,
    ),
    SkillSpec(
        name="lowcode_editor_assistant",
        description="低代码后台管理与编辑器辅助",
        summary="后台/低代码平台管理员可用；用于页面配置、编辑器字段解释、后台管理和受控前端操作。",
        allowed_agents=("retrieval", "file_analysis", "frontend_control"),
        platforms=("lowcode", "yaai_admin"),
        roles=("admin",),
        current_pages=("*",),
        page_types=("*",),
        tags=("lowcode", "admin"),
        prompt_loader=_lowcode_assistant_prompt,
    ),
    SkillSpec(
        name="activity_publish_skill",
        description="活动创建页发布辅助示例",
        summary="仅管理员在后台活动创建页可用；用于活动发布前的信息检索、去重和填表建议。",
        allowed_agents=("retrieval", "frontend_control"),
        platforms=("yaai_admin", "lowcode"),
        roles=("admin",),
        current_pages=("activity_create",),
        page_types=("activity_create",),
        tags=("activity", "admin"),
        prompt_loader=_activity_publish_prompt,
    ),
]
