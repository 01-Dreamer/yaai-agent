from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.core.context import RuntimeContext
from src.tools.base import ToolResult, ToolSpec

ActionSender = Callable[[RuntimeContext, str, dict[str, Any]], Awaitable[dict[str, Any]]]


BROWSER_ACTION_DESCRIPTIONS = {
    "navigate": (
        "请求浏览器跳转页面。参数：path（目标路径，必填）；need_confirm（是否需要确认，默认需要）。"
        "返回：跳转后的路径。示例：{\"path\":\"/conference\",\"need_confirm\":true}。"
        "限制：只能跳转当前平台已注册的前端路由，路径必须由 Supervisor 根据页面目录生成并校验，不能跳转外部站点。"
    ),
    "fill": (
        "请求浏览器填写当前页面表单或低代码编辑器字段。参数：values（要填入的字段和值，必填）；"
        "diff（展示给用户的修改前后对比数组，用于确认弹窗预览）；need_confirm（默认需要确认）。"
        "返回：成功填入的字段数、匹配到的字段数、未匹配到的字段名。"
        "示例：{\"values\":{\"姓名\":\"李四\"},\"diff\":[{\"field\":\"姓名\",\"to\":\"李四\"}]}。"
        "限制：填表前必须先调用页面查看工具获取表单字段名和可选值，不能提交、保存或执行任意脚本。"
    ),
    "highlight": (
        "请求浏览器高亮元素或标红页面文本。元素高亮：selector（目标元素选择器）、持续时长。"
        "文本标红：标记模式、标记列表（每条包含目标原文和周围上下文）、高亮颜色、背景色，默认不需要确认。"
        "返回：高亮的元素数量或标记的文本数量。"
        "示例：{\"标记模式\":\"文本标红\",\"标记列表\":[{\"上下文\":\"服务开展活动。学会致力于\",\"目标\":\"开展活动\"}],\"高亮颜色\":\"#dc2626\"}。"
        "限制：文本标红前必须先查看页面结构，标记列表中的目标文本必须来自页面原文，不能凭空编造或用正则全页匹配。"
    ),
    "inspect_html": (
        "只读查看当前页面结构。参数：最大字符数、最大字段数、目标字段关键词、是否需要确认（默认不需要）。"
        "返回：页面标题、网址、路径、表单字段列表（含标签名、字段名、选择器、类型、必填项、可选项）、按钮列表、页面文本、受限的页面快照。"
        "用途：为填表生成字段参数，为高亮生成原文标记，为页面问答理解当前页面内容。"
        "限制：不读取浏览器存储信息，不读取输入框中已填的值，不执行任何脚本。"
    ),
}


class BrowserActionTool:
    def __init__(self, action: str, sender: ActionSender | None = None) -> None:
        self.action = action
        self.name = f"browser.{action}_tool"
        self.description = BROWSER_ACTION_DESCRIPTIONS.get(action, f"请求浏览器执行前端 action：{action}")
        self.spec = ToolSpec(
            name=self.name,
            description=self.description,
            namespace="browser",

            platforms=("frontend", "lowcode"),
            capabilities=(action,),
        )
        self._sender = sender

    def bind_sender(self, sender: ActionSender) -> None:
        self._sender = sender

    async def run(self, context: RuntimeContext, **kwargs: Any) -> ToolResult:
        if self._sender is None:
            return ToolResult(False, error="browser action sender is not bound")
        result = await self._sender(context, self.action, kwargs)
        return ToolResult(bool(result.get("success")), data=result, error=result.get("error"))
