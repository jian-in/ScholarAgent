"""工具层:给大模型装上"手"。

大模型本身只会生成文字。它"调用工具"的真相是:
模型生成一段 JSON(工具名 + 参数),由我们的代码解析并真正执行,
再把执行结果作为文字塞回对话,模型看着结果继续想下一步。
本文件定义所有工具的统一形状,以及集中管理工具的登记处。
"""


# 工具可用这个前缀告诉 Agent:错误来自外部服务,继续换参数调用也没有意义。
# Agent 会在本轮后续步骤中移除该工具,避免限流/断网时陷入重试循环。
STOP_RETRY_PREFIX = "[本轮停止重试]"


class Tool:
    """所有工具的基类:子类填三个类属性 + 实现 run() 即可。"""

    name = ""          # 工具名,模型靠它点名调用
    description = ""   # 给模型看的说明书,写得越清楚模型用得越准
    parameters = {"type": "object", "properties": {}, "required": []}  # 参数的 JSON Schema

    def start_run(self):
        """开始新一轮 Agent 任务时重置工具的临时会话状态。"""

    def completion_ready(self) -> bool:
        """需要任务级完成校验的工具可覆盖此方法。"""
        return True

    def run(self, **kwargs) -> str:
        """真正干活的地方。必须返回字符串,因为结果最终要变成文字回给模型。"""
        raise NotImplementedError

    def schema(self) -> dict:
        """输出 OpenAI 工具协议要求的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具登记处:Agent 从这里查询有哪些工具,并代替模型执行它们。"""

    def __init__(self, tools=None, artifacts=None):
        self._tools = {}
        self._run_tool_calls = 0
        # 可选:本轮产物收集器(工作台用来展示论文/笔记/记忆)
        self.artifacts = artifacts
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool):
        if not tool.name:
            raise ValueError(f"工具 {type(tool).__name__} 没有设置 name")
        if tool.name in self._tools:
            raise ValueError(f"工具名重复:{tool.name}")
        self._tools[tool.name] = tool

    def start_run(self):
        """通知本登记处内的工具开始一轮新的任务。"""
        self._run_tool_calls = 0
        for tool in self._tools.values():
            tool.start_run()

    @property
    def run_tool_calls(self) -> int:
        """当前任务已实际执行的工具调用次数。"""
        return self._run_tool_calls

    def complete_required_arguments(self, name: str, arguments: dict,
                                    remembered: dict) -> dict:
        """用本轮已知的同名参数补齐模型偶尔漏掉的必填字段。"""
        tool = self._tools.get(name)
        if tool is None:
            return dict(arguments)
        completed = dict(arguments)
        for key in tool.parameters.get("required", []):
            if key not in completed and key in remembered:
                completed[key] = remembered[key]
        return completed

    def pending_completions(self, names) -> list:
        """返回尚未达到完成条件的指定工具名。"""
        return [name for name in names
                if name in self._tools
                and not self._tools[name].completion_ready()]

    def schemas(self, exclude=None) -> list:
        """返回可用工具的 JSON Schema,可临时排除本轮已熔断的工具。"""
        excluded = set(exclude or ())
        return [tool.schema() for tool in self._tools.values()
                if tool.name not in excluded]

    def subset(self, names: list) -> "ToolRegistry":
        """按名单挑出部分工具,组成一个新的登记处。

        M4 多智能体的关键机制:一个"角色" = 系统提示词 + 工具白名单。
        检索员只发搜索的工具、写作员只发笔记的工具 —— 各司其职,
        既省上下文(工具清单更短),也防止角色越权乱用工具。
        """
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise ValueError(f"登记处里没有这些工具:{', '.join(missing)}")
        return ToolRegistry(
            [self._tools[n] for n in names], artifacts=self.artifacts
        )

    def call(self, name: str, arguments: dict) -> str:
        """执行工具。关键设计:出错不抛异常,而是把错误文字返回给模型。

        模型看到错误信息后,往往能自己修正参数重试 ——
        这是 Agent 具备"纠错能力"的来源之一。
        """
        self._run_tool_calls += 1
        tool = self._tools.get(name)
        if tool is None:
            return f"错误:不存在名为 {name} 的工具。可用工具:{', '.join(self._tools)}"
        try:
            result = str(tool.run(**arguments))
        except Exception as exc:  # 故意兜住一切异常,转成文字回传给模型
            result = f"工具 {name} 执行出错:{type(exc).__name__}: {exc}"
        if self.artifacts is not None:
            try:
                self.artifacts.record(name, arguments, result)
            except Exception:
                pass
        return result
