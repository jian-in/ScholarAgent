"""工具层:给大模型装上"手"。

大模型本身只会生成文字。它"调用工具"的真相是:
模型生成一段 JSON(工具名 + 参数),由我们的代码解析并真正执行,
再把执行结果作为文字塞回对话,模型看着结果继续想下一步。
本文件定义所有工具的统一形状,以及集中管理工具的登记处。

工具内部仍可返回旧式字符串；登记处会把它适配成 ``ToolResult``，让
控制流、产物收集和运行事件不再依赖中文成功文案。
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


# 工具可用这个前缀告诉 Agent:错误来自外部服务,继续换参数调用也没有意义。
# Agent 会在本轮后续步骤中移除该工具,避免限流/断网时陷入重试循环。
STOP_RETRY_PREFIX = "[本轮停止重试]"


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结构化结果。

    ``text`` 是模型看到的兼容观察文本；其余字段只供运行时、Web 和
    产物收集使用。
    """

    text: str
    success: bool = True
    stop_retry: bool = False
    artifacts: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    diagnostic: Mapping[str, Any] | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["artifacts"] = [dict(item) for item in self.artifacts]
        return data


def adapt_tool_result(value: Any) -> ToolResult:
    """把旧工具的字符串/字典结果归一化为 ``ToolResult``。"""
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, Mapping) and "text" in value:
        artifacts = value.get("artifacts") or ()
        if isinstance(artifacts, Mapping):
            artifacts = (artifacts,)
        return ToolResult(
            text=str(value.get("text") or ""),
            success=bool(value.get("success", True)),
            stop_retry=bool(value.get("stop_retry", False)),
            artifacts=tuple(artifacts),
            diagnostic=value.get("diagnostic"),
        )

    text = str(value or "")
    failure = _looks_like_failure(text)
    return ToolResult(
        text=text,
        success=not failure,
        stop_retry=text.startswith(STOP_RETRY_PREFIX),
        diagnostic={"legacy": True} if failure else None,
    )


def _looks_like_failure(text: str) -> bool:
    """识别旧工具的错误出口，集中在兼容层而非 ArtifactCollector。"""
    return (
        text.startswith("错误:")
        or "执行出错:" in text
        or text.startswith("工具 ") and "执行出错" in text
        or "不是 PDF" in text
        or "无法解析" in text
        or "还没下载" in text
        or "不存在名为" in text
    )


class Tool:
    """所有工具的基类:子类填三个类属性 + 实现 run() 即可。"""

    name = ""          # 工具名,模型靠它点名调用
    description = ""   # 给模型看的说明书,写得越清楚模型用得越准
    parameters = {"type": "object", "properties": {}, "required": []}  # 参数的 JSON Schema
    license = "MIT"

    def start_run(self):
        """开始新一轮 Agent 任务时重置工具的临时会话状态。"""

    def completion_ready(self) -> bool:
        """需要任务级完成校验的工具可覆盖此方法。"""
        return True

    def run(self, **kwargs) -> str:
        """真正干活的地方。必须返回字符串,因为结果最终要变成文字回给模型。"""
        raise NotImplementedError

    def run_result(self, **kwargs) -> ToolResult:
        """结构化调用入口；旧工具只需实现 ``run`` 即可兼容。"""
        return adapt_tool_result(self.run(**kwargs))

    def artifact_metadata(self, arguments: Mapping[str, Any],
                          result: ToolResult) -> list[Mapping[str, Any]]:
        """工具可覆盖此方法，为成功副作用提供结构化元数据。"""
        return list(result.artifacts)

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

    def __init__(self, tools=None, artifacts=None, workspace=None):
        self._tools = {}
        self._run_tool_calls = 0
        # 可选:本轮产物收集器(工作台用来展示论文/笔记/记忆)
        self.artifacts = artifacts
        self.workspace = workspace
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
            [self._tools[n] for n in names], artifacts=self.artifacts,
            workspace=self.workspace,
        )

    def call_result(self, name: str, arguments: dict, context=None) -> ToolResult:
        """执行工具并返回结构化结果。

        模型看到错误信息后,往往能自己修正参数重试 ——
        这是 Agent 具备"纠错能力"的来源之一。
        """
        self._run_tool_calls += 1
        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(
                text=f"错误:不存在名为 {name} 的工具。可用工具:{', '.join(self._tools)}",
                success=False,
                diagnostic={"kind": "unknown_tool", "name": name},
            )
            self._record_result(name, arguments, result, context)
            return result
        if context is not None:
            context.emit("tool_started", name=name, arguments=arguments)
            context.metrics.record_tool_call()
        try:
            result = adapt_tool_result(tool.run_result(**arguments))
        except Exception as exc:  # 故意兜住一切异常,转成文字回传给模型
            result = ToolResult(
                text=f"工具 {name} 执行出错:{type(exc).__name__}: {exc}",
                success=False,
                diagnostic={
                    "kind": "exception",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
        metadata = list(result.artifacts)
        if not metadata:
            try:
                metadata = list(tool.artifact_metadata(arguments, result))
            except Exception:
                metadata = []
        if metadata:
            result = ToolResult(
                text=result.text,
                success=result.success,
                stop_retry=result.stop_retry,
                artifacts=tuple(metadata),
                diagnostic=result.diagnostic,
            )
        self._record_result(name, arguments, result, context)
        return result

    def _record_result(self, name: str, arguments: dict,
                       result: ToolResult, context=None) -> None:
        if context is not None:
            context.emit(
                "tool_completed",
                name=name,
                success=result.success,
                stop_retry=result.stop_retry,
                observation_preview=result.text[:240],
                diagnostic=result.diagnostic,
            )
            for artifact in result.artifacts:
                try:
                    context.evidence.ingest_artifact(artifact)
                except (TypeError, ValueError):
                    # 来源锚点是审计旁路；坏元数据不能把原本可用的工具
                    # 结果升级成运行失败。
                    pass
                context.emit("artifact", **dict(artifact))
        if self.artifacts is not None:
            try:
                self.artifacts.record_result(name, arguments, result)
            except Exception:
                # 产物展示是旁路能力，不能让工具调用失败。
                pass

    def call(self, name: str, arguments: dict, context=None) -> str:
        """旧兼容入口：只返回模型可见文本。"""
        return self.call_result(name, arguments, context=context).text

    def discover_plugins(self, group: str = "scholaragent.tools",
                         entry_points=None) -> dict:
        """发现并隔离加载外部工具插件。

        返回 ``loaded`` 与 ``errors``，坏插件或重复名称不会阻断核心启动。
        ``entry_points`` 参数只用于测试和自定义适配器。
        """
        report = {"loaded": [], "errors": []}
        try:
            points = entry_points
            if points is None:
                points = importlib.metadata.entry_points()
            if hasattr(points, "select"):
                points = points.select(group=group)
            elif isinstance(points, Mapping):
                points = points.get(group, ())
            else:
                points = [point for point in points if getattr(point, "group", group) == group]
        except Exception as exc:
            report["errors"].append({"name": group, "error": f"发现失败: {exc}"})
            return report

        for point in points:
            name = getattr(point, "name", "<unknown>")
            try:
                candidate = point.load()
                if isinstance(candidate, type):
                    candidate = candidate()
                elif callable(candidate) and not isinstance(candidate, Tool):
                    candidate = candidate()
                if not isinstance(candidate, Tool):
                    raise TypeError("插件必须返回 scholaragent.tool.Tool 实例")
                plugin_classes = [base for base in type(candidate).__mro__ if base is not Tool]
                if not any("license" in base.__dict__ and base.__dict__["license"]
                           for base in plugin_classes):
                    raise ValueError("插件缺少显式 license 信息")
                if not candidate.description:
                    raise ValueError("插件缺少 description 信息")
                if not isinstance(candidate.parameters, Mapping):
                    raise ValueError("插件 parameters 必须是 JSON Schema 对象")
                if type(candidate).run is Tool.run:
                    raise ValueError("插件缺少 run() 实现")
                self.register(candidate)
                report["loaded"].append(candidate.name)
            except Exception as exc:
                report["errors"].append({
                    "name": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return report
