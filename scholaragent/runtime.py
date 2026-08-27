"""统一执行运行时与组装入口。

CLI、Web、评测和案例都从这里获得同一套模型、工具、记忆、三种执行模式和
Auto 路由。旧的 ``main.build_agent`` / ``build_runners`` 只是本模块的兼容
转发，不再各自复制组装规则。
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from .agent import CANCELLED_ANSWER, Agent
from .artifacts import ArtifactCollector
from .events import MemoryEventSink, RunContext, RunEvent
from .workflow import WorkflowRegistry
from .llm import LLMClient, ScriptedLLM
from .memory import ConversationMemory, MemoryStore
from .metrics import MetricsCollector, RunMetrics
from .planner import Planner
from .routing import CostAwareRouter
from .team import ResearchTeam
from .tool import ToolRegistry
from .tools import build_builtin_tools
from .workspace import Workspace, workspace_for


RUNTIME_MODES = ("react", "plan", "team")


def detect_ollama(prefer: str = None):
    """探测本机可聊天且支持工具调用的 Ollama 模型。"""
    candidates = [item["name"] for item in list_ollama_models()]
    if not candidates:
        return None
    return prefer if prefer in candidates else candidates[0]


def list_ollama_models() -> list[dict]:
    """读取本机 Ollama 模型目录，失败时返回空列表而不是阻断工作台。"""
    import httpx

    try:
        # Ollama 只监听本机；不应继承系统代理，否则 Windows 下可能先
        # 构造一个无关的 HTTPS 代理/证书上下文，拖慢模型目录和任务启动。
        response = httpx.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=2,
            trust_env=False,
        )
        response.raise_for_status()
        raw_models = response.json().get("models", [])
    except Exception:
        return []

    models = []
    for raw in raw_models:
        name = str(raw.get("name") or "").strip()
        if not name or any(marker in name.lower() for marker in ("embed", "rerank")):
            continue
        details = raw.get("details") or {}
        capabilities = raw.get("capabilities")
        if capabilities is None:
            capabilities = ["completion", "tools"]
        capabilities = [str(item) for item in capabilities]
        models.append({
            "name": name,
            "size_bytes": raw.get("size"),
            "context_length": details.get("context_length"),
            "capabilities": capabilities,
            "supports_tools": "tools" in capabilities,
        })
    return models


DEMO_SCRIPT = [
    {
        "content": "这个乘法我需要用计算器算一下。",
        "tool_calls": [
            {"id": "call_demo_1", "name": "calculator",
             "arguments": {"expression": "(3+5)*12"}},
        ],
    },
    {"content": "计算完成:(3+5)*12 = 96。", "tool_calls": []},
]


RESEARCH_SYSTEM_PROMPT = (
    "你是一个严谨的科研文献调研助手。建议的工作流程:"
    "先用 arxiv_search 检索相关论文;需要细读时用 download_paper 下载,"
    "再用 read_paper 分段阅读(结尾会提示下一页,可多次调用翻页);"
    "重要发现随手用 save_note 记录;任务开始时可用 read_notes 回顾之前的笔记。"
    "值得跨对话记住的结论用 remember 存入长期记忆,想不起来的事用 recall 检索。"
    "回答时必须注明信息来自哪篇论文(标题 + arXiv 编号);"
    "论文里没有的内容不要编造;遇到计算用 calculator,问时间用 current_time。"
)


@dataclass(frozen=True)
class RunResult:
    """一次运行的统一结果，供所有入口消费。"""

    run_id: str
    status: str
    requested_mode: str
    mode: str
    answer: str
    metrics: RunMetrics
    artifacts: Mapping[str, Any]
    routing: Mapping[str, Any] | None
    events: tuple[Mapping[str, Any], ...]
    error: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    workflow: str | None = None
    source_format: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        data["artifacts"] = dict(self.artifacts)
        data["routing"] = dict(self.routing) if self.routing else None
        data["events"] = [dict(event) for event in self.events]
        data["evidence"] = dict(self.evidence)
        data["seconds"] = self.metrics.seconds
        data["cancelled"] = self.status == "cancelled"
        return data


class RuntimeExecutionError(RuntimeError):
    """旧 ``run(task) -> str`` 适配器使用的失败异常。"""

    def __init__(self, result: RunResult):
        self.result = result
        super().__init__(result.error or "运行失败")


def _invoke_runner(runner, task: str, context: RunContext):
    return context.invoke(runner, task)


def assemble_runners(agent: Agent, on_progress=None, should_stop=None,
                     artifacts=None, team_require_full_paper=True,
                     research_llm=None, summary_llm=None) -> dict:
    """唯一的三模式组装实现，供兼容入口和运行时共同使用。"""
    research_llm = research_llm or getattr(agent, "research_llm", None) or agent.llm
    summary_llm = summary_llm or getattr(agent, "summary_llm", None) or research_llm
    agent.research_llm = research_llm
    agent.summary_llm = summary_llm
    if on_progress is not None:
        agent.on_progress = on_progress
    if should_stop is not None:
        agent.should_stop = should_stop
    if artifacts is not None and getattr(agent, "tools", None) is not None:
        agent.tools.artifacts = artifacts

    worker = Agent(
        research_llm,
        agent.tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        max_steps=15,
        tool_call_limits={"arxiv_search": 3},
        metrics_mode="plan",
        on_progress=on_progress,
        should_stop=should_stop,
        summary_llm=summary_llm,
    )
    return {
        "react": agent,
        "plan": Planner(
            research_llm,
            worker,
            on_progress=on_progress,
            should_stop=should_stop,
            summary_llm=summary_llm,
        ),
        "team": ResearchTeam(
            research_llm,
            agent.tools,
            on_progress=on_progress,
            should_stop=should_stop,
            require_full_paper=team_require_full_paper,
            summary_llm=summary_llm,
        ),
    }


def execute_runners(task: str, mode: str, runners: Mapping[str, object],
                    router, workspace: Workspace,
                    artifacts: ArtifactCollector, *, run_id: str = None,
                    metrics_collector: MetricsCollector = None,
                    event_sink=None, on_progress=None,
                    should_stop=None, workflow_registry=None,
                    workflow: str | None = None,
                    source=None) -> RunResult:
    """执行已组装的 runner，并生成唯一的 ``RunResult``。

    Web 的兼容注入和 ``ExecutionRuntime`` 都走这里，因此计时、事件、
    取消、失败和指标不会在入口层各自复制一套。
    """
    if mode not in (*RUNTIME_MODES, "auto"):
        raise ValueError(f"未知模式: {mode}")
    metrics = metrics_collector or MetricsCollector(mode)
    if metrics_collector is not None:
        metrics.restart()
    workflow_registry = workflow_registry or WorkflowRegistry.default()
    selection = workflow_registry.select(
        task,
        source=source,
        requested=workflow,
    )
    context = RunContext(
        mode=mode,
        run_id=run_id,
        workspace=workspace,
        metrics=metrics,
        event_sink=event_sink,
        should_stop=should_stop,
    )
    for runner in runners.values():
        if hasattr(runner, "on_progress") and on_progress is not None:
            runner.on_progress = on_progress
        if hasattr(runner, "should_stop") and should_stop is not None:
            runner.should_stop = should_stop
        for child in (
            getattr(runner, "agent", None),
            getattr(runner, "searcher", None),
            getattr(runner, "reader", None),
            getattr(runner, "writer", None),
        ):
            if child is None:
                continue
            if on_progress is not None and hasattr(child, "on_progress"):
                child.on_progress = on_progress
            if should_stop is not None and hasattr(child, "should_stop"):
                child.should_stop = should_stop

    context.emit("run_started", requested_mode=mode, task_preview=task[:240])
    context.emit(
        "workflow_selected",
        workflow=selection.name,
        version=selection.spec.version,
        source_format=selection.source_format,
        reason=selection.reason,
        outputs=list(selection.spec.outputs),
        validators=list(selection.spec.validators),
    )
    decision = None
    selected_mode = mode
    answer = ""
    error = None
    status = "completed"
    try:
        if mode == "auto":
            decision = router.route(task)
            selected_mode = decision.mode
            context.set_mode(selected_mode)
            context.emit("mode_selected", **decision.to_dict())
        else:
            context.set_mode(mode)
            context.emit(
                "mode_selected",
                mode=mode,
                reason="显式选择执行模式",
                policy_version="explicit",
            )
        runner = runners[selected_mode]
        answer = _invoke_runner(runner, task, context)
        if answer == CANCELLED_ANSWER or context.is_cancelled():
            status = "cancelled"
            answer = CANCELLED_ANSWER
            context.mark_terminal("cancelled", reason=context.token.reason or "协作式停止")
        else:
            context.mark_terminal("completed", answer_length=len(str(answer or "")))
    except Exception as exc:
        if context.is_cancelled():
            status = "cancelled"
            answer = CANCELLED_ANSWER
            context.mark_terminal("cancelled", reason=context.token.reason or "协作式停止")
        else:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            context.mark_terminal("failed", error=error)
    finally:
        metrics = metrics.finish()

    return RunResult(
        run_id=context.run_id,
        status=status,
        requested_mode=mode,
        mode=selected_mode,
        answer=str(answer or ""),
        metrics=metrics,
        artifacts=artifacts.to_dict(),
        routing=decision.to_dict() if decision is not None else None,
        events=tuple(event.to_dict() for event in context.event_list),
        error=error,
        evidence=context.evidence.to_dict(),
        workflow=selection.name,
        source_format=selection.source_format,
    )


class ExecutionRuntime:
    """一次运行所需的完整依赖图。"""

    def __init__(self, demo: bool = False,
                 workspace: Workspace | str | None = None,
                 artifacts: ArtifactCollector | None = None,
                 llm=None,
                 research_llm=None,
                 summary_llm=None,
                 conversation=True,
                 auto_recall: bool = True,
                 team_require_full_paper: bool = True,
                 discover_plugins: bool = False):
        self.workspace = workspace_for(workspace)
        self.artifacts = artifacts or ArtifactCollector(self.workspace)
        self.memory = MemoryStore(workspace=self.workspace)
        base_llm = llm or (ScriptedLLM(DEMO_SCRIPT) if demo else LLMClient())
        self.research_llm = research_llm or base_llm
        self.summary_llm = summary_llm or self.research_llm
        # 保留旧属性：外部入口仍可把 runtime.llm 当作主模型读取。
        self.llm = self.research_llm
        tools = build_builtin_tools(self.workspace, memory_store=self.memory)
        self.registry = ToolRegistry(
            tools,
            artifacts=self.artifacts,
            workspace=self.workspace,
        )
        self.plugin_report = {"loaded": [], "errors": []}
        if discover_plugins:
            self.plugin_report = self.registry.discover_plugins()
        if conversation is True:
            conversation = ConversationMemory()
        self.agent = Agent(
            self.research_llm,
            self.registry,
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            max_steps=15,
            conversation=conversation if conversation is not False else None,
            long_memory=self.memory,
            auto_recall=auto_recall,
            tool_call_limits={"arxiv_search": 3},
            summary_llm=self.summary_llm,
        )
        self.runners = assemble_runners(
            self.agent,
            artifacts=self.artifacts,
            team_require_full_paper=team_require_full_paper,
            research_llm=self.research_llm,
            summary_llm=self.summary_llm,
        )
        self.router = CostAwareRouter(str(self.workspace.router_policy_path))
        self.workflow_registry = WorkflowRegistry.default()
        # 兼容旧的 ``build_agent`` 返回值，同时让 CLI 能回到统一 RunResult
        # 接缝，而不需要重新组装第二套执行图。
        self.agent._runtime = self

    def discover_tool_plugins(self) -> dict:
        """显式发现本机已安装的工具插件；默认运行不会加载外部代码。"""
        self.plugin_report = self.registry.discover_plugins()
        return self.plugin_report

    def set_callbacks(self, on_progress=None, should_stop=None) -> None:
        """把本次运行的兼容回调安装到所有执行层。"""
        for runner in self.runners.values():
            if hasattr(runner, "on_progress") and on_progress is not None:
                runner.on_progress = on_progress
            if hasattr(runner, "should_stop") and should_stop is not None:
                runner.should_stop = should_stop
            for child in (
                getattr(runner, "agent", None),
                getattr(runner, "searcher", None),
                getattr(runner, "reader", None),
                getattr(runner, "writer", None),
            ):
                if child is None:
                    continue
                if on_progress is not None and hasattr(child, "on_progress"):
                    child.on_progress = on_progress
                if should_stop is not None and hasattr(child, "should_stop"):
                    child.should_stop = should_stop

    def runner(self, mode: str, metrics_collector: MetricsCollector = None,
               event_sink=None, on_progress=None, should_stop=None):
        if mode not in (*RUNTIME_MODES, "auto"):
            raise ValueError(f"未知模式: {mode}")
        return RuntimeRunner(
            self,
            mode,
            metrics_collector=metrics_collector,
            event_sink=event_sink,
            on_progress=on_progress,
            should_stop=should_stop,
        )

    def run(self, task: str, mode: str = "react", *, run_id: str = None,
            metrics_collector: MetricsCollector = None,
            event_sink=None, on_progress=None, should_stop=None,
            workflow: str | None = None, source=None) -> RunResult:
        task = str(task or "").strip()
        if not task:
            raise ValueError("任务不能为空。")
        return execute_runners(
            task,
            mode,
            self.runners,
            self.router,
            self.workspace,
            self.artifacts,
            run_id=run_id,
            metrics_collector=metrics_collector,
            event_sink=event_sink,
            on_progress=on_progress,
            should_stop=should_stop,
            workflow_registry=self.workflow_registry,
            workflow=workflow,
            source=source,
        )


class RuntimeRunner:
    """把统一 ``RunResult`` 适配为旧的 ``run(task) -> str``。"""

    def __init__(self, runtime: ExecutionRuntime, mode: str,
                 metrics_collector=None, event_sink=None,
                 on_progress=None, should_stop=None):
        self.runtime = runtime
        self.mode = mode
        self.metrics_collector = metrics_collector
        self.event_sink = event_sink
        self.on_progress = on_progress
        self.should_stop = should_stop
        self.last_result: RunResult | None = None
        self.last_metrics: RunMetrics | None = None

    def run(self, task: str, context: RunContext = None,
            workflow: str | None = None, source=None) -> str:
        # 外部已经提供上下文时直接委派，供 Planner/Team 的嵌套兼容使用。
        if context is not None:
            runner = self.runtime.runners[self.mode]
            return context.invoke(runner, task)
        self.last_result = self.runtime.run(
            task,
            self.mode,
            metrics_collector=self.metrics_collector,
            event_sink=self.event_sink,
            on_progress=self.on_progress,
            should_stop=self.should_stop,
            workflow=workflow,
            source=source,
        )
        self.last_metrics = self.last_result.metrics
        if self.last_result.status == "failed":
            raise RuntimeExecutionError(self.last_result)
        return self.last_result.answer


def create_runtime(**kwargs) -> ExecutionRuntime:
    return ExecutionRuntime(**kwargs)


def build_agent(demo: bool, artifacts=None, workspace=None, llm=None,
                research_llm=None, summary_llm=None,
                discover_plugins: bool = False) -> Agent:
    """旧 main/web 入口的兼容适配器。"""
    return ExecutionRuntime(
        demo=demo,
        workspace=workspace,
        artifacts=artifacts,
        llm=llm,
        research_llm=research_llm,
        summary_llm=summary_llm,
        discover_plugins=discover_plugins,
    ).agent


def build_runners(agent: Agent, on_progress=None, should_stop=None,
                  artifacts=None) -> dict:
    """旧 main/web 入口的兼容适配器。"""
    return assemble_runners(
        agent,
        on_progress=on_progress,
        should_stop=should_stop,
        artifacts=artifacts,
        research_llm=getattr(agent, "research_llm", None),
        summary_llm=getattr(agent, "summary_llm", None),
    )
