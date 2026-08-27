"""重构护栏：CLI、Web、评测共享同一组装契约。"""

from pathlib import Path

import main
import webapp
from evals import run_eval
from scholaragent import config
from scholaragent.events import MemoryEventSink, RunContext
from scholaragent.llm import ScriptedLLM
from scholaragent.metrics import MetricsCollector
from scholaragent.routing import AdaptiveRunner, RoutingDecision, ROUTING_MODES
from scholaragent.runtime import RUNTIME_MODES, create_runtime
from scholaragent.tool import Tool, ToolRegistry
from scholaragent.workspace import TemporaryWorkspace


class EchoTool(Tool):
    name = "echo"

    def run(self, text=""):
        return text


def test_cli_web_eval_share_the_same_explicit_mode_contract():
    assert tuple(main.CLI_MODES[:-1]) == (*RUNTIME_MODES,)
    assert tuple(webapp.VALID_MODES[:-1]) == (*RUNTIME_MODES,)
    runtime = create_runtime(
        demo=True,
        workspace=TemporaryWorkspace(Path(".pytest-runtime-contract")),
        conversation=False,
    )
    assert tuple(runtime.runners) == RUNTIME_MODES
    # 不把测试临时目录留在项目中；runtime 只在工具真正写入时创建它。


def test_adaptive_runner_executes_only_the_selected_mode():
    calls = []

    class Runner:
        def __init__(self, mode):
            self.mode = mode

        def run(self, task):
            calls.append((self.mode, task))
            return self.mode

    class Router:
        def route(self, task):
            return RoutingDecision(
                mode="plan",
                predicted_utility={mode: float(mode == "plan") for mode in ROUTING_MODES},
                features={"bias": 1.0},
                reason="契约测试",
                policy_version="test",
            )

    runner = AdaptiveRunner(
        Router(), {mode: Runner(mode) for mode in ROUTING_MODES}
    )
    assert runner.run("任务") == "plan"
    assert calls == [("plan", "任务")]


def test_context_events_have_stable_run_identity_and_no_api_key():
    sink = MemoryEventSink()
    context = RunContext(
        mode="react",
        workspace=TemporaryWorkspace(Path(".pytest-event-contract")),
        event_sink=sink,
    )
    context.emit("run_started", api_key="must-not-leak", task_preview="任务")
    context.set_mode("plan")
    context.mark_terminal("completed")
    context.mark_terminal("failed", error="不能覆盖完成")

    assert len(sink.events) == 2
    assert {event.run_id for event in sink.events} == {context.run_id}
    assert sink.events[0].payload["api_key"] == "<redacted>"
    assert sink.events[-1].type == "completed"


def test_cancellation_and_failure_or_completion_are_mutually_exclusive():
    context = RunContext(mode="react")
    context.request_cancel("测试取消")
    context.mark_terminal("cancelled", reason="测试取消")
    context.mark_terminal("completed")
    terminal = [event.type for event in context.event_list
                if event.type in {"cancelled", "failed", "completed"}]
    assert terminal == ["cancelled"]


def test_missing_token_usage_remains_none():
    collector = MetricsCollector("react")
    collector.record_llm_call(None)
    metrics = collector.finish()
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None


def test_eval_state_uses_workspace_without_mutating_global_data_dir(tmp_path):
    original = config.DATA_DIR
    workspace = run_eval.isolate_eval_state("contract")
    try:
        assert isinstance(workspace, TemporaryWorkspace)
        assert config.DATA_DIR == original
        assert workspace.root == (Path("evals") / "results" / "data_contract").resolve()
    finally:
        # isolate_eval_state 只创建对象，不会产生目录或运行数据。
        pass


def test_runtime_uses_explicit_workspace_for_all_persistent_tools(tmp_path):
    workspace = TemporaryWorkspace(tmp_path)
    runtime = create_runtime(
        demo=True,
        workspace=workspace,
        conversation=False,
    )
    assert runtime.memory.path == str(workspace.memory_path)
    assert runtime.registry.workspace is workspace
    assert runtime.artifacts.workspace is workspace
