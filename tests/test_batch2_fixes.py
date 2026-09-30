"""第二批修复(P1-4/P2-1/P2-2/P2-6/P2-8)的离线回归测试。

风格与已有测试一致:pytest 函数为主,不联网、不花钱。
"""

import time

import pytest

from scholaragent import config
from scholaragent.agent import Agent
from scholaragent.events import RunContext
from scholaragent.llm import ScriptedLLM
from scholaragent.metrics import MetricsCollector
from scholaragent.planner import Planner
from scholaragent.tool import (
    STOP_RETRY_PREFIX, Tool, ToolRegistry, ToolResult,
)
from scholaragent.workspace import TemporaryWorkspace


class FlakyTool(Tool):
    """永远失败的工具,用来验证熔断。"""
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def run(self):
        self.calls += 1
        return ToolResult(text="错误:上游服务挂了", success=False)


class SometimesOkTool(Tool):
    """按剧本成功/失败,用来验证熔断计数器清零。"""
    name = "sometimes"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self):
        self.calls += 1
        ok = self.outcomes.pop(0) if self.outcomes else True
        return ToolResult(text="ok" if ok else "错误:炸了", success=ok)


class EchoTool(Tool):
    name = "echo"

    def run(self, text=""):
        return text


def _tool_call_replies(name, count, arguments=None):
    return [
        {"content": None, "tool_calls": [
            {"id": f"call_{i}", "name": name,
             "arguments": dict(arguments or {})}]}
        for i in range(count)
    ]


# ―― P1-4:工具连续失败熔断 ――――――――――――――――――――――――――――――――――


def test_circuit_breaker_disables_tool_after_3_consecutive_failures():
    tool = FlakyTool()
    llm = ScriptedLLM(
        _tool_call_replies(tool.name, 6) + [{"content": "done", "tool_calls": []}])
    agent = Agent(llm, ToolRegistry([tool]), verbose=False, max_steps=10)
    assert agent.run("test fuse") == "done"
    # 只真实执行 3 次,第 4 次起走停用短路
    assert tool.calls == 3
    observations = [m for m in llm.last_messages if m.get("role") == "tool"]
    assert len(observations) == 6
    assert all(not o["content"].startswith(STOP_RETRY_PREFIX)
               for o in observations[:3])
    assert all(o["content"].startswith(STOP_RETRY_PREFIX)
               and "已停用" in o["content"] for o in observations[3:])
    # 停用短路同样记账(P2-1 联动)
    assert agent.last_metrics.tool_calls == 6


def test_circuit_breaker_resets_streak_on_success():
    tool = SometimesOkTool([False, False, True, False, False])
    llm = ScriptedLLM(
        _tool_call_replies(tool.name, 5) + [{"content": "done", "tool_calls": []}])
    agent = Agent(llm, ToolRegistry([tool]), verbose=False, max_steps=10)
    assert agent.run("test reset") == "done"
    # 失败,失败,成功(清零),失败,失败 —— 从未连续 3 次,不熔断
    assert tool.calls == 5
    observations = [m for m in llm.last_messages if m.get("role") == "tool"]
    assert not any("已停用" in o["content"] for o in observations)


def test_circuit_breaker_threshold_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CONSECUTIVE_FAILURE_LIMIT", 2)
    tool = FlakyTool()
    llm = ScriptedLLM(
        _tool_call_replies(tool.name, 4) + [{"content": "done", "tool_calls": []}])
    agent = Agent(llm, ToolRegistry([tool]), verbose=False, max_steps=10)
    assert agent.run("test threshold") == "done"
    assert tool.calls == 2


def test_circuit_breaker_config_default_and_env(monkeypatch):
    assert config.TOOL_CONSECUTIVE_FAILURE_LIMIT == 3
    monkeypatch.setenv("SCHOLARAGENT_TOOL_CONSECUTIVE_FAILURE_LIMIT", "5")
    assert config._positive_int(
        "SCHOLARAGENT_TOOL_CONSECUTIVE_FAILURE_LIMIT", 3) == 5


# ―― P2-1:短路分支记账统一 ――――――――――――――――――――――――――――――――――


def test_unknown_tool_call_is_counted(tmp_path):
    registry = ToolRegistry([])
    ctx = RunContext(mode="react", workspace=TemporaryWorkspace(str(tmp_path)))
    result = registry.call_result("ghost", {"x": 1}, context=ctx)
    assert not result.success
    assert result.diagnostic["kind"] == "unknown_tool"
    assert registry.run_tool_calls == 1
    assert ctx.metrics.finish().tool_calls == 1
    types = [e.type for e in ctx.event_list]
    assert "tool_started" in types
    assert "tool_completed" in types


def test_agent_counts_hallucinated_tool_calls(tmp_path):
    llm = ScriptedLLM(
        _tool_call_replies("ghost", 2) + [{"content": "done", "tool_calls": []}])
    agent = Agent(llm, ToolRegistry([]), verbose=False)
    assert agent.run("test") == "done"
    # 以前这里是 0(漏记),现在 2 次幻觉调用都要记账
    assert agent.last_metrics.tool_calls == 2


def test_invalid_arguments_short_circuit_is_counted():
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [
            {"id": "e1", "name": "echo", "arguments": {},
             "error": "工具参数不是合法 JSON"}]},
        {"content": "done", "tool_calls": []},
    ])
    agent = Agent(llm, ToolRegistry([EchoTool()]), verbose=False)
    assert agent.run("test") == "done"
    assert agent.last_metrics.tool_calls == 1


def test_tool_limit_short_circuits_are_counted():
    llm = ScriptedLLM(
        _tool_call_replies("echo", 3, {"text": "x"})
        + [{"content": "done", "tool_calls": []}])
    agent = Agent(llm, ToolRegistry([EchoTool()]), verbose=False,
                  tool_call_limits={"echo": 1})
    assert agent.run("test") == "done"
    # 1 次真实执行 + 1 次超限短路 + 1 次停用短路
    assert agent.last_metrics.tool_calls == 3


# ―― P2-2:重试成功的 usage 计入总量 ――――――――――――――――――――――――――――


def test_retry_success_records_usage_but_marks_incomplete():
    collector = MetricsCollector("react")
    collector.record_llm_call(
        {"prompt_tokens": 100, "completion_tokens": 20}, request_attempts=3)
    metrics = collector.finish()
    assert metrics.prompt_tokens == 100
    assert metrics.completion_tokens == 20
    assert metrics.request_attempts == 3
    assert metrics.llm_calls == 1
    # 早前失败 attempt 的 token 不可知,总量仍标记不完整
    assert metrics.token_accounting_complete is False


def test_single_attempt_keeps_complete_accounting():
    collector = MetricsCollector("react")
    collector.record_llm_call(
        {"prompt_tokens": 100, "completion_tokens": 20}, request_attempts=1)
    metrics = collector.finish()
    assert metrics.token_accounting_complete is True
    assert metrics.prompt_tokens == 100


def test_retry_without_usage_still_marks_missing():
    collector = MetricsCollector("react")
    collector.record_llm_call(None, request_attempts=2)
    metrics = collector.finish()
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.token_accounting_complete is False


# ―― P2-6:软超时下沉到 RunContext ――――――――――――――――――――――――――――


def test_soft_timeout_triggers_cancel_once():
    ctx = RunContext(mode="react", soft_timeout_seconds=0.05)
    assert not ctx.is_cancelled()
    time.sleep(0.09)
    assert ctx.is_cancelled()
    assert "软超时" in ctx.token.reason
    cancel_events = [e for e in ctx.event_list if e.type == "cancel_requested"]
    assert len(cancel_events) == 1
    # 只触发一次,重复查询不再补发事件
    assert ctx.is_cancelled()
    assert len([e for e in ctx.event_list
                if e.type == "cancel_requested"]) == 1


def test_soft_timeout_not_triggered_before_deadline():
    ctx = RunContext(mode="react", soft_timeout_seconds=60)
    assert not ctx.is_cancelled()
    assert ctx.soft_timeout_remaining() > 0


def test_soft_timeout_disabled_by_default():
    ctx = RunContext(mode="react")
    assert ctx.soft_timeout_seconds is None
    assert ctx.soft_timeout_remaining() is None
    assert not ctx.is_cancelled()


@pytest.mark.parametrize("bad_value", [0, -5, "not-a-number", None])
def test_soft_timeout_bad_values_are_ignored(bad_value):
    ctx = RunContext(mode="react", soft_timeout_seconds=bad_value)
    assert ctx.soft_timeout_seconds is None
    assert not ctx.is_cancelled()


def test_execute_runners_forwards_soft_timeout(tmp_path):
    from scholaragent.artifacts import ArtifactCollector
    from scholaragent.runtime import execute_runners

    seen = {}

    class FakeRunner:
        def run(self, task, context=None):
            seen["soft_timeout"] = context.soft_timeout_seconds
            return "ok"

    workspace = TemporaryWorkspace(str(tmp_path))
    result = execute_runners(
        "hi", "react", {"react": FakeRunner()}, router=None,
        workspace=workspace, artifacts=ArtifactCollector(workspace),
        soft_timeout_seconds=12,
    )
    assert seen["soft_timeout"] == 12
    assert result.status == "completed"


def test_execute_runners_soft_timeout_cancels_long_run(tmp_path):
    from scholaragent.artifacts import ArtifactCollector
    from scholaragent.runtime import execute_runners

    class SlowRunner:
        def run(self, task, context=None):
            while not context.is_cancelled():
                time.sleep(0.01)
            return "unfinished"

    workspace = TemporaryWorkspace(str(tmp_path))
    result = execute_runners(
        "hi", "react", {"react": SlowRunner()}, router=None,
        workspace=workspace, artifacts=ArtifactCollector(workspace),
        soft_timeout_seconds=0.05,
    )
    assert result.status == "cancelled"


# ―― P2-8:反思解析连续失败不再静默放行 ―――――――――――――――――――――――


class FakeStepAgent:
    def run(self, task, context=None):
        return "步骤完成"


def _planner(summary_replies):
    return Planner(
        llm=ScriptedLLM([]),
        agent=FakeStepAgent(),
        summary_llm=ScriptedLLM(summary_replies),
        verbose=False,
    )


def test_reflect_single_parse_failure_passes_without_issue():
    planner = _planner([{"content": "我觉得不太行", "tool_calls": []}])
    ctx = RunContext(mode="plan")
    assert planner._reflect("步骤", "结果", ctx) == (True, "")
    assert ctx.completion_issues == []


def test_reflect_two_consecutive_failures_recorded_once():
    planner = _planner([
        {"content": "不行", "tool_calls": []},
        {"content": "还是不行", "tool_calls": []},
        {"content": "依然不行", "tool_calls": []},
    ])
    ctx = RunContext(mode="plan")
    assert planner._reflect("s", "r", ctx) == (True, "")
    assert ctx.completion_issues == []
    assert planner._reflect("s", "r", ctx) == (True, "")
    assert len(ctx.completion_issues) == 1
    assert ctx.completion_issues[0]["reason"] == "reflect_parse_failed"
    # 每个连续失败 streak 只记一次,不刷屏
    assert planner._reflect("s", "r", ctx) == (True, "")
    assert len(ctx.completion_issues) == 1


def test_reflect_success_resets_failure_streak():
    planner = _planner([
        {"content": "不行", "tool_calls": []},
        {"content": '{"ok": true}', "tool_calls": []},
        {"content": "不行", "tool_calls": []},
    ])
    ctx = RunContext(mode="plan")
    planner._reflect("s", "r", ctx)
    ok, _ = planner._reflect("s", "r", ctx)
    assert ok is True
    planner._reflect("s", "r", ctx)
    assert ctx.completion_issues == []


def test_reflect_failure_kinds_all_counted():
    planner = _planner([
        {"content": "纯文字无 JSON", "tool_calls": []},
        {"content": "{不是合法 json", "tool_calls": []},
    ])
    ctx = RunContext(mode="plan")
    planner._reflect("s", "r", ctx)
    planner._reflect("s", "r", ctx)
    assert len(ctx.completion_issues) == 1
    assert ctx.completion_issues[0]["reason"] == "reflect_parse_failed"


def test_run_resets_reflect_failure_counter():
    planner = _planner([{"content": "胡言乱语", "tool_calls": []}])
    planner._reflect_parse_failures = 5  # 模拟上一轮的残留
    plan_llm_replies = [{"content": '["做一件事"]', "tool_calls": []}]
    planner.llm = ScriptedLLM(plan_llm_replies)
    answer = planner.run("测试任务")
    assert answer == "步骤完成"
    # run() 开始时重置,本轮 1 次解析失败
    assert planner._reflect_parse_failures == 1
