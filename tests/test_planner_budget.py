"""离线回归:步骤预算叠加及临时 Agent 配置的恢复。"""

import pytest

from scholaragent.agent import CANCELLED_ANSWER, JobCancelled
from scholaragent.events import RunContext
from scholaragent.gap_survey import FAST_GAP_MAX_STEPS, FAST_GAP_TOOL_NAMES
from scholaragent.llm import ScriptedLLM
from scholaragent.planner import Planner
from scholaragent.workspace import TemporaryWorkspace


FAST_TASK = (
    "补齐四项资料缺口:推理能力 CoT、ToT; Agent 工具使用;"
    "2024-2025 最新进展系统性综述;硬件与推理效率优化。"
    "全部开始调研,快速摘要筛选,不下载 PDF。"
)


class Registry:
    def __init__(self, names=("arxiv_search", "download_paper", "recall"), error=None):
        self.names = tuple(names)
        self.error = error

    def subset(self, names):
        if self.error is not None:
            raise self.error
        return Registry(names)


class RecordingAgent:
    def __init__(self, max_steps=10, error=None):
        self.max_steps = max_steps
        self.tools = Registry()
        self.error = error
        self.seen = []

    def run(self, prompt, context=None):
        self.seen.append((self.max_steps, self.tools.names, prompt, context))
        if self.error is not None:
            raise self.error
        if context is not None and context.is_cancelled():
            return CANCELLED_ANSWER
        return "执行结果"


@pytest.mark.parametrize("max_steps", [1, 2, 5, 10, 15])
@pytest.mark.parametrize("fast_gap", [False, True])
@pytest.mark.parametrize("is_retry", [False, True])
@pytest.mark.parametrize("use_context", [False, True])
def test_step_budget_composes_caps_and_restores_configuration(
        tmp_path, max_steps, fast_gap, is_retry, use_context):
    agent = RecordingAgent(max_steps)
    original_tools = agent.tools
    planner = Planner(None, agent, verbose=False)
    task = FAST_TASK if fast_gap else "普通任务"
    context = RunContext(workspace=TemporaryWorkspace(tmp_path)) if use_context else None

    assert planner._execute_step(
        task, ["检索资料"], [], 1, "补充证据", context=context, is_retry=is_retry,
    ) == "执行结果"

    expected = min(max_steps, FAST_GAP_MAX_STEPS) if fast_gap else max_steps
    if is_retry:
        expected = max(1, expected // 2)
    cap, tools, prompt, seen_context = agent.seen[0]
    assert cap == expected
    assert tools == (tuple(FAST_GAP_TOOL_NAMES) if fast_gap else original_tools.names)
    assert "补充证据" in prompt
    assert seen_context is context
    assert agent.max_steps == max_steps
    assert agent.tools is original_tools


def test_planner_retry_uses_half_budget_and_next_step_uses_original_budget():
    llm = ScriptedLLM([
        {"content": '["检索资料", "整理证据"]', "tool_calls": []},
        {"content": '{"ok": false, "advice": "补充出处"}', "tool_calls": []},
        {"content": '{"ok": true}', "tool_calls": []},
        {"content": "最终综述", "tool_calls": []},
    ])
    agent = RecordingAgent(15)
    original_tools = agent.tools
    planner = Planner(llm, agent, verbose=False)

    assert planner.run("调研某方向") == "最终综述"
    assert [seen[0] for seen in agent.seen] == [15, 7, 15]
    assert "补充出处" in agent.seen[1][2]
    assert agent.max_steps == 15
    assert agent.tools is original_tools
    assert planner._active_context is None


@pytest.mark.parametrize("task", ["普通任务", FAST_TASK])
@pytest.mark.parametrize("use_context", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, JobCancelled])
def test_step_restores_configuration_after_exception(tmp_path, task, use_context, error_type):
    error = error_type("执行中断")
    agent = RecordingAgent(15, error=error)
    original_tools = agent.tools
    context = RunContext(workspace=TemporaryWorkspace(tmp_path)) if use_context else None

    with pytest.raises(error_type) as caught:
        Planner(None, agent)._execute_step(
            task, ["检索资料"], [], 1, "", context=context, is_retry=True,
        )

    assert caught.value is error
    assert agent.max_steps == 15
    assert agent.tools is original_tools


def test_step_restores_budget_when_tool_subset_raises():
    error = RuntimeError("筛选工具失败")
    agent = RecordingAgent(15)
    agent.tools = Registry(error=error)
    original_tools = agent.tools

    with pytest.raises(RuntimeError) as caught:
        Planner(None, agent)._execute_step(FAST_TASK, ["检索资料"], [], 1, "")

    assert caught.value is error
    assert agent.max_steps == 15
    assert agent.tools is original_tools
    assert agent.seen == []


@pytest.mark.parametrize("task", ["普通任务", FAST_TASK])
def test_step_restores_configuration_after_context_cancellation(tmp_path, task):
    context = RunContext(workspace=TemporaryWorkspace(tmp_path))
    context.request_cancel()
    agent = RecordingAgent(15)
    original_tools = agent.tools

    assert Planner(None, agent)._execute_step(
        task, ["检索资料"], [], 1, "", context=context, is_retry=True,
    ) == CANCELLED_ANSWER
    assert agent.max_steps == 15
    assert agent.tools is original_tools


def test_retry_does_not_add_tools_attribute_to_legacy_runner():
    class LegacyRunner:
        max_steps = 5

        def run(self, prompt):
            assert self.max_steps == 2
            return "旧版结果"

    agent = LegacyRunner()
    assert Planner(None, agent)._execute_step(
        "普通任务", ["检索资料"], [], 1, "", is_retry=True,
    ) == "旧版结果"
    assert agent.max_steps == 5
    assert not hasattr(agent, "tools")


@pytest.mark.parametrize("task", ["普通任务", FAST_TASK])
def test_retry_supports_legacy_runner_without_step_budget(task):
    class LegacyRunner:
        def run(self, prompt):
            return "旧版结果"

    agent = LegacyRunner()
    assert Planner(None, agent)._execute_step(
        task, ["检索资料"], [], 1, "", is_retry=True,
    ) == "旧版结果"
    assert vars(agent) == {}
