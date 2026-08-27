"""统一运行上下文的离线指标契约。"""

from pathlib import Path

from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import create_runtime
from scholaragent.workspace import TemporaryWorkspace


def final(text):
    return {"content": text, "tool_calls": []}


def test_react_plan_team_and_auto_each_have_one_metrics_owner(tmp_path):
    cases = {
        "react": [final("react answer")],
        "plan": [
            final('["第一步", "第二步"]'),
            final("结果一"),
            final('{"ok": true}'),
            final("结果二"),
            final('{"ok": true}'),
            final("plan answer"),
        ],
        "team": [final("检索报告"), final("精读笔记"), final("team answer")],
        "auto": [final("auto answer")],
    }

    for mode, replies in cases.items():
        runtime = create_runtime(
            llm=ScriptedLLM(replies),
            workspace=TemporaryWorkspace(tmp_path / mode),
            conversation=False,
            auto_recall=False,
            team_require_full_paper=False,
        )
        result = runtime.run("一个离线任务", mode=mode)
        assert result.status == "completed"
        assert result.metrics.llm_calls == (6 if mode == "plan" else 3 if mode == "team" else 1)
        assert result.metrics.tool_calls == 0
        assert result.metrics.prompt_tokens is None
        assert result.metrics.completion_tokens is None
        assert result.events[-1]["type"] == "completed"


def test_failed_and_cancelled_runs_still_return_terminal_metrics(tmp_path):
    failed = create_runtime(
        llm=ScriptedLLM([]),
        workspace=TemporaryWorkspace(tmp_path / "failed"),
        conversation=False,
        auto_recall=False,
    )
    failed.runners["react"] = object()  # 通过统一入口观察失败，而非入口自建指标
    result = failed.run("失败任务", mode="react")
    assert result.status == "failed"
    assert result.metrics.llm_calls == 0
    assert result.events[-1]["type"] == "failed"

    cancel = create_runtime(
        llm=ScriptedLLM([final("不会被调用")]),
        workspace=TemporaryWorkspace(tmp_path / "cancelled"),
        conversation=False,
        auto_recall=False,
    )
    result = cancel.run("取消任务", mode="react", should_stop=lambda: True)
    assert result.status == "cancelled"
    assert result.metrics.llm_calls == 0
    assert result.events[-1]["type"] == "cancelled"


def test_callback_event_adapter_and_run_result_share_the_same_event_stream(tmp_path):
    received = []
    runtime = create_runtime(
        llm=ScriptedLLM([final("回答")]),
        workspace=TemporaryWorkspace(tmp_path / "callback"),
        conversation=False,
        auto_recall=False,
    )
    result = runtime.run("任务", mode="react", event_sink=received.append)

    assert [event.type for event in received] == [event["type"] for event in result.events]
    assert result.events[-1]["type"] == "completed"


def test_runtime_wires_research_and_summary_models_to_their_roles(tmp_path):
    """统一运行时应把云端调研模型和本地总结模型接到正确的执行层。"""
    research = ScriptedLLM([])
    summary = ScriptedLLM([])
    runtime = create_runtime(
        research_llm=research,
        summary_llm=summary,
        workspace=TemporaryWorkspace(tmp_path / "split-models"),
        conversation=False,
        auto_recall=False,
    )

    assert runtime.agent.llm is research
    assert runtime.agent.summary_llm is summary
    assert runtime.runners["plan"].llm is research
    assert runtime.runners["plan"].summary_llm is summary
    assert runtime.runners["team"].searcher.llm is research
    assert runtime.runners["team"].writer.llm is summary


def test_model_step_event_exposes_safe_model_role_metadata(tmp_path):
    """轨迹应显示模型角色，但不能携带 API Key 等秘密。"""
    research = ScriptedLLM(
        [final("回答")],
        model="cloud-research",
        provider="cloud",
        role="research",
    )
    runtime = create_runtime(
        llm=research,
        workspace=TemporaryWorkspace(tmp_path / "model-event"),
        conversation=False,
        auto_recall=False,
    )

    result = runtime.run("任务", mode="react")
    model_event = next(event for event in result.events if event["type"] == "model_step")
    assert model_event["payload"]["step"] == 1
    assert model_event["payload"]["role"] == "research"
    assert model_event["payload"]["provider"] == "cloud"
    assert model_event["payload"]["model"] == "cloud-research"
    assert "api_key" not in str(model_event).lower()
    assert result.metrics.llm_usage_by_role["research"]["llm_calls"] == 1
    assert result.metrics.llm_usage_by_role["research"]["prompt_tokens"] is None
