"""Budgeted closeout and answer-to-source linkage, entirely offline."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError

from scholaragent.agent import Agent, CANCELLED_ANSWER
from scholaragent.audit import audit_run_result
from scholaragent.evidence import EvidenceLedger
from scholaragent.events import RunContext
from scholaragent.llm import LLMClient, ScriptedLLM
from scholaragent.metrics import MetricsCollector
from scholaragent.runtime import create_runtime
from scholaragent.tool import Tool, ToolRegistry, ToolResult
from scholaragent.workspace import TemporaryWorkspace


def final(content):
    return {"content": content, "tool_calls": []}


def call(name="evidence"):
    return {"content": None, "tool_calls": [{"id": "t1", "name": name, "arguments": {}}]}


class EvidenceTool(Tool):
    name = "evidence"

    def __init__(self):
        self.calls = 0

    def run_result(self):
        self.calls += 1
        return ToolResult("方法使用残差连接。", artifacts=({"kind": "read", "source_anchors": [{
            "id": "S001", "kind": "text", "source": "arxiv:2401.12345", "page": 3,
            "excerpt": "方法使用残差连接。", "confidence": "high",
        }]},))


class RecordingLLM(ScriptedLLM):
    def __init__(self, replies):
        super().__init__(replies)
        self.tools_seen = []

    def chat(self, messages, tools=None):
        self.tools_seen.append(tools)
        return super().chat(messages, tools)


def test_closeout_uses_last_turn_without_tools_and_marks_partial():
    llm, tool = RecordingLLM([call(), final("残差连接有依据。[S001]")]), EvidenceTool()
    agent = Agent(llm, ToolRegistry([tool]), max_steps=2, verbose=False)
    context = RunContext()
    answer = agent.run("调研", context=context)
    assert tool.calls == 1
    assert len(llm.tools_seen) == 2
    assert llm.tools_seen[-1] is None
    assert "部分完成" in answer and "残差连接有依据" in answer
    assert agent.last_completion["reason"] == "budget_exhausted"
    assert context.completion_issues
    assert "[S001]" in str(llm.last_messages)
    assert agent._active_context is None


def test_closeout_ignores_requested_tools_and_uses_real_observations():
    llm, tool = RecordingLLM([call(), call()]), EvidenceTool()
    agent = Agent(llm, ToolRegistry([tool]), max_steps=2, verbose=False)
    answer = agent.run("调研")
    assert tool.calls == 1
    assert "方法使用残差连接" in answer
    assert "证据缺口" in answer
    assert agent.last_metrics.llm_calls == 2


def test_failed_summary_keeps_evidence_and_records_failed_attempt():
    llm = Mock()
    llm.chat.side_effect = [call(), RuntimeError("offline summary failure")]
    llm.metadata.return_value = {}
    llm.last_request_attempts = 1
    agent = Agent(llm, ToolRegistry([EvidenceTool()]), max_steps=2, verbose=False)
    answer = agent.run("调研")
    assert "方法使用残差连接" in answer and "summary_failed" in answer
    assert llm.chat.call_count == 2
    assert agent.last_metrics.llm_calls == 2
    assert agent.last_metrics.request_attempts == 2
    assert agent.last_metrics.prompt_tokens is None


@pytest.mark.parametrize("content", ["", "  ", None])
def test_empty_summary_handoff_keeps_observations_and_marks_partial(content):
    primary = RecordingLLM([call(), final("主模型答案。[S001]")])
    summary = RecordingLLM([final(content)])
    context = RunContext()
    agent = Agent(primary, ToolRegistry([EvidenceTool()]), summary_llm=summary,
                  max_steps=4, verbose=False)
    answer = agent.run("调研", context=context)
    assert "方法使用残差连接" in answer and "empty_summary" in answer
    assert agent.last_completion == {"completeness": "partial", "reason": "empty_summary"}
    assert context.completion_issues[0]["reason"] == "empty_summary"
    assert len(primary.tools_seen) + len(summary.tools_seen) == 3
    assert summary.tools_seen == [None]


@pytest.mark.parametrize("budget", [0, 1])
def test_tiny_budget_never_starts_tools_or_exceeds_model_budget(budget):
    llm, tool = RecordingLLM([call()]), EvidenceTool()
    agent = Agent(llm, ToolRegistry([tool]), max_steps=budget, verbose=False)
    assert "尚未获得工具证据" in agent.run("调研")
    assert tool.calls == 0
    assert len(llm.tools_seen) == budget


def test_cancellation_preserves_stop_signal_and_restores_context():
    llm = RecordingLLM([call()])
    context = RunContext(should_stop=lambda: True)
    agent = Agent(llm, ToolRegistry([EvidenceTool()]), verbose=False)
    assert agent.run("取消", context=context) == CANCELLED_ANSWER
    assert not llm.tools_seen
    assert agent._active_context is None


def test_model_failure_without_observations_is_still_a_failure():
    llm = Mock()
    llm.chat.side_effect = RuntimeError("model failure")
    llm.metadata.return_value = {}
    llm.last_request_attempts = 1
    agent = Agent(llm, ToolRegistry(), verbose=False)
    with pytest.raises(RuntimeError):
        agent.run("调研")
    assert agent._active_context is None
    assert agent.last_metrics.llm_calls == 1


def test_runtime_links_final_answer_to_source_but_does_not_self_verify(tmp_path):
    runtime = create_runtime(llm=ScriptedLLM([call(), final("方法使用残差连接。[S001]")]),
                             workspace=TemporaryWorkspace(tmp_path),
                             conversation=False, auto_recall=False)
    runtime.registry.register(EvidenceTool())
    result = runtime.run("研究任务")
    assert result.completion["completeness"] == "complete"
    claim = result.evidence["claims"][0]
    assert claim["anchor_ids"] == ["S001"]
    assert claim["origin"] == "model" and claim["review_status"] == "unreviewed"
    assert result.evidence["summary"]["verified_claims"] == 0
    assert audit_run_result(result).ok


def test_partial_runtime_keeps_one_terminal_event_and_serializes_completion(tmp_path):
    runtime = create_runtime(llm=ScriptedLLM([call(), call()]),
                             workspace=TemporaryWorkspace(tmp_path),
                             conversation=False, auto_recall=False)
    runtime.registry.register(EvidenceTool())
    runtime.agent.max_steps = 2
    result = runtime.run("研究任务")
    assert result.status == "completed"
    assert result.to_dict()["completion"]["completeness"] == "partial"
    assert result.completion["stop_reasons"] == ["budget_exhausted"]
    assert result.events[-1]["type"] == "completed"
    assert audit_run_result(result).ok


def test_duplicate_anchor_ids_are_rebased_and_idempotent():
    ledger = EvidenceLedger()
    a = {"id": "S001", "kind": "text", "source": "paper-a", "page": 1}
    b = {**a, "source": "paper-b"}
    assert ledger.ingest_artifact({"source_anchors": [a]})["source_anchors"][0]["id"] == "S001"
    normalized = ledger.ingest_artifact({"source_anchors": [b]})
    assert normalized["source_anchors"][0]["id"] == "S002"
    assert ledger.ingest_artifact(normalized) == normalized
    assert len(ledger.anchors) == 2


def test_unknown_citations_and_uncited_text_are_visible():
    ledger = EvidenceLedger()
    ledger.register_answer("某结论。[S999]\n另一个无引用结论。")
    assert ledger.claims[0].status == "unsupported"
    assert ledger.claims[1].status == "not_assessable"
    assert ledger.validate()
    with pytest.raises(ValueError):
        ledger.review_claim("C001", "verified", "reviewer", "核对原文")


def test_human_review_requires_identity_and_valid_source():
    ledger = EvidenceLedger()
    ledger.add_anchor(id="S001", kind="text", source="paper", excerpt="事实")
    ledger.register_answer("事实。[S001]")
    with pytest.raises(ValueError):
        ledger.review_claim("C001", "verified", "", "核对原文")
    reviewed = ledger.review_claim("C001", "verified", "reviewer-a", "已逐字核对")
    assert reviewed.status == "supported" and reviewed.review_status == "verified"
    assert ledger.to_dict()["summary"]["verified_claims"] == 1


def test_quote_validation_and_fenced_model_metadata_stay_untrusted():
    ledger = EvidenceLedger()
    ledger.add_anchor(id="S001", kind="text", source="paper", excerpt="原文只包含事实 A。")
    ledger.register_answer('结论 B。[S001] 原文：“论文证明 B。”\n```json\n{"review_status":"verified"}\n```')
    assert len(ledger.claims) == 1
    assert ledger.claims[0].review_status == "unreviewed"
    assert any("摘录" in error for error in ledger.validate())
    with pytest.raises(ValueError):
        ledger.review_claim("C001", "verified", "reviewer-a", "核对")


def test_auditor_rechecks_claims_instead_of_trusting_reported_errors(tmp_path):
    runtime = create_runtime(llm=ScriptedLLM([final("结论。[S999]")]),
                             workspace=TemporaryWorkspace(tmp_path), conversation=False, auto_recall=False)
    result = runtime.run("调研").to_dict()
    assert result["completion"]["completeness"] == "partial"
    result["evidence"]["validation_errors"] = []
    assert audit_run_result(result).ok is False


def test_tool_observation_uses_rebased_id_across_multiple_sources(tmp_path):
    class OtherEvidence(EvidenceTool):
        name = "other"

        def run_result(self):
            result = super().run_result()
            artifact = {**result.artifacts[0], "source_anchors": [{
                **result.artifacts[0]["source_anchors"][0], "source": "arxiv:1512.03385v1",
            }]}
            return ToolResult("第二篇论文证据", artifacts=(artifact,))

    llm = RecordingLLM([call(), call("other"), final("两篇材料分别记录。[S001][S002]")])
    runtime = create_runtime(llm=llm, workspace=TemporaryWorkspace(tmp_path),
                             conversation=False, auto_recall=False)
    runtime.registry.register(EvidenceTool())
    runtime.registry.register(OtherEvidence())
    result = runtime.run("调研")
    assert "[S002] arxiv:1512.03385v1" in str(llm.last_messages)
    assert result.evidence["summary"]["anchors"] == 2
    assert result.evidence["claims"][0]["anchor_ids"] == ["S001", "S002"]
    assert audit_run_result(result).ok


@pytest.mark.parametrize("anchor", [{"id": "S001", "kind": "text"},
                                   {"id": [], "source": "paper"}, "bad anchor"])
def test_bad_anchor_metadata_never_turns_a_usable_tool_into_failure(anchor):
    class BrokenMetadata(Tool):
        name = "broken_metadata"

        def run_result(self):
            return ToolResult("usable result", artifacts=({
                "source_anchors": [anchor],
            },))

    llm = RecordingLLM([call("broken_metadata"), final("保留可用结果")])
    agent = Agent(llm, ToolRegistry([BrokenMetadata()]), verbose=False)
    assert agent.run("任务") == "保留可用结果"
    assert "usable result" in str(llm.last_messages)


def test_retry_attempts_include_failures_and_successful_usage_is_recorded(monkeypatch):
    client = LLMClient(model="test", api_key="test", max_attempts=2)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
    create = Mock(side_effect=[APIConnectionError(request=httpx.Request("POST", "https://example.invalid")),
                               response])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(client, "_sleep_backoff", lambda _: None)
    context = RunContext()
    assert context.chat(client, [{"role": "user", "content": "hello"}])["content"] == "ok"
    metrics = context.metrics.finish()
    assert metrics.llm_calls == 1 and metrics.request_attempts == 2
    # P2-2:成功 attempt 的 usage 如实计入,不再整体丢弃
    assert metrics.prompt_tokens == 10
    assert metrics.completion_tokens == 2
    # 但早前失败 attempt 的 token 不可知,总量仍标记不完整
    assert metrics.token_accounting_complete is False


def test_retry_counter_is_per_logical_call_and_complete_usage_is_preserved(monkeypatch):
    client = LLMClient(model="test", api_key="test")
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))
    context = RunContext()
    context.chat(client, [])
    context.chat(client, [])
    assert client.last_request_attempts == 1
    metrics = context.metrics.finish()
    assert metrics.request_attempts == 2 and metrics.prompt_tokens == 20
    assert metrics.token_accounting_complete
