"""阶段 A：运行指标与路由数据接口的离线测试。"""

from scholaragent.agent import Agent
from scholaragent.llm import ScriptedLLM
from scholaragent.metrics import MetricsCollector
from scholaragent.routing import (
    FEATURE_VERSION,
    POLICY_FORMAT_VERSION,
    ROUTING_MODES,
    RoutingDecision,
    empty_policy_document,
)
from scholaragent.tool import Tool, ToolRegistry


class EchoTool(Tool):
    name = "echo"

    def run(self, text=""):
        return text


def test_agent_records_real_call_counts_and_missing_tokens():
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "echo", "arguments": {"text": "ok"}},
        ]},
        {"content": "done", "tool_calls": []},
    ])
    agent = Agent(llm, ToolRegistry([EchoTool()]), verbose=False)

    assert agent.run("test") == "done"
    metrics = agent.last_metrics
    assert metrics.mode == "react"
    assert metrics.llm_calls == 2
    assert metrics.tool_calls == 1
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.seconds >= 0


def test_metrics_collector_keeps_complete_api_usage():
    collector = MetricsCollector("plan")
    collector.record_llm_call({"prompt_tokens": 12, "completion_tokens": 3})
    collector.record_llm_call({"prompt_tokens": 8, "completion_tokens": 7})

    metrics = collector.finish(tool_calls=4)
    assert metrics.to_dict()["prompt_tokens"] == 20
    assert metrics.completion_tokens == 10
    assert metrics.llm_calls == 2
    assert metrics.tool_calls == 4


def test_metrics_break_down_token_usage_by_model_role():
    collector = MetricsCollector("plan")
    collector.record_llm_call(
        {"prompt_tokens": 12, "completion_tokens": 3},
        role="research", provider="cloud", model="cloud-model",
    )
    collector.record_llm_call(
        {"prompt_tokens": 8, "completion_tokens": 7},
        role="summary", provider="ollama", model="local-model",
    )

    by_role = collector.finish().llm_usage_by_role
    assert by_role["research"] == {
        "provider": "cloud", "model": "cloud-model", "llm_calls": 1,
        "prompt_tokens": 12, "completion_tokens": 3,
    }
    assert by_role["summary"]["provider"] == "ollama"
    assert by_role["summary"]["prompt_tokens"] == 8


def test_routing_interfaces_have_versioned_policy_shape():
    decision = RoutingDecision(
        mode="react", predicted_utility={"react": 0.5}, features={"bias": 1.0},
        reason="冷启动", policy_version=FEATURE_VERSION,
    )
    assert decision.to_dict()["mode"] == "react"

    policy = empty_policy_document()
    assert policy["format_version"] == POLICY_FORMAT_VERSION
    assert policy["feature_version"] == FEATURE_VERSION
    assert tuple(policy["modes"]) == ROUTING_MODES
