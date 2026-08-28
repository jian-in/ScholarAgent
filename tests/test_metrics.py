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
        # 上游未返回缓存明细时如实为 None
        "cache_hit_tokens": None, "cache_miss_tokens": None,
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


def test_metrics_aggregate_cache_hit_and_tolerate_missing():
    """缓存明细应逐次累加;上游不区分缓存时如实记 None,不估算。"""
    collector = MetricsCollector("react")
    collector.record_llm_call({
        "prompt_tokens": 100, "completion_tokens": 10,
        "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40,
    })
    collector.record_llm_call({
        "prompt_tokens": 200, "completion_tokens": 20,
        "prompt_cache_hit_tokens": 160, "prompt_cache_miss_tokens": 40,
    })
    metrics = collector.finish()
    assert metrics.cache_hit_tokens == 220
    assert metrics.cache_miss_tokens == 80

    other = MetricsCollector("react")
    other.record_llm_call({"prompt_tokens": 50, "completion_tokens": 5})
    snapshot = other.finish()
    assert snapshot.cache_hit_tokens is None
    assert snapshot.cache_miss_tokens is None


def test_llm_client_extracts_cache_fields_from_usage():
    """模型层应把 DeepSeek 风格的缓存明细透传进 usage。"""
    from scholaragent import llm as llm_module

    class FakeUsage:
        prompt_tokens = 100
        completion_tokens = 8
        prompt_cache_hit_tokens = 64
        prompt_cache_miss_tokens = 36

    class FakeMessage:
        content = "ok"
        tool_calls = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    client = llm_module.LLMClient.__new__(llm_module.LLMClient)
    client.model = "test"
    client._client = type("C", (), {})()
    client._client.chat = type(
        "Chat", (), {"completions": type("Comp", (), {"create": None})})()
    client._client.chat.completions = FakeCompletions()

    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply["usage"]["prompt_cache_hit_tokens"] == 64
    assert reply["usage"]["prompt_cache_miss_tokens"] == 36


def test_llm_client_derives_cache_miss_for_openai_style_usage():
    """OpenAI 兼容实现只给 prompt_tokens_details.cached_tokens:
    应作 fallback 提取,miss 由 prompt-hit 推导(两家语义一致:
    prompt_tokens 均包含命中部分)。
    """
    from scholaragent import llm as llm_module

    class FakeDetails:
        cached_tokens = 64

    class FakeUsage:
        prompt_tokens = 100
        completion_tokens = 8
        prompt_tokens_details = FakeDetails()

    class FakeMessage:
        content = "ok"
        tool_calls = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    client = llm_module.LLMClient.__new__(llm_module.LLMClient)
    client.model = "test"
    client._client = type("C", (), {})()
    client._client.chat = type(
        "Chat", (), {"completions": type("Comp", (), {"create": None})})()
    client._client.chat.completions = FakeCompletions()

    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply["usage"]["prompt_cache_hit_tokens"] == 64
    assert reply["usage"]["prompt_cache_miss_tokens"] == 36
