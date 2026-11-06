"""兼容把标准响应包在 {"data": ...} 里的 OpenAI 兼容网关(离线)。"""

from types import SimpleNamespace
from unittest.mock import Mock

from openai.types.chat import ChatCompletion

from scholaragent import llm

MESSAGES = [{"role": "user", "content": "算 365*24"}]

COMPLETION = {
    "id": "gen_1",
    "object": "chat.completion",
    "created": 1,
    "model": "gateway/model",
    "choices": [{
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning": "需要调用计算器",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "calculator", "arguments": "{\"expression\":\"365*24\"}"},
            }],
        },
    }],
    "usage": {
        "prompt_tokens": 293,
        "completion_tokens": 54,
        "total_tokens": 347,
        "prompt_tokens_details": {"cached_tokens": 200},
        "cost": 0.0001527,
    },
}


def enveloped():
    """模拟 SDK 解析网关响应后的形态:顶层字段全为 None,内容在 data 里。"""
    return ChatCompletion.model_construct(choices=None, data=COMPLETION)


def client_returning(*responses):
    client = llm.LLMClient(model="gateway/model", api_key="test", max_attempts=2)
    create = Mock(side_effect=list(responses))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._sleep_backoff = Mock()
    return client, create


def test_enveloped_response_is_unwrapped_with_tool_calls_and_usage():
    client, create = client_returning(enveloped())
    reply = client.chat(MESSAGES, tools=[{"type": "function"}])
    assert create.call_count == 1
    assert reply["content"] is None
    assert reply["tool_calls"] == [{
        "id": "call-1", "name": "calculator",
        "arguments": {"expression": "365*24"}, "error": None,
    }]
    assert reply["usage"] == {
        "prompt_tokens": 293, "completion_tokens": 54,
        "prompt_cache_hit_tokens": 200, "prompt_cache_miss_tokens": 93,
        "cost": 0.0001527,
    }


def test_standard_response_is_returned_unchanged():
    standard = ChatCompletion.model_validate(COMPLETION)
    assert llm._unwrap_data_envelope(standard) is standard


def test_envelope_without_choices_still_counts_as_empty_and_retries():
    empty = ChatCompletion.model_construct(choices=None, data={"choices": []})
    client, create = client_returning(empty, enveloped())
    assert client.chat(MESSAGES)["tool_calls"][0]["name"] == "calculator"
    assert create.call_count == 2


def test_malformed_envelope_is_left_for_the_empty_choices_guard():
    broken = ChatCompletion.model_construct(choices=None, data={"choices": [{"bogus": 1}]})
    assert llm._unwrap_data_envelope(broken) is broken


def test_usage_without_cost_reports_none():
    plain = dict(COMPLETION, usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6})
    client, _ = client_returning(ChatCompletion.model_validate(plain))
    assert client.chat(MESSAGES)["usage"]["cost"] is None
