"""Offline regression coverage for one shared LLM request retry budget."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError

from scholaragent import llm


MESSAGES = [{"role": "user", "content": "hello"}]


def response(choices=True):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
        if choices else [],
        usage=None,
    )


def status_error(status):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    error_type = RateLimitError if status == 429 else APIStatusError
    return error_type("test error", response=httpx.Response(status, request=request), body=None)


def client_with_results(monkeypatch, results, max_attempts=3):
    client = llm.LLMClient(model="test", api_key="test", max_attempts=max_attempts)
    create = Mock(side_effect=results)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sleep = Mock()
    monkeypatch.setattr(client, "_sleep_backoff", sleep)
    return client, create, sleep


def test_sdk_is_lazy_and_receives_timeout_without_builtin_retries(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(llm, "OpenAI", factory)
    client = llm.LLMClient(base_url="https://example.invalid/v1", api_key="test", timeout=7.5)
    factory.assert_not_called()
    assert client._client_instance() is client._client_instance()
    factory.assert_called_once_with(base_url="https://example.invalid/v1", api_key="test",
                                    max_retries=0, timeout=7.5)


def test_default_timeout_and_attempts():
    client = llm.LLMClient(api_key="test")
    assert client._timeout == 60.0
    assert client._max_attempts == 3


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_status_retries_then_succeeds(monkeypatch, status):
    client, create, sleep = client_with_results(monkeypatch, [status_error(status), response()])
    assert client.chat(MESSAGES)["content"] == "ok"
    assert create.call_count == 2
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError])
def test_connection_errors_retry(monkeypatch, error_type):
    error = error_type(request=httpx.Request("POST", "https://example.invalid"))
    client, create, sleep = client_with_results(monkeypatch, [error, response()])
    assert client.chat(MESSAGES)["content"] == "ok"
    assert create.call_count == 2
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_status_is_raised_without_retry(monkeypatch, status):
    error = status_error(status)
    client, create, sleep = client_with_results(monkeypatch, [error])
    with pytest.raises(APIStatusError) as caught:
        client.chat(MESSAGES)
    assert caught.value is error
    assert create.call_count == 1
    sleep.assert_not_called()


def test_unrelated_exception_is_not_retried(monkeypatch):
    error = ValueError("bad input")
    client, create, sleep = client_with_results(monkeypatch, [error])
    with pytest.raises(ValueError) as caught:
        client.chat(MESSAGES)
    assert caught.value is error
    assert create.call_count == 1
    sleep.assert_not_called()


def test_empty_choices_are_retried(monkeypatch):
    client, create, sleep = client_with_results(monkeypatch, [response(False), response()])
    assert client.chat(MESSAGES)["content"] == "ok"
    assert create.call_count == 2
    sleep.assert_called_once_with(1)


def test_mixed_network_and_empty_responses_share_budget(monkeypatch):
    client, create, sleep = client_with_results(
        monkeypatch, [status_error(503), response(False), status_error(429), response()])
    with pytest.raises(RateLimitError):
        client.chat(MESSAGES)
    assert create.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(1,), (2,)]


def test_empty_choices_exhaust_budget_without_final_sleep(monkeypatch):
    client, create, sleep = client_with_results(monkeypatch, [response(False)] * 3)
    with pytest.raises(RuntimeError, match="choices"):
        client.chat(MESSAGES)
    assert create.call_count == 3
    assert sleep.call_count == 2


@pytest.mark.parametrize("max_attempts", [0, 1, -2])
def test_single_attempt_and_clamped_values(monkeypatch, max_attempts):
    client, create, sleep = client_with_results(monkeypatch, [response(False)], max_attempts)
    with pytest.raises(RuntimeError, match="choices"):
        client.chat(MESSAGES)
    assert create.call_count == 1
    sleep.assert_not_called()


def test_request_arguments_preserved_and_empty_tools_normalized(monkeypatch):
    client, create, _ = client_with_results(monkeypatch, [response(), response()])
    client.chat(MESSAGES, tools=[])
    create.assert_called_with(model="test", messages=MESSAGES, tools=None)
    tools = [{"type": "function", "function": {"name": "test"}}]
    client.chat(MESSAGES, tools=tools)
    create.assert_called_with(model="test", messages=MESSAGES, tools=tools)


def test_backoff_is_capped_and_jittered(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(llm.time, "sleep", sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda low, high: 0.25)
    llm.LLMClient._sleep_backoff(1)
    llm.LLMClient._sleep_backoff(5)
    assert [call.args for call in sleep.call_args_list] == [(2.25,), (8.25,)]
