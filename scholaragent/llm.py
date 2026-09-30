"""模型层:整个系统里唯一和大模型 API 打交道的地方。

上层(Agent)只依赖本文件的 chat() 接口,不关心背后是 DeepSeek、
通义千问还是本地 Ollama。以后想换模型,只改 .env 配置,不改代码。
"""

import json
import random
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from . import config

# 单次 API 请求的超时(秒):防止某个请求 hang 住,把整个 Agent 循环拖死。
DEFAULT_REQUEST_TIMEOUT = 60.0
# 网络抖动/空响应的最大尝试次数:指数退避,失败后自动重试。
DEFAULT_MAX_ATTEMPTS = 3


def _infer_provider(base_url):
    normalized = str(base_url or "").lower().rstrip("/")
    if normalized in {
        "http://localhost:11434/v1",
        "http://127.0.0.1:11434/v1",
    }:
        return "ollama"
    return "cloud" if normalized else "unknown"


def _is_retryable(exc: Exception) -> bool:
    """只有瞬时故障值得重试:401 鉴权失败、400 参数错误,重试多少次都没用。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


class LLMClient:
    """真实的大模型客户端,走 OpenAI 兼容协议(国内主流厂商都支持)。"""

    def __init__(self, base_url=None, api_key=None, model=None,
                 provider=None, role="general",
                 timeout=None, max_attempts=None):
        self.model = model or config.LLM_MODEL
        self.provider = provider or _infer_provider(base_url or config.LLM_BASE_URL)
        self.role = role or "general"
        self._base_url = base_url or config.LLM_BASE_URL
        self._api_key = api_key or config.LLM_API_KEY
        self._timeout = DEFAULT_REQUEST_TIMEOUT if timeout is None else timeout
        self._max_attempts = (DEFAULT_MAX_ATTEMPTS
                              if max_attempts is None else max(1, int(max_attempts)))
        # 延迟创建 SDK 客户端：Windows 首次构造 httpx/SSL 代理上下文可能
        # 很慢，不能让工作台的“提交任务”或后台线程在真正运行前卡住。
        self._client = None

    def _client_instance(self):
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                # 重试策略由本文件的 _create_completion 统一拥有,
                # SDK 内置重试关掉,避免两层退避叠加、难以归因。
                max_retries=0,
                timeout=self._timeout,
            )
        return self._client

    def metadata(self):
        """返回可写入运行轨迹的安全元数据，不包含密钥。"""
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
        }

    @staticmethod
    def _sleep_backoff(attempt):
        # 指数退避 + 抖动:第 n 次重试前睡 2**n 秒(封顶 8 秒),避免惊群。
        time.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))

    def _create_completion(self, messages, tools):
        """带重试的单次请求:瞬时网络故障或空 choices 会触发重试。

        非重试型错误(鉴权失败、参数错误)直接抛出;重试耗尽后抛出
        最后一个异常 —— 不在这里吞掉,调用方需要知道这次是真失败了。
        """
        last_error = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client_instance().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,  # 空列表有些厂商会报错,统一转成 None
                )
            except Exception as exc:  # noqa: BLE001 —— 重试判定见 _is_retryable
                last_error = exc
                if attempt < self._max_attempts and _is_retryable(exc):
                    self._sleep_backoff(attempt)
                    continue
                raise
        raise last_error  # 防御性:正常不会走到这里

    def chat(self, messages, tools=None):
        """发送整段对话历史,拿回模型的一条回复。

        参数:
            messages: 对话历史,OpenAI 消息格式的字典列表
            tools:    可用工具的 JSON Schema 列表,None 表示不提供工具

        返回统一格式(屏蔽 SDK 细节,方便替换模型和离线测试):
            {"content": 文本回复或 None,
             "tool_calls": [{"id": ..., "name": ..., "arguments": 参数字典,
                             "error": None 或参数解析失败时的提示文字}]}
        """
        response = self._request_with_guard(messages, tools)
        message = response.choices[0].message
        usage = getattr(response, "usage", None)

        tool_calls = []
        for tc in message.tool_calls or []:
            # 模型给出的参数是 JSON 字符串,这里就地解析成字典。
            # 真实模型偶尔会生成不合法的 JSON(截断、单引号等):
            # 遵循设计原则"错误回传而非崩溃",解析失败不让程序挂掉,
            # 而是把错误标记出来,由 Agent 回传给模型让它自己重试。
            raw_args = tc.function.arguments or "{}"
            try:
                arguments, error = json.loads(raw_args), None
            except json.JSONDecodeError:
                arguments = {}
                error = f"工具参数不是合法 JSON,请修正后重新调用。原始内容:{raw_args[:200]}"
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": arguments,
                "error": error,
            })
        # 缓存明细:DeepSeek 直接给 hit/miss;OpenAI 兼容实现只给
        # prompt_tokens_details.cached_tokens,miss 由 prompt-hit 推导。
        # 语义一致:两家 prompt_tokens 都包含命中部分
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
        if cache_hit is None:
            cache_hit = getattr(
                getattr(usage, "prompt_tokens_details", None),
                "cached_tokens", None)
        if cache_hit is not None and cache_miss is None:
            prompt_total = getattr(usage, "prompt_tokens", None)
            if isinstance(prompt_total, int) and isinstance(cache_hit, int):
                cache_miss = max(0, prompt_total - cache_hit)
        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "prompt_cache_hit_tokens": cache_hit,
                "prompt_cache_miss_tokens": cache_miss,
            } if usage is not None else None,
        }

    def _request_with_guard(self, messages, tools):
        """请求 + 空 choices 守卫:把"响应结构异常"也纳入重试语义。"""
        last_error = None
        for attempt in range(1, self._max_attempts + 1):
            response = self._create_completion(messages, tools)
            choices = getattr(response, "choices", None) or []
            if choices:
                return response
            last_error = RuntimeError("模型返回的 choices 为空")
            if attempt < self._max_attempts:
                self._sleep_backoff(attempt)
        raise last_error


class ScriptedLLM:
    """按剧本回话的假模型:测试和离线演示用,不花钱、不联网。

    它和 LLMClient 有完全相同的 chat() 接口(鸭子类型),
    所以 Agent 完全感觉不到自己面对的是真模型还是假模型 ——
    这正是把模型单独封装成一层的最大好处。
    """

    def __init__(self, replies, model="scripted", provider="offline",
                 role="general"):
        self._replies = list(replies)
        self.last_messages = None  # 记录最近一次收到的对话历史,供测试断言
        self.model = model
        self.provider = provider
        self.role = role or "general"

    def metadata(self):
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
        }

    def chat(self, messages, tools=None):
        self.last_messages = [dict(m) for m in messages]
        if not self._replies:
            return {"content": "(剧本演完了)", "tool_calls": []}
        return self._replies.pop(0)


def assistant_message(reply):
    """把 chat() 返回的统一格式转回 OpenAI 消息格式,用于追加进对话历史。

    对话历史必须完整保留模型的每一次工具调用记录,
    模型下一轮才能"记得"自己刚才调了什么工具。
    """
    message = {"role": "assistant", "content": reply["content"]}
    if reply["tool_calls"]:
        message["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
            for tc in reply["tool_calls"]
        ]
    return message
