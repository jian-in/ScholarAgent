"""运行指标采集：记录一次执行的真实资源消耗。"""

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Mapping, Optional


@dataclass(frozen=True)
class RunMetrics:
    """一次任务执行的可序列化指标。

    token 字段为 ``None`` 表示上游模型 API 没有返回该数据，绝不估算或伪造。
    """

    mode: str
    seconds: float
    llm_calls: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    tool_calls: int
    # 按模型职责保留一份 token 分解，便于核验云端调研与本地总结的成本边界。
    llm_usage_by_role: Mapping[str, Mapping] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCollector:
    """轻量级、无外部依赖的单次运行指标采集器。"""

    def __init__(self, mode: str):
        self.mode = mode
        self.restart()

    def restart(self) -> None:
        """从当前时刻开始一段新的、独立的执行观测。"""
        self._started_at = perf_counter()
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._missing_prompt_tokens = False
        self._missing_completion_tokens = False
        self._tool_calls = 0
        self._llm_usage_by_role = {}

    def set_mode(self, mode: str) -> None:
        """在 Auto 路由完成后，把指标归属切换到实际执行模式。"""
        self.mode = mode

    def record_llm_call(self, usage: Optional[Mapping] = None, role=None,
                        provider=None, model=None) -> None:
        self._llm_calls += 1
        role_key = str(role or "general")
        bucket = self._llm_usage_by_role.setdefault(role_key, {
            "provider": provider,
            "model": model,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "missing_prompt_tokens": False,
            "missing_completion_tokens": False,
        })
        bucket["llm_calls"] += 1
        if bucket.get("provider") is None and provider is not None:
            bucket["provider"] = provider
        if bucket.get("model") is None and model is not None:
            bucket["model"] = model
        if not usage:
            self._missing_prompt_tokens = True
            self._missing_completion_tokens = True
            bucket["missing_prompt_tokens"] = True
            bucket["missing_completion_tokens"] = True
            return

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
            self._prompt_tokens += prompt_tokens
            bucket["prompt_tokens"] += prompt_tokens
        else:
            self._missing_prompt_tokens = True
            bucket["missing_prompt_tokens"] = True
        if isinstance(completion_tokens, int) and completion_tokens >= 0:
            self._completion_tokens += completion_tokens
            bucket["completion_tokens"] += completion_tokens
        else:
            self._missing_completion_tokens = True
            bucket["missing_completion_tokens"] = True

    def record_tool_call(self) -> None:
        self._tool_calls += 1

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    def snapshot(self, tool_calls: Optional[int] = None) -> RunMetrics:
        """返回当前观测快照，不重置采集器。"""
        return self._build_metrics(tool_calls=tool_calls)

    def _build_metrics(self, tool_calls: Optional[int] = None) -> RunMetrics:
        usage_by_role = {
            role: {
                "provider": bucket.get("provider"),
                "model": bucket.get("model"),
                "llm_calls": bucket["llm_calls"],
                "prompt_tokens": (
                    None if bucket["missing_prompt_tokens"]
                    else bucket["prompt_tokens"]
                ),
                "completion_tokens": (
                    None if bucket["missing_completion_tokens"]
                    else bucket["completion_tokens"]
                ),
            }
            for role, bucket in self._llm_usage_by_role.items()
        }
        return RunMetrics(
            mode=self.mode,
            seconds=perf_counter() - self._started_at,
            llm_calls=self._llm_calls,
            prompt_tokens=None if self._missing_prompt_tokens else self._prompt_tokens,
            completion_tokens=(
                None if self._missing_completion_tokens else self._completion_tokens
            ),
            tool_calls=self._tool_calls if tool_calls is None else tool_calls,
            llm_usage_by_role=usage_by_role,
        )

    def finish(self, tool_calls: Optional[int] = None) -> RunMetrics:
        return self._build_metrics(tool_calls=tool_calls)


class InstrumentedLLM:
    """旧评测适配器；新运行时通过 :class:`RunContext` 统一记账。

    保留它是为了兼容外部脚本和旧案例数据，不再在生产入口使用，避免
    模型调用被上下文和包装器重复计数。
    """

    def __init__(self, llm, collector: MetricsCollector):
        self._llm = llm
        self._collector = collector

    def chat(self, messages, tools=None):
        reply = self._llm.chat(messages, tools=tools)
        self._collector.record_llm_call(reply.get("usage"))
        return reply


class InstrumentedTool:
    """旧评测适配器；新运行时由 ToolRegistry 直接记账。"""

    def __init__(self, tool, collector: MetricsCollector):
        self._tool = tool
        self._collector = collector
        self.name = tool.name
        self.description = tool.description
        self.parameters = tool.parameters

    def start_run(self):
        return self._tool.start_run()

    def completion_ready(self) -> bool:
        return self._tool.completion_ready()

    def schema(self) -> dict:
        return self._tool.schema()

    def run(self, **kwargs):
        self._collector.record_tool_call()
        return self._tool.run(**kwargs)
