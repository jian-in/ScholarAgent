"""运行指标采集：记录一次执行的真实资源消耗。"""

from dataclasses import asdict, dataclass
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

    def record_llm_call(self, usage: Optional[Mapping] = None) -> None:
        self._llm_calls += 1
        if not usage:
            self._missing_prompt_tokens = True
            self._missing_completion_tokens = True
            return

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
            self._prompt_tokens += prompt_tokens
        else:
            self._missing_prompt_tokens = True
        if isinstance(completion_tokens, int) and completion_tokens >= 0:
            self._completion_tokens += completion_tokens
        else:
            self._missing_completion_tokens = True

    def record_tool_call(self) -> None:
        self._tool_calls += 1

    def finish(self, tool_calls: Optional[int] = None) -> RunMetrics:
        return RunMetrics(
            mode=self.mode,
            seconds=perf_counter() - self._started_at,
            llm_calls=self._llm_calls,
            prompt_tokens=None if self._missing_prompt_tokens else self._prompt_tokens,
            completion_tokens=(
                None if self._missing_completion_tokens else self._completion_tokens
            ),
            tool_calls=self._tool_calls if tool_calls is None else tool_calls,
        )


class InstrumentedLLM:
    """在不改变 LLM ``chat`` 契约的前提下，记录跨执行器的调用。"""

    def __init__(self, llm, collector: MetricsCollector):
        self._llm = llm
        self._collector = collector

    def chat(self, messages, tools=None):
        reply = self._llm.chat(messages, tools=tools)
        self._collector.record_llm_call(reply.get("usage"))
        return reply


class InstrumentedTool:
    """保留原工具行为，同时把真实执行次数记入共享采集器。"""

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
