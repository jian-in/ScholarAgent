"""最小运行事件协议与运行上下文。

事件是 CLI、Web、评测和回放之间的稳定接缝。核心执行器只发结构化事件，
旧的 ``on_progress(text)`` 仍作为兼容适配器保留在各执行器上。
"""

from __future__ import annotations

import inspect
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

from .evidence import EvidenceLedger
from .metrics import MetricsCollector
from .workspace import Workspace, default_workspace


EVENT_TYPES = frozenset({
    "run_started",
    "workflow_selected",
    "mode_selected",
    "model_step",
    "tool_started",
    "tool_completed",
    "artifact",
    "cancel_requested",
    "cancelled",
    "failed",
    "completed",
})


class EventSink(Protocol):
    def emit(self, event: "RunEvent") -> None:
        ...


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_payload(value: Any, key: str = "") -> Any:
    """递归清理不应进入事件的秘密字段。"""
    lowered = key.lower()
    if any(secret in lowered for secret in (
        "api_key", "apikey", "authorization", "password", "secret",
    )):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): _safe_payload(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item, key) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class RunEvent:
    """一次运行事件的可序列化表示。"""

    type: str
    run_id: str
    timestamp: str
    mode: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"未知运行事件类型: {self.type}")
        object.__setattr__(self, "payload", _safe_payload(self.payload))

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryEventSink:
    """内存事件适配器，供测试、评测和单次运行结果使用。"""

    def __init__(self):
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class CallbackEventSink:
    """把事件转发给外部存储适配器。"""

    def __init__(self, callback: Callable[[RunEvent], None]):
        self.callback = callback

    def emit(self, event: RunEvent) -> None:
        self.callback(event)


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self.reason = ""

    def request(self, reason: str = "用户请求") -> None:
        self.reason = reason
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class RunContext:
    """一次运行共享的上下文：事件、指标、工作区和取消信号。"""

    def __init__(self, mode: str = "react", run_id: str | None = None,
                 workspace: Workspace | None = None,
                 metrics: MetricsCollector | None = None,
                 event_sink: EventSink | Callable[[RunEvent], None] | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 evidence: EvidenceLedger | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self.mode = mode
        self.workspace = workspace or default_workspace()
        self.metrics = metrics or MetricsCollector(mode)
        if event_sink is None:
            self.events: EventSink = MemoryEventSink()
        elif hasattr(event_sink, "emit"):
            self.events = event_sink  # type: ignore[assignment]
        else:
            self.events = CallbackEventSink(event_sink)  # type: ignore[arg-type]
        # 无论外部适配器是内存还是回调，运行结果都要保留一份本地事件投影；
        # 否则 Web 的 JobStore 能看到事件，RunResult 却会出现空事件流。
        self._events: list[RunEvent] = []
        self.evidence = evidence or EvidenceLedger()
        self.token = CancellationToken()
        self._should_stop = should_stop
        self._terminal_event: str | None = None

    @property
    def event_list(self) -> list[RunEvent]:
        return list(self._events)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        setter = getattr(self.metrics, "set_mode", None)
        if setter:
            setter(mode)

    def emit(self, event_type: str, **payload: Any) -> RunEvent:
        event = RunEvent(
            type=event_type,
            run_id=self.run_id,
            timestamp=_timestamp(),
            mode=self.mode,
            payload=payload,
        )
        self._events.append(event)
        self.events.emit(event)
        return event

    def is_cancelled(self) -> bool:
        if self.token.is_cancelled():
            return True
        if self._should_stop is not None:
            try:
                if bool(self._should_stop()):
                    self.request_cancel("外部停止信号")
                    return True
            except Exception:
                # 停止探测器属于兼容回调，异常不能破坏主任务。
                pass
        return False

    def request_cancel(self, reason: str = "用户请求") -> None:
        if self.token.is_cancelled():
            return
        self.token.request(reason)
        self.emit("cancel_requested", reason=reason)

    def mark_terminal(self, event_type: str, **payload: Any) -> None:
        if event_type not in {"cancelled", "failed", "completed"}:
            raise ValueError(f"不是终态事件: {event_type}")
        if self._terminal_event is not None:
            return
        self._terminal_event = event_type
        self.emit(event_type, **payload)

    def chat(self, llm, messages: list, tools=None, step: int | None = None) -> dict:
        """统一执行一次模型调用并记账。"""
        metadata = {}
        describe = getattr(llm, "metadata", None)
        if callable(describe):
            try:
                raw_metadata = describe()
                if isinstance(raw_metadata, Mapping):
                    metadata = {
                        key: raw_metadata.get(key)
                        for key in ("role", "provider", "model")
                        if raw_metadata.get(key) is not None
                    }
            except Exception:
                # 元数据只是可观测性增强，不能影响真正的模型调用。
                metadata = {}
        self.emit(
            "model_step",
            step=step,
            message_count=len(messages),
            tool_schema_count=len(tools or []),
            **metadata,
        )
        reply = llm.chat(messages, tools=tools)
        self.metrics.record_llm_call(
            reply.get("usage"),
            role=metadata.get("role"),
            provider=metadata.get("provider"),
            model=metadata.get("model"),
        )
        return reply

    def invoke(self, runner, task: str):
        """向旧 runner 传递上下文；不支持新参数的对象仍可运行。"""
        run = getattr(runner, "run", runner)
        try:
            signature = inspect.signature(run)
            accepts_context = (
                "context" in signature.parameters
                or any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in signature.parameters.values())
            )
        except (TypeError, ValueError):
            accepts_context = True
        if accepts_context:
            return run(task, context=self)
        return run(task)


def event_message(event: RunEvent) -> str:
    """兼容旧日志界面的简短人类可读投影。"""
    payload = event.payload
    if event.type == "run_started":
        return f"[运行] 开始 · 模式 {event.mode.upper()}"
    if event.type == "workflow_selected":
        return f"[工作流] {payload.get('workflow', 'research-review')} · {payload.get('reason', '')}"
    if event.type == "mode_selected":
        return f"[路由] 实际模式 {payload.get('mode', event.mode).upper()} · {payload.get('reason', '')}"
    if event.type == "model_step":
        step = payload.get("step")
        role = payload.get("role", "模型")
        model = payload.get("model")
        provider = payload.get("provider")
        suffix = f" · {provider}/{model}" if provider or model else ""
        return f"[模型] 第 {step} 步 · {role}{suffix}"
    if event.type == "tool_started":
        return f"[工具] 开始 {payload.get('name', '')}"
    if event.type == "tool_completed":
        state = "成功" if payload.get("success") else "失败"
        return f"[工具] 完成 {payload.get('name', '')} · {state}"
    if event.type == "artifact":
        return f"[产物] {payload.get('kind', 'artifact')}"
    if event.type == "cancel_requested":
        return f"[取消] {payload.get('reason', '已请求停止')}"
    if event.type == "cancelled":
        return "[取消] 任务已取消"
    if event.type == "failed":
        return f"[失败] {payload.get('error', '执行失败')}"
    if event.type == "completed":
        return "[完成] 任务执行完成"
    return f"[{event.type}]"
