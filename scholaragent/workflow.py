"""声明式科研工作流清单与来源格式路由。

这个模块借鉴了成熟科研技能包的“短路由器 + 静态清单 + 按需参考”思路，
但只保存 ScholarAgent 自己的工作流契约，不执行清单中的任意代码。真正的
模型、工具、事件和指标执行仍由 :mod:`scholaragent.runtime` 负责。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .gap_survey import is_gap_survey_task


WORKFLOW_SCHEMA_VERSION = "workflow-manifest-v1"
SOURCE_FORMATS = (
    "task-only",
    "pdf-text",
    "scanned-pdf",
    "html",
    "doi-arxiv",
    "pasted-text",
)
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_ARXIV_ID = re.compile(r"^(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)


def detect_source_format(source: str | Path | None) -> str:
    """根据用户提供的来源，返回可审计的输入格式标签。

    这里只做轻量、确定性的识别，不访问网络，也不判断 PDF 是否真的需要
    OCR；后者由具体读取工具在获得文件后报告。
    """
    text = str(source or "").strip()
    if not text:
        return "task-only"
    lower = text.lower()
    if _DOI.match(text) or _ARXIV_ID.match(text):
        return "doi-arxiv"
    if "arxiv.org/" in lower or lower.startswith("arxiv:"):
        return "doi-arxiv"
    if lower.endswith(".pdf") or ".pdf?" in lower:
        return "pdf-text"
    if lower.endswith((".html", ".htm")) or lower.startswith(("http://", "https://")):
        return "html"
    if "\n" in text or len(text) >= 120:
        return "pasted-text"
    return "task-only"


@dataclass(frozen=True)
class WorkflowSpec:
    """一个不含可执行代码的工作流契约。"""

    name: str
    version: str
    description: str
    input_formats: tuple[str, ...]
    outputs: tuple[str, ...]
    always_load: tuple[str, ...]
    references: tuple[Mapping[str, str], ...]
    evidence_requirements: tuple[str, ...]
    validators: tuple[str, ...]
    status: str = "contract"

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(f"工作流名称不安全: {self.name}")
        if not self.version.strip():
            raise ValueError("工作流必须声明版本")
        if not self.description.strip():
            raise ValueError(f"工作流 {self.name} 缺少 description")
        for field_name in ("input_formats", "outputs", "always_load",
                           "references", "evidence_requirements", "validators"):
            value = tuple(getattr(self, field_name))
            comparable = (
                tuple((item.get("condition"), item.get("path"))
                      for item in value)
                if field_name == "references" else value
            )
            if len(value) != len(set(comparable)):
                raise ValueError(f"工作流 {self.name} 的 {field_name} 不能重复")
            object.__setattr__(self, field_name, value)
        unknown = set(self.input_formats).difference(SOURCE_FORMATS)
        if unknown:
            raise ValueError(f"工作流 {self.name} 含未知来源格式: {sorted(unknown)}")
        clean_refs = []
        for reference in self.references:
            if not isinstance(reference, Mapping):
                raise ValueError(f"工作流 {self.name} 的 references 必须是对象")
            condition = str(reference.get("condition") or "").strip()
            path = str(reference.get("path") or "").strip()
            if not condition or not path:
                raise ValueError(f"工作流 {self.name} 的 reference 缺少 condition/path")
            clean_refs.append({"condition": condition, "path": path})
        object.__setattr__(self, "references", tuple(clean_refs))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowSpec":
        required = ("name", "version", "description", "input_formats", "outputs",
                    "always_load", "references", "evidence_requirements", "validators")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"工作流清单缺少字段: {', '.join(missing)}")
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            description=str(value["description"]),
            input_formats=tuple(value["input_formats"]),
            outputs=tuple(value["outputs"]),
            always_load=tuple(value["always_load"]),
            references=tuple(value["references"]),
            evidence_requirements=tuple(value["evidence_requirements"]),
            validators=tuple(value["validators"]),
            status=str(value.get("status") or "contract"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowSelection:
    """一次任务的工作流选择结果，供运行结果和证据包记录。"""

    spec: WorkflowSpec
    source_format: str
    reason: str

    @property
    def name(self) -> str:
        return self.spec.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "version": self.spec.version,
            "source_format": self.source_format,
            "reason": self.reason,
            "status": self.spec.status,
        }


class WorkflowRegistry:
    """工作流清单登记处；默认只加载仓库内的静态 JSON。"""

    def __init__(self, specs: Iterable[WorkflowSpec] = ()):
        self._specs: dict[str, WorkflowSpec] = {}
        for spec in specs:
            self.register(spec)

    @classmethod
    def default(cls) -> "WorkflowRegistry":
        path = Path(__file__).with_name("workflows") / "manifests.json"
        return cls.from_path(path)

    @classmethod
    def from_path(cls, path: str | Path) -> "WorkflowRegistry":
        manifest_path = Path(path)
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"工作流清单必须使用 {WORKFLOW_SCHEMA_VERSION}")
        rows = value.get("workflows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("工作流清单必须包含 workflows 数组")
        return cls(WorkflowSpec.from_dict(row) for row in rows)

    def register(self, spec: WorkflowSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"工作流名称重复: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> WorkflowSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"不存在名为 {name} 的工作流") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def select(self, task: str, source: str | Path | None = None,
               requested: str | None = None) -> WorkflowSelection:
        """根据显式请求或任务意图选择工作流，不执行外部代码。"""
        source_format = detect_source_format(source)
        if requested:
            spec = self.get(requested)
            reason = "用户显式指定工作流"
            return WorkflowSelection(spec, source_format, reason)

        text = str(task or "").lower()
        if is_gap_survey_task(task):
            name, reason = "gap-survey", "任务要求补全四类资料缺口"
        elif any(marker in text for marker in ("paper card", "论文卡", "论文卡片", "证据链审查")):
            name, reason = "paper-card", "任务要求论文卡片或结论证据链审查"
        elif any(marker in text for marker in ("中英文对照", "全文翻译", "markdown reader", "图文对应")):
            name, reason = "paper-reading", "任务要求全文阅读、翻译或图文对应产物"
        elif any(marker in text for marker in ("每日文献", "文献推送", "literature pipeline", "定时检索")):
            name, reason = "literature-pipeline", "任务要求批量发现、筛选和归档"
        elif any(marker in text for marker in ("查文献", "找文献", "引用核验", "文献检索")):
            name, reason = "literature-search", "任务要求文献发现或引用核验"
        else:
            name, reason = "research-review", "未命中特定工作流，使用通用科研调研契约"
        return WorkflowSelection(self.get(name), source_format, reason)
