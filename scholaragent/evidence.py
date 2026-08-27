"""来源锚点与结论—证据账本。

科研调研的可解释性不能只停在“调用过某个搜索工具”。这个模块提供一个
小而稳定的接口，让工具、运行结果和后续审查可以共享同一组来源锚点，
同时明确哪些结论有证据、证据不足或无法判断。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping
import re


EVIDENCE_SCHEMA_VERSION = "evidence-ledger-v1"
ANCHOR_KINDS = (
    "text",
    "caption",
    "figure",
    "table",
    "equation",
    "metadata",
    "page",
)
CONFIDENCE_LEVELS = ("high", "medium", "low")
CLAIM_STATUSES = ("supported", "partial", "unsupported", "not_assessable")
_ANCHOR_ID = re.compile(r"^[A-Z][A-Z0-9_-]{0,15}\d{1,6}$")


def _clip(text: Any, limit: int = 500) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


@dataclass(frozen=True)
class SourceAnchor:
    """一个可回到原文的结构化来源位置。"""

    id: str
    kind: str
    source: str
    page: int | None = None
    section: str | None = None
    locator: str | None = None
    confidence: str = "medium"
    excerpt: str = ""

    def __post_init__(self) -> None:
        if not _ANCHOR_ID.fullmatch(self.id):
            raise ValueError(f"来源锚点 ID 不安全: {self.id}")
        if self.kind not in ANCHOR_KINDS:
            raise ValueError(f"未知来源锚点类型: {self.kind}")
        if not str(self.source).strip():
            raise ValueError("来源锚点必须声明 source")
        if self.page is not None and (not isinstance(self.page, int) or self.page < 1):
            raise ValueError("来源页码必须是正整数")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"未知来源置信度: {self.confidence}")
        object.__setattr__(self, "excerpt", _clip(self.excerpt))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceAnchor":
        return cls(
            id=str(value.get("id") or ""),
            kind=str(value.get("kind") or "text"),
            source=str(value.get("source") or ""),
            page=value.get("page"),
            section=value.get("section"),
            locator=value.get("locator"),
            confidence=str(value.get("confidence") or "medium"),
            excerpt=str(value.get("excerpt") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimEvidence:
    """一个结论及其来源支持关系。"""

    id: str
    claim: str
    anchor_ids: tuple[str, ...] = field(default_factory=tuple)
    status: str = "supported"
    note: str = ""

    def __post_init__(self) -> None:
        if not _ANCHOR_ID.fullmatch(self.id):
            raise ValueError(f"结论 ID 不安全: {self.id}")
        if not str(self.claim).strip():
            raise ValueError("结论不能为空")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"未知结论证据状态: {self.status}")
        anchors = tuple(str(anchor_id) for anchor_id in self.anchor_ids)
        if len(anchors) != len(set(anchors)):
            raise ValueError(f"结论 {self.id} 的来源锚点不能重复")
        object.__setattr__(self, "anchor_ids", anchors)
        object.__setattr__(self, "note", _clip(self.note))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["anchor_ids"] = list(self.anchor_ids)
        return data


class EvidenceLedger:
    """一次运行内的来源锚点和结论证据登记处。"""

    def __init__(self, anchors: Iterable[SourceAnchor] = (),
                 claims: Iterable[ClaimEvidence] = ()):
        self._anchors: dict[str, SourceAnchor] = {}
        self._claims: dict[str, ClaimEvidence] = {}
        for anchor in anchors:
            self.add_anchor(anchor=anchor)
        for claim in claims:
            self.add_claim(claim=claim)

    @property
    def anchors(self) -> tuple[SourceAnchor, ...]:
        return tuple(self._anchors.values())

    @property
    def claims(self) -> tuple[ClaimEvidence, ...]:
        return tuple(self._claims.values())

    def add_anchor(self, anchor: SourceAnchor | Mapping[str, Any] | None = None,
                   **kwargs: Any) -> SourceAnchor:
        if anchor is None:
            anchor = SourceAnchor.from_dict(kwargs)
        elif not isinstance(anchor, SourceAnchor):
            anchor = SourceAnchor.from_dict(anchor)
        previous = self._anchors.get(anchor.id)
        if previous is not None and previous != anchor:
            raise ValueError(f"来源锚点 ID 重复且内容不同: {anchor.id}")
        self._anchors[anchor.id] = anchor
        return anchor

    def add_claim(self, value: ClaimEvidence | Mapping[str, Any] | None = None,
                  **kwargs: Any) -> ClaimEvidence:
        # ``claim`` 保留为关键字字段，便于自然地写
        # ``add_claim(id=..., claim='...', anchor_ids=[...])``；同时也兼容
        # ``add_claim(claim=ClaimEvidence(...))`` 这种对象入口。
        claim = value
        if claim is None and isinstance(kwargs.get("claim"), (ClaimEvidence, Mapping)):
            claim = kwargs.pop("claim")
        if claim is None:
            claim = ClaimEvidence(
                id=str(kwargs.get("id") or ""),
                claim=str(kwargs.get("claim") or ""),
                anchor_ids=tuple(kwargs.get("anchor_ids") or ()),
                status=str(kwargs.get("status") or "supported"),
                note=str(kwargs.get("note") or ""),
            )
        elif not isinstance(claim, ClaimEvidence):
            claim = ClaimEvidence(
                id=str(claim.get("id") or ""),
                claim=str(claim.get("claim") or ""),
                anchor_ids=tuple(claim.get("anchor_ids") or ()),
                status=str(claim.get("status") or "supported"),
                note=str(claim.get("note") or ""),
            )
        previous = self._claims.get(claim.id)
        if previous is not None and previous != claim:
            raise ValueError(f"结论 ID 重复且内容不同: {claim.id}")
        self._claims[claim.id] = claim
        return claim

    def ingest_artifact(self, artifact: Mapping[str, Any]) -> None:
        """吸收工具产物中的 ``source_anchor(s)``，重复内容保持幂等。"""
        raw = artifact.get("source_anchors")
        if raw is None and artifact.get("source_anchor") is not None:
            raw = [artifact["source_anchor"]]
        if isinstance(raw, Mapping):
            raw = [raw]
        for item in raw or ():
            self.add_anchor(item)

    def validate(self) -> list[str]:
        """返回可读的结构错误；空列表表示账本自洽。"""
        errors = []
        for claim in self.claims:
            missing = [anchor_id for anchor_id in claim.anchor_ids
                       if anchor_id not in self._anchors]
            if missing:
                errors.append(
                    f"结论 {claim.id} 引用了不存在的来源锚点: {', '.join(missing)}"
                )
            if claim.status in {"supported", "partial"} and not claim.anchor_ids:
                errors.append(f"结论 {claim.id} 没有来源锚点")
        return errors

    def to_dict(self) -> dict[str, Any]:
        statuses = Counter(claim.status for claim in self.claims)
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "claims": [claim.to_dict() for claim in self.claims],
            "summary": {
                "anchors": len(self._anchors),
                "claims": len(self._claims),
                "supported_claims": statuses.get("supported", 0),
                "partial_claims": statuses.get("partial", 0),
                "unsupported_claims": statuses.get("unsupported", 0),
                "not_assessable_claims": statuses.get("not_assessable", 0),
                "validation_errors": len(self.validate()),
            },
            "validation_errors": self.validate(),
        }
