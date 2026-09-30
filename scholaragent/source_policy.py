"""Immutable, run-scoped paper version constraints, independent of prompts."""
import re
from dataclasses import dataclass


class SourceMismatch(ValueError):
    def __init__(self, kind, requested, pinned):
        self.kind = kind
        self.requested = requested
        self.pinned = pinned
        super().__init__(f"文献约束 {kind}: 请求 {requested!r}；固定版本: {', '.join(pinned)}")


@dataclass(frozen=True)
class SourcePolicy:
    pinned: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.pinned, (str, bytes)):
            raise ValueError("pinned_sources 应为带版本编号的列表")
        values = tuple(self.pinned)
        if any(not isinstance(x, str) or not re.fullmatch(r"\d{4}\.\d{4,5}v[1-9]\d*", x) for x in values):
            raise ValueError("pinned_sources 必须是带版本的 arXiv 编号")
        object.__setattr__(self, "pinned", tuple(dict.fromkeys(values)))

    def resolve(self, requested):
        if not self.pinned:
            return requested
        value = requested.strip() if isinstance(requested, str) else ""
        if value in self.pinned:
            return value
        matches = [x for x in self.pinned if x.split("v", 1)[0] == value.split("v", 1)[0]]
        if len(matches) == 1 and value == matches[0].split("v", 1)[0]:
            return matches[0]
        kind = "source_version_mismatch" if matches else "source_not_pinned"
        raise SourceMismatch(kind, requested, self.pinned)

    def instruction(self):
        if not self.pinned:
            return ""
        return ("\n运行级固定文献约束：" + ", ".join(self.pinned) +
                "。下载和阅读只使用这些版本；省略版本号时工具补齐唯一固定版本，"
                "其他版本或文献返回不匹配诊断。检索元数据的版本需另行核对，摘要不是正文。")
