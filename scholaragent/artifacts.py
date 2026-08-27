"""本轮产物摘要:从工具调用中收集论文、笔记、长期记忆。

不扫描整个 data/ 目录(会混入历史文件),只记录本轮实际成功的工具副作用。
"""

from __future__ import annotations

from typing import Optional

from .tool import ToolResult, adapt_tool_result
from .workspace import Workspace, workspace_for


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class ArtifactCollector:
    """线程内单任务使用;由 ToolRegistry 在成功调用后写入。"""

    def __init__(self, workspace: Workspace | str | None = None):
        self.workspace = workspace_for(workspace)
        self.papers: list[dict] = []
        self.notes: list[dict] = []
        self.memories: list[dict] = []
        self.read_ids: list[str] = []
        self._paper_ids: set[str] = set()
        self._note_keys: set[str] = set()
        self._memory_keys: set[str] = set()
        self._read_ids: set[str] = set()

    def record_result(self, tool_name: str, arguments: Optional[dict],
                      result: ToolResult) -> None:
        """只消费结构化成功状态与产物元数据。"""
        arguments = arguments or {}
        if not result.success:
            return

        metadata = list(result.artifacts)
        if not metadata:
            metadata = self._legacy_metadata(tool_name, arguments, result)
        for item in metadata:
            self._record_metadata(tool_name, arguments, item, result)

    def record(self, tool_name: str, arguments: Optional[dict], result) -> None:
        """旧字符串入口，集中在边界处适配，供外部工具兼容。"""
        self.record_result(tool_name, arguments, adapt_tool_result(result))

    def _legacy_metadata(self, tool_name: str, arguments: dict,
                         result: ToolResult) -> list[dict]:
        """为仍返回字符串的旧工具提供元数据适配，不参与控制流。"""
        if tool_name == "download_paper":
            arxiv_id = str(arguments.get("arxiv_id") or "").strip()
            return ([{
                "kind": "paper",
                "arxiv_id": arxiv_id,
                "path": str(self.workspace.paper_path(arxiv_id)),
                "summary": _clip(result.text, 120),
            }] if arxiv_id else [])
        if tool_name == "read_paper":
            arxiv_id = str(arguments.get("arxiv_id") or "").strip()
            return ([{"kind": "read", "arxiv_id": arxiv_id}]
                    if arxiv_id else [])
        if tool_name == "save_note":
            title = str(arguments.get("title") or "").strip() or "未命名笔记"
            return [{
                "kind": "note",
                "title": title,
                "path": str(self.workspace.notes_path),
                "summary": _clip(arguments.get("content") or result.text, 140),
            }]
        if tool_name == "remember":
            content = str(arguments.get("content") or "").strip()
            if not content:
                return []
            return [{
                "kind": "memory",
                "text": _clip(content, 180),
                "source": str(arguments.get("source") or "").strip(),
                "path": str(self.workspace.memory_path),
            }]
        return []

    def _record_metadata(self, tool_name: str, arguments: dict,
                         metadata: dict, result: ToolResult) -> None:
        kind = metadata.get("kind")
        if kind == "paper" or tool_name == "download_paper":
            arxiv_id = str(metadata.get("arxiv_id") or arguments.get("arxiv_id") or "").strip()
            if not arxiv_id or arxiv_id in self._paper_ids:
                return
            self._paper_ids.add(arxiv_id)
            self.papers.append({
                "arxiv_id": arxiv_id,
                "path": str(metadata.get("path") or self.workspace.paper_path(arxiv_id)),
                "exists": bool(metadata.get("exists", False)),
                "summary": _clip(metadata.get("summary") or result.text, 120),
            })
            return

        if kind == "read" or tool_name == "read_paper":
            arxiv_id = str(metadata.get("arxiv_id") or arguments.get("arxiv_id") or "").strip()
            if not arxiv_id or arxiv_id in self._read_ids:
                return
            self._read_ids.add(arxiv_id)
            self.read_ids.append(arxiv_id)
            return

        if kind == "note" or tool_name == "save_note":
            title = str(metadata.get("title") or arguments.get("title") or "").strip() or "未命名笔记"
            key = title
            if key in self._note_keys:
                return
            self._note_keys.add(key)
            self.notes.append({
                "title": title,
                "path": str(metadata.get("path") or self.workspace.notes_path),
                "summary": _clip(metadata.get("summary") or arguments.get("content") or result.text, 140),
            })
            return

        if kind == "memory" or tool_name == "remember":
            content = str(metadata.get("text") or arguments.get("content") or "").strip()
            if not content:
                return
            source = str(metadata.get("source") or arguments.get("source") or "").strip()
            key = content[:80] + "|" + source
            if key in self._memory_keys:
                return
            self._memory_keys.add(key)
            self.memories.append({
                "text": _clip(content, 180),
                "source": source,
                "path": str(metadata.get("path") or self.workspace.memory_path),
            })

    def merge(self, other: "ArtifactCollector") -> None:
        if not other:
            return
        for paper in other.papers:
            arxiv_id = paper.get("arxiv_id")
            if not arxiv_id or arxiv_id in self._paper_ids:
                continue
            self._paper_ids.add(arxiv_id)
            self.papers.append(dict(paper))
        for note in other.notes:
            title = note.get("title") or "未命名笔记"
            if title in self._note_keys:
                continue
            self._note_keys.add(title)
            self.notes.append(dict(note))
        for memory in other.memories:
            key = (memory.get("text") or "")[:80] + "|" + (memory.get("source") or "")
            if key in self._memory_keys:
                continue
            self._memory_keys.add(key)
            self.memories.append(dict(memory))
        for arxiv_id in other.read_ids:
            if arxiv_id in self._read_ids:
                continue
            self._read_ids.add(arxiv_id)
            self.read_ids.append(arxiv_id)

    def to_dict(self) -> dict:
        return {
            "root": str(self.workspace.root),
            "papers": list(self.papers),
            "notes": list(self.notes),
            "memories": list(self.memories),
            "read_ids": list(self.read_ids),
            "counts": {
                "papers": len(self.papers),
                "notes": len(self.notes),
                "memories": len(self.memories),
                "read": len(self.read_ids),
            },
        }

    def is_empty(self) -> bool:
        return not (self.papers or self.notes or self.memories or self.read_ids)
