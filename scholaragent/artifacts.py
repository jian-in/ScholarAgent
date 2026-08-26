"""本轮产物摘要:从工具调用中收集论文、笔记、长期记忆。

不扫描整个 data/ 目录(会混入历史文件),只记录本轮实际成功的工具副作用。
"""

from __future__ import annotations

import os
from typing import Optional

from . import config
from .tools.notes import _notes_file
from .tools.papers import _pdf_path


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class ArtifactCollector:
    """线程内单任务使用;由 ToolRegistry 在成功调用后写入。"""

    def __init__(self):
        self.papers: list[dict] = []
        self.notes: list[dict] = []
        self.memories: list[dict] = []
        self.read_ids: list[str] = []
        self._paper_ids: set[str] = set()
        self._note_keys: set[str] = set()
        self._memory_keys: set[str] = set()
        self._read_ids: set[str] = set()

    def record(self, tool_name: str, arguments: Optional[dict], result: str) -> None:
        arguments = arguments or {}
        result = str(result or "")
        if result.startswith("错误:") or "执行出错:" in result:
            return

        if tool_name == "download_paper":
            arxiv_id = str(arguments.get("arxiv_id") or "").strip()
            if not arxiv_id or "已就绪" not in result:
                return
            if arxiv_id in self._paper_ids:
                return
            self._paper_ids.add(arxiv_id)
            path = _pdf_path(arxiv_id)
            self.papers.append({
                "arxiv_id": arxiv_id,
                "path": path,
                "exists": os.path.exists(path),
                "summary": _clip(result, 120),
            })
            return

        if tool_name == "read_paper":
            arxiv_id = str(arguments.get("arxiv_id") or "").strip()
            if not arxiv_id or arxiv_id in self._read_ids:
                return
            # 读失败(未下载等)不记
            if "请先" in result or "不存在" in result or "出错" in result:
                return
            self._read_ids.add(arxiv_id)
            self.read_ids.append(arxiv_id)
            return

        if tool_name == "save_note":
            title = str(arguments.get("title") or "").strip() or "未命名笔记"
            if "已保存" not in result:
                return
            key = title
            if key in self._note_keys:
                return
            self._note_keys.add(key)
            path = _notes_file()
            self.notes.append({
                "title": title,
                "path": path,
                "summary": _clip(arguments.get("content") or result, 140),
            })
            return

        if tool_name == "remember":
            content = str(arguments.get("content") or "").strip()
            if not content or "已存入" not in result:
                return
            source = str(arguments.get("source") or "").strip()
            key = content[:80] + "|" + source
            if key in self._memory_keys:
                return
            self._memory_keys.add(key)
            self.memories.append({
                "text": _clip(content, 180),
                "source": source,
                "path": os.path.join(config.DATA_DIR, "memory", "memories.jsonl"),
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
