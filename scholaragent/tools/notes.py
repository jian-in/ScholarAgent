"""文献工具 4/5:研究笔记。

Agent 每次 run() 都是全新对话(记忆系统要到 M2 才有),
笔记就是它现阶段唯一的"长期记忆":读到重要内容随手记下,
下次任务开始时先 read_notes 回顾之前的发现。
这也是给 M2 埋的伏笔 —— 到时候会看到,"记忆"本质上就是
"写下来 + 需要时捞回来",笔记是它最朴素的形态。
"""

import os
from datetime import datetime

from .. import config  # 旧测试/外部工具通过此兼容属性调整默认目录
from ..tool import Tool, ToolResult
from ..workspace import Workspace, workspace_for


def _notes_file(workspace: Workspace | str | None = None) -> str:
    return str(workspace_for(workspace).notes_path)


class SaveNoteTool(Tool):
    name = "save_note"
    description = "把一条研究笔记追加保存到本地笔记本(markdown 格式,永久保留)"
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "这条笔记的标题,例如某篇论文的题目或一个主题",
            },
            "content": {
                "type": "string",
                "description": "笔记正文:发现、要点、引用(注明 arXiv 编号)",
            },
        },
        "required": ["title", "content"],
    }

    def __init__(self, workspace: Workspace | str | None = None):
        self.workspace = workspace

    def _active_workspace(self) -> Workspace:
        return workspace_for(self.workspace)

    def run(self, title: str, content: str) -> str:
        path = _notes_file(self.workspace)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## {title}({stamp})\n\n{content.strip()}\n")
        return f"笔记「{title}」已保存"

    def artifact_metadata(self, arguments, result: ToolResult):
        if not result.success:
            return []
        return [{
            "kind": "note",
            "title": str(arguments.get("title") or "未命名笔记"),
            "path": str(self._active_workspace().notes_path),
            "summary": str(arguments.get("content") or ""),
        }]


class ReadNotesTool(Tool):
    name = "read_notes"
    description = "查看之前保存的研究笔记,适合在任务开始时回顾已有发现"
    parameters = {"type": "object", "properties": {}, "required": []}

    MAX_CHARS = 4000  # 和 read_paper 一样,守住上下文预算

    def __init__(self, workspace: Workspace | str | None = None):
        self.workspace = workspace

    def run(self) -> str:
        path = _notes_file(self.workspace)
        if not os.path.exists(path):
            return "笔记本还是空的,读到重要内容可以用 save_note 记下来"
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if len(text) > self.MAX_CHARS:
            text = "(笔记太长,只显示最近的部分)\n…" + text[-self.MAX_CHARS:]
        return text
