"""记忆工具(M2):让模型自己决定"记住什么、想起什么"。

和 notes.py 的分工:
    笔记(save_note/read_notes) —— 给"人"看的成果沉淀,整篇读取
    记忆(remember/recall)      —— 给"模型"用的检索仓库,按相关性捞

recall 背后是手写的 BM25(见 memory.py),这正是 RAG(检索增强生成)
的最小实现:先检索、再生成,答案有出处、模型少编造。
"""

from ..memory import MemoryStore
from ..tool import Tool


class RememberTool(Tool):
    name = "remember"
    description = (
        "把一条重要结论存入长期记忆(跨对话永久保留)。"
        "适合存:论文核心结论、关键数据、用户的偏好和要求"
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要记住的内容,一条一个完整的事实",
            },
            "source": {
                "type": "string",
                "description": "内容出处,如论文的 arXiv 编号,可不填",
            },
        },
        "required": ["content"],
    }

    def __init__(self, store: MemoryStore = None):
        self.store = store or MemoryStore()

    def run(self, content: str, source: str = "") -> str:
        return self.store.add(content, source)


class RecallTool(Tool):
    name = "recall"
    description = "用关键词从长期记忆中检索相关内容,适合在任务开始时先回忆"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词,中英文都可以",
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: MemoryStore = None):
        self.store = store or MemoryStore()

    def run(self, query: str) -> str:
        hits = self.store.search(query, top_k=3)
        if not hits:
            return "长期记忆里没有找到相关内容"
        lines = []
        for h in hits:
            # .get 兜底:记忆文件被手改漏了字段时,降级显示而不是崩溃
            source = f"(出处:{h['source']})" if h.get("source") else ""
            lines.append(f"- [{h.get('time', '?')}] {h.get('text', '')}{source}")
        return "检索到的相关记忆:\n" + "\n".join(lines)
