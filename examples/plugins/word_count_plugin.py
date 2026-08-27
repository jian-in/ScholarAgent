"""最小外部工具插件示例，可作为独立分发包的入口点目标。"""

from scholaragent.tool import Tool


class WordCountTool(Tool):
    name = "word_count"
    description = "统计一段文本按空白切分后的词数。"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "待统计文本"}},
        "required": ["text"],
    }
    license = "MIT"

    def run(self, text: str = "") -> str:
        return str(len(str(text).split()))

