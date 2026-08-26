"""内置工具 2:时钟。

大模型的知识停在训练数据截止的那一天,它并不知道"现在"是什么时候。
凡是模型自身不具备、又随时在变化的信息,都适合做成工具。
"""

from datetime import datetime

from ..tool import Tool


class ClockTool(Tool):
    name = "current_time"
    description = "获取当前的日期、时间和星期几"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self) -> str:
        now = datetime.now()
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        return now.strftime("%Y-%m-%d %H:%M:%S") + f" 星期{weekdays[now.weekday()]}"
