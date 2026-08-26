"""内置工具集。

新增一个工具的步骤:
    1. 在本目录新建文件,写一个继承 Tool 的类(参考 calculator.py)
    2. 在下面 import 它,并加进 BUILTIN_TOOLS 列表
"""

from .arxiv_search import ArxivSearchTool
from .calculator import CalculatorTool
from .clock import ClockTool
from .memory_tools import RecallTool, RememberTool
from .notes import ReadNotesTool, SaveNoteTool
from .papers import DownloadPaperTool, ReadPaperTool

BUILTIN_TOOLS = [
    CalculatorTool(),
    ClockTool(),
    # M1 文献工具集
    ArxivSearchTool(),
    DownloadPaperTool(),
    ReadPaperTool(),
    SaveNoteTool(),
    ReadNotesTool(),
    # M2 记忆工具
    RememberTool(),
    RecallTool(),
]
