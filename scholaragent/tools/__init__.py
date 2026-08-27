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
from ..memory import MemoryStore
from ..workspace import Workspace

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


def build_builtin_tools(workspace: Workspace | str | None = None,
                        memory_store: MemoryStore | None = None) -> list:
    """为一次运行创建隔离的内置工具实例。

    ``BUILTIN_TOOLS`` 保留给旧代码和简单示例；运行时、评测和 Web 都应
    调用这个工厂，避免工具在不同任务间共享游标或持久化路径。
    """
    store = memory_store or MemoryStore(workspace=workspace)
    return [
        CalculatorTool(),
        ClockTool(),
        ArxivSearchTool(),
        DownloadPaperTool(workspace),
        ReadPaperTool(workspace),
        SaveNoteTool(workspace),
        ReadNotesTool(workspace),
        RememberTool(store),
        RecallTool(store),
    ]
