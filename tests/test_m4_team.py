"""M4 多智能体流水线的离线测试。

运行方式(项目根目录下):
    python -m pytest tests -q    (推荐)
    python tests/test_m4_team.py (没装 pytest 时直接跑)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.llm import ScriptedLLM
from scholaragent.team import ResearchTeam
from scholaragent.tool import Tool, ToolRegistry
from scholaragent.tools import BUILTIN_TOOLS


class RecordingLLM(ScriptedLLM):
    """记录每一次收到的完整对话(含 system 提示词与工具清单)。"""

    def __init__(self, replies):
        super().__init__(replies)
        self.history = []       # 每次 chat 收到的 messages
        self.tools_seen = []    # 每次 chat 收到的工具清单

    def chat(self, messages, tools=None):
        self.history.append([dict(m) for m in messages])
        self.tools_seen.append([t["function"]["name"] for t in (tools or [])])
        return super().chat(messages, tools)


def final(content):
    return {"content": content, "tool_calls": []}


# ―― ToolRegistry.subset ――――――――――――――――――――――――――――――


def test_subset_picks_only_named_tools():
    registry = ToolRegistry(BUILTIN_TOOLS)
    sub = registry.subset(["arxiv_search", "recall"])
    names = [s["function"]["name"] for s in sub.schemas()]
    assert names == ["arxiv_search", "recall"]
    # 白名单外的工具在子登记处里必须调不到
    assert "不存在" in sub.call("download_paper", {})


def test_subset_unknown_name_raises():
    registry = ToolRegistry(BUILTIN_TOOLS)
    try:
        registry.subset(["no_such_tool"])
        raise AssertionError("不该放行不存在的工具名")
    except ValueError:
        pass


# ―― 流水线 ――――――――――――――――――――――――――――――――――――――


def test_team_pipeline_hands_off_artifacts():
    """检索报告要传给精读员,报告+笔记要传给写作员;各角色工具白名单正确。"""
    llm = RecordingLLM([
        final("检索报告:推荐精读 ReAct(2210.03629)"),   # 检索员
        final("精读笔记:ReAct 提出推理与行动交替"),      # 精读员
        final("综述:ReAct(2210.03629)开创了……"),      # 写作员
    ])
    team = ResearchTeam(llm, ToolRegistry(BUILTIN_TOOLS), verbose=False,
                        require_full_paper=False)
    review = team.run("LLM Agent")

    assert review.startswith("综述")

    # 三次调用分别来自三个角色:看各自的 system 提示词
    systems = [h[0]["content"] for h in llm.history]
    assert "检索员" in systems[0]
    assert "精读员" in systems[1]
    assert "写作员" in systems[2]

    # 工具白名单:检索员只有搜索类,写作员碰不到下载
    assert set(llm.tools_seen[0]) == {"arxiv_search", "recall"}
    assert "download_paper" in llm.tools_seen[1]
    assert "download_paper" not in llm.tools_seen[2]

    # 交接物:精读员拿到检索报告,写作员拿到报告和笔记
    assert "2210.03629" in llm.history[1][-1]["content"]
    writer_prompt = llm.history[2][-1]["content"]
    assert "检索报告" in writer_prompt and "精读笔记" in writer_prompt
    assert "推理与行动交替" in writer_prompt


def test_team_clips_long_handoffs():
    """超长交接物必须截断,不能把下一个角色的上下文挤爆。"""
    llm = RecordingLLM([
        final("检索报告:" + "x" * 20000),
        final("精读笔记"),
        final("综述"),
    ])
    team = ResearchTeam(llm, ToolRegistry(BUILTIN_TOOLS), verbose=False,
                        require_full_paper=False)
    team.run("主题")
    reader_prompt = llm.history[1][-1]["content"]
    assert "已截断" in reader_prompt
    assert len(reader_prompt) < 13000


if __name__ == "__main__":
    # 不依赖 pytest 的极简测试运行器
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过:{name}")
    print("全部测试通过")
