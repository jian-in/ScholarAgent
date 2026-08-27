"""M3 规划层的离线测试:用剧本假模型走通 计划→执行→反思→汇总。

运行方式(项目根目录下):
    python -m pytest tests -q       (推荐)
    python tests/test_m3_planner.py (没装 pytest 时直接跑)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.agent import Agent
from scholaragent.llm import ScriptedLLM
from scholaragent.planner import Planner, parse_plan
from scholaragent.tool import ToolRegistry
from scholaragent.tools.calculator import CalculatorTool


class RecordingLLM(ScriptedLLM):
    """在 ScriptedLLM 基础上记录每一次收到的完整对话,供测试断言。"""

    def __init__(self, replies):
        super().__init__(replies)
        self.history = []

    def chat(self, messages, tools=None):
        self.history.append([dict(m) for m in messages])
        return super().chat(messages, tools)


def final(content):
    """造一条"直接给最终回答"的剧本回复。"""
    return {"content": content, "tool_calls": []}


def make_planner(replies):
    llm = RecordingLLM(replies)
    agent = Agent(llm, ToolRegistry([CalculatorTool()]), verbose=False)
    return Planner(llm, agent, verbose=False), llm


# ―― parse_plan ――――――――――――――――――――――――――――――――――――


def test_parse_plan_extracts_json_from_chatter():
    text = '好的,计划如下:["检索论文", "阅读论文"] 请确认。'
    assert parse_plan(text) == ["检索论文", "阅读论文"]


def test_parse_plan_caps_at_six_steps():
    steps = parse_plan(str([f"步骤{i}" for i in range(10)]).replace("'", '"'))
    assert len(steps) == 6


def test_parse_plan_garbage_returns_none():
    assert parse_plan("我觉得这个任务应该分三步走") is None
    assert parse_plan("") is None
    assert parse_plan('["", "  "]') is None  # 全是空步骤等于没计划


# ―― 主流程 ――――――――――――――――――――――――――――――――――――――


def test_planner_happy_path_two_steps():
    """两步计划:计划→执行1→反思1→执行2→反思2→汇总,共 6 次模型调用。"""
    planner, llm = make_planner([
        final('["查资料", "写总结"]'),        # 计划
        final("资料内容 A"),                   # 执行第 1 步(内层 Agent)
        final('{"ok": true}'),                # 反思第 1 步
        final("总结草稿 B"),                   # 执行第 2 步
        final('{"ok": true}'),                # 反思第 2 步
        final("最终综述 C"),                   # 汇总
    ])
    answer = planner.run("帮我调研某方向")
    assert answer == "最终综述 C"
    assert len(llm.history) == 6
    # 第 2 步执行时,提示词里必须带着第 1 步的结果摘要
    step2_prompt = llm.history[3][-1]["content"]
    assert "资料内容 A" in step2_prompt
    assert "只执行第 2 步" in step2_prompt


def test_planner_reflect_retry_carries_advice():
    """反思不通过时:该步带着改进建议重试一次,重试提示里要有建议原文。"""
    planner, llm = make_planner([
        final('["查资料"]'),                          # 计划(单步)
        final("敷衍的结果"),                           # 执行(第一次)
        final('{"ok": false, "advice": "要注明论文编号"}'),  # 反思:不合格
        final("认真的结果,含编号 2210.03629"),          # 重试执行
    ])
    answer = planner.run("查一下")
    assert answer == "认真的结果,含编号 2210.03629"  # 单步任务直接返回,不再汇总
    retry_prompt = llm.history[3][-1]["content"]
    assert "要注明论文编号" in retry_prompt          # 改进建议传进了重试


def test_planner_bad_plan_degrades_to_single_step():
    """计划解析失败:退化为单步执行(等价纯 ReAct),不能崩。"""
    planner, llm = make_planner([
        final("我觉得应该先这样再那样"),   # 计划输出不是 JSON
        final("直接执行的结果"),           # 整个任务当一步执行
        final('{"ok": true}'),            # 反思
    ])
    assert planner.run("随便一个任务") == "直接执行的结果"


def test_planner_reflect_parses_fenced_json():
    """本地小模型爱把 JSON 包进```围栏——包着的"未达标"也必须触发重试。"""
    planner, llm = make_planner([
        final('["查资料"]'),
        final("敷衍的结果"),
        final('好的,我的判断是:\n```json\n{"ok": false, "advice": "补充出处"}\n```'),
        final("认真的结果"),
    ])
    assert planner.run("查") == "认真的结果"
    assert "补充出处" in llm.history[3][-1]["content"]


def test_planner_reflect_missing_ok_key_passes():
    """解析出的 JSON 没有 ok 字段时不采信"未达标",避免空建议白白重试。"""
    planner, llm = make_planner([
        final('["查资料"]'),
        final("结果"),
        final('{"OK": true}'),  # 键名大小写不对,视为没有 ok 字段
    ])
    assert planner.run("查") == "结果"
    assert len(llm.history) == 3  # 不应产生第 4 次调用(重试)


def test_planner_reflect_gibberish_passes():
    """质检员输出解析不了时放行,不能把主流程卡死。"""
    planner, llm = make_planner([
        final('["查资料"]'),
        final("结果"),
        final("这个嘛,我觉得还行吧"),  # 反思输出不是 JSON
    ])
    assert planner.run("查") == "结果"


def test_planner_uses_cloud_for_plan_and_local_for_reflection_and_synthesis():
    """Plan 的外部调研步骤走主模型，反思和最终汇总走本地模型。"""
    research = RecordingLLM([
        final('["检索资料", "整理证据"]'),
        final("资料结果 A"),
        final("证据结果 B"),
    ])
    summary = RecordingLLM([
        final('{"ok": true}'),
        final('{"ok": true}'),
        final("本地最终综述"),
    ])
    agent = Agent(research, ToolRegistry([CalculatorTool()]), verbose=False)
    planner = Planner(
        research,
        agent,
        summary_llm=summary,
        verbose=False,
    )

    assert planner.run("调研并整理资料") == "本地最终综述"
    assert len(research.history) == 3
    assert len(summary.history) == 3


if __name__ == "__main__":
    # 不依赖 pytest 的极简测试运行器
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过:{name}")
    print("全部测试通过")


def test_synthesize_keeps_step_conclusion_tail():
    """汇总材料超预算时必须保头保尾:步骤结论(常在结尾)不能被砍掉。

    曾用"只保开头 1200 字符"导致最终回答看不到各步结论,
    输出显得残缺——这正是本测试守护的回归点。
    """
    tail_marker = "TAIL_CONCLUSION_结论在结尾"
    middle_marker = "MIDDLE_GAP_中间应被砍掉"
    long_result = "x" * 2500 + middle_marker + "y" * 1500 + tail_marker
    assert len(long_result) > 4000

    research = RecordingLLM([
        final('["阅读论文", "撰写综述"]'),
        final(long_result),
        final("第二步的普通结果"),
    ])
    summary = RecordingLLM([
        final('{"ok": true}'),
        final('{"ok": true}'),
        final("最终综述"),
    ])
    agent = Agent(research, ToolRegistry([CalculatorTool()]), verbose=False)
    planner = Planner(research, agent, summary_llm=summary, verbose=False)

    assert planner.run("调研并整理资料") == "最终综述"
    synthesize_prompt = summary.history[-1][-1]["content"]
    assert tail_marker in synthesize_prompt
    assert middle_marker not in synthesize_prompt
