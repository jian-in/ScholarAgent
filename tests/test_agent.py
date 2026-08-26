"""核心框架的离线测试:不联网、不花钱,验证 Agent 循环的关键行为。

运行方式(项目根目录下):
    python -m pytest tests -q      (推荐)
    python tests/test_agent.py     (没装 pytest 时直接跑)
"""

import os
import sys

# 保证无论从哪个目录运行,都能 import 到项目根目录下的 scholaragent 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.agent import Agent
from scholaragent.llm import ScriptedLLM
from scholaragent.tool import STOP_RETRY_PREFIX, Tool, ToolRegistry
from scholaragent.tools.calculator import CalculatorTool, safe_eval


def make_agent(replies):
    """搭一个用假模型驱动的 Agent,供各测试复用。"""
    llm = ScriptedLLM(replies)
    agent = Agent(llm, ToolRegistry([CalculatorTool()]), verbose=False)
    return agent, llm


def test_react_loop():
    """主流程:模型先调工具、后作答,工具结果应回填进对话历史。"""
    agent, llm = make_agent([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "calculator",
             "arguments": {"expression": "(3+5)*12"}}]},
        {"content": "计算结果是 96", "tool_calls": []},
    ])
    answer = agent.run("帮我算 (3+5)*12")

    assert answer == "计算结果是 96"
    # 假模型最后一次收到的对话历史里,应有且仅有一条工具结果,内容是 96
    tool_msgs = [m for m in llm.last_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "96"
    assert tool_msgs[0]["tool_call_id"] == "call_1"


def test_unknown_tool_reported_not_crashed():
    """模型点名不存在的工具时,应把错误当结果回传,而不是程序崩溃。"""
    agent, llm = make_agent([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "no_such_tool", "arguments": {}}]},
        {"content": "好的", "tool_calls": []},
    ])
    agent.run("随便说点什么")
    tool_msg = [m for m in llm.last_messages if m.get("role") == "tool"][0]
    assert "不存在" in tool_msg["content"]


def test_max_steps_fuse():
    """模型永远要求调工具时,max_steps 保险丝应中止循环。"""
    endless = {"content": None, "tool_calls": [
        {"id": "x", "name": "calculator", "arguments": {"expression": "1+1"}}]}
    agent, _ = make_agent([dict(endless) for _ in range(99)])
    agent.max_steps = 3
    answer = agent.run("死循环测试")
    assert "最大步数" in answer


def test_temporary_tool_failure_disables_retries_for_current_run():
    """外部服务熔断后,模型再次点名也不应真的执行工具。"""
    class TemporaryFailureTool(Tool):
        name = "temporary"

        def __init__(self):
            self.calls = 0

        def run(self) -> str:
            self.calls += 1
            return f"{STOP_RETRY_PREFIX} 服务繁忙"

    tool = TemporaryFailureTool()
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "temporary", "arguments": {}}]},
        {"content": None, "tool_calls": [
            {"id": "call_2", "name": "temporary", "arguments": {}}]},
        {"content": "已说明资料缺口", "tool_calls": []},
    ])
    answer = Agent(llm, ToolRegistry([tool]), verbose=False).run("测试熔断")

    assert answer == "已说明资料缺口"
    assert tool.calls == 1
    tool_messages = [m for m in llm.last_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert all(m["content"].startswith(STOP_RETRY_PREFIX)
               for m in tool_messages)


def test_per_tool_call_limit_prevents_execution():
    """达到单轮预算后,同一工具的新调用只回传说明,不继续执行。"""
    class CountingTool(Tool):
        name = "counting"

        def __init__(self):
            self.calls = 0

        def run(self) -> str:
            self.calls += 1
            return "ok"

    tool = CountingTool()
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "counting", "arguments": {}}]},
        {"content": None, "tool_calls": [
            {"id": "call_2", "name": "counting", "arguments": {}}]},
        {"content": "结束", "tool_calls": []},
    ])
    agent = Agent(llm, ToolRegistry([tool]), verbose=False,
                  tool_call_limits={"counting": 1})

    assert agent.run("测试调用预算") == "结束"
    assert tool.calls == 1


def test_required_arguments_are_reused_across_tools():
    """下载后读取时模型漏掉同名编号，Agent 应沿用本轮已知参数。"""
    class FirstTool(Tool):
        name = "first"
        parameters = {
            "type": "object", "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
        }

        def run(self, item_id):
            return f"ready:{item_id}"

    class SecondTool(FirstTool):
        name = "second"

        def __init__(self):
            self.received = None

        def run(self, item_id):
            self.received = item_id
            return "read"

    second = SecondTool()
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "first", "arguments": {"item_id": "P1"}}]},
        {"content": None, "tool_calls": [
            {"id": "call_2", "name": "second", "arguments": {}}]},
        {"content": "完成", "tool_calls": []},
    ])

    answer = Agent(llm, ToolRegistry([FirstTool(), second]),
                   verbose=False).run("测试参数继承")

    assert answer == "完成"
    assert second.received == "P1"


def test_empty_model_reply_does_not_end_run():
    llm = ScriptedLLM([
        {"content": None, "tool_calls": []},
        {"content": "补全后的答案", "tool_calls": []},
    ])

    answer = Agent(llm, ToolRegistry([]), verbose=False).run("请回答")

    assert answer == "补全后的答案"
    assert "上一条回复为空" in llm.last_messages[-1]["content"]


def test_required_tool_completion_rejects_premature_final_answer():
    class ReadingTool(Tool):
        name = "reading"

        def __init__(self):
            self.done = False

        def completion_ready(self):
            return self.done

        def run(self):
            self.done = True
            return "全文读完"

    tool = ReadingTool()
    llm = ScriptedLLM([
        {"content": "我已经读完了", "tool_calls": []},
        {"content": None, "tool_calls": [
            {"id": "call_read", "name": "reading", "arguments": {}}]},
        {"content": "基于全文的结论", "tool_calls": []},
    ])
    agent = Agent(llm, ToolRegistry([tool]), verbose=False,
                  required_tool_completions=["reading"])

    assert agent.run("精读") == "基于全文的结论"
    assert tool.done


def test_short_final_answer_is_prompted_to_finish():
    llm = ScriptedLLM([
        {"content": "稍后总结", "tool_calls": []},
        {"content": "完整结论" * 20, "tool_calls": []},
    ])
    agent = Agent(llm, ToolRegistry([]), verbose=False, min_final_chars=20)

    answer = agent.run("精读")

    assert len(answer) >= 20
    assert "精读结论尚不完整" in llm.last_messages[-1]["content"]


def test_calculator_rejects_code():
    """安全边界:计算器绝不能执行数字运算以外的任何东西。"""
    for evil in ["__import__('os')", "open('x')", "1 if 1 else 2", "'a'*3"]:
        try:
            safe_eval(evil)
            raise AssertionError(f"不该放行:{evil}")
        except ValueError:
            pass  # 正确:被拒绝了


def test_calculator_rejects_huge_pow():
    """安全边界:超大乘方应被快速拒绝,而不是把 CPU/内存烧死。"""
    for bomb in ["9**9**9", "2**10000", "10**10**10"]:
        try:
            safe_eval(bomb)
            raise AssertionError(f"不该放行:{bomb}")
        except ValueError:
            pass  # 正确:被拒绝了


def test_bad_tool_arguments_reported_not_crashed():
    """模型给的参数不是合法 JSON 时(error 字段非空),错误应回传给模型。"""
    agent, llm = make_agent([
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "calculator", "arguments": {},
             "error": "工具参数不是合法 JSON,请修正后重新调用"}]},
        {"content": "好的", "tool_calls": []},
    ])
    agent.run("测试参数解析失败")
    tool_msg = [m for m in llm.last_messages if m.get("role") == "tool"][0]
    assert "不是合法 JSON" in tool_msg["content"]


def test_calculator_basic():
    assert safe_eval("(3+5)*12") == 96
    assert safe_eval("-2**3") == -8
    assert safe_eval("7//2 + 7%2") == 4
    assert safe_eval("10/4") == 2.5


if __name__ == "__main__":
    # 不依赖 pytest 的极简测试运行器
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过:{name}")
    print("全部测试通过")
