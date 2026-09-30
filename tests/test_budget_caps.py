"""单步调用上限与 run 级字符预算的离线回归。

P1-2 的两个护栏:
- max_tool_calls_per_step:一轮内模型返回再多 tool_call,也只执行前 N 个,
  超出的按短路分支回传文字说明(不执行、但不把工具整轮停用)。
- run_char_budget:单轮内 messages 只追加不裁剪,累计字符超限走
  _closeout 部分报告通道,而不是继续烧钱。
"""

import unittest

from scholaragent import config
from scholaragent.agent import Agent
from scholaragent.llm import ScriptedLLM
from scholaragent.tool import Tool, ToolRegistry


class EchoTool(Tool):
    name = "echo"

    def __init__(self):
        self.calls = 0

    def run(self, text=""):
        self.calls += 1
        return f"echo:{text}"


class BudgetCapsTests(unittest.TestCase):
    def test_config_defaults(self):
        self.assertEqual(config.AGENT_MAX_TOOL_CALLS_PER_STEP, 4)
        self.assertEqual(config.AGENT_RUN_CHAR_BUDGET, 200000)
        self.assertEqual(config.PAPER_READER_MAX_STEPS, 20)
        self.assertEqual(config.PAPER_READER_CHUNK_CHARS, 4800)

    def test_agent_param_falls_back_to_config(self):
        registry = ToolRegistry([EchoTool()])
        agent = Agent(ScriptedLLM([]), registry, verbose=False)
        self.assertEqual(agent.max_tool_calls_per_step,
                         config.AGENT_MAX_TOOL_CALLS_PER_STEP)
        self.assertEqual(agent.run_char_budget, config.AGENT_RUN_CHAR_BUDGET)

    def test_explicit_zero_is_clamped_not_disabled(self):
        registry = ToolRegistry([EchoTool()])
        agent = Agent(ScriptedLLM([]), registry, verbose=False,
                      max_tool_calls_per_step=0)
        self.assertEqual(agent.max_tool_calls_per_step, 1)

    def test_per_step_cap_executes_only_first_n(self):
        tool = EchoTool()
        registry = ToolRegistry([tool])
        llm = ScriptedLLM([
            {"content": None, "tool_calls": [
                {"id": f"c{i}", "name": "echo", "arguments": {"text": f"t{i}"}}
                for i in range(6)]},
            {"content": None, "tool_calls": [
                {"id": "c6", "name": "echo", "arguments": {"text": "again"}}]},
            {"content": "done", "tool_calls": []},
        ])
        agent = Agent(llm, registry, verbose=False, system_prompt="s",
                      max_tool_calls_per_step=2)
        answer = agent.run("cap")
        self.assertEqual(answer, "done")
        # 第一步只执行 2 个,第二步的 1 个正常执行(工具没有被整轮停用)
        self.assertEqual(tool.calls, 3)
        self.assertEqual(registry.run_tool_calls, 3)
        tool_messages = [m for m in llm.last_messages
                         if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 7)
        dropped = [m for m in tool_messages if "单步上限" in m["content"]]
        self.assertEqual(len(dropped), 4)
        # 被丢弃的调用仍有 tool_call_id 配对,历史结构完整
        self.assertEqual(
            [m["tool_call_id"] for m in dropped], ["c2", "c3", "c4", "c5"])
        executed = [m["content"] for m in tool_messages[:2]]
        self.assertEqual(executed, ["echo:t0", "echo:t1"])

    def test_run_char_budget_triggers_closeout(self):
        tool = EchoTool()
        registry = ToolRegistry([tool])
        llm = ScriptedLLM([
            {"content": None, "tool_calls": [
                {"id": "c1", "name": "echo",
                 "arguments": {"text": "y" * 5000}}]},
            {"content": "收尾摘要", "tool_calls": []},
        ])
        agent = Agent(llm, registry, verbose=False, system_prompt="s",
                      max_steps=10, run_char_budget=3000)
        answer = agent.run("budget")
        # 工具只执行了一次,第二步开始前预算检查触发 _closeout
        self.assertEqual(tool.calls, 1)
        self.assertIn("## 阶段结果", answer)
        self.assertIn("budget_exhausted", answer)
        self.assertIn("收尾摘要", answer)
        self.assertEqual(agent.last_completion["reason"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
