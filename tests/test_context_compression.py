"""上下文定稿式裁剪的离线测试。

核心契约:工具结果在创建时即裁剪定稿,请求历史只追加、永不改写 ——
这是模型前缀缓存(命中部分约 1/4 价格)全程有效的条件。

运行方式(项目根目录下):
    python -m pytest tests/test_context_compression.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent import config
from scholaragent.agent import COMPRESSED_MARK, Agent
from scholaragent.llm import ScriptedLLM
from scholaragent.memory import ConversationMemory
from scholaragent.tool import Tool, ToolRegistry


class ChunkTool(Tool):
    """模拟 read_paper:每次调用返回一大段"论文片段"。"""

    name = "read_chunk"
    description = "读取一大段文本(测试用)"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, chunk_chars=6000):
        self.chunk_chars = chunk_chars
        self.calls = 0

    def run(self) -> str:
        self.calls += 1
        head = f"--- 第 {self.calls} 段 ---\n"
        return head + "x" * self.chunk_chars + f"\n(第 {self.calls} 段结束)"


class RecordingLLM(ScriptedLLM):
    """记录每次 chat 收到的完整历史快照与工具清单,供逐次断言。"""

    def __init__(self, replies):
        super().__init__(replies)
        self.calls = []
        self.tools_seen = []

    def chat(self, messages, tools=None):
        # 浅拷贝即可:content 是字符串,后续替换的是新字符串,
        # 不会污染这里存下的快照
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append([dict(t) for t in (tools or [])])
        return super().chat(messages, tools=tools)


def make_run(n_tool_steps, chunk_chars=6000):
    """跑一次 n_tool_steps 次工具调用后收尾的任务,返回 (llm, tool)。"""
    replies = [
        {"content": None, "tool_calls": [
            {"id": f"call_{i}", "name": "read_chunk", "arguments": {}}]}
        for i in range(1, n_tool_steps + 1)
    ] + [{"content": "完成", "tool_calls": []}]
    llm = RecordingLLM(replies)
    tool = ChunkTool(chunk_chars)
    Agent(llm, ToolRegistry([tool]), verbose=False).run("读完全部段落")
    return llm, tool


def test_observation_finalized_at_creation():
    """大结果在创建当步就已定稿,不存在"先发全文、事后改写"。"""
    llm, _ = make_run(4)
    for step_index, snapshot in enumerate(llm.calls):
        # 第 k 次请求时,历史里只有 k-1 条工具结果
        tool_msgs = [m for m in snapshot if m.get("role") == "tool"]
        for msg in tool_msgs:
            assert COMPRESSED_MARK in msg["content"], (
                f"第 {step_index + 1} 次请求时发现未定稿的工具结果")


def test_history_is_append_only():
    """缓存契约:任意一次请求的消息序列是下一次请求的严格前缀。

    任何事后改写(压缩/重排)都会破坏该性质并打失效前缀缓存。
    """
    llm, _ = make_run(6)
    for prev, curr in zip(llm.calls, llm.calls[1:]):
        assert curr[:len(prev)] == prev, (
            "检测到历史消息被改写:请求前缀不稳定,前缀缓存将失效")


def test_message_structure_and_pairing_preserved():
    """裁剪只改内容:消息顺序、角色序列和 tool_call_id 配对必须完整。"""
    llm, _ = make_run(4)
    last = llm.calls[-1]
    roles = [m.get("role") for m in last]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles.count("assistant") == 4  # 4 次工具调用的思考
    assert roles.count("tool") == 4
    assistant_ids = {
        tc["id"] for m in last if m.get("role") == "assistant"
        for tc in m.get("tool_calls", [])
    }
    for msg in last:
        if msg.get("role") == "tool":
            assert msg["tool_call_id"] in assistant_ids


def test_finalize_is_idempotent():
    """已定稿的内容再次进入裁剪不会二次截断。"""
    _, tool = make_run(2)
    agent = Agent(ScriptedLLM([]), ToolRegistry([tool]), verbose=False)
    sample = "y" * 9000
    once = agent._finalize_observation(sample)
    twice = agent._finalize_observation(once)
    assert once == twice


def test_prune_disabled_when_threshold_zero(monkeypatch):
    """AGENT_CONTEXT_PRUNE_THRESHOLD=0 时完全裁剪关闭,历史发原文。"""
    monkeypatch.setattr(config, "AGENT_CONTEXT_PRUNE_THRESHOLD", 0)
    llm, _ = make_run(3)
    for snapshot in llm.calls:
        for msg in snapshot:
            if msg.get("role") == "tool":
                assert COMPRESSED_MARK not in msg["content"]
                assert len(msg["content"]) > 5000


def test_measured_char_saving_with_tight_budget(monkeypatch):
    """收紧预算后,同一剧本模型收到的总字符量应大幅下降。"""
    monkeypatch.setattr(config, "AGENT_CONTEXT_PRUNE_THRESHOLD", 1000)
    monkeypatch.setattr(config, "AGENT_CONTEXT_PRUNE_HEAD", 600)
    monkeypatch.setattr(config, "AGENT_CONTEXT_PRUNE_TAIL", 300)
    llm_pruned, _ = make_run(6)
    total_pruned = sum(
        len(str(m.get("content") or ""))
        for call in llm_pruned.calls for m in call)

    monkeypatch.setattr(config, "AGENT_CONTEXT_PRUNE_THRESHOLD", 0)
    llm_plain, _ = make_run(6)
    total_plain = sum(
        len(str(m.get("content") or ""))
        for call in llm_plain.calls for m in call)

    assert total_pruned < total_plain * 0.5, (
        f"裁剪后 {total_pruned} 字符,未裁剪 {total_plain} 字符")


def test_conversation_memory_receives_finalized_history():
    """写回会话记忆的也是定稿后历史,跨轮预算同步受益。"""
    replies = [
        {"content": None, "tool_calls": [
            {"id": f"call_{i}", "name": "read_chunk", "arguments": {}}]}
        for i in range(1, 4)
    ] + [{"content": "完成", "tool_calls": []}]
    llm = ScriptedLLM(replies)
    memory = ConversationMemory()
    Agent(llm, ToolRegistry([ChunkTool()]), verbose=False,
          conversation=memory).run("读完全部段落")
    tool_msgs = [m for m in memory.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    for msg in tool_msgs:
        assert COMPRESSED_MARK in msg["content"]


def test_tool_schema_list_stays_stable_across_steps():
    """停用工具不再从请求的 tools 参数中移除,保持缓存前缀稳定。

    模型调用停用工具时会收到文字回传,约束照常生效。
    """
    from scholaragent.tool import STOP_RETRY_PREFIX
    from scholaragent.tools.calculator import CalculatorTool

    class TemporaryFailureTool(Tool):
        name = "temporary"

        def run(self) -> str:
            return f"{STOP_RETRY_PREFIX} 服务繁忙"

    replies = [
        {"content": None, "tool_calls": [
            {"id": "call_1", "name": "temporary", "arguments": {}}]},
        # 第二次调用时 temporary 已被停用,模型仍点名它
        {"content": None, "tool_calls": [
            {"id": "call_2", "name": "temporary", "arguments": {}}]},
        {"content": "好", "tool_calls": []},
    ]
    llm = RecordingLLM(replies)
    agent = Agent(llm, ToolRegistry([TemporaryFailureTool(), CalculatorTool()]),
                  verbose=False, tool_call_limits={"temporary": 1})
    answer = agent.run("测试")

    assert answer == "好"
    assert len(llm.tools_seen) == 3
    first = llm.tools_seen[0]
    for tools in llm.tools_seen[1:]:
        assert tools == first, "tools 参数在步间变化会打失效前缀缓存"
