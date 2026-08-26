"""上下文压缩的离线测试:验证旧工具结果被压缩、最近结果保持原文、
消息配对不被破坏,并实测压缩前后的发送字符量。

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
    """记录每次 chat 收到的完整历史快照,供逐次断言。"""

    def __init__(self, replies):
        super().__init__(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        # 浅拷贝即可:content 是字符串,后续压缩替换的是新字符串,
        # 不会污染这里存下的快照
        self.calls.append([dict(m) for m in messages])
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


def test_old_observations_compressed_recent_kept():
    """6 次工具调用后:最早 4 条被压缩(带标记),最近 2 条保持原文。"""
    llm, _ = make_run(6)
    last = llm.calls[-1]
    tool_msgs = [m for m in last if m.get("role") == "tool"]
    assert len(tool_msgs) == 6
    for msg in tool_msgs[:4]:
        assert COMPRESSED_MARK in msg["content"]
        assert len(msg["content"]) < 1000  # 6000 字符压到远小于原文
    for msg in tool_msgs[4:]:
        assert COMPRESSED_MARK not in msg["content"]
        assert len(msg["content"]) > 5000  # 最近两条原样


def test_message_structure_and_pairing_preserved():
    """压缩只改内容:消息顺序、角色序列和 tool_call_id 配对必须完整。"""
    llm, _ = make_run(4)
    last = llm.calls[-1]
    roles = [m.get("role") for m in last]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles.count("assistant") == 4  # 4 次工具调用的思考
    assert roles.count("tool") == 4
    # 每条 tool 消息的 id 都能在某条 assistant 消息的 tool_calls 里找到
    assistant_ids = {
        tc["id"] for m in last if m.get("role") == "assistant"
        for tc in m.get("tool_calls", [])
    }
    for msg in last:
        if msg.get("role") == "tool":
            assert msg["tool_call_id"] in assistant_ids


def test_compression_is_idempotent():
    """已压缩的消息再次进入压缩不会二次截断。"""
    llm, tool = make_run(4)
    last = llm.calls[-1]
    compressed = [m for m in last if m.get("role") == "tool"
                  and COMPRESSED_MARK in m["content"]]
    assert compressed
    lengths_before = [len(m["content"]) for m in compressed]
    Agent(ScriptedLLM([]), ToolRegistry([tool]),
          verbose=False)._compress_history(last)
    lengths_after = [len(m["content"]) for m in compressed]
    assert lengths_after == lengths_before


def test_compression_disabled_when_config_zero(monkeypatch):
    """AGENT_CONTEXT_OLD_OBSERVATION_CHARS=0 时完全不压缩。"""
    monkeypatch.setattr(config, "AGENT_CONTEXT_OLD_OBSERVATION_CHARS", 0)
    llm, _ = make_run(4)
    last = llm.calls[-1]
    tool_msgs = [m for m in last if m.get("role") == "tool"]
    assert len(tool_msgs) == 4
    for msg in tool_msgs:
        assert COMPRESSED_MARK not in msg["content"]
        assert len(msg["content"]) > 5000


def test_measured_char_saving():
    """同一段剧本,压缩开启 vs 关闭,统计模型实际收到的总字符量。"""
    llm_compressed, _ = make_run(9)
    total_compressed = sum(
        len(str(m.get("content") or ""))
        for call in llm_compressed.calls for m in call)

    original = config.AGENT_CONTEXT_OLD_OBSERVATION_CHARS
    config.AGENT_CONTEXT_OLD_OBSERVATION_CHARS = 0
    try:
        llm_plain, _ = make_run(9)
    finally:
        config.AGENT_CONTEXT_OLD_OBSERVATION_CHARS = original
    total_plain = sum(
        len(str(m.get("content") or ""))
        for call in llm_plain.calls for m in call)

    assert total_compressed < total_plain * 0.5, (
        f"压缩后 {total_compressed} 字符,未压缩 {total_plain} 字符,"
        "节省应超过一半"
    )


def test_conversation_memory_receives_compressed_history():
    """写回会话记忆的也是压缩后历史,跨轮预算同步受益。"""
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
    assert COMPRESSED_MARK in tool_msgs[0]["content"]  # 最早一条已压缩
    assert COMPRESSED_MARK not in tool_msgs[2]["content"]  # 最近一条原文
