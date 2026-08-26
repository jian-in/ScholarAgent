"""M2 记忆系统的离线测试。

运行方式(项目根目录下):
    python -m pytest tests -q      (推荐)
    python tests/test_m2_memory.py (没装 pytest 时直接跑)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.agent import Agent
from scholaragent.llm import ScriptedLLM
from scholaragent.memory import BM25Index, ConversationMemory, MemoryStore, tokenize
from scholaragent.tool import ToolRegistry
from scholaragent.tools.calculator import CalculatorTool

# ―― 分词 ――――――――――――――――――――――――――――――――――――――――


def test_tokenize_mixed():
    tokens = tokenize("ReAct 智能体框架")
    assert "react" in tokens          # 英文转小写
    assert "智能" in tokens            # 中文二字组合
    assert "框架" in tokens
    assert "智" not in tokens          # 单字刻意不收录(太常见,会捞回无关文档)
    assert tokenize("猫") == ["猫"]     # 整段只有一个字时才保留单字


# ―― BM25 ――――――――――――――――――――――――――――――――――――――――


def test_bm25_ranks_relevant_doc_first():
    docs = [
        "今天食堂的红烧肉很好吃",
        "ReAct 论文提出推理与行动交替的智能体框架",
        "BM25 是经典的文本检索排序算法",
    ]
    index = BM25Index()
    index.build(docs)

    top = index.search("智能体 框架", top_k=2)
    assert top, "应该有命中"
    assert top[0][1] == 1  # 最相关的是第 2 篇(下标 1)

    assert index.search("量子力学", top_k=3) == []  # 无关词不硬凑结果


def test_bm25_empty_corpus():
    index = BM25Index()
    index.build([])
    assert index.search("任何词") == []


# ―― 长期记忆 ――――――――――――――――――――――――――――――――――――


def test_memory_store_persist_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memories.jsonl")

        store = MemoryStore(path)
        store.add("ReAct 论文的 arXiv 编号是 2210.03629", source="2210.03629")
        store.add("用户喜欢蓝色")

        # 新开一个实例模拟重启程序:数据应该还在(持久化)
        store2 = MemoryStore(path)
        hits = store2.search("ReAct 编号")
        assert hits and "2210.03629" in hits[0]["text"]
        assert store2.search("不存在的主题") == []


def test_memory_store_sees_other_instances_writes():
    """同一文件的两个实例:一个写入后,另一个(已加载过缓存的)要能看到。

    这正是 main.py 里的真实场景:remember 工具和 Agent 的自动回忆
    各持有一个 MemoryStore 实例,共用同一个 JSONL 文件。
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.jsonl")
        reader, writer = MemoryStore(path), MemoryStore(path)

        reader.search("先随便搜一下")  # 触发 reader 加载(此时文件为空)
        writer.add("ReAct 论文的 arXiv 编号是 2210.03629")

        hits = reader.search("ReAct 编号")  # reader 必须发现文件变了
        assert hits and "2210.03629" in hits[0]["text"]


# ―― 会话记忆裁剪 ――――――――――――――――――――――――――――――――――


def test_conversation_trim_keeps_system_and_rounds():
    memory = ConversationMemory(max_chars=600)
    messages = [{"role": "system", "content": "系统提示词"}]
    # 造 5 轮对话,每轮约 200 字符,总量远超预算
    for i in range(5):
        messages.append({"role": "user", "content": f"问题{i}:" + "x" * 100})
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "calculator",
                                                      "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "96"})
        messages.append({"role": "assistant", "content": f"回答{i}:" + "y" * 80})

    memory.save(messages)
    trimmed = memory.messages

    assert trimmed[0]["role"] == "system"           # system 永远保留
    assert trimmed[1]["role"] == "user"             # 裁剪必须落在轮的边界上
    user_contents = [m["content"] for m in trimmed if m.get("role") == "user"]
    assert any(c.startswith("问题4") for c in user_contents)      # 最新一轮一定在
    assert not any(c.startswith("问题0") for c in user_contents)  # 最旧一轮被丢弃
    # 不能出现"孤儿"工具消息:每条 tool 前面必须有带 tool_calls 的 assistant
    for i, m in enumerate(trimmed):
        if m.get("role") == "tool":
            assert trimmed[i - 1].get("tool_calls"), "tool 消息被拆散了"


def test_memory_store_skips_corrupt_lines():
    """JSONL 里有坏行时:跳过坏行,其余照常工作,绝不整体瘫痪。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"text": "完整的一条记忆 ReAct", "source": "", "time": "t"}\n')
            f.write('{"text": "被断电截断的半行 ...\n')      # 坏行
            f.write('{"没有text字段": true}\n')              # 缺字段的行

        store = MemoryStore(path)
        hits = store.search("ReAct 记忆")
        assert hits and "完整的一条" in hits[0]["text"]
        # 坏文件上继续追加也要正常
        store.add("新的一条")
        assert store.search("新的一条")


def test_conversation_trim_without_user_rounds():
    """退化情况:超预算但没有任何 user 轮,只留 system,不能崩溃。"""
    memory = ConversationMemory(max_chars=10)
    memory.save([{"role": "system", "content": "x" * 100}])
    assert memory.messages == [{"role": "system", "content": "x" * 100}]


def test_auto_recall_not_persisted_into_history():
    """注入的回忆只服务当轮:写回会话记忆的 user 消息必须是原话。"""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(os.path.join(tmp, "m.jsonl"))
        store.add("ReAct 论文的 arXiv 编号是 2210.03629")

        conv = ConversationMemory()
        agent, llm = make_agent(
            [{"content": "好", "tool_calls": []}],
            conversation=conv, long_memory=store, auto_recall=True,
        )
        agent.run("帮我查 ReAct 论文的编号")

        # 发给模型的那一份带注入(RAG 生效)……
        sent = [m for m in llm.last_messages if m.get("role") == "user"][0]
        assert "长期记忆" in sent["content"]
        # ……但存进会话记忆的那一份必须是用户原话(不污染历史)
        saved = [m for m in conv.messages if m.get("role") == "user"][0]
        assert saved["content"] == "帮我查 ReAct 论文的编号"


def test_conversation_no_trim_when_small():
    memory = ConversationMemory(max_chars=99999)
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
    memory.save(messages)
    assert len(memory.messages) == 3  # 没超预算就一条都不动


# ―― Agent 集成:多轮对话 + 自动回忆 ――――――――――――――――――――


def make_agent(replies, **kwargs):
    llm = ScriptedLLM(replies)
    agent = Agent(llm, ToolRegistry([CalculatorTool()]), verbose=False, **kwargs)
    return agent, llm


def test_agent_multi_turn_conversation():
    """第二次 run() 时,模型应能看到第一次的对话内容。"""
    agent, llm = make_agent(
        [{"content": "第一答", "tool_calls": []},
         {"content": "第二答", "tool_calls": []}],
        conversation=ConversationMemory(),
    )
    agent.run("我叫严谨温")
    agent.run("我叫什么?")

    history = [str(m.get("content")) for m in llm.last_messages]
    assert any("我叫严谨温" in c for c in history), "上一轮的内容丢了"
    assert any("第一答" in c for c in history)


def test_agent_auto_recall_injects_memory():
    """开启 auto_recall 时,相关长期记忆应自动附在任务后面。"""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(os.path.join(tmp, "m.jsonl"))
        store.add("ReAct 论文的 arXiv 编号是 2210.03629")

        agent, llm = make_agent(
            [{"content": "好", "tool_calls": []}],
            long_memory=store, auto_recall=True,
        )
        agent.run("帮我查 ReAct 论文的编号")

        user_msg = [m for m in llm.last_messages if m.get("role") == "user"][0]
        assert "长期记忆" in user_msg["content"]
        assert "2210.03629" in user_msg["content"]


def test_agent_no_recall_when_disabled():
    """默认不开 auto_recall:行为和 M0 完全一致,老测试不受影响。"""
    agent, llm = make_agent([{"content": "好", "tool_calls": []}])
    agent.run("随便问问")
    user_msg = [m for m in llm.last_messages if m.get("role") == "user"][0]
    assert "长期记忆" not in user_msg["content"]


if __name__ == "__main__":
    # 不依赖 pytest 的极简测试运行器
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过:{name}")
    print("全部测试通过")
