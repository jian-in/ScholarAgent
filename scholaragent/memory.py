"""记忆层(M2):会话记忆 + 长期记忆。

"记忆"听起来玄,拆开只有两件事:写下来 + 需要时捞回来。
本文件实现两种形态:

1. ConversationMemory(会话记忆)
   保存对话消息列表,让 Agent 的多次 run() 连成一段连续对话。
   难点在"裁剪":对话越攒越长,必须丢弃旧内容守住上下文预算,
   而且不能把 工具结果(role=tool)和它对应的 assistant 消息拆散,
   否则发给 API 会直接报错 —— 所以只能按"轮"为单位整块丢弃。

2. MemoryStore(长期记忆)
   把重要结论写进 JSONL 文件持久保存,用 BM25 算法按相关性检索。
   BM25 是搜索引擎沿用几十年的经典算法,这里从零手写(含中文分词),
   不依赖任何第三方库 —— 它就是"检索增强"(RAG)最朴素的底座。
   以后想换向量检索,只要实现同样的 search() 接口即可(M5 再做)。
"""

import json
import math
import os
import re
import time

from . import config

# ―――――――――――――――――― 会话记忆 ――――――――――――――――――


class ConversationMemory:
    """跨 run() 保存对话消息,超预算时按"轮"裁剪。

    一"轮" = 一条 user 消息带出的所有后续消息(assistant、tool)。
    裁剪时永远保留 system 提示词,然后从最旧的轮开始整块丢弃。
    """

    def __init__(self, max_chars: int = None):
        self.max_chars = max_chars or config.CONVERSATION_MAX_CHARS
        self.messages = []  # OpenAI 消息格式的字典列表

    def load(self, system_prompt: str) -> list:
        """取出对话历史;第一次使用时用 system 提示词初始化。"""
        if not self.messages:
            self.messages = [{"role": "system", "content": system_prompt}]
        return self.messages

    def save(self, messages: list):
        """写回对话历史,并在超出预算时裁剪。"""
        self.messages = self._trim(messages)

    def _trim(self, messages: list) -> list:
        def total_chars(msgs):
            return sum(len(str(m.get("content") or "")) for m in msgs)

        if total_chars(messages) <= self.max_chars:
            return messages

        system, rest = messages[0], messages[1:]
        # 找出每一轮的起点(user 消息的下标)
        round_starts = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        if not round_starts:
            return [system]  # 退化情况:没有任何 user 轮,只留 system
        # 从最旧的轮开始丢,直到预算够用;至少保留最后一轮
        for start in round_starts[1:]:
            kept = [system] + rest[start:]
            if total_chars(kept) <= self.max_chars:
                return kept
        return [system] + rest[round_starts[-1]:]


# ―――――――――――――――――― 分词 ――――――――――――――――――

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list:
    """极简中英文分词:英文按单词,中文按"相邻两字"。

    中文不用装分词库:把"智能体框架"切成 智能/能体/体框/框架,
    BM25 靠二字组合就能把相关文档排到前面 —— 工程上叫字符 bigram,
    是中文检索里性价比极高的土办法。

    刻意不收录单字:单字太常见(比如"主"既在"主题"也在"主线"里),
    会把不相关的文档也捞回来。只有当一段连续汉字本身只有一个字时
    (即被英文、数字或标点隔开的孤字,如"猫 cat"里的"猫")才保留单字。
    """
    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    for chunk in _CJK_RE.findall(text):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return tokens


# ―――――――――――――――――― BM25 检索 ――――――――――――――――――


class BM25Index:
    """从零实现的 BM25 相关性排序(k1、b 用最常见的默认值)。

    直觉:一个词在文档里出现越多次越相关(但收益递减,由 k1 控制),
    这个词在越少的文档里出现越有区分度(IDF),长文档要吃点惩罚(b)。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = []   # 每篇文档的词列表
        self.doc_freq = {}      # 词 -> 出现过该词的文档数
        self.avg_len = 0.0

    def build(self, docs: list):
        """docs 是字符串列表,建立倒排统计。"""
        self.docs_tokens = [tokenize(d) for d in docs]
        self.doc_freq = {}
        for tokens in self.docs_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        n = len(self.docs_tokens)
        self.avg_len = (sum(len(t) for t in self.docs_tokens) / n) if n else 0.0

    def search(self, query: str, top_k: int = 3) -> list:
        """返回 [(得分, 文档下标)],按相关性从高到低,得分为 0 的不要。"""
        n = len(self.docs_tokens)
        if n == 0:
            return []
        scores = [0.0] * n
        for token in set(tokenize(query)):
            df = self.doc_freq.get(token)
            if not df:
                continue  # 语料里根本没有这个词
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            for i, tokens in enumerate(self.docs_tokens):
                tf = tokens.count(token)
                if not tf:
                    continue
                length_norm = 1 - self.b + self.b * len(tokens) / (self.avg_len or 1)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0), reverse=True)
        return ranked[:top_k]


# ―――――――――――――――――― 长期记忆 ――――――――――――――――――


class MemoryStore:
    """长期记忆:JSONL 文件持久化 + BM25 相关性检索。

    JSONL = 每行一个 JSON 对象,追加写入不破坏旧数据,人眼能直接读。
    它的可靠性来自"逐行独立":就算某一行被写坏(断电、手改出错),
    也只损失那一行 —— 前提是加载时必须容错地跳过坏行,而不是崩溃。
    """

    def __init__(self, path: str = None):
        self.path = path or os.path.join(config.DATA_DIR, "memory", "memories.jsonl")
        self._entries = None   # 惰性加载
        self._index = None
        self._loaded_size = -1  # 上次加载时的文件大小,用于检测"别人写了新内容"

    def _load(self):
        """加载记忆文件;若文件在加载后又被写过(别的实例/进程),自动重读。

        为什么要这样:同一个文件可能被多个 MemoryStore 实例共用
        (比如 remember 工具一个、Agent 的自动回忆一个),只认第一次
        加载的内存缓存会读到过期数据 —— 缓存失效是工程里最常见的坑之一,
        对策是每次读之前先核对文件有没有变。
        """
        if os.path.exists(self.path):
            stat = os.stat(self.path)
            current_stamp = (stat.st_size, stat.st_mtime_ns)  # 大小+修改时间双重核对
        else:
            current_stamp = None
        if self._entries is not None and current_stamp == self._loaded_size:
            return  # 文件没变,缓存可信
        self._entries = []
        skipped = 0
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1  # 坏一行只跳过一行,这正是 JSONL 的价值
                        continue
                    self._entries.append(entry)
        if skipped:
            print(f"(长期记忆文件里有 {skipped} 行损坏,已跳过,其余正常加载)")
        self._loaded_size = current_stamp
        self._rebuild()

    def _rebuild(self):
        self._index = BM25Index()
        # e.get 而不是 e["text"]:手改文件漏了字段也只是检索不到,不崩溃
        self._index.build([e.get("text", "") for e in self._entries])

    def add(self, text: str, source: str = "") -> str:
        self._load()
        entry = {
            "text": text.strip(),
            "source": source,
            "time": time.strftime("%Y-%m-%d %H:%M"),
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # 若上次写入被打断留了半行,先补一个换行,防止新旧两条粘成一行
        needs_newline = False
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._entries.append(entry)
        stat = os.stat(self.path)
        self._loaded_size = (stat.st_size, stat.st_mtime_ns)
        self._rebuild()  # 记忆量不大,每次重建索引最简单可靠;大了再优化
        return f"已存入长期记忆(现共 {len(self._entries)} 条)"

    def search(self, query: str, top_k: int = 3) -> list:
        """返回最相关的记忆条目列表(dict),没有相关的返回空列表。"""
        self._load()
        return [self._entries[i] for _, i in self._index.search(query, top_k)]
