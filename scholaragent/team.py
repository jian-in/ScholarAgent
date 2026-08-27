"""多智能体层(M4):检索员 → 精读员 → 写作员 的流水线协作。

为什么要多个 Agent:单个 Agent 身兼数职时,系统提示词越写越长、
工具越挂越多,模型顾此失彼。多智能体的思路是"分工":

    一个角色 = 一份专属系统提示词 + 一份工具白名单(见 tool.subset)

    检索员  只管搜索与筛选,产出「检索报告」
    精读员  只管下载与精读,产出「精读笔记」
    写作员  不碰检索与下载,只根据前两者的产出写「综述」

角色之间不共享对话历史,只传递书面交接物(报告/笔记)——
这既是团队协作的朴素模型,也天然控制了每个角色的上下文规模。
本模块用固定流水线编排(顺序确定、结果可复现、方便 M6 评测);
"由协调者模型动态派活"是它的进阶形态,留作论文的展望部分。
"""

from . import config
from .agent import Agent
from .events import RunContext

SEARCHER_PROMPT = (
    "你是文献检索员。只负责用 arxiv_search 检索并筛选论文,不下载不精读。"
    "产出一份检索报告,格式要求:第一行先写「推荐精读:」并给出最值得精读的"
    " 1-2 篇(标题 + arXiv 编号),然后再列出其余相关论文(标题、arXiv 编号、"
    "一句话概括、相关度高/中/低)。推荐必须放在最前面 —— 报告过长时"
    "结尾可能被截断,开头是最安全的位置。"
    "可以先用 recall 看看长期记忆里有没有已知的相关结论。"
)

READER_PROMPT = (
    "你是论文精读员。根据检索报告,用 download_paper 下载推荐的论文,"
    "用 read_paper 分段阅读,严格按结尾的位置参数持续翻页,直到看到全文读完。"
    "产出一份精读笔记:研究问题、方法、关键结论、局限性,注明页码;"
    "重要结论用 remember 存入长期记忆,笔记用 save_note 保存。"
    "论文里没有的内容不要编造。"
)

WRITER_PROMPT = (
    "你是综述写作员。根据检索报告和精读笔记撰写一份简明综述:"
    "研究背景、主要方法脉络、代表工作对比、开放问题。"
    "每篇代表论文都要交代研究问题、核心机制、实验/评价结论和局限性，"
    "让读者能看出论文到底解决了什么，而不是只复述摘要或给出空泛趋势。"
    "每个论点注明来源论文(标题 + arXiv 编号);"
    "材料里没有的内容不要编造,材料不足就明确说明缺口。"
    "需要回顾更多细节时可用 read_notes 和 recall 查阅。"
)


def _clip(text: str, limit: int = None) -> str:
    """交接物传给下一个角色前做截断,守住上下文预算。

    保头 + 保尾、砍中间:报告的推荐在开头,笔记的结论常在结尾,
    只保开头会把"结论在结尾"文体的关键信息整段砍掉。
    """
    text = str(text)
    limit = limit or config.TEAM_HANDOFF_MAX_CHARS
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.3)
    return text[:head] + "\n…(中间已截断)…\n" + text[-tail:]


class ResearchTeam:
    """固定流水线:检索员 → 精读员 → 写作员。"""

    def __init__(self, llm, registry, verbose=True, require_full_paper=True,
                 on_progress=None, should_stop=None, summary_llm=None):
        self.verbose = verbose
        self.on_progress = on_progress
        self.should_stop = should_stop
        self.research_llm = llm
        self.summary_llm = summary_llm or llm
        # 检索和精读由调研模型驱动；每个角色完成工具工作后，
        # Agent 会把证据交给本地总结模型，写作员始终只使用总结模型。
        self.searcher = Agent(
            self.research_llm, registry.subset(["arxiv_search", "recall"]),
            system_prompt=SEARCHER_PROMPT, max_steps=8, verbose=verbose,
            tool_call_limits={"arxiv_search": 2}, on_progress=on_progress,
            should_stop=should_stop, summary_llm=self.summary_llm)
        self.reader = Agent(
            self.research_llm, registry.subset(["download_paper", "read_paper",
                                                "save_note", "remember"]),
            system_prompt=READER_PROMPT,
            max_steps=config.PAPER_READER_MAX_STEPS, verbose=verbose,
            required_tool_completions=(
                ["read_paper"] if require_full_paper else []),
            min_final_chars=300 if require_full_paper else 0,
            on_progress=on_progress, should_stop=should_stop,
            summary_llm=self.summary_llm)
        self.writer = Agent(
            self.summary_llm, registry.subset(["read_notes", "recall"]),
            system_prompt=WRITER_PROMPT, max_steps=6, verbose=verbose,
            on_progress=on_progress, should_stop=should_stop)
        self.last_metrics = None

    def run(self, topic: str, context: RunContext = None) -> str:
        """围绕一个主题跑完整的调研流水线,返回综述。"""
        from .agent import CANCELLED_ANSWER

        owns_context = context is None
        context = context or RunContext(mode="team", should_stop=self.should_stop)
        self._active_context = context
        self._context_external = not owns_context
        try:
            if self._stop_requested():
                self._log("[取消] 用户已请求停止")
                return CANCELLED_ANSWER

            self._log(f"[团队] 检索员开工:{topic}")
            report = context.invoke(
                self.searcher,
                f"围绕主题「{topic}」检索并筛选论文,产出检索报告",
            )
            if report == CANCELLED_ANSWER or self._stop_requested():
                self._log("[取消] 流水线在检索阶段后停止")
                return CANCELLED_ANSWER

            self._log("[团队] 精读员开工")
            notes = context.invoke(
                self.reader,
                f"这是检索员的报告:\n{_clip(report)}\n\n"
                f"请精读其中推荐的论文,产出精读笔记",
            )
            if notes == CANCELLED_ANSWER or self._stop_requested():
                self._log("[取消] 流水线在精读阶段后停止")
                return CANCELLED_ANSWER

            self._log("[团队] 写作员开工")
            return context.invoke(
                self.writer,
                f"主题:{topic}\n\n检索报告:\n{_clip(report)}\n\n"
                f"精读笔记:\n{_clip(notes)}\n\n请撰写综述",
            )
        finally:
            self.last_metrics = (
                context.metrics.finish() if owns_context else context.metrics.snapshot()
            )
            self._active_context = None
            self._context_external = False

    def _stop_requested(self) -> bool:
        if getattr(self, "_active_context", None) is not None:
            return self._active_context.is_cancelled()
        if not self.should_stop:
            return False
        try:
            return bool(self.should_stop())
        except Exception:
            return False

    def _log(self, text: str):
        if self.verbose and not getattr(self, "_context_external", False):
            print(text)
        if self.on_progress:
            try:
                self.on_progress(text)
            except Exception:
                pass
