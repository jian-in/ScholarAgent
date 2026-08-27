"""面向资料缺口的四路文献调研契约。

普通调研允许模型自己决定下一步；资料缺口补全不应依赖模型临场记忆，
否则很容易只写一段泛泛的“后续工作”。本模块只负责确定性识别、拆题和
输出要求，实际检索、下载、阅读仍由既有 Agent 工具链完成。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class GapTrack:
    """一个独立的研究缺口方向。"""

    key: str
    title: str
    query: str
    focus: str

    def plan_step(self, depth: str = "deep") -> str:
        if depth == "fast":
            return (
                f"{self.title}：使用英文关键词“{self.query}”检索至少 3 篇候选论文，"
                "只做摘要级筛选，不下载 PDF、不调用 read_paper。"
                f"重点回答{self.focus}。每篇候选按“论文标题/arXiv 编号/年份、"
                "摘要中能确认的核心问题、核心方法或机制、摘要给出的实验/评价结论、"
                "局限性或未提供的信息、摘要证据”形成快速论文卡片；摘要没有支持的内容"
                "必须标记为“摘要未提供”，不能用常识补写。"
            )
        return (
            f"{self.title}：使用英文关键词“{self.query}”检索至少 3 篇相关论文，"
            "筛选至少 1 篇代表性论文下载并用 read_paper 精读到全文读完。"
            f"重点回答{self.focus}。每篇论文必须按“论文标题/arXiv 编号/年份、"
            "核心问题、核心方法或机制、实验设置与关键结论、局限性、证据页码或摘要”"
            "形成论文核心卡片；检索或阅读失败要明确记录，不能用常识补写。"
        )


GAP_TRACKS = (
    GapTrack(
        key="reasoning",
        title="推理能力（CoT、ToT 等）",
        query="LLM reasoning chain of thought tree of thoughts inference",
        focus="模型如何分解、搜索、验证和修正推理过程，以及论文实验证明了什么",
    ),
    GapTrack(
        key="tool-use",
        title="Agent 工具使用（Tool Use）",
        query="LLM agent tool use function calling tool learning",
        focus="工具选择、参数生成、调用反馈和错误恢复分别由什么机制实现",
    ),
    GapTrack(
        key="recent-survey",
        title="2024-2025 年最新进展与系统性综述",
        query="LLM agent survey 2024 2025 systematic review",
        focus="综述覆盖的研究范围、分类框架、代表工作、评价结论和仍未解决的问题",
    ),
    GapTrack(
        key="inference-efficiency",
        title="硬件与推理效率（inference optimization）",
        query="LLM inference optimization hardware serving quantization speculative decoding",
        focus="优化作用于模型、显存、并行/服务系统的哪一层，以及速度、显存和质量的权衡",
    ),
)


_PRIMARY_MARKERS = (
    r"资料缺口",
    r"研究缺口",
    r"补齐(?:这|上述|当前)?(?:四|4)项",
    r"全部开始(?:调研)?",
)
_TRACK_MARKERS = (
    r"推理能力|chain[ -]?of[ -]?thought|\bCoT\b|\bToT\b|tree[ -]?of[ -]?thought",
    r"工具使用|tool use|tool-use|function calling",
    r"2024\s*[-~至到]\s*2025|2024.*2025|systematic review|系统性综述",
    r"推理效率|inference optimization|量化|quantization|推测解码|speculative decoding|硬件",
)


def is_gap_survey_task(task: str) -> bool:
    """判断任务是否明确要求补全四类资料缺口。"""

    text = str(task or "")
    primary_hit = any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _PRIMARY_MARKERS
    )
    if primary_hit and re.search(
        r"补齐|补充|调研|检索|启动|开始|全部|逐一|四项|4项|后续",
        text,
        flags=re.IGNORECASE,
    ):
        track_hits = sum(
            bool(re.search(pattern, text, flags=re.IGNORECASE))
            for pattern in _TRACK_MARKERS
        )
        if track_hits >= 3 or re.search(
            r"四项|4项|全部|各个|逐一",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return sum(
        bool(re.search(pattern, text, flags=re.IGNORECASE))
        for pattern in _TRACK_MARKERS
    ) >= 3


_FAST_MARKERS = (
    r"快速",
    r"摘要(?:筛选|扫描|级)",
    r"不下载",
    r"不(?:做|进行)全文精读",
    r"只看摘要",
)
_DEEP_MARKERS = (
    r"深度",
    r"全文精读",
    r"精读到全文",
    r"页码级",
    r"下载\s*PDF",
    r"下载论文",
)


def is_fast_gap_survey_task(task: str) -> bool:
    """判断缺口任务是否应采用摘要级快速扫描。

    快速是资料缺口任务的安全默认值；只有用户明确要求深度/全文证据时
    才进入高成本模式。
    """

    text = str(task or "")
    if not is_gap_survey_task(text):
        return False
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _FAST_MARKERS):
        return True
    return not any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _DEEP_MARKERS
    )


def build_gap_survey_plan(task: str, depth: str | None = None) -> list[str]:
    """返回固定四路计划，避免模型把多个缺口压成一个泛化步骤。"""

    if not is_gap_survey_task(task):
        raise ValueError("任务没有明确要求补全四类资料缺口")
    depth = depth or ("fast" if is_fast_gap_survey_task(task) else "deep")
    if depth not in {"fast", "deep"}:
        raise ValueError("缺口调研深度只能是 fast 或 deep")
    return [track.plan_step(depth=depth) for track in GAP_TRACKS]


def synthesis_instruction(task: str) -> str:
    """返回缺口调研的强制汇总格式；普通任务不加载这段约束。"""

    if not is_gap_survey_task(task):
        return ""
    fast_note = (
        "当前是快速摘要筛选：所有实验、局限和机制只能写摘要明确支持的内容；"
        "没有全文证据的地方必须标记为摘要级或未提供。\n"
        if is_fast_gap_survey_task(task) else ""
    )
    return (
        "这是一次四路资料缺口补全，不要把结果写成泛泛的未来展望。最终回答必须严格包含：\n"
        + fast_note
        + "1. 调研范围与执行状态：明确四个方向是否都已启动、每个方向检索/精读了多少篇；\n"
        + "2. 四个方向分别成节，每节先给出代表论文表，再逐篇写论文核心卡片：研究问题、"
        + "核心机制、实验与关键结论、局限性、arXiv 编号和证据页码/摘要；\n"
        + "3. 横向对比表：比较四个方向解决的问题、方法层次、证据强度和与本项目的关系；\n"
        + "4. 仍然存在的资料缺口：只列工具实际没有查到或没有精读的内容，并说明下一步；\n"
        + "5. 对本项目的可落地启示：每条启示必须能回指到具体论文，区分论文事实与你的推断。\n"
        + "如果某一方向没有足够证据，保留该方向标题并标记“未完成/证据不足”，不要用模板化"
        + "表述假装已经完成。"
    )


GAP_HANDOFF_MAX_CHARS = 3200
FAST_GAP_HANDOFF_MAX_CHARS = 1800
FAST_GAP_MAX_STEPS = 6
FAST_GAP_TOOL_NAMES = ("arxiv_search", "recall")


_TRACK_SECTION_ALIASES = (
    ("推理能力（CoT、ToT 等）", ("推理能力", "CoT", "ToT", "chain of thought", "tree of thoughts")),
    ("Agent 工具使用（Tool Use）", ("工具使用", "Tool Use", "tool-use", "function calling")),
    ("2024-2025 年最新进展与系统性综述", ("2024-2025", "2024", "2025", "系统性综述", "systematic review")),
    ("硬件与推理效率（inference optimization）", ("推理效率", "inference optimization", "量化", "quantization", "硬件")),
)
_CORE_FIELD_MARKERS = (
    ("核心问题", ("核心问题", "研究问题", "research question")),
    ("核心方法/机制", ("核心方法", "核心机制", "方法机制", "method", "mechanism")),
    ("实验/评价结论", ("实验", "评价", "结果", "结论", "experiment", "evaluation")),
    ("局限性", ("局限", "限制", "limitation")),
)


def missing_synthesis_sections(answer: str) -> list[str]:
    """检查缺口调研汇总是否至少具备四方向和论文核心字段。"""

    text = str(answer or "")
    missing = []
    for title, aliases in _TRACK_SECTION_ALIASES:
        if not any(re.search(re.escape(alias), text, flags=re.IGNORECASE) for alias in aliases):
            missing.append(title)
    for title, aliases in _CORE_FIELD_MARKERS:
        if not any(re.search(re.escape(alias), text, flags=re.IGNORECASE) for alias in aliases):
            missing.append(title)
    return missing
