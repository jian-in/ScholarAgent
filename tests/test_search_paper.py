"""论文内检索 search_paper 与 read_paper 调用上限的离线测试。"""

from types import SimpleNamespace

import pytest

from scholaragent import config
from scholaragent.agent import Agent
from scholaragent.events import RunContext
from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import research_tool_limits
from scholaragent.team import ResearchTeam
from scholaragent.tool import Tool, ToolRegistry
from scholaragent.tools import BUILTIN_TOOLS, build_builtin_tools, papers
from scholaragent.tools.papers import PASSAGE_CHARS, SearchPaperTool

PAGES = [
    "ReAct: Synergizing Reasoning and Acting. Shunyu Yao, Princeton University; "
    "Google Research, Brain team.",
    "Related work on chain-of-thought prompting and action plan generation.",
    "On ALFWorld, the best ReAct trial achieves an average success rate of 71%, "
    "outperforming Act (45%) and BUTLER (37%). " + "Details of the setup. " * 60,
    "",  # 没有文字层的页面
]


@pytest.fixture
def paper(tmp_path, monkeypatch):
    """在工作区放一个占位 PDF,解析时返回固定的分页文字。"""
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "2210.03629v1.pdf").write_bytes(b"%PDF placeholder")
    reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda t=t: t) for t in PAGES])
    monkeypatch.setattr(papers, "PdfReader", lambda path: reader)
    return SearchPaperTool(workspace=str(tmp_path))


def test_search_returns_relevant_passages_with_pages_and_jump_hint(paper):
    result = paper.run("2210.03629v1", "ALFWorld success rate", top_k=2)
    assert result.splitlines()[0] == "在论文 2210.03629v1 中检索「ALFWorld success rate」,最相关的 1 段:"
    assert "--- 第 3 页 ---" in result and "71%" in result
    assert 'read_paper(arxiv_id="2210.03629v1", start_page=页码)' in result
    assert "另有 1 页没有文字层" in result


def test_overlapping_passages_on_the_same_page_are_not_repeated(paper):
    result = paper.run("2210.03629v1", "setup details", top_k=8)
    hits = [line for line in result.splitlines() if line.startswith("--- 第 3 页")]
    assert 1 <= len(hits) <= -(-len(PAGES[2]) // PASSAGE_CHARS)


def test_search_anchors_carry_page_and_excerpt(paper):
    from scholaragent.tool import ToolResult

    text = paper.run("2210.03629v1", "Princeton Brain team authors", top_k=1)
    [artifact] = paper.artifact_metadata({"arxiv_id": "2210.03629v1"}, ToolResult(text=text))
    [anchor] = artifact["source_anchors"]
    assert artifact["kind"] == "search"
    assert anchor["page"] == 1 and anchor["source"] == "arxiv:2210.03629v1"
    assert "Princeton University" in anchor["excerpt"]
    assert "只返回了相关片段" not in anchor["excerpt"]


def test_search_reports_misses_and_bad_inputs_in_plain_words(paper, tmp_path, monkeypatch):
    assert "没有找到" in paper.run("2210.03629v1", "quantum chromodynamics")
    assert "请给出要检索的关键词" in paper.run("2210.03629v1", "  ")
    assert "还没下载" in paper.run("2401.00001", "anything")

    blank = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "")])
    monkeypatch.setattr(papers, "PdfReader", lambda path: blank)
    assert "改用 read_paper" in SearchPaperTool(workspace=str(tmp_path)).run("2210.03629v1", "x")


def test_search_is_registered_and_bound_by_pinned_versions(paper):
    assert any(tool.name == "search_paper" for tool in BUILTIN_TOOLS)
    assert "search_paper" in {tool.name for tool in build_builtin_tools()}

    context = RunContext(pinned_sources=["2210.03629v1"])
    registry = ToolRegistry([paper])
    ok = registry.call_result("search_paper", {"arxiv_id": "2210.03629", "query": "ALFWorld"},
                              context)
    assert ok.success and "71%" in ok.text
    assert any(anchor.page == 3 for anchor in context.evidence.anchors)
    blocked = registry.call_result("search_paper", {"arxiv_id": "2210.03629v3", "query": "x"},
                                   context)
    assert blocked.diagnostic["kind"] == "source_version_mismatch"


class Pages(Tool):
    name = "read_paper"
    description = "fixture"

    def run(self, arxiv_id="x"):
        return "--- 第 1 页 ---\n正文"


class Search(Tool):
    name = "search_paper"
    description = "fixture"

    def run(self, arxiv_id="x", query=""):
        return "命中"


def _reads(count):
    calls = [{"content": None, "tool_calls": [{"id": f"c{i}", "name": "read_paper",
                                               "arguments": {"arxiv_id": "x"}, "error": None}]}
             for i in range(count)]
    return calls + [{"content": "完成", "tool_calls": []}]


def _limit_message(tools):
    llm = ScriptedLLM(_reads(3))
    agent = Agent(llm, ToolRegistry(tools), max_steps=10, verbose=False,
                  tool_call_limits={"read_paper": 2})
    agent.run("读论文")
    notes = [m.get("content") or "" for m in llm.last_messages if m.get("role") == "tool"]
    [limited] = [note for note in notes if "调用上限" in note]
    return limited


def test_read_paper_limit_points_to_search_when_available():
    assert "改用 search_paper" in _limit_message([Pages(), Search()])
    assert "请根据已有信息作答" in _limit_message([Pages()])


def test_read_paper_limit_is_configurable_and_skipped_for_full_reads(monkeypatch):
    monkeypatch.setattr(config, "READ_PAPER_CALL_LIMIT", 10)
    assert research_tool_limits() == {"arxiv_search": 4, "read_paper": 10}
    monkeypatch.setattr(config, "READ_PAPER_CALL_LIMIT", 0)
    assert research_tool_limits() == {"arxiv_search": 4}

    monkeypatch.setattr(config, "READ_PAPER_CALL_LIMIT", 10)
    registry = ToolRegistry(BUILTIN_TOOLS)
    task_driven = ResearchTeam(ScriptedLLM([]), registry, verbose=False, require_full_paper=False)
    full = ResearchTeam(ScriptedLLM([]), registry, verbose=False, require_full_paper=True)
    assert task_driven.reader.tool_call_limits == {"read_paper": 10}
    assert full.reader.tool_call_limits == {}
    assert "search_paper" in {s["function"]["name"] for s in task_driven.reader.tools.schemas()}
