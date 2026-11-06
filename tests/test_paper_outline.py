"""论文目录(章节 → 页码)的离线测试:书签优先,没有书签时识别标题行。"""

import io
from types import SimpleNamespace

from pypdf import PdfReader, PdfWriter

from scholaragent.tools import papers
from scholaragent.tools.papers import DownloadPaperTool, format_outline, paper_outline


def pdf_with_bookmarks(pages=6):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    intro = writer.add_outline_item("1 Introduction", 0)
    writer.add_outline_item("2 Method", 2)
    sub = writer.add_outline_item("2.1 Setup", 2, parent=intro)
    writer.add_outline_item("2.1.1 Too deep", 3, parent=sub)   # 三级标题不列出
    writer.add_outline_item("3 Experiments", 4)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def fake_reader(page_texts):
    """没有书签、只有文字的“PDF”:每页按给定文字返回。"""
    return SimpleNamespace(
        outline=[],
        pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in page_texts],
    )


def test_bookmarks_give_titles_pages_and_depth():
    source, entries = paper_outline(PdfReader(io.BytesIO(pdf_with_bookmarks())))
    assert source == "来自 PDF 书签"
    titles = [title for _, title, _ in entries]
    assert titles == ["1 Introduction", "2.1 Setup", "2 Method", "3 Experiments"]
    assert (2, "2.1 Setup", 3) in entries
    assert "2.1.1 Too deep" not in titles


def test_headings_are_detected_when_there_are_no_bookmarks():
    reader = fake_reader([
        "Deep Residual Learning\nAbstract\nWe present a residual learning framework.\n1. Introduction\nDeep networks",
        "text\n2. Related Work\nResidual Representations.",
        "3. Deep Residual Learning\n3.1. Residual Learning\n3.1.1. Too Deep Detail\nThis is a sentence.",
        "References\n[1] Y. Bengio.",
    ])
    source, entries = paper_outline(reader)
    assert source == "自动识别，可能不全"
    assert entries == [
        (1, "Abstract", 1), (1, "1. Introduction", 1), (1, "2. Related Work", 2),
        (1, "3. Deep Residual Learning", 3), (2, "3.1. Residual Learning", 3),
        (1, "References", 4),
    ]


def test_format_outline_points_to_jump_reading_and_caps_length(monkeypatch):
    monkeypatch.setattr(papers, "OUTLINE_MAX_ENTRIES", 2)
    text = format_outline("2210.03629v1", PdfReader(io.BytesIO(pdf_with_bookmarks())))
    assert "目录(来自 PDF 书签)" in text
    assert "- 1 Introduction …… 第 1 页" in text
    assert "(其余 2 条省略)" in text
    assert 'start_page=页码' in text and "不必从头读完全文" in text

    assert format_outline("x", fake_reader(["no headings here."])) == ""


def test_download_result_includes_outline_for_cached_pdf(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "2401.22222.pdf").write_bytes(pdf_with_bookmarks())
    result = DownloadPaperTool(workspace=str(tmp_path)).run("2401.22222")
    assert result.startswith("论文 2401.22222 已就绪")
    assert "3 Experiments …… 第 5 页" in result


def test_outline_failure_never_breaks_download(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "2401.33333.pdf").write_bytes(pdf_with_bookmarks())

    def boom(*args, **kwargs):
        raise RuntimeError("bad outline")

    monkeypatch.setattr(papers, "format_outline", boom)
    result = DownloadPaperTool(workspace=str(tmp_path)).run("2401.33333")
    assert result.startswith("论文 2401.33333 已就绪(")
    assert result.endswith("共 6 页),可以用 read_paper 开始阅读")
    assert "目录" not in result


def test_prompts_steer_agents_to_jump_reading_instead_of_full_reads():
    """真实实验中顺序翻页占了约七成工具调用;提示词要引导按目录跳读、够用就停。"""
    from scholaragent.planner import PLAN_SYSTEM_PROMPT
    from scholaragent.runtime import RESEARCH_SYSTEM_PROMPT
    from scholaragent.team import READER_PROMPT_TASK_DRIVEN

    assert "不需要阅读全文" in PLAN_SYSTEM_PROMPT
    assert "同一篇论文只在一个步骤里下载和阅读" in PLAN_SYSTEM_PROMPT
    assert "通读全文" in PLAN_SYSTEM_PROMPT  # 以“不要写通读全文”的形式出现
    for prompt in (RESEARCH_SYSTEM_PROMPT, READER_PROMPT_TASK_DRIVEN):
        assert "目录" in prompt and "start_page" in prompt
