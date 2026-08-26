"""轻量 Markdown 渲染的离线测试。"""

from scholaragent.markdown_lite import render_markdown


def test_headings_lists_and_paragraphs():
    html = render_markdown(
        "# 标题\n\n一段 **加粗** 与 *斜体*。\n\n- 项目一\n- 项目二\n\n1. 首先\n2. 其次"
    )
    assert "<h1>标题</h1>" in html
    assert "<strong>加粗</strong>" in html
    assert "<em>斜体</em>" in html
    assert "<ul>" in html and "<li>项目一</li>" in html
    assert "<ol>" in html and "<li>首先</li>" in html


def test_code_block_and_inline_code_escape_html():
    html = render_markdown("用 `a<b>` 表示。\n\n```python\nprint('<script>')\n```")
    assert "<code>a&lt;b&gt;</code>" in html
    assert "<pre><code" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_links_only_allow_http_and_escape():
    html = render_markdown("见 [ReAct](https://arxiv.org/abs/2210.03629) 与 [坏](javascript:alert(1))")
    assert 'href="https://arxiv.org/abs/2210.03629"' in html
    assert 'rel="noopener noreferrer"' in html
    # 非 http(s) 链接不生成 <a>,原文按纯文本转义保留
    assert 'href="javascript:' not in html
    assert "<a " in html  # 只有合法 https 链接
    assert html.count("<a ") == 1


def test_blockquote_hr_and_empty():
    html = render_markdown("> 引用一句\n\n---\n\n结尾")
    assert "<blockquote>" in html
    assert "<hr>" in html
    assert "结尾" in html
    assert render_markdown("") == ""
    assert render_markdown(None) == ""


def test_raw_html_is_escaped_not_executed():
    html = render_markdown("<img src=x onerror=alert(1)> **安全**")
    assert "<img" not in html
    assert "&lt;img" in html
    assert "<strong>安全</strong>" in html
