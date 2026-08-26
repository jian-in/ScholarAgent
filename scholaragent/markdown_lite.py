"""轻量、安全的 Markdown 子集渲染。

不引入第三方库:只覆盖科研综述常见写法,所有文本先转义再插标签,
避免 XSS。支持标题、加粗/斜体、行内代码、代码块、列表、引用、链接、分割线。
"""

from __future__ import annotations

import html
import re


_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_BOLD_UNDER_RE = re.compile(r"__(.+?)__")
_ITALIC_UNDER_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")


def escape(text: str) -> str:
    return html.escape(text or "", quote=True)


def _format_inline(text: str) -> str:
    """先抽出代码与链接占位,再转义,最后还原带标签片段。"""
    text = text or ""
    slots: list[str] = []

    def park(fragment: str) -> str:
        slots.append(fragment)
        return f"\x00MD{len(slots) - 1}\x00"

    def code_sub(match: re.Match) -> str:
        return park(f"<code>{escape(match.group(1))}</code>")

    def link_sub(match: re.Match) -> str:
        label = escape(match.group(1))
        href = escape(match.group(2))
        return park(
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">{label}</a>'
        )

    text = _CODE_INLINE_RE.sub(code_sub, text)
    text = _LINK_RE.sub(link_sub, text)
    text = escape(text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _BOLD_UNDER_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    text = _ITALIC_UNDER_RE.sub(r"<em>\1</em>", text)

    for index, fragment in enumerate(slots):
        text = text.replace(f"\x00MD{index}\x00", fragment)
    return text


def render_markdown(source: str) -> str:
    """把 Markdown 子集渲染成 HTML 片段(不含外层包装)。"""
    if source is None:
        return ""
    text = str(source).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ""

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_ul = False
    in_ol = False
    in_blockquote = False
    paragraph: list[str] = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            out.append("</blockquote>")
            in_blockquote = False

    def flush_paragraph():
        nonlocal paragraph
        if not paragraph:
            return
        close_lists()
        body = _format_inline(" ".join(paragraph).strip())
        if body:
            out.append(f"<p>{body}</p>")
        paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            flush_paragraph()
            close_lists()
            close_blockquote()
            lang = stripped[3:].strip()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # closing fence
            lang_attr = f' class="language-{escape(lang)}"' if lang else ""
            out.append(
                f"<pre><code{lang_attr}>{escape(chr(10).join(code_lines))}</code></pre>"
            )
            continue

        if not stripped:
            flush_paragraph()
            close_lists()
            close_blockquote()
            i += 1
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_paragraph()
            close_lists()
            close_blockquote()
            out.append("<hr>")
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_lists()
            close_blockquote()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_format_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            close_lists()
            quote = stripped[1:].lstrip()
            if not in_blockquote:
                out.append("<blockquote>")
                in_blockquote = True
            out.append(f"<p>{_format_inline(quote)}</p>")
            i += 1
            continue
        close_blockquote()

        ul_item = re.match(r"^[-*+]\s+(.+)$", stripped)
        if ul_item:
            flush_paragraph()
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_format_inline(ul_item.group(1))}</li>")
            i += 1
            continue

        ol_item = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if ol_item:
            flush_paragraph()
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_format_inline(ol_item.group(2))}</li>")
            i += 1
            continue

        # ordinary paragraph text (may soft-wrap)
        close_lists()
        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    close_lists()
    close_blockquote()
    return "".join(out)
