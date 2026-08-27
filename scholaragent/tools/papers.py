"""文献工具 2/3:论文下载与分段阅读。

两个贯穿全项目的要点在这里第一次登场:

安全要点 —— arxiv_id 来自模型输出,绝不能直接拼进文件路径:
    先用正则校验格式,防止 ../../ 这类路径穿越攻击。
    (和 calculator 用 ast 白名单是同一个思想:模型给的一切输入都要设边界)

上下文预算要点 —— 一篇论文全文可能有几十万字符,一次性塞给模型
    会挤爆上下文窗口(还费钱)。read_paper 按"页 + 页内偏移"分段返回,
    每段结尾告诉模型下一段从哪继续,多次调用就能完整读完全文 ——
    这是所有生产级 Agent 处理长文档的标准做法。
"""

import os
import re

import httpx
from pypdf import PdfReader

from .. import config
from ..ocr import OCRPage, PageOCR, default_ocr
from ..tool import Tool, ToolResult
from ..workspace import Workspace, workspace_for

# 新式编号如 2401.12345 / 2401.12345v2,老式编号如 cs/0112017 / math.GT/0309136
_ID_PATTERN = re.compile(
    r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?)$"
)


def _validate_id(arxiv_id: str) -> str:
    """校验 arXiv 编号格式,不合法直接抛错(会被 ToolRegistry 转成文字回给模型)。"""
    arxiv_id = arxiv_id.strip()
    if not _ID_PATTERN.match(arxiv_id):
        raise ValueError(f"不是合法的 arXiv 编号:{arxiv_id}(应形如 2401.12345)")
    return arxiv_id


def _pdf_path(arxiv_id: str, workspace: Workspace | str | None = None) -> str:
    return str(workspace_for(workspace).paper_path(arxiv_id))


class DownloadPaperTool(Tool):
    name = "download_paper"
    description = "按 arXiv 编号下载论文 PDF 到本地。下载完成后才能用 read_paper 阅读"
    parameters = {
        "type": "object",
        "properties": {
            "arxiv_id": {
                "type": "string",
                "description": "论文的 arXiv 编号,例如 2401.12345",
            },
        },
        "required": ["arxiv_id"],
    }

    def __init__(self, workspace: Workspace | str | None = None):
        self.workspace = workspace

    def _active_workspace(self) -> Workspace:
        return workspace_for(self.workspace)

    def run(self, arxiv_id: str) -> str:
        arxiv_id = _validate_id(arxiv_id)
        path = _pdf_path(arxiv_id, self.workspace)

        if not os.path.exists(path):
            response = httpx.get(
                f"https://arxiv.org/pdf/{arxiv_id}",
                timeout=60,
                follow_redirects=True,
            )
            response.raise_for_status()
            # arXiv 对不存在的编号有时返回 200 + HTML 错误页,不能只看状态码
            if not response.content.startswith(b"%PDF"):
                return f"下载到的内容不是 PDF,编号 {arxiv_id} 可能不存在"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # 先写临时文件再原子替换:中途断电/Ctrl+C 也不会留下半截的正式文件
            temp_path = path + ".part"
            with open(temp_path, "wb") as f:
                f.write(response.content)
            os.replace(temp_path, path)

        try:
            pages = len(PdfReader(path).pages)
        except Exception as exc:
            # 缓存的 PDF 解析不了就删掉,让"重试"真正重新下载,
            # 否则这个编号会被坏文件永久卡死
            os.remove(path)
            return (f"论文 {arxiv_id} 的 PDF 无法解析({type(exc).__name__}),"
                    f"已清理坏缓存,可以重新调用 download_paper 再试一次")
        size_kb = os.path.getsize(path) // 1024
        return (f"论文 {arxiv_id} 已就绪({size_kb} KB,共 {pages} 页),"
                f"可以用 read_paper 开始阅读")

    def artifact_metadata(self, arguments, result: ToolResult):
        arxiv_id = str(arguments.get("arxiv_id") or "").strip()
        path = self._active_workspace().paper_path(arxiv_id) if arxiv_id else None
        if not result.success or not arxiv_id or path is None or not path.exists():
            return []
        return [{
            "kind": "paper",
            "arxiv_id": arxiv_id,
            "path": str(path),
            "exists": True,
            "summary": result.text,
        }]


class ReadPaperTool(Tool):
    name = "read_paper"
    description = (
        "阅读已下载论文的文字内容。每次返回一段(默认约 6000 字符);"
        "结尾会给出下一段的位置参数;同一任务内后续调用即使省略位置，"
        "也会从上次结束处继续，直到完整读完全文。遇到没有文字层的页面会"
        "自动尝试 OCR，并用 [OCR] 标注可能存在的识别误差"
    )
    parameters = {
        "type": "object",
        "properties": {
            "arxiv_id": {
                "type": "string",
                "description": "论文的 arXiv 编号,例如 2401.12345",
            },
            "start_page": {
                "type": "integer",
                "description": "从第几页开始读,默认第 1 页",
            },
            "start_char": {
                "type": "integer",
                "description": "页内起始字符位置(上一次调用的结尾会给出),默认 0",
            },
        },
        "required": ["arxiv_id"],
    }

    # 32K 模型下 6000 个英文字符通常约 1500 tokens，既能守住单轮预算，
    # 又能减少长论文所需的模型往返次数。仍可通过环境变量按模型调整。
    MAX_CHARS = config.PAPER_READER_CHUNK_CHARS

    def __init__(self, workspace: Workspace | str | None = None,
                 ocr: PageOCR | None = None):
        self.workspace = workspace
        self.ocr = ocr if ocr is not None else default_ocr()
        self._next_positions = {}
        self._ocr_cache = {}
        self._completion_met = False
        self._anchor_counter = 0
        self._anchor_ids = {}

    def start_run(self):
        self._next_positions.clear()
        self._ocr_cache.clear()
        self._completion_met = False
        self._anchor_counter = 0
        self._anchor_ids.clear()

    def completion_ready(self) -> bool:
        return self._completion_met

    def _ocr_page(self, path, page: int) -> OCRPage:
        key = (str(path), page)
        cached = self._ocr_cache.get(key)
        if cached is not None:
            return cached
        try:
            result = self.ocr.read_page(path, page)
            if not isinstance(result, OCRPage):
                result = OCRPage(page=page, text=str(result or ""))
        except Exception as exc:
            result = OCRPage(
                page=page,
                text="",
                confidence="low",
                diagnostic=f"OCR 适配器失败: {type(exc).__name__}: {exc}",
            )
        self._ocr_cache[key] = result
        return result

    def run(self, arxiv_id: str, start_page: int = None,
            start_char: int = None) -> str:
        arxiv_id = _validate_id(arxiv_id)
        path = _pdf_path(arxiv_id, self.workspace)
        if not os.path.exists(path):
            return f"论文 {arxiv_id} 还没下载,请先调用 download_paper"

        try:
            reader = PdfReader(path)
        except Exception as exc:
            return (f"论文 {arxiv_id} 的 PDF 无法解析({type(exc).__name__}),"
                    f"请重新调用 download_paper(它会自动清理坏文件)")
        total = len(reader.pages)
        if start_page is None:
            # 缺少页码时，页内偏移本身没有可靠含义，采用工具上一段
            # 亲自算出的完整位置；首次读取则从第一页开始。
            page, offset = self._next_positions.get(arxiv_id, (1, 0))
        else:
            page = max(1, int(start_page))
            offset = max(0, int(start_char or 0))
        if page > total:
            return f"起始页 {page} 超出范围:这篇论文一共只有 {total} 页"

        chunks, used, any_text = [], 0, False
        next_pos = None  # 下一段的位置 (页, 页内偏移);None 表示读完了
        ocr_attempted = False
        while page <= total:
            full = (reader.pages[page - 1].extract_text() or "").strip()
            ocr_page = None
            if not full:
                ocr_attempted = True
                ocr_page = self._ocr_page(path, page)
                full = ocr_page.text
            body = full[offset:] if offset < len(full) else ""
            header = f"--- 第 {page} 页 ---\n"
            if ocr_page is not None and ocr_page.text:
                header += "[OCR] 本页由 OCR 提取，可能存在识别误差。\n"
            budget = self.MAX_CHARS - used - len(header)
            if budget <= 0:
                next_pos = (page, offset)
                break
            if body and len(body) > budget:
                if not chunks:
                    # 第一段就超预算:切一刀,并记住页内续读位置
                    any_text = True
                    chunks.append(header + body[:budget])
                    used += len(header) + budget
                    next_pos = (page, offset + budget)
                else:
                    next_pos = (page, offset)  # 整段留给下一次调用
                break
            if body:
                any_text = True
                chunks.append(header + body)
            else:
                # 图表页/扫描页没有文字层,如实说明(也计入预算,防止刷屏)
                detail = "(本页没有可提取的文字,可能是图表或扫描页)"
                if ocr_page is not None and ocr_page.diagnostic:
                    detail += f"; OCR 未能获得文字: {ocr_page.diagnostic}"
                chunks.append(header + detail)
            used += len(chunks[-1])
            page += 1
            offset = 0

        if not any_text:
            # 扫描版属于终态：继续调同一工具也不会产生文字，允许 Agent
            # 结束并如实说明资料限制。
            self._completion_met = True
            ocr_hint = "，已尝试 OCR" if ocr_attempted else ""
            return (f"论文 {arxiv_id} 的这些页面没有可提取的文字层"
                    f"(可能是扫描版 PDF{ocr_hint}),read_paper 读不了这份文件")
        if next_pos is None:
            self._next_positions.pop(arxiv_id, None)
            self._completion_met = True
            tail = f"\n(第 {total}/{total} 页,全文读完)"
        else:
            p, c = next_pos
            self._next_positions[arxiv_id] = next_pos
            tail = (f"\n(还没读完;继续阅读请再次调用 read_paper,"
                    f"参数 start_page={p}, start_char={c})")
        return "\n".join(chunks) + tail

    def artifact_metadata(self, arguments, result: ToolResult):
        arxiv_id = str(arguments.get("arxiv_id") or "").strip()
        if not result.success or not arxiv_id:
            return []
        metadata = {
            "kind": "read",
            "arxiv_id": arxiv_id,
            "completed": self.completion_ready(),
        }
        anchors = []
        pages = list(re.finditer(r"---\s*第\s*(\d+)\s*页\s*---", result.text))
        for index, match in enumerate(pages):
            page = int(match.group(1))
            end = pages[index + 1].start() if index + 1 < len(pages) else len(result.text)
            excerpt = result.text[match.end():end].strip()
            anchor_key = (arxiv_id, page)
            anchor_id = self._anchor_ids.get(anchor_key)
            if anchor_id is None:
                self._anchor_counter += 1
                anchor_id = f"S{self._anchor_counter:03d}"
                self._anchor_ids[anchor_key] = anchor_id
            no_text = not excerpt or "没有可提取的文字" in excerpt
            from_ocr = "[OCR]" in excerpt
            anchors.append({
                "id": anchor_id,
                "kind": "page" if no_text else "text",
                "source": f"arxiv:{arxiv_id}",
                "page": page,
                "locator": f"pdf-page:{page}:ocr" if from_ocr else f"pdf-page:{page}",
                "confidence": "low" if no_text else ("medium" if from_ocr else "high"),
                "excerpt": excerpt[:300],
            })
        if anchors:
            metadata["source_anchors"] = anchors
        return [metadata]
