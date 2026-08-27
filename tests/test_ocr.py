"""OCR 增强的公开行为契约。"""

from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfWriter

from scholaragent.ocr import OCRPage
from scholaragent.ocr import TesseractOCR
from scholaragent.tools.papers import ReadPaperTool
from scholaragent.workspace import TemporaryWorkspace


class FakeOCR:
    def __init__(self, text="扫描页正文"):
        self.text = text
        self.calls = []

    def read_page(self, pdf_path, page):
        self.calls.append((pdf_path, page))
        return OCRPage(page=page, text=self.text, confidence="medium")


def test_read_paper_uses_ocr_for_a_textless_page_and_keeps_page_evidence(tmp_path):
    workspace = TemporaryWorkspace(tmp_path)
    workspace.ensure("papers")
    pdf_path = workspace.paper_path("2401.88888")
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    ocr = FakeOCR()
    tool = ReadPaperTool(workspace, ocr=ocr)
    text = tool.run("2401.88888")
    metadata = tool.artifact_metadata(
        {"arxiv_id": "2401.88888"},
        type("Result", (), {"success": True, "text": text})(),
    )

    assert "扫描页正文" in text
    assert "OCR" in text
    assert "全文读完" in text
    assert [(str(path), page) for path, page in ocr.calls] == [(str(pdf_path), 1)]
    assert metadata[0]["source_anchors"][0]["confidence"] == "medium"
    assert metadata[0]["source_anchors"][0]["locator"] == "pdf-page:1:ocr"

    repeated = tool.artifact_metadata(
        {"arxiv_id": "2401.88888"},
        type("Result", (), {"success": True, "text": text})(),
    )
    assert repeated[0]["source_anchors"][0]["id"] == "S001"


def test_tesseract_reports_missing_dependencies_without_crashing(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"not a parsed PDF")
    ocr = TesseractOCR(
        tesseract_cmd=tmp_path / "missing-tesseract.exe",
        renderer_cmd=tmp_path / "missing-pdftoppm.exe",
    )

    assert ocr.available is False
    result = ocr.read_page(pdf_path, 1)
    assert result.text == ""
    assert "OCR 依赖不可用" in (result.diagnostic or "")


def test_tesseract_renders_one_page_then_recognizes_it(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"placeholder")
    tesseract = tmp_path / "tesseract.exe"
    renderer = tmp_path / "pdftoppm.exe"
    tesseract.write_bytes(b"")
    renderer.write_bytes(b"")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == str(renderer):
            output_base = Path(command[-1])
            output_base.with_suffix(".png").write_bytes(b"png")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="识别出来的论文文字\n", stderr="")

    monkeypatch.setattr("scholaragent.ocr.subprocess.run", fake_run)
    ocr = TesseractOCR(
        tesseract_cmd=tesseract,
        renderer_cmd=renderer,
        language="chi_sim+eng",
        dpi=240,
        psm=6,
    )

    result = ocr.read_page(pdf_path, 2)

    assert result.text == "识别出来的论文文字"
    assert result.confidence == "medium"
    assert len(commands) == 2
    assert commands[0][0] == str(renderer)
    assert commands[0][commands[0].index("-f") + 1] == "2"
    assert commands[0][commands[0].index("-r") + 1] == "240"
    assert commands[1][0] == str(tesseract)
    assert commands[1][commands[1].index("-l") + 1] == "chi_sim+eng"
    assert commands[1][commands[1].index("--psm") + 1] == "6"
