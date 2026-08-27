"""可选的 PDF OCR 适配器。

OCR 位于论文阅读器的一个小接口之后：阅读器只需要按页取得
``OCRPage``，不需要知道 PDF 如何栅格化、Tesseract 安装在哪里，或临时
图片如何清理。默认适配器不把二进制工具打进项目，而是从环境变量、PATH
和当前项目所在磁盘的常见位置发现它们。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import config


OCR_CONFIDENCE = ("high", "medium", "low")


@dataclass(frozen=True)
class OCRPage:
    """一页 OCR 结果；空文本也保留诊断信息，便于如实降级。"""

    page: int
    text: str
    confidence: str = "medium"
    engine: str = "tesseract"
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.page, int) or self.page < 1:
            raise ValueError("OCR 页码必须是正整数")
        if self.confidence not in OCR_CONFIDENCE:
            raise ValueError(f"未知 OCR 置信度: {self.confidence}")
        object.__setattr__(self, "text", str(self.text or "").strip())
        if self.diagnostic is not None:
            object.__setattr__(self, "diagnostic", _clip(self.diagnostic))


class PageOCR(Protocol):
    """论文阅读器需要的最小 OCR 接口。"""

    def read_page(self, pdf_path: str | Path, page: int) -> OCRPage:
        ...


def _clip(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _project_drive_candidates(*parts: str) -> list[Path]:
    """寻找与项目位于同一磁盘的工具，避免把用户机器路径写死。"""
    try:
        project_drive = Path(__file__).resolve().parents[2]
    except (IndexError, OSError):
        return []
    return [project_drive.joinpath(*parts)]


def _resolve_command(
    name: str,
    explicit: str | Path | None,
    candidates: list[Path],
) -> str | None:
    configured = str(explicit or os.getenv(
        "SCHOLARAGENT_TESSERACT_CMD" if name == "tesseract"
        else "SCHOLARAGENT_PDF_RENDERER"
    ) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return str(configured_path.resolve())
        found = shutil.which(configured)
        return found

    found = shutil.which(name)
    if found:
        return found
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _tesseract_candidates() -> list[Path]:
    return [
        *_project_drive_candidates("Tesseract", "tesseract.exe"),
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ]


def _renderer_candidates() -> list[Path]:
    return [
        *_project_drive_candidates("bin", "pdftoppm.exe"),
        *_project_drive_candidates("Poppler", "Library", "bin", "pdftoppm.exe"),
    ]


class TesseractOCR:
    """使用 Tesseract + ``pdftoppm`` 读取单页 PDF 的适配器。

    ``tesseract_cmd`` 和 ``renderer_cmd`` 都可显式注入，测试与其他机器不
    需要依赖本机 PATH。生产默认顺序是：显式参数/环境变量、PATH、常见
    Windows 安装位置和项目所在磁盘的 ``Tesseract`` 目录。
    """

    def __init__(self, tesseract_cmd: str | Path | None = None,
                 renderer_cmd: str | Path | None = None,
                 language: str | None = None, dpi: int | None = None,
                 timeout: int | None = None, psm: int | None = None):
        self.tesseract_cmd = _resolve_command(
            "tesseract", tesseract_cmd, _tesseract_candidates()
        )
        self.renderer_cmd = _resolve_command(
            "pdftoppm", renderer_cmd, _renderer_candidates()
        )
        self.language = str(language or config.OCR_LANGUAGE or "chi_sim+eng")
        self.dpi = max(72, int(dpi or config.OCR_DPI))
        self.timeout = max(1, int(timeout or config.OCR_TIMEOUT))
        self.psm = max(0, int(config.OCR_PSM if psm is None else psm))

    @property
    def available(self) -> bool:
        return bool(self.tesseract_cmd and self.renderer_cmd)

    def diagnostics(self) -> dict[str, str | bool | None]:
        """返回可展示的本机能力信息，不执行外部命令。"""
        return {
            "available": self.available,
            "tesseract_cmd": self.tesseract_cmd,
            "renderer_cmd": self.renderer_cmd,
            "language": self.language,
            "dpi": self.dpi,
        }

    def read_page(self, pdf_path: str | Path, page: int) -> OCRPage:
        if not isinstance(page, int) or page < 1:
            raise ValueError("OCR 页码必须是正整数")
        source = Path(pdf_path)
        if not source.is_file():
            return OCRPage(page, "", confidence="low", diagnostic="PDF 文件不存在")
        if not self.available:
            missing = []
            if not self.tesseract_cmd:
                missing.append("tesseract")
            if not self.renderer_cmd:
                missing.append("pdftoppm")
            return OCRPage(
                page,
                "",
                confidence="low",
                diagnostic="OCR 依赖不可用: " + ", ".join(missing),
            )

        with tempfile.TemporaryDirectory(prefix="scholaragent-ocr-") as temporary:
            output_base = Path(temporary) / "page"
            render_command = [
                self.renderer_cmd,
                "-f", str(page),
                "-l", str(page),
                "-r", str(self.dpi),
                "-png",
                "-singlefile",
                str(source),
                str(output_base),
            ]
            try:
                rendered = subprocess.run(
                    render_command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return OCRPage(
                    page, "", confidence="low",
                    diagnostic=f"PDF 页面渲染失败: {type(exc).__name__}: {exc}",
                )
            image_path = output_base.with_suffix(".png")
            if rendered.returncode != 0 or not image_path.is_file():
                detail = rendered.stderr or rendered.stdout
                return OCRPage(
                    page, "", confidence="low",
                    diagnostic=f"PDF 页面渲染失败: {_clip(detail) or rendered.returncode}",
                )

            command = [
                self.tesseract_cmd,
                str(image_path),
                "stdout",
                "-l", self.language,
                "--psm", str(self.psm),
            ]
            tessdata = Path(self.tesseract_cmd).parent / "tessdata"
            if tessdata.is_dir():
                command.extend(["--tessdata-dir", str(tessdata)])
            try:
                recognized = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return OCRPage(
                    page, "", confidence="low",
                    diagnostic=f"Tesseract 执行失败: {type(exc).__name__}: {exc}",
                )
            if recognized.returncode != 0:
                return OCRPage(
                    page, "", confidence="low",
                    diagnostic=f"Tesseract 识别失败: {_clip(recognized.stderr or recognized.stdout)}",
                )
            text = recognized.stdout.strip()
            return OCRPage(
                page,
                text,
                confidence="medium" if text else "low",
                diagnostic=None if text else "Tesseract 未识别出文字",
            )


def default_ocr() -> TesseractOCR:
    """创建默认 OCR 适配器；发现失败时仍返回可解释的不可用对象。"""
    return TesseractOCR()


__all__ = ["OCRPage", "PageOCR", "TesseractOCR", "default_ocr"]
