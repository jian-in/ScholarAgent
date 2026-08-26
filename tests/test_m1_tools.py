"""M1 文献工具集的离线测试:不联网,网络部分只测纯函数与安全边界。

运行方式(项目根目录下):
    python -m pytest tests -q      (推荐)
    python tests/test_m1_tools.py  (没装 pytest 时直接跑)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent import config
import httpx

from scholaragent.tool import STOP_RETRY_PREFIX
from scholaragent.tools.arxiv_search import (
    ArxivSearchTool,
    _parse_atom,
    _parse_openalex,
)
from scholaragent.tools.notes import ReadNotesTool, SaveNoteTool
from scholaragent.tools.papers import ReadPaperTool, _validate_id

# ―― arXiv Atom XML 解析 ――――――――――――――――――――――――――――――――

# 从真实 arXiv API 响应精简出来的样本(保留命名空间和折行标题等特征)
SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query Results</title>
  <entry>
    <id>http://arxiv.org/abs/2210.03629v3</id>
    <published>2022-10-06T18:00:00Z</published>
    <title>ReAct: Synergizing Reasoning
      and Acting in Language Models</title>
    <summary>  We explore the use of LLMs to generate both reasoning
      traces and task-specific actions.  </summary>
    <author><name>Shunyu Yao</name></author>
    <author><name>Jeffrey Zhao</name></author>
    <author><name>Dian Yu</name></author>
    <author><name>Karthik Narasimhan</name></author>
  </entry>
</feed>"""

SAMPLE_OPENALEX = {
    "results": [{
        "title": "A-MEM: Agentic Memory for LLM Agents",
        "publication_date": "2025-02-17",
        "doi": "https://doi.org/10.48550/arxiv.2502.12110",
        "primary_location": {
            "landing_page_url": "http://arxiv.org/abs/2502.12110",
            "pdf_url": "https://arxiv.org/pdf/2502.12110",
        },
        "authorships": [
            {"author": {"display_name": "Wujiang Xu"}},
            {"author": {"display_name": "Zujie Liang"}},
        ],
        "abstract_inverted_index": {
            "Agentic": [0], "memory": [1], "organizes": [2],
            "long-term": [3], "experiences": [4],
        },
    }],
}


def test_parse_atom():
    """标题折行要压平,编号取最后一段,作者全部提取。"""
    papers = _parse_atom(SAMPLE_ATOM)
    assert len(papers) == 1
    p = papers[0]
    assert p["id"] == "2210.03629v3"
    assert p["title"] == "ReAct: Synergizing Reasoning and Acting in Language Models"
    assert p["published"] == "2022-10-06"
    assert len(p["authors"]) == 4
    assert "reasoning" in p["summary"]


def test_parse_openalex():
    papers = _parse_openalex(SAMPLE_OPENALEX)
    assert papers[0]["id"] == "2502.12110"
    assert papers[0]["authors"] == ["Wujiang Xu", "Zujie Liang"]
    assert papers[0]["summary"] == "Agentic memory organizes long-term experiences"


def test_arxiv_search_uses_primary_when_available(monkeypatch):
    """官方接口正常时仍优先采用 arXiv Atom 结果。"""
    request = httpx.Request("GET", "https://export.arxiv.org/api/query")
    response = httpx.Response(200, text=SAMPLE_ATOM, request=request)

    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get",
                        lambda *args, **kwargs: response)
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    result = ArxivSearchTool().run("LLM agent memory")

    assert "ReAct: Synergizing Reasoning and Acting" in result


def test_arxiv_search_stops_after_bounded_timeouts(monkeypatch):
    """主源和备用源都超时时,只各请求一次并发出熔断信号。"""
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout(
            "timed out",
            request=httpx.Request("GET", "https://export.arxiv.org/api/query"),
        )

    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", timeout)
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    result = ArxivSearchTool().run("LLM agent memory")

    assert result.startswith(STOP_RETRY_PREFIX)
    assert len(calls) == 2  # arXiv 与 OpenAlex 各一次


def test_arxiv_search_429_immediately_uses_fallback(monkeypatch):
    """429 直接切备用源,不把时间浪费在原接口的重复请求上。"""
    request = httpx.Request("GET", "https://export.arxiv.org/api/query")
    rate_limit_response = httpx.Response(
        429, headers={"Retry-After": "120"}, request=request)
    calls = []

    def rate_limited(*args, **kwargs):
        calls.append(1)
        return rate_limit_response

    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", rate_limited)
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    result = ArxivSearchTool().run("LLM agent memory")

    assert result.startswith(STOP_RETRY_PREFIX)
    assert len(calls) == 2  # arXiv 一次 + OpenAlex 一次


def test_arxiv_search_falls_back_to_openalex(monkeypatch):
    """arXiv 长时间限流时,OpenAlex 仍应返回可供下载的 arXiv 编号。"""
    request = httpx.Request("GET", "https://example.test")
    responses = [
        httpx.Response(429, headers={"Retry-After": "120"}, request=request),
        httpx.Response(200, json=SAMPLE_OPENALEX, request=request),
    ]
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get",
                        lambda *args, **kwargs: responses.pop(0))
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    result = ArxivSearchTool().run("LLM agent memory", max_results=5)

    assert "OpenAlex" in result
    assert "A-MEM: Agentic Memory for LLM Agents" in result
    assert "arXiv编号: 2502.12110" in result


def test_arxiv_search_can_use_openalex_directly(monkeypatch):
    """配置备用源为默认后,不再先碰容易限流的 arXiv API。"""
    request = httpx.Request("GET", "https://api.openalex.org/works")
    calls = []

    params_seen = {}

    def openalex(*args, **kwargs):
        calls.append(args[0])
        params_seen.update(kwargs["params"])
        return httpx.Response(200, json=SAMPLE_OPENALEX, request=request)

    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", openalex)
    monkeypatch.setenv("ARXIV_SEARCH_BACKEND", "openalex")

    result = ArxivSearchTool().run("LLM agent memory")

    assert calls == ["https://api.openalex.org/works"]
    assert "title_and_abstract.search:LLM agent memory" in params_seen["filter"]
    assert "直接检索" in result
    assert "自动备用" not in result
    assert "arXiv编号: 2502.12110" in result


# ―― arXiv 编号校验(安全边界)――――――――――――――――――――――――――――


def test_validate_id_accepts_real_ids():
    for good in ["2401.12345", "2210.03629v3", "cs/0112017", "math.GT/0309136"]:
        assert _validate_id(good) == good


def test_validate_id_rejects_path_traversal():
    """模型给的编号绝不能变成任意文件路径。"""
    for evil in ["../../etc/passwd", "..\\..\\windows", "2401.12345; rm -rf /",
                 "http://evil.com/x.pdf", ""]:
        try:
            _validate_id(evil)
            raise AssertionError(f"不该放行:{evil}")
        except ValueError:
            pass  # 正确:被拒绝了


# ―― PDF 分段阅读 ――――――――――――――――――――――――――――――――――――


def make_minimal_pdf(text: str) -> bytes:
    """亲手拼一个最小的合法单页 PDF(含一行文字)。

    好处:测试不依赖网络、仓库里也不用放二进制文件;
    顺带能看清 PDF 的真实结构 —— 对象表 + 交叉引用表 + 尾部。
    """
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF").encode()
    return bytes(out)


def _with_temp_data_dir(fn):
    """把 config.DATA_DIR 临时指到一个空目录,跑完恢复。"""
    original = config.DATA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.DATA_DIR = tmp
        try:
            fn(tmp)
        finally:
            config.DATA_DIR = original


def test_read_paper():
    def scenario(tmp):
        papers_dir = os.path.join(tmp, "papers")
        os.makedirs(papers_dir)
        with open(os.path.join(papers_dir, "2401.11111.pdf"), "wb") as f:
            f.write(make_minimal_pdf("ScholarAgent test paper"))

        tool = ReadPaperTool()
        result = tool.run(arxiv_id="2401.11111")
        assert "ScholarAgent test paper" in result
        assert "全文读完" in result

        # 起始页越界要给人话提示,而不是崩溃
        assert "超出范围" in tool.run(arxiv_id="2401.11111", start_page=99)

    _with_temp_data_dir(scenario)


def test_read_paper_not_downloaded():
    def scenario(tmp):
        result = ReadPaperTool().run(arxiv_id="2401.99999")
        assert "还没下载" in result

    _with_temp_data_dir(scenario)


def test_read_paper_oversize_page_resumable():
    """单页超过预算时:切段返回,并且靠 start_char 能把剩余部分读完。"""
    def scenario(tmp):
        papers_dir = os.path.join(tmp, "papers")
        os.makedirs(papers_dir)
        long_text = "A" * 2500 + "MIDDLE" + "B" * 2500  # 约 5000 字符的一页
        with open(os.path.join(papers_dir, "2401.22222.pdf"), "wb") as f:
            f.write(make_minimal_pdf(long_text))

        tool = ReadPaperTool()
        tool.MAX_CHARS = 3000  # 固定小预算，专测页内续读，不受默认配置影响
        part1 = tool.run(arxiv_id="2401.22222")
        assert "全文读完" not in part1              # 没读完就不能谎报读完
        assert "start_char=" in part1               # 必须给出续读位置

        import re
        match = re.search(r"start_page=(\d+), start_char=(\d+)", part1)
        part2 = tool.run(arxiv_id="2401.22222",
                         start_page=int(match.group(1)),
                         start_char=int(match.group(2)))
        assert "全文读完" in part2
        # 两段拼起来必须覆盖全文(中间标志词至少出现在某一段里)
        assert "MIDDLE" in (part1 + part2)
        assert part2.count("B") >= 2000             # 第二段拿到了剩余内容

    _with_temp_data_dir(scenario)


def test_read_paper_reuses_trusted_cursor_when_model_omits_fields():
    """续读参数缺失一半时，使用工具记录的游标而不是回到第一页。"""
    def scenario(tmp):
        papers_dir = os.path.join(tmp, "papers")
        os.makedirs(papers_dir)
        long_text = "A" * 2500 + "MIDDLE" + "B" * 2500
        with open(os.path.join(papers_dir, "2401.55555.pdf"), "wb") as f:
            f.write(make_minimal_pdf(long_text))

        tool = ReadPaperTool()
        tool.MAX_CHARS = 3000
        part1 = tool.run(arxiv_id="2401.55555")
        assert "全文读完" not in part1
        assert not tool.completion_ready()

        # 模拟本地模型只给了一个明显错误的 start_char。
        part2 = tool.run(arxiv_id="2401.55555", start_char=999999999)
        assert "全文读完" in part2
        assert tool.completion_ready()
        assert "MIDDLE" in (part1 + part2)

    _with_temp_data_dir(scenario)


def test_read_paper_textless_pdf():
    """没有文字层的 PDF 要如实说明,而不是谎报"全文读完"。"""
    def scenario(tmp):
        papers_dir = os.path.join(tmp, "papers")
        os.makedirs(papers_dir)
        with open(os.path.join(papers_dir, "2401.33333.pdf"), "wb") as f:
            f.write(make_minimal_pdf(""))  # 一页但没有文字

        result = ReadPaperTool().run(arxiv_id="2401.33333")
        assert "没有可提取的文字层" in result
        assert "全文读完" not in result

    _with_temp_data_dir(scenario)


def test_download_paper_heals_corrupt_cache():
    """缓存里是坏 PDF 时,download_paper 要删掉它并说明,不能永久卡死。"""
    def scenario(tmp):
        papers_dir = os.path.join(tmp, "papers")
        os.makedirs(papers_dir)
        bad = os.path.join(papers_dir, "2401.44444.pdf")
        with open(bad, "wb") as f:
            f.write(b"%PDF-1.4 corrupted garbage")  # 文件存在 => 不会走网络

        from scholaragent.tools.papers import DownloadPaperTool
        result = DownloadPaperTool().run(arxiv_id="2401.44444")
        assert "已清理坏缓存" in result
        assert not os.path.exists(bad)  # 坏文件必须被删掉,重试才能真正重新下载

    _with_temp_data_dir(scenario)


# ―― 研究笔记 ――――――――――――――――――――――――――――――――――――――


def test_notes_roundtrip():
    def scenario(tmp):
        save, read = SaveNoteTool(), ReadNotesTool()
        assert "空的" in read.run()
        save.run(title="ReAct 论文要点", content="推理与行动交替,arXiv 2210.03629")
        save.run(title="第二条", content="内容二")
        text = read.run()
        assert "ReAct 论文要点" in text
        assert "2210.03629" in text
        assert "第二条" in text

    _with_temp_data_dir(scenario)


if __name__ == "__main__":
    # 不依赖 pytest 的极简测试运行器
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过:{name}")
    print("全部测试通过")
