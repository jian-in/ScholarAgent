"""arxiv_search 检索增强的离线测试:分类过滤、按时间排序、日期过滤,
以及参数校验的错误回传行为。全部不联网。

运行方式(项目根目录下):
    python -m pytest tests/test_arxiv_search_refinement.py -q
"""

import json
import re

import httpx

from scholaragent.tools.arxiv_search import (
    ArxivSearchTool,
    _build_arxiv_search_query,
    _clean_categories,
    _normalize_date,
)

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.99999v1</id>
    <title>  A Fresh Agent Paper </title>
    <author><name>Ada Lovelace</name></author>
    <published>2026-01-15T00:00:00Z</published>
    <summary>新论文摘要。</summary>
  </entry>
</feed>"""

SAMPLE_OPENALEX = {
    "results": [{
        "title": "Fresh Agent Paper (OA)",
        "publication_date": "2026-01-20",
        "authorships": [{"author": {"display_name": "A. Turing"}}],
        "abstract_inverted_index": {"新论文": [0], "摘要": [1]},
        "primary_location": {"landing_page_url":
                             "https://arxiv.org/abs/2501.99999"},
        "doi": None,
    }],
}


def _fake_arxiv_response(captured):
    def fake_get(url, params=None, headers=None, timeout=None,
                 follow_redirects=None):
        captured.append({"url": url, "params": params})
        request = httpx.Request("GET", url)
        return httpx.Response(200, text=SAMPLE_ATOM, request=request)
    return fake_get


def _fake_openalex_response(captured):
    def fake_get(url, params=None, headers=None, timeout=None,
                 follow_redirects=None):
        captured.append({"url": url, "params": params})
        request = httpx.Request("GET", url)
        return httpx.Response(200, text=json.dumps(SAMPLE_OPENALEX),
                              request=request)
    return fake_get


# ―― 纯函数:查询构建与参数清洗 ――――――――――――――――――――――


def test_build_query_default_stays_plain():
    """不带新参数时检索式与旧版完全一致,保证行为兼容。"""
    assert _build_arxiv_search_query("LLM agents") == "all:LLM agents"


def test_build_query_with_categories_and_date():
    built = _build_arxiv_search_query(
        "LLM agents", categories=["cs.AI", "cs.CL"],
        date_range="submittedDate:[202501010000 TO 202602012359]")
    assert built == ("all:LLM agents AND (cat:cs.AI OR cat:cs.CL)"
                     " AND submittedDate:[202501010000 TO 202602012359]")


def test_normalize_date_validates():
    assert _normalize_date("2025-06-30") == "20250630"
    for bad in ("2025/06/30", "2025-13-01", "昨天", "2025-02-30"):
        try:
            _normalize_date(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} 应被拒绝")


def test_clean_categories_accepts_list_and_csv():
    assert _clean_categories(["cs.AI", " cs.CL ", "cs.AI"]) == ["cs.AI", "cs.CL"]
    assert _clean_categories("cs.AI,cs.LG") == ["cs.AI", "cs.LG"]
    assert _clean_categories(None) == []
    try:
        _clean_categories(["不是分类"])
    except ValueError:
        pass
    else:
        raise AssertionError("非法分类号应被拒绝")


# ―― run():主源参数组装与错误回传 ―――――――――――――――――――


def test_run_maps_sort_categories_and_date(monkeypatch):
    captured = []
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get",
                        _fake_arxiv_response(captured))
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    result = ArxivSearchTool().run(
        "LLM agents", sort_by="submitted_date",
        categories=["cs.AI"], from_date="2025-01-01")

    assert "Fresh Agent Paper" in result
    params = captured[0]["params"]
    assert params["sortBy"] == "submittedDate"
    assert params["search_query"].startswith(
        "all:LLM agents AND (cat:cs.AI) AND submittedDate:[202501010000 TO ")
    # 区间上界是"明天",只校验结构,不写死日期保证测试确定性
    assert re.search(r"TO \d{8}2359\]$", params["search_query"])


def test_run_returns_error_text_on_bad_params(monkeypatch):
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get",
                        _fake_arxiv_response([]))
    tool = ArxivSearchTool()

    bad_date = tool.run("agents", from_date="2025/01/01")
    assert bad_date.startswith("参数无效") and "YYYY-MM-DD" in bad_date

    bad_sort = tool.run("agents", sort_by="cheapest")
    assert "sort_by" in bad_sort

    bad_category = tool.run("agents", categories=["不是分类"])
    assert bad_category.startswith("参数无效")


def test_openalex_fallback_carries_sort_date_and_category_terms(monkeypatch):
    captured = []
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    def fake_get(url, params=None, headers=None, timeout=None,
                 follow_redirects=None):
        captured.append({"url": url, "params": params})
        if "export.arxiv.org" in url:
            request = httpx.Request("GET", url)
            return httpx.Response(429, text="", request=request)
        return _fake_openalex_response([])(url, params, headers,
                                           timeout, follow_redirects)

    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", fake_get)

    result = ArxivSearchTool().run(
        "LLM agents", sort_by="submitted_date",
        categories=["cs.AI", "eess.SP"], from_date="2025-06-01")

    assert "Fresh Agent Paper (OA)" in result
    oa_params = captured[-1]["params"]
    assert oa_params["sort"] == "publication_date:desc"
    assert "from_publication_date:2025-06-01" in oa_params["filter"]
    # cs.AI 映射为关键词;eess.SP 无映射,按小写原文并入,不静默丢弃
    assert "artificial intelligence" in oa_params["filter"]
    assert "eess sp" in oa_params["filter"]


def test_schema_teaches_refinement_strategies():
    """工具描述必须教会模型"分类限定/按时间排序"这两个改写策略。"""
    function = ArxivSearchTool().schema()["function"]
    description = function["description"]
    assert "categories" in description
    assert "submitted_date" in description
    props = function["parameters"]["properties"]
    assert props["sort_by"]["enum"] == ["relevance", "submitted_date"]
    assert props["categories"]["type"] == "array"
    assert "YYYY-MM-DD" in props["from_date"]["description"]


def test_run_result_forwards_refinement_kwargs(monkeypatch):
    """ToolRegistry 以 run_result(**模型参数) 派发:新参数必须能透传,
    否则模型一旦使用增强参数就会收到 TypeError(真实回归)。"""
    captured = []
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get",
                        _fake_arxiv_response(captured))
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)

    result = ArxivSearchTool().run_result(
        "LLM agents", max_results=5, sort_by="submitted_date",
        categories=["cs.AI"], from_date="2026-01-01")

    assert result.success
    assert "Fresh Agent Paper" in result.text
    assert captured[0]["params"]["sortBy"] == "submittedDate"
