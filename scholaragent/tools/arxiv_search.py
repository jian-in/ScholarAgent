"""文献工具 1:arXiv 搜索。

arXiv 提供免费的公开 API(不用注册、不要 Key):
    GET https://export.arxiv.org/api/query?search_query=...
返回 Atom 格式的 XML。这里用标准库 ElementTree 亲手解析,
不引第三方 arxiv 包 —— 亲手解析一次,才知道"搜索工具"底层长什么样。

网络说明:httpx 默认识别 HTTP_PROXY/HTTPS_PROXY 环境变量,
访问 arXiv 慢的话设好代理环境变量即可,代码不用改。
"""

import os
import re
import xml.etree.ElementTree as ET

import httpx

from ..tool import STOP_RETRY_PREFIX, Tool, ToolResult, adapt_tool_result

API_URL = "https://export.arxiv.org/api/query"
OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_ARXIV_SOURCE_ID = "S4306400194"

# Atom XML 里所有标签都带命名空间,查找时必须带上,否则什么都找不到
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def _temporary_failure(error: str) -> str:
    return (
        f"{STOP_RETRY_PREFIX} arXiv 服务连续请求失败（{error}）。"
        "本轮请不要再次调用 arxiv_search；"
        "请说明检索服务暂时繁忙，稍后再运行。"
    )


def _text(node, path: str) -> str:
    """安全地取出子节点的文字:节点不存在时返回空串而不是崩溃。"""
    child = node.find(path, _ATOM)
    return (child.text or "").strip() if child is not None else ""


def _parse_atom(xml_text: str) -> list:
    """把 arXiv 返回的 Atom XML 解析成论文信息列表(纯函数,方便离线测试)。"""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", _ATOM):
        # <id> 形如 http://arxiv.org/abs/2401.12345v2,只要最后一段编号
        arxiv_id = _text(entry, "atom:id").rsplit("/", 1)[-1]
        papers.append({
            "id": arxiv_id,
            # 标题在 XML 里常被折行,压掉换行和连续空格
            "title": " ".join(_text(entry, "atom:title").split()),
            "authors": [_text(a, "atom:name")
                        for a in entry.findall("atom:author", _ATOM)],
            "published": _text(entry, "atom:published")[:10],
            "summary": " ".join(_text(entry, "atom:summary").split()),
        })
    return papers


def _abstract_from_inverted_index(index) -> str:
    """OpenAlex 用「单词 -> 位置列表」保存摘要,这里还原为普通文本。"""
    if not isinstance(index, dict) or not index:
        return ""
    max_position = max(
        (position for positions in index.values() for position in positions),
        default=-1,
    )
    words = [""] * (max_position + 1)
    for word, positions in index.items():
        for position in positions:
            if 0 <= position < len(words):
                words[position] = word
    return " ".join(word for word in words if word)


def _openalex_arxiv_id(work: dict) -> str:
    """从 OpenAlex 的 arXiv 落地页或 DOI 中提取可下载的 arXiv 编号。"""
    location = work.get("primary_location") or {}
    for url in (location.get("landing_page_url"), location.get("pdf_url")):
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?$", url or "",
                          flags=re.IGNORECASE)
        if match:
            return match.group(1).split("?", 1)[0]
    doi = work.get("doi") or ""
    match = re.search(r"10\.48550/arxiv\.(.+)$", doi, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _parse_openalex(payload: dict) -> list:
    """把 OpenAlex 的 arXiv 索引结果转换成与 Atom 解析器相同的结构。"""
    papers = []
    for work in payload.get("results", []):
        arxiv_id = _openalex_arxiv_id(work)
        if not arxiv_id:
            continue
        papers.append({
            "id": arxiv_id,
            "title": " ".join((work.get("title") or "").split()),
            "authors": [
                authorship.get("author", {}).get("display_name", "")
                for authorship in work.get("authorships", [])
                if authorship.get("author", {}).get("display_name")
            ],
            "published": work.get("publication_date") or "",
            "summary": _abstract_from_inverted_index(
                work.get("abstract_inverted_index")),
        })
    return papers


def _format_papers(papers: list, source: str = "arXiv") -> str:
    lines = [f"[检索源:{source}]"]
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " 等"
        lines.append(
            f"{i}. {paper['title']}\n"
            f"   arXiv编号: {paper['id']}  日期: {paper['published']}  作者: {authors}\n"
            f"   摘要: {paper['summary'][:300]}"
        )
    return "\n".join(lines)


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "搜索 arXiv 学术论文,返回标题、作者、arXiv 编号、日期和摘要。"
        "支持用 OpenAlex 的 arXiv 索引避开官方接口限流。"
        "查询请用英文关键词,例如 'LLM agent tool use'"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "英文搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "返回论文数量,默认 5,最多 10",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 5) -> str:
        max_results = max(1, min(int(max_results), 10))  # 防模型狮子大开口挤爆上下文
        if os.getenv("ARXIV_SEARCH_BACKEND", "auto").strip().lower() == "openalex":
            return self._fallback_result(
                query, max_results, "已按配置使用 OpenAlex",
                source="OpenAlex 的 arXiv 索引（直接检索）")
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        headers = {
            "Accept": "application/atom+xml",
            "User-Agent": os.getenv(
                "ARXIV_USER_AGENT",
                "ScholarAgent/1.0 (educational research client)",
            ),
        }

        try:
            response = httpx.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=httpx.Timeout(8.0, connect=4.0),
                follow_redirects=True,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                return self._fallback_result(
                    query, max_results, f"HTTP {response.status_code}")
            response.raise_for_status()
        except httpx.RequestError as exc:
            return self._fallback_result(
                query, max_results,
                f"{type(exc).__name__}: {exc}")

        papers = _parse_atom(response.text)
        if not papers:
            return f"没有找到与「{query}」相关的论文,请换个英文关键词试试"
        return _format_papers(papers[:max_results])

    def run_result(self, query: str, max_results: int = 5) -> ToolResult:
        """把旧文字出口转换为结构化临时失败，不把控制前缀交给模型。"""
        legacy = self.run(query, max_results)
        if legacy.startswith(STOP_RETRY_PREFIX):
            return ToolResult(
                text=legacy[len(STOP_RETRY_PREFIX):].lstrip(),
                success=False,
                stop_retry=True,
                diagnostic={"kind": "temporary_service_failure"},
            )
        return adapt_tool_result(legacy)

    def _fallback_result(self, query: str, max_results: int,
                         arxiv_error: str,
                         source: str = "OpenAlex 的 arXiv 索引（自动备用）") -> str:
        """arXiv 持续限流时改查 OpenAlex 的 arXiv 索引。"""
        try:
            papers = self._search_openalex(query, max_results)
        except (httpx.HTTPError, ValueError) as exc:
            return _temporary_failure(
                f"arXiv: {arxiv_error}; OpenAlex: {type(exc).__name__}: {exc}")
        if not papers:
            return (f"没有找到与「{query}」相关且带 arXiv 编号的论文"
                    f"（arXiv 接口状态:{arxiv_error}）")
        return _format_papers(papers[:max_results], source=source)

    def _search_openalex(self, query: str, max_results: int) -> list:
        headers = {
            "Accept": "application/json",
            "User-Agent": os.getenv(
                "ARXIV_USER_AGENT",
                "ScholarAgent/1.0 (educational research client)",
            ),
        }
        # OpenAlex 的普通 search 会把正文偶然提到关键词的论文排得很靠前。
        # 限定在标题与摘要后，文献检索相关度明显更稳定。过滤器用逗号/冒号
        # 分隔语法，先清掉这些控制符，避免模型生成的标点破坏参数结构。
        safe_query = " ".join(re.findall(r"[A-Za-z0-9.+_/-]+", query)) or query
        params = {
            "filter": (
                f"primary_location.source.id:{OPENALEX_ARXIV_SOURCE_ID},"
                f"title_and_abstract.search:{safe_query}"
            ),
            "per-page": max_results,
            "select": ("title,publication_date,authorships,"
                       "abstract_inverted_index,primary_location,doi"),
        }
        email = os.getenv("OPENALEX_EMAIL", "").strip()
        if email:
            params["mailto"] = email
        response = httpx.get(
            OPENALEX_API_URL,
            params=params,
            headers=headers,
            timeout=httpx.Timeout(12.0, connect=6.0),
            follow_redirects=True,
        )
        response.raise_for_status()
        return _parse_openalex(response.json())
