"""文献工具 1:arXiv 搜索。

arXiv 提供免费的公开 API(不用注册、不要 Key):
    GET https://export.arxiv.org/api/query?search_query=...
返回 Atom 格式的 XML。这里用标准库 ElementTree 亲手解析,
不引第三方 arxiv 包 —— 亲手解析一次,才知道"搜索工具"底层长什么样。

网络说明:httpx 默认识别 HTTP_PROXY/HTTPS_PROXY 环境变量,
访问 arXiv 慢的话设好代理环境变量即可,代码不用改。
"""

import datetime
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

# 备用源 OpenAlex 不认识 arXiv 分类号,这里做近似关键词映射;
# 没映射到的分类按小写原文并入查询,并在结果里如实标注来源
CATEGORY_TERM_MAP = {
    "cs.AI": "artificial intelligence",
    "cs.CL": "natural language processing",
    "cs.LG": "machine learning",
    "stat.ML": "machine learning",
    "cs.CV": "computer vision",
    "cs.IR": "information retrieval",
    "cs.HC": "human-computer interaction",
    "cs.CR": "security cryptography",
    "cs.MA": "multi-agent systems",
}
_CATEGORY_PATTERN = re.compile(r"[a-z-]+(?:\.[A-Z]{2})?")
_SORT_API_NAMES = {"relevance": "relevance", "submitted_date": "submittedDate"}


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


def _normalize_date(value: str) -> str:
    """校验 YYYY-MM-DD 并返回紧凑形式 YYYYMMDD(arXiv 区间语法用)。"""
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(value).strip())
    if not match:
        raise ValueError(f"日期格式应为 YYYY-MM-DD,收到:{value!r}")
    year, month, day = (int(part) for part in match.groups())
    try:
        datetime.date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"日期不合法:{value!r}({exc})") from exc
    return f"{year:04d}{month:02d}{day:02d}"


def _clean_categories(categories) -> list:
    """接受列表或逗号分隔字符串,去空、去重、限 4 个并校验分类号格式。"""
    if categories is None:
        return []
    if isinstance(categories, str):
        raw = categories.split(",")
    else:
        raw = list(categories)
    cleaned = []
    for item in raw:
        category = str(item).strip()
        if not category:
            continue
        if not _CATEGORY_PATTERN.fullmatch(category):
            raise ValueError(
                f"分类号格式不合法:{category!r}(应形如 cs.AI、stat.ML)")
        if category not in cleaned:
            cleaned.append(category)
    return cleaned[:4]


def _build_arxiv_search_query(query: str, categories=None,
                              date_range: str = None) -> str:
    """组装 arXiv 检索式;纯函数,便于离线测试。

    形如:all:关键词 AND (cat:cs.AI OR cat:cs.CL) AND submittedDate:[... TO ...]
    """
    search = f"all:{query}"
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        search += f" AND ({cats})"
    if date_range:
        search += f" AND {date_range}"
    return search


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "搜索 arXiv 学术论文,返回标题、作者、arXiv 编号、日期和摘要。"
        "用法建议:宽泛主题(如 'AI'、'agent')先用 categories 限定分类"
        "(如 cs.AI、cs.CL)再搜,避免噪声;用户要最新论文时用 "
        "sort_by='submitted_date';只看某日期之后的用 from_date='YYYY-MM-DD'。"
        "查询请用英文关键词;一个主题角度一次查询,重要主题可分多次从不同"
        "角度检索。官方接口繁忙时自动改用 OpenAlex 的 arXiv 索引。"
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
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "submitted_date"],
                "description": "排序方式:relevance=相关度(默认);"
                               "submitted_date=最新提交优先",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "arXiv 分类过滤,如 ['cs.AI'] 或 "
                               "['cs.CL','cs.LG'],最多 4 个",
            },
            "from_date": {
                "type": "string",
                "description": "只返回该日期(YYYY-MM-DD)之后提交的论文",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 5,
            sort_by: str = "relevance", categories=None,
            from_date: str = None) -> str:
        max_results = max(1, min(int(max_results), 10))  # 防模型狮子大开口挤爆上下文
        sort_by = str(sort_by or "relevance").strip().lower()
        if sort_by not in _SORT_API_NAMES:
            return (f"参数无效:sort_by 只支持 relevance 或 submitted_date,"
                    f"收到:{sort_by!r}")
        try:
            cats = _clean_categories(categories)
            from_compact = _normalize_date(from_date) if from_date else None
        except ValueError as exc:
            # 错误回传而非崩溃:参数问题让模型自己修正后重试
            return f"参数无效:{exc}"
        date_range = None
        if from_compact:
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            date_range = (f"submittedDate:[{from_compact}0000 TO "
                          f"{tomorrow:%Y%m%d}2359]")
        if os.getenv("ARXIV_SEARCH_BACKEND", "auto").strip().lower() == "openalex":
            return self._fallback_result(
                query, max_results, "已按配置使用 OpenAlex",
                source="OpenAlex 的 arXiv 索引（直接检索）",
                sort_by=sort_by, categories=cats, from_date=from_date)
        params = {
            "search_query": _build_arxiv_search_query(query, cats, date_range),
            "start": 0,
            "max_results": max_results,
            "sortBy": _SORT_API_NAMES[sort_by],
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
                    query, max_results, f"HTTP {response.status_code}",
                    sort_by=sort_by, categories=cats, from_date=from_date)
            response.raise_for_status()
        except httpx.RequestError as exc:
            return self._fallback_result(
                query, max_results,
                f"{type(exc).__name__}: {exc}",
                sort_by=sort_by, categories=cats, from_date=from_date)

        papers = _parse_atom(response.text)
        if not papers:
            return f"没有找到与「{query}」相关的论文,请换个英文关键词试试"
        return _format_papers(papers[:max_results])

    def run_result(self, query: str, max_results: int = 5,
                   **kwargs) -> ToolResult:
        """把旧文字出口转换为结构化临时失败，不把控制前缀交给模型。

        新增检索参数(sort_by/categories/from_date)经 **kwargs 原样转发:
        ToolRegistry 会把模型给出的全部参数透传进来,签名写死会导致
        增强参数触发 TypeError。
        """
        legacy = self.run(query, max_results, **kwargs)
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
                         source: str = "OpenAlex 的 arXiv 索引（自动备用）",
                         sort_by: str = "relevance", categories=None,
                         from_date: str = None) -> str:
        """arXiv 持续限流时改查 OpenAlex 的 arXiv 索引。"""
        try:
            papers = self._search_openalex(
                query, max_results, sort_by=sort_by,
                categories=categories, from_date=from_date)
        except (httpx.HTTPError, ValueError) as exc:
            return _temporary_failure(
                f"arXiv: {arxiv_error}; OpenAlex: {type(exc).__name__}: {exc}")
        if not papers:
            return (f"没有找到与「{query}」相关且带 arXiv 编号的论文"
                    f"（arXiv 接口状态:{arxiv_error}）")
        return _format_papers(papers[:max_results], source=source)

    def _search_openalex(self, query: str, max_results: int,
                         sort_by: str = "relevance", categories=None,
                         from_date: str = None) -> list:
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
        if categories:
            # OpenAlex 不认识 arXiv 分类号,用映射表并入近似关键词;
            # 未映射的分类按小写原文并入,无害且不静默丢弃语义
            terms = " ".join(
                CATEGORY_TERM_MAP.get(c, c.replace(".", " ").lower())
                for c in categories)
            safe_query = f"{safe_query} {terms}"
        arxiv_filter = (
            f"primary_location.source.id:{OPENALEX_ARXIV_SOURCE_ID},"
            f"title_and_abstract.search:{safe_query}"
        )
        if from_date:
            arxiv_filter += f",from_publication_date:{from_date}"
        params = {
            "filter": arxiv_filter,
            "per-page": max_results,
            "select": ("title,publication_date,authorships,"
                       "abstract_inverted_index,primary_location,doi"),
        }
        if sort_by == "submitted_date":
            params["sort"] = "publication_date:desc"
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
