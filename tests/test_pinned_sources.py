"""Run-level version constraints and registered-source handoff, offline."""
import pytest
import httpx

from scholaragent.events import RunContext
from scholaragent.tool import Tool, ToolRegistry
from scholaragent.tools.arxiv_search import ArxivSearchTool
from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import create_runtime
from scholaragent.workspace import TemporaryWorkspace


class PaperTool(Tool):
    name = "download_paper"

    def __init__(self):
        self.ids = []

    def run(self, arxiv_id):
        self.ids.append(arxiv_id)
        return "fixture paper"


def test_version_constraint_reaches_subset_and_corrects_omitted_version():
    tool = PaperTool()
    registry = ToolRegistry([tool]).subset(["download_paper"])
    context = RunContext(pinned_sources=["2210.03629v1"])
    result = registry.call_result("download_paper", {"arxiv_id": "2210.03629"}, context)
    assert result.success
    assert tool.ids == ["2210.03629v1"]
    blocked = registry.call_result("download_paper", {"arxiv_id": "2210.03629v3"}, context)
    assert not blocked.success and not blocked.stop_retry
    assert blocked.diagnostic["kind"] == "source_version_mismatch"
    assert tool.ids == ["2210.03629v1"]


ATOM = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2210.03629v3</id><title>ReAct fixture</title>
<author><name>Fixture Author</name></author><published>2022-10-06</published>
<summary>Abstract evidence, not full text.</summary></entry></feed>'''


def test_search_metadata_is_registered_with_its_actual_version(monkeypatch):
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", lambda url, **kw:
                        httpx.Response(200, text=ATOM, request=httpx.Request("GET", url)))
    context = RunContext(pinned_sources=["2210.03629v1"])
    result = ToolRegistry([ArxivSearchTool()]).call_result("arxiv_search", {"query": "ReAct"}, context)
    assert result.success
    assert len(context.evidence.anchors) == 1
    anchor = context.evidence.anchors[0]
    assert anchor.kind == "metadata" and anchor.source == "arxiv:2210.03629v3"
    assert anchor.page is None and "Abstract evidence" in anchor.excerpt
    assert result.artifacts[0]["source_anchors"][0]["id"] == anchor.id


class RecordingLLM(ScriptedLLM):
    def __init__(self, replies):
        super().__init__(replies)
        self.history = []

    def chat(self, messages, tools=None):
        self.history.append([dict(x) for x in messages])
        return super().chat(messages, tools)


def final(text):
    return {"content": text, "tool_calls": []}


def call(name, arguments):
    return {"content": None, "tool_calls": [{"id": "t1", "name": name, "arguments": arguments}]}


def test_runtime_team_keeps_goal_pins_and_actual_anchor_catalog(tmp_path, monkeypatch):
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", lambda url, **kw:
                        httpx.Response(200, text=ATOM, request=httpx.Request("GET", url)))
    downloaded = []
    monkeypatch.setattr("scholaragent.tools.papers.DownloadPaperTool.run",
                        lambda self, arxiv_id: downloaded.append(arxiv_id) or "fixture download")
    llm = RecordingLLM([
        call("arxiv_search", {"query": "ReAct"}), final("推荐精读：2210.03629。报告未传约束或锚点。"),
        call("download_paper", {"arxiv_id": "2210.03629"}), final("正文尚未读取。"),
        final("检索元数据与固定版本不同。[S001]"),
    ])
    runtime = create_runtime(llm=llm, workspace=TemporaryWorkspace(tmp_path),
                             conversation=False, auto_recall=False, team_require_full_paper=False)
    result = runtime.run("仅核对固定版本，勿引用新论文补证", mode="team",
                         pinned_sources=["2210.03629v1"])
    assert downloaded == ["2210.03629v1"]
    for history in (llm.history[2], llm.history[4]):
        user_text = "\n".join(x.get("content") or "" for x in history if x["role"] == "user")
        assert "仅核对固定版本，勿引用新论文补证" in user_text
        assert "2210.03629v1" in user_text
        assert "[S001]" in user_text and "arxiv:2210.03629v3" in user_text
        assert "metadata" in user_text
    assert not result.evidence["validation_errors"]
    assert result.evidence["claims"][0]["review_status"] == "unreviewed"


@pytest.mark.parametrize("name", ["download_paper", "read_paper"])
@pytest.mark.parametrize("requested", ["2210.03629v3", "2405.13966", "2210.03629v0", None])
def test_paper_constraints_stop_external_work_before_execution(name, requested):
    tool = PaperTool()
    tool.name = name
    context = RunContext(pinned_sources=["2210.03629v1"])
    result = ToolRegistry([tool]).call_result(name, {"arxiv_id": requested}, context)
    assert not result.success and not result.artifacts and not tool.ids
    assert context.metrics.snapshot().tool_calls == 1
    assert any(e.type == "tool_completed" and not e.payload["success"] for e in context.event_list)


def test_multiple_pinned_versions_need_explicit_version_and_do_not_leak():
    tool = PaperTool()
    registry = ToolRegistry([tool])
    context = RunContext(pinned_sources=["2210.03629v1", "2210.03629v3"])
    assert not registry.call_result(tool.name, {"arxiv_id": "2210.03629"}, context).success
    assert registry.call_result(tool.name, {"arxiv_id": "2210.03629v3"}, context).success
    assert registry.call_result(tool.name, {"arxiv_id": "2405.13966"}, RunContext()).success
    assert tool.ids == ["2210.03629v3", "2405.13966"]


@pytest.mark.parametrize("mode", ["react", "plan", "team"])
def test_all_modes_share_the_run_level_constraint(tmp_path, monkeypatch, mode):
    downloads = []
    monkeypatch.setattr("scholaragent.tools.papers.DownloadPaperTool.run",
                        lambda self, arxiv_id: downloads.append(arxiv_id) or "fixture")
    worker = [call("download_paper", {"arxiv_id": "2210.03629v3"}), final("版本不匹配，未取得正文。")]
    replies = {"react": worker,
               "plan": [final('["核对论文版本"]'), *worker, final('{"ok":true}'), final("阶段结论")],
               "team": [final("推荐精读：2210.03629v3"), *worker, final("阶段结论")]}[mode]
    runtime = create_runtime(llm=ScriptedLLM(replies), workspace=TemporaryWorkspace(tmp_path),
                             conversation=False, auto_recall=False, team_require_full_paper=False)
    result = runtime.run("核对版本", mode=mode, pinned_sources=["2210.03629v1"])
    assert not downloads
    failures = [e for e in result.events if e["type"] == "tool_completed"]
    assert failures and failures[0]["payload"]["diagnostic"]["kind"] == "source_version_mismatch"
    assert result.events[0]["payload"]["pinned_sources"] == ["2210.03629v1"]


def test_openalex_records_get_metadata_anchors_without_claiming_body(monkeypatch):
    monkeypatch.setenv("ARXIV_SEARCH_BACKEND", "openalex")
    payload = {"results": [{"title": "Indexed fixture", "publication_date": "2022-10-06",
              "primary_location": {"landing_page_url": "https://arxiv.org/abs/2210.03629"},
              "authorships": [], "abstract_inverted_index": {"Abstract": [0], "only": [1]}}]}
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", lambda url, **kw:
                        httpx.Response(200, json=payload, request=httpx.Request("GET", url)))
    context = RunContext()
    result = ToolRegistry([ArxivSearchTool()]).call_result("arxiv_search", {"query": "fixture"}, context)
    assert result.success and len(context.evidence.anchors) == 1
    anchor = context.evidence.anchors[0]
    assert anchor.source == "arxiv:2210.03629" and anchor.page is None
    assert anchor.kind == "metadata" and "OpenAlex" in anchor.locator


def test_empty_search_does_not_create_anchors(monkeypatch):
    monkeypatch.delenv("ARXIV_SEARCH_BACKEND", raising=False)
    monkeypatch.setattr("scholaragent.tools.arxiv_search.httpx.get", lambda url, **kw:
                        httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>',
                                       request=httpx.Request("GET", url)))
    context = RunContext()
    ToolRegistry([ArxivSearchTool()]).call_result("arxiv_search", {"query": "empty"}, context)
    assert not context.evidence.anchors


def test_experiment_enforces_structured_task_sources(tmp_path, monkeypatch):
    import json
    from evals.run_experiment import execute_experiment
    downloads = []
    monkeypatch.setattr("scholaragent.tools.papers.DownloadPaperTool.run",
                        lambda self, arxiv_id: downloads.append(arxiv_id) or "fixture")
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(json.dumps({"id": "pinned", "task": "核对来源",
                                    "sources": ["2210.03629v1"]}), encoding="utf-8")
    def factory(**kwargs):
        return create_runtime(llm=ScriptedLLM([
            call("download_paper", {"arxiv_id": "2210.03629v3"}), final("资料未取得。")]), **kwargs)
    paths = execute_experiment({"schema_version": "experiment-definition-v1", "id": "pinned-offline",
                                "tasks": str(task_file), "modes": ["react"]},
                               tmp_path / "bundle", runtime_factory=factory)
    row = json.loads(paths["runs"].read_text(encoding="utf-8"))
    assert not downloads and row["pinned_sources"] == ["2210.03629v1"]
    assert row["events"][0]["payload"]["pinned_sources"] == ["2210.03629v1"]
