"""本轮产物收集的离线测试。"""

from scholaragent.artifacts import ArtifactCollector
from scholaragent.tool import Tool, ToolRegistry


class DummyTool(Tool):
    name = "dummy"
    description = "x"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        return "ok"


def test_collector_records_successful_side_effects(tmp_path, monkeypatch):
    from scholaragent import config
    from scholaragent.tools import notes as notes_mod
    from scholaragent.tools import papers as papers_mod

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(notes_mod.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(papers_mod.config, "DATA_DIR", str(tmp_path))

    collector = ArtifactCollector()
    collector.record(
        "download_paper",
        {"arxiv_id": "2401.12345"},
        "论文 2401.12345 已就绪(12 KB,共 3 页),可以用 read_paper 开始阅读",
    )
    collector.record("read_paper", {"arxiv_id": "2401.12345"}, "第 1 页内容…")
    collector.record(
        "save_note",
        {"title": "ReAct 要点", "content": "推理与行动交替"},
        "笔记「ReAct 要点」已保存",
    )
    collector.record(
        "remember",
        {"content": "ReAct 结合推理与行动", "source": "2401.12345"},
        "已存入长期记忆(现共 1 条)",
    )
    # 失败结果不记
    collector.record("download_paper", {"arxiv_id": "9999.99999"}, "下载到的内容不是 PDF")
    collector.record("save_note", {"title": "坏", "content": "x"}, "工具 save_note 执行出错")

    data = collector.to_dict()
    assert data["counts"]["papers"] == 1
    assert data["counts"]["notes"] == 1
    assert data["counts"]["memories"] == 1
    assert data["counts"]["read"] == 1
    assert data["papers"][0]["arxiv_id"] == "2401.12345"
    assert "papers" in data["papers"][0]["path"].replace("\\", "/")
    assert data["notes"][0]["title"] == "ReAct 要点"
    assert data["memories"][0]["source"] == "2401.12345"


def test_registry_forwards_artifacts_on_call():
    collector = ArtifactCollector()
    registry = ToolRegistry([DummyTool()], artifacts=collector)
    result = registry.call("dummy", {})
    assert result == "ok"
    # dummy 不在产物白名单,不应写入
    assert collector.is_empty()

    # 直接验证 registry 会调用 record
    class NoteLike(Tool):
        name = "save_note"
        description = "n"
        parameters = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
            "required": ["title", "content"],
        }

        def run(self, title, content):
            return f"笔记「{title}」已保存"

    collector2 = ArtifactCollector()
    reg2 = ToolRegistry([NoteLike()], artifacts=collector2)
    reg2.call("save_note", {"title": "A", "content": "B"})
    assert collector2.to_dict()["counts"]["notes"] == 1


def test_subset_shares_artifacts_collector():
    collector = ArtifactCollector()
    registry = ToolRegistry([DummyTool()], artifacts=collector)
    sub = registry.subset(["dummy"])
    assert sub.artifacts is collector


def test_workspace_returns_artifacts(monkeypatch):
    import webapp
    from scholaragent.artifacts import ArtifactCollector

    class ArtifactRunner:
        def __init__(self):
            self.verbose = True
            self.last_metrics = None
            self.calls = []

        def run(self, task):
            self.calls.append(task)
            # 模拟工具登记处写入
            # build_agent 会挂 collector;这里直接在返回后由 webapp 汇总
            return "## 综述\n\n完成"

    def fake_build_agent(demo=False, artifacts=None):
        agent = type("A", (), {"verbose": True, "tools": None})()
        if artifacts is not None:
            artifacts.record(
                "save_note",
                {"title": "笔记1", "content": "要点"},
                "笔记「笔记1」已保存",
            )
            artifacts.record(
                "remember",
                {"content": "事实", "source": "2401.1"},
                "已存入长期记忆(现共 1 条)",
            )
        return agent

    def fake_build_runners(agent, on_progress=None, should_stop=None, artifacts=None):
        runner = ArtifactRunner()
        return {"react": runner, "plan": ArtifactRunner(), "team": ArtifactRunner()}

    monkeypatch.setattr(webapp, "build_agent", fake_build_agent)
    monkeypatch.setattr(webapp, "build_runners", fake_build_runners)
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")

    result = webapp.LocalWorkspace().run("调研", "react")
    assert result["artifacts"]["counts"]["notes"] == 1
    assert result["artifacts"]["counts"]["memories"] == 1
