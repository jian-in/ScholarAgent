"""本地工作台的无模型测试。"""

import json
import threading
import time
from urllib.request import Request, urlopen

import webapp


class FakeRunner:
    def __init__(self, answer, delay=0.0, logs=None, check_stop_each=0.05):
        self.answer = answer
        self.delay = delay
        self.logs = list(logs or [])
        self.calls = []
        self.on_progress = None
        self.should_stop = None
        self.verbose = True
        self.last_metrics = None
        self.check_stop_each = check_stop_each

    def run(self, task):
        from scholaragent.agent import CANCELLED_ANSWER

        self.calls.append(task)
        if self.on_progress:
            for message in self.logs:
                if self.should_stop and self.should_stop():
                    if self.on_progress:
                        self.on_progress("[取消] 假执行器在步骤边界停止")
                    return CANCELLED_ANSWER
                self.on_progress(message)
        remaining = self.delay
        while remaining > 0:
            if self.should_stop and self.should_stop():
                if self.on_progress:
                    self.on_progress("[取消] 假执行器在等待中停止")
                return CANCELLED_ANSWER
            step = min(self.check_stop_each, remaining)
            time.sleep(step)
            remaining -= step
        return self.answer


def test_workspace_reuses_only_the_explicit_runner(monkeypatch):
    runners = {mode: FakeRunner(mode) for mode in ("react", "plan", "team")}
    monkeypatch.setattr(webapp, "build_agent", lambda demo, artifacts=None: object())
    monkeypatch.setattr(
        webapp,
        "build_runners",
        lambda agent, on_progress=None, should_stop=None, artifacts=None: runners,
    )
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")

    result = webapp.LocalWorkspace().run("查论文", "plan")
    assert result["answer"] == "plan"
    assert result["mode"] == "plan"
    assert runners["plan"].calls == ["查论文"]
    assert runners["react"].calls == []
    assert runners["team"].calls == []


def test_workspace_rejects_empty_or_unknown_tasks():
    workspace = webapp.LocalWorkspace()
    try:
        workspace.run("", "auto")
        raise AssertionError("空任务不应提交")
    except ValueError as exc:
        assert "不能为空" in str(exc)
    try:
        workspace.run("任务", "unknown")
        raise AssertionError("未知模式不应提交")
    except ValueError as exc:
        assert "未知" in str(exc)


def test_server_serves_page_and_status_endpoint():
    server = webapp.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base_url}/api/status") as response:
            status = json.loads(response.read().decode("utf-8"))
        with urlopen(base_url) as response:
            page = response.read().decode("utf-8")
        assert "model" in status
        assert "ScholarAgent" in page
        assert "/app.js" in page
        assert "progress-log" in page
        assert "cancel-task" in page
        assert "copy-answer" in page
        assert "artifacts-panel" in page
    finally:
        server.shutdown()
        server.server_close()


def test_job_streams_progress_logs(monkeypatch):
    """后台任务应能被轮询到步骤日志，再拿到最终答案。"""

    def fake_build_runners(agent, on_progress=None, should_stop=None, artifacts=None):
        runners = {
            mode: FakeRunner(
                f"answer-{mode}",
                delay=0.15,
                logs=[f"[{mode}] 开始", f"[{mode}] 调用工具 calculator"],
            )
            for mode in ("react", "plan", "team")
        }
        for runner in runners.values():
            if on_progress:
                runner.on_progress = on_progress
            if should_stop:
                runner.should_stop = should_stop
        return runners

    monkeypatch.setattr(webapp, "build_agent", lambda demo, artifacts=None: type("A", (), {"verbose": True})())
    monkeypatch.setattr(webapp, "build_runners", fake_build_runners)
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")

    workspace = webapp.LocalWorkspace()
    job_id = workspace.start_job("帮我算一下", "react")

    saw_logs = False
    final = None
    for _ in range(40):
        snap = workspace.job_status(job_id, after=0)
        if snap["logs"]:
            saw_logs = True
        if snap["status"] in ("done", "error", "cancelled"):
            final = snap
            break
        time.sleep(0.05)

    assert saw_logs, "执行过程中应能看到步骤日志"
    assert final is not None
    assert final["status"] == "done"
    assert final["answer"] == "answer-react"
    assert final["mode"] == "react"
    assert final.get("answer_html")
    assert "answer-react" in final["answer_html"]
    messages = [entry["message"] for entry in final["logs"]]
    assert any("调用工具" in message for message in messages)
    assert any("工作台" in message for message in messages)


def test_job_api_over_http(monkeypatch):
    def fake_build_runners(agent, on_progress=None, should_stop=None, artifacts=None):
        runner = FakeRunner("http-ok", delay=0.05, logs=["[react] step"])
        if on_progress:
            runner.on_progress = on_progress
        if should_stop:
            runner.should_stop = should_stop
        return {"react": runner, "plan": FakeRunner("p"), "team": FakeRunner("t")}

    monkeypatch.setattr(webapp, "build_agent", lambda demo, artifacts=None: type("A", (), {"verbose": True})())
    monkeypatch.setattr(webapp, "build_runners", fake_build_runners)
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")

    # 每个 create_server 共用类级 workspace,测试里重置以免互相污染
    webapp.WorkspaceRequestHandler.workspace = webapp.LocalWorkspace()
    server = webapp.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        req = Request(
            f"{base_url}/api/jobs",
            data=json.dumps({"task": "测试", "mode": "react"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as response:
            created = json.loads(response.read().decode("utf-8"))
        job_id = created["job_id"]
        assert job_id

        final = None
        for _ in range(40):
            with urlopen(f"{base_url}/api/jobs/{job_id}?after=0") as response:
                snap = json.loads(response.read().decode("utf-8"))
            if snap["status"] in ("done", "error", "cancelled"):
                final = snap
                break
            time.sleep(0.05)
        assert final is not None
        assert final["status"] == "done"
        assert final["answer"] == "http-ok"
    finally:
        server.shutdown()
        server.server_close()


def test_cancel_job_stops_cooperatively(monkeypatch):
    """取消请求应在下一步边界生效,状态变为 cancelled。"""

    def fake_build_runners(agent, on_progress=None, should_stop=None, artifacts=None):
        runner = FakeRunner(
            "should-not-finish",
            delay=2.0,
            logs=["[react] 开始长任务"],
            check_stop_each=0.05,
        )
        if on_progress:
            runner.on_progress = on_progress
        if should_stop:
            runner.should_stop = should_stop
        return {
            "react": runner,
            "plan": FakeRunner("p"),
            "team": FakeRunner("t"),
        }

    monkeypatch.setattr(webapp, "build_agent", lambda demo, artifacts=None: type("A", (), {"verbose": True})())
    monkeypatch.setattr(webapp, "build_runners", fake_build_runners)
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")

    workspace = webapp.LocalWorkspace()
    job_id = workspace.start_job("很长的调研", "react")

    # 等任务真正开始跑
    for _ in range(40):
        snap = workspace.job_status(job_id)
        if snap["status"] == "running":
            break
        time.sleep(0.05)

    result = workspace.cancel_job(job_id)
    assert result["ok"] is True

    final = None
    for _ in range(60):
        snap = workspace.job_status(job_id, after=0)
        if snap["status"] in ("done", "error", "cancelled"):
            final = snap
            break
        time.sleep(0.05)

    assert final is not None
    assert final["status"] == "cancelled"
    assert "取消" in (final["answer"] or "")
    messages = [entry["message"] for entry in final["logs"]]
    assert any("取消" in message for message in messages)


def test_cancel_finished_job_is_rejected(monkeypatch):
    def fake_build_runners(agent, on_progress=None, should_stop=None, artifacts=None):
        runner = FakeRunner("done-already", delay=0.0)
        return {"react": runner, "plan": FakeRunner("p"), "team": FakeRunner("t")}

    monkeypatch.setattr(webapp, "build_agent", lambda demo, artifacts=None: type("A", (), {"verbose": True})())
    monkeypatch.setattr(webapp, "build_runners", fake_build_runners)
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")

    workspace = webapp.LocalWorkspace()
    job_id = workspace.start_job("短任务", "react")
    for _ in range(40):
        if workspace.job_status(job_id)["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)

    result = workspace.cancel_job(job_id)
    assert result["ok"] is False


def test_snapshot_marks_slow_jobs(monkeypatch):
    workspace = webapp.LocalWorkspace()
    job_id = workspace.jobs.create("慢任务", "team")
    workspace.jobs.mark_running(job_id)
    # 把 started_at 拨回,模拟已运行很久
    with workspace.jobs._lock:
        workspace.jobs._jobs[job_id]["started_at"] = time.time() - (
            webapp.SLOW_HINT_SECONDS + 5
        )
    snap = workspace.job_status(job_id)
    assert snap["slow"] is True
    assert snap["elapsed"] >= webapp.SLOW_HINT_SECONDS


def test_run_includes_answer_html(monkeypatch):
    runners = {
        "react": FakeRunner("## 结论\n\n- 要点一"),
        "plan": FakeRunner("p"),
        "team": FakeRunner("t"),
    }
    monkeypatch.setattr(webapp, "build_agent", lambda demo, artifacts=None: object())
    monkeypatch.setattr(
        webapp,
        "build_runners",
        lambda agent, on_progress=None, should_stop=None, artifacts=None: runners,
    )
    monkeypatch.setattr(webapp.config, "LLM_API_KEY", "test-key")
    result = webapp.LocalWorkspace().run("写综述", "react")
    assert result["answer"].startswith("## 结论")
    assert "<h2>结论</h2>" in result["answer_html"]
    assert "<li>要点一</li>" in result["answer_html"]
    assert "artifacts" in result


def test_progress_callback_on_agent():
    """Agent 的 on_progress 应收到工具调用日志,且不影响回答。"""
    from scholaragent.agent import Agent
    from scholaragent.llm import ScriptedLLM
    from scholaragent.tool import ToolRegistry
    from scholaragent.tools.calculator import CalculatorTool

    logs = []
    llm = ScriptedLLM([
        {"content": "算一下", "tool_calls": [
            {"id": "c1", "name": "calculator",
             "arguments": {"expression": "1+1"}}]},
        {"content": "等于 2", "tool_calls": []},
    ])
    agent = Agent(
        llm, ToolRegistry([CalculatorTool()]),
        verbose=False, on_progress=logs.append,
    )
    answer = agent.run("1+1")
    assert answer == "等于 2"
    assert any("调用工具" in line for line in logs)
    assert any("工具返回" in line for line in logs)


def test_agent_stops_when_should_stop():
    from scholaragent.agent import CANCELLED_ANSWER, Agent
    from scholaragent.llm import ScriptedLLM
    from scholaragent.tool import ToolRegistry
    from scholaragent.tools.calculator import CalculatorTool

    stop = {"flag": False}
    logs = []
    llm = ScriptedLLM([
        {"content": "先算", "tool_calls": [
            {"id": "c1", "name": "calculator",
             "arguments": {"expression": "1+1"}}]},
        {"content": "再算", "tool_calls": [
            {"id": "c2", "name": "calculator",
             "arguments": {"expression": "2+2"}}]},
        {"content": "等于 4", "tool_calls": []},
    ])

    def should_stop():
        return stop["flag"]

    # 第一次工具返回后请求取消
    original_call = ToolRegistry.call

    def flaky_call(self, name, arguments):
        result = original_call(self, name, arguments)
        stop["flag"] = True
        return result

    registry = ToolRegistry([CalculatorTool()])
    registry.call = flaky_call.__get__(registry, ToolRegistry)
    agent = Agent(
        llm, registry, verbose=False,
        on_progress=logs.append, should_stop=should_stop,
    )
    answer = agent.run("算")
    assert answer == CANCELLED_ANSWER
    assert any("取消" in line for line in logs)
