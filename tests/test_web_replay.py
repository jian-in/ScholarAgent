"""工作台实时与已保存案例接口的契约。"""

import json
import threading
from urllib.request import urlopen

import webapp
from scholaragent.replay import SavedCaseStore


def test_workspace_exposes_saved_case_projection(tmp_path):
    bundle = tmp_path / "case-one"
    bundle.mkdir()
    (bundle / "runs.jsonl").write_text(json.dumps({
        "run_id": "case-one:react:1",
        "case_id": "case-one",
        "task": "离线",
        "mode": "react",
        "answer": "答复",
        "metrics": {"llm_calls": 0, "tool_calls": 0, "seconds": 0.0},
        "artifacts": {},
        "error": None,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    workspace = webapp.LocalWorkspace(SavedCaseStore(tmp_path))

    assert workspace.list_saved_cases()[0]["id"] == "case-one"
    replay = workspace.replay_case("case-one")
    assert replay["source"] == "saved_case"
    assert replay["selected"]["answer"] == "答复"


def test_http_exposes_cases_and_marks_replay_source(tmp_path):
    bundle = tmp_path / "case-http"
    bundle.mkdir()
    (bundle / "runs.jsonl").write_text(json.dumps({
        "run_id": "case-http:plan:1", "case_id": "case-http",
        "task": "离线", "mode": "plan", "answer": "回放",
        "metrics": {}, "artifacts": {}, "error": None,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    webapp.WorkspaceRequestHandler.workspace = webapp.LocalWorkspace(SavedCaseStore(tmp_path))
    server = webapp.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/api/cases") as response:
            cases = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base}/api/cases/case-http") as response:
            replay = json.loads(response.read().decode("utf-8"))
        assert cases["cases"][0]["id"] == "case-http"
        assert replay["selected"]["source"] == "saved_case"
        assert replay["selected"]["mode"] == "plan"
    finally:
        server.shutdown()
        server.server_close()
