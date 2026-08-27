"""科研产物和运行证据的确定性审查契约。"""

import json

from scholaragent.audit import audit_evidence_bundle, audit_run_result
from scholaragent.experiments import write_evidence_bundle
from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import create_runtime
from scholaragent.workspace import TemporaryWorkspace


def test_audit_accepts_a_consistent_run_result_and_rejects_mixed_run_ids(tmp_path):
    runtime = create_runtime(
        llm=ScriptedLLM([{"content": "回答", "tool_calls": []}]),
        workspace=TemporaryWorkspace(tmp_path / "valid"),
        conversation=False,
        auto_recall=False,
    )
    result = runtime.run("一个任务", mode="react")

    report = audit_run_result(result)
    assert report.ok is True
    assert report.errors == ()

    broken = result.to_dict()
    broken["events"] = [dict(event) for event in broken["events"]]
    broken["events"][0]["run_id"] = "other-run"
    invalid = audit_run_result(broken)
    assert invalid.ok is False
    assert any("run_id" in error for error in invalid.errors)


def test_audit_checks_a_whole_evidence_bundle(tmp_path):
    runtime = create_runtime(
        llm=ScriptedLLM([{"content": "回答", "tool_calls": []}]),
        workspace=TemporaryWorkspace(tmp_path / "state"),
        conversation=False,
        auto_recall=False,
    )
    result = runtime.run("一个任务", mode="react")
    paths = write_evidence_bundle(
        [result.to_dict()],
        tmp_path / "bundle",
        experiment_id="audit-demo",
        tasks=[{"id": "task-1", "task": "一个任务"}],
    )

    assert audit_evidence_bundle(paths["directory"]).ok is True

    row = json.loads(paths["runs"].read_text(encoding="utf-8").strip())
    row["events"][0]["run_id"] = "mixed-run"
    paths["runs"].write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    report = audit_evidence_bundle(paths["directory"])
    assert report.ok is False
    assert any("run_id" in error for error in report.errors)
