"""实验清单与不可变证据包的公开行为。"""

import json

import pytest

from scholaragent.experiments import (
    EXPERIMENT_SCHEMA_VERSION,
    build_manifest,
    task_set_hash,
    write_evidence_bundle,
)


def test_manifest_contains_reproducibility_fields_and_omits_secret_config(tmp_path):
    tasks = [{"id": "a", "task": "比较两篇论文"}, {"id": "b", "task": "总结方法"}]

    manifest = build_manifest(
        "demo-v1",
        tasks=tasks,
        model="offline-scripted",
        strategy_version="router-v1",
        config={
            "max_steps": 15,
            "LLM_API_KEY": "must-not-be-written",
            "nested": {"temperature": 0.0, "access_token": "also-secret"},
        },
        started_at="2026-08-27T00:00:00+00:00",
    )

    data = manifest.to_dict()
    assert data["schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert data["experiment_id"] == "demo-v1"
    assert data["task_set_hash"] == task_set_hash(tasks)
    assert data["git_commit"]
    assert data["started_at"] == "2026-08-27T00:00:00+00:00"
    assert data["ended_at"] is None
    serialized = json.dumps(data, ensure_ascii=False)
    assert "must-not-be-written" not in serialized
    assert "also-secret" not in serialized
    assert "LLM_API_KEY" not in serialized


def test_evidence_bundle_is_self_describing_and_scores_are_separate(tmp_path):
    rows = [{
        "run_id": "demo:react:1",
        "task_id": "demo-task",
        "task": "离线演示",
        "mode": "react",
        "answer": "完成",
        "events": [{"type": "completed", "run_id": "demo:react:1"}],
        "metrics": {"mode": "react", "llm_calls": 1, "tool_calls": 0},
        "artifacts": {"counts": {"papers": 0, "notes": 0, "memories": 0}},
        "error": None,
    }]

    paths = write_evidence_bundle(
        rows,
        tmp_path / "bundle",
        experiment_id="demo-v1",
        tasks=[{"id": "demo-task", "task": "离线演示"}],
        model="offline-scripted",
        config={"api_key": "hidden"},
        ended_at="2026-08-27T00:01:00+00:00",
    )

    assert {path.name for path in paths.values()} >= {
        "manifest.json", "runs.jsonl", "scores.template.jsonl", "summary.md",
    }
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    run = json.loads(paths["runs"].read_text(encoding="utf-8").strip())
    score_template = json.loads(paths["scores"].read_text(encoding="utf-8").strip())
    assert manifest["ended_at"] == "2026-08-27T00:01:00+00:00"
    assert manifest["score_states"] == {"unscored": 1, "author_scored": 0, "independently_scored": 0}
    assert run["score_state"] == "unscored"
    assert score_template["run_id"] == run["run_id"]
    assert "hidden" not in paths["manifest"].read_text(encoding="utf-8")
    assert "未评分" in paths["summary"].read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_evidence_bundle(rows, tmp_path / "bundle", experiment_id="demo-v1")


def test_score_state_distinguishes_author_and_independent_scores(tmp_path):
    rows = [
        {"run_id": "a", "task": "a", "mode": "react", "answer": "a"},
        {"run_id": "b", "task": "b", "mode": "plan", "answer": "b"},
        {"run_id": "c", "task": "c", "mode": "team", "answer": "c"},
    ]
    scores = [
        {"run_id": "a", "scorer_type": "author", "task_completion": 0.5},
        {"run_id": "b", "scorer_type": "independent", "task_completion": 0.6},
    ]

    paths = write_evidence_bundle(
        rows,
        tmp_path / "scored",
        experiment_id="scored-v1",
        scores=scores,
    )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    run_rows = [json.loads(line) for line in paths["runs"].read_text(encoding="utf-8").splitlines()]
    assert manifest["score_states"] == {"unscored": 1, "author_scored": 1, "independently_scored": 1}
    assert {row["run_id"]: row["score_state"] for row in run_rows} == {
        "a": "author_scored", "b": "independently_scored", "c": "unscored",
    }
    assert paths["scores"].name == "scores.template.jsonl"
