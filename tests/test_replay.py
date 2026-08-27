"""公开案例 JSONL 回放适配器的行为。"""

import json

import pytest

from scholaragent.replay import SavedCaseStore


def _write_bundle(root):
    bundle = root / "case-one"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "experiment_id": "case-one",
        "schema_version": "experiment-bundle-v1",
        "score_states": {"unscored": 1, "author_scored": 0, "independently_scored": 0},
    }), encoding="utf-8")
    (bundle / "runs.jsonl").write_text(json.dumps({
        "run_id": "case-one:react:1",
        "case_id": "case-one",
        "task": "离线案例",
        "mode": "react",
        "answer": "真实回答",
        "trace": ["[第 1 步] 最终回答:<完整内容见 answer 字段>"],
        "metrics": {"llm_calls": 1, "tool_calls": 0, "seconds": 0.1},
        "artifacts": {"counts": {"papers": 0, "notes": 0, "memories": 0, "read": 0}},
        "score_state": "unscored",
        "error": None,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return bundle


def test_saved_case_store_projects_jsonl_into_realtime_result_shape(tmp_path):
    _write_bundle(tmp_path)
    store = SavedCaseStore(tmp_path)

    cases = store.list_cases()
    replay = store.get_case("case-one")

    assert cases == [{
        "id": "case-one",
        "title": "case-one",
        "runs": 1,
        "score_states": {"unscored": 1, "author_scored": 0, "independently_scored": 0},
    }]
    assert replay["source"] == "saved_case"
    assert replay["selected"]["status"] == "completed"
    assert replay["selected"]["answer"] == "真实回答"
    assert replay["selected"]["score_state"] == "unscored"
    assert replay["selected"]["events"][-1]["type"] == "completed"


def test_saved_case_store_rejects_path_traversal_and_missing_cases(tmp_path):
    store = SavedCaseStore(tmp_path)

    with pytest.raises(KeyError):
        store.get_case("../secret")
    with pytest.raises(KeyError):
        store.get_case("missing")


def test_saved_case_store_reads_author_score_state_from_separate_file(tmp_path):
    bundle = _write_bundle(tmp_path)
    (bundle / "scores.jsonl").write_text(json.dumps({
        "run_id": "case-one:react:1",
        "task_completion": 0.8,
        "factual_correctness": 0.8,
        "citation_validity": 0.8,
        "output_completeness": 0.8,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    replay = SavedCaseStore(tmp_path).get_case("case-one")
    assert replay["selected"]["score_state"] == "author_scored"
