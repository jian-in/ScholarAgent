"""阶段 D：路由实验报告保留原始指标与人工评分状态。"""

import json
from pathlib import Path

from evals.calibrate_router import load_calibration_tasks, load_completed_run_ids
from evals.evaluate_rule_router import evaluate_tasks, summarize_observations
from evals.run_routing_eval import load_holdout_tasks
from scholaragent.routing_evaluation import attach_quality, summarize, write_report


def _rows():
    return [{
        "run_id": "a:rule:1", "task_id": "a", "strategy": "rule", "category": "simple",
        "selected_mode": "react", "routing_decision": {"reason": "单目标"},
        "metrics": {"seconds": 1.5, "llm_calls": 2, "tool_calls": 1},
        "total_tokens": None,
    }]


def test_report_marks_missing_human_quality_instead_of_inventing_it(tmp_path):
    rows = attach_quality(_rows(), {})
    assert rows[0]["quality"] is None
    summary = summarize(rows)
    assert summary["rule"]["quality"] is None

    path = tmp_path / "report.md"
    write_report(rows, str(path))
    assert "未评分" in path.read_text(encoding="utf-8")
    raw = path.with_suffix(".jsonl").read_text(encoding="utf-8")
    assert json.loads(raw)["total_tokens"] is None


def test_report_uses_quality_formula_from_independent_scores():
    scores = {"a:rule:1": {
        "task_completion": 1.0, "factual_correctness": 0.5,
        "citation_validity": 0.5, "output_completeness": 1.0,
    }}
    rows = attach_quality(_rows(), scores)
    assert rows[0]["quality"] == 0.75


def test_router_task_set_has_fixed_balanced_and_isolated_splits():
    path = Path(__file__).parents[1] / "evals" / "router_tasks.jsonl"
    calibration = load_calibration_tasks(str(path))
    holdout = load_holdout_tasks(str(path))
    assert len(calibration) == 18
    assert len(holdout) == 18
    assert {task["id"] for task in calibration}.isdisjoint(task["id"] for task in holdout)
    assert {task["category"] for task in calibration} == {"simple", "medium", "complex"}
    assert {task["category"] for task in holdout} == {"simple", "medium", "complex"}


def test_calibration_resume_loads_unique_completed_run_ids(tmp_path):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        '{"run_id":"simple-01:react:1"}\n'
        '{"run_id":"simple-01:plan:1"}\n',
        encoding="utf-8",
    )
    assert load_completed_run_ids(str(path)) == {
        "simple-01:react:1", "simple-01:plan:1",
    }


def test_calibration_resume_rejects_duplicate_run_ids(tmp_path):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        '{"run_id":"simple-01:react:1"}\n'
        '{"run_id":"simple-01:react:1"}\n',
        encoding="utf-8",
    )
    try:
        load_completed_run_ids(str(path))
    except ValueError as exc:
        assert "重复 run_id" in str(exc)
    else:
        raise AssertionError("重复 run_id 应被拒绝")


def test_offline_rule_evaluation_does_not_claim_answer_quality():
    rows = evaluate_tasks([{
        "id": "simple-1", "split": "holdout", "category": "simple",
        "task": "查询论文年份",
    }])
    assert rows[0]["selected_mode"] == "react"
    assert rows[0]["matches_category_label"] is True
    assert "quality" not in rows[0]


def test_observation_summary_preserves_real_costs(tmp_path):
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps({
        "mode": "team", "error": None,
        "metrics": {
            "llm_calls": 3, "prompt_tokens": 100,
            "completion_tokens": 20, "seconds": 4.5,
        },
    }) + "\n", encoding="utf-8")
    summary = summarize_observations(str(path))
    assert summary["team"] == {
        "runs": 1, "llm_calls": 3, "prompt_tokens": 100,
        "completion_tokens": 20, "seconds": 4.5, "errors": 0,
    }
