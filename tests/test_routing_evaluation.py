"""阶段 D：路由实验报告保留原始指标与人工评分状态。"""

import json
from pathlib import Path

from evals.calibrate_router import load_calibration_tasks
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
