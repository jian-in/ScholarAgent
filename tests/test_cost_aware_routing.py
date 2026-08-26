"""阶段 C：离线训练、策略加载与安全回退的离线测试。"""

import json

import pytest

from scholaragent.router_training import fit_policy
from scholaragent.routing import CostAwareRouter, FEATURE_NAMES, FEATURE_VERSION, TaskFeatureExtractor, empty_policy_document


def _observations():
    extractor = TaskFeatureExtractor()
    rows = []
    for index, mode in enumerate(("react", "plan", "team")):
        for repeat in range(2):
            task = f"检索论文并总结 {mode} {repeat}"
            rows.append({
                "run_id": f"{mode}-{repeat}",
                "split": "calibration",
                "mode": mode,
                "feature_version": FEATURE_VERSION,
                "features": extractor.extract(task),
                "quality": 0.5 + 0.1 * index,
                "metrics": {
                    "seconds": 1.0, "llm_calls": 2, "prompt_tokens": None,
                    "completion_tokens": None, "tool_calls": 1,
                },
            })
    return rows


def test_training_is_deterministic_with_fixed_inputs_and_seed():
    first = fit_policy(_observations(), seed=7, trained_at="2026-01-01T00:00:00Z")
    second = fit_policy(_observations(), seed=7, trained_at="2026-01-01T00:00:00Z")
    assert first == second
    assert first["normalization"]["seconds"]["std"] == 0.0


def test_token_cost_rejects_missing_usage_instead_of_inventing_values():
    with pytest.raises(ValueError, match="token 缺失"):
        fit_policy(_observations(), weights={"lambda_token": 0.1})


def test_training_rejects_holdout_observation():
    rows = _observations()
    rows[0]["split"] = "holdout"
    with pytest.raises(ValueError, match="calibration"):
        fit_policy(rows)


def test_cost_aware_router_loads_policy_and_breaks_ties_deterministically(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(fit_policy(_observations(), trained_at="fixed")), encoding="utf-8")
    router = CostAwareRouter(str(policy_path), alpha=0)
    decision = router.route("查询论文年份")
    assert decision.mode == "team"  # 训练样本中 team 的质量更高
    assert decision.policy_version.startswith("cost-aware-router")


@pytest.mark.parametrize("contents", ["{", "{}"])
def test_missing_or_invalid_policy_safely_falls_back_to_rules(tmp_path, contents):
    path = tmp_path / "policy.json"
    path.write_text(contents, encoding="utf-8")
    decision = CostAwareRouter(str(path)).route("查询 ReAct 论文年份")
    assert decision.mode == "react"
    assert "已回退规则路由" in decision.reason


def test_missing_policy_safely_falls_back_to_rules(tmp_path):
    decision = CostAwareRouter(str(tmp_path / "missing.json")).route("查询 ReAct 论文年份")
    assert decision.mode == "react"
    assert "已回退规则路由" in decision.reason


def test_cost_aware_router_breaks_equal_utility_ties_by_mode_order(tmp_path):
    policy = empty_policy_document()
    policy["feature_names"] = list(FEATURE_NAMES)
    policy["models"] = {
        mode: {"coefficients": [0.0] * len(FEATURE_NAMES), "covariance_inverse": None}
        for mode in ("react", "plan", "team")
    }
    path = tmp_path / "tie.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert CostAwareRouter(str(path)).route("任何任务").mode == "react"
