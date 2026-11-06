"""实验执行指标汇总的离线测试:缺失不当 0、费用不混写估算。"""

import pytest

from evals.summarize_runs import main as summarize_main
from scholaragent.execution_report import (
    completeness_by_category,
    execution_summary,
    render_markdown,
)


def run(mode, category="locate", status="completed", completeness="complete",
        stop=(), **metrics):
    base = {"seconds": 10.0, "llm_calls": 2, "tool_calls": 3,
            "prompt_tokens": 1000, "completion_tokens": 100,
            "cache_hit_tokens": 600, "cache_miss_tokens": 400}
    base.update(metrics)
    return {"mode": mode, "category": category, "status": status,
            "completion": {"completeness": completeness, "stop_reasons": list(stop)},
            "metrics": base, "model_metadata": {"model": "m"}}


def test_summary_counts_status_completeness_and_stop_reasons():
    rows = [run("react"), run("react", completeness="partial", stop=["budget_exhausted"]),
            run("plan", status="failed", completeness="partial")]
    summary = execution_summary(rows)
    assert list(summary) == ["react", "plan"]
    assert summary["react"]["completed"] == 2
    assert summary["react"]["complete"] == 1
    assert summary["react"]["stop_reasons"] == {"budget_exhausted": 1}
    assert summary["plan"]["failed"] == 1


def test_missing_observations_are_not_counted_as_zero():
    rows = [run("team", prompt_tokens=1000), run("team", prompt_tokens=None, completion_tokens=None)]
    item = execution_summary(rows)["team"]
    assert item["prompt_tokens"] == {"mean": 1000, "observed": 1}
    assert item["completion_tokens"]["observed"] == 1


def test_cache_hit_rate_is_aggregate_and_unknown_when_missing():
    rows = [run("react", cache_hit_tokens=600, cache_miss_tokens=400),
            run("react", cache_hit_tokens=0, cache_miss_tokens=1000)]
    assert execution_summary(rows)["react"]["cache_hit_rate"] == pytest.approx(0.3)

    missing = [run("react", cache_hit_tokens=None, cache_miss_tokens=None)]
    assert execution_summary(missing)["react"]["cache_hit_rate"] is None


def test_reported_cost_only_from_service_and_estimate_only_with_prices():
    rows = [run("plan", reported_cost=0.01), run("plan")]
    item = execution_summary(rows)["plan"]
    assert item["reported_cost"]["total"] == pytest.approx(0.01)
    assert item["reported_cost"]["observed"] == 1
    assert item["estimated_cost"] is None

    priced = execution_summary(rows, price_in_per_m=0.3, price_out_per_m=1.2)["plan"]
    # 每次 1000 输入 + 100 输出:0.0003 + 0.00012
    assert priced["estimated_cost"]["mean"] == pytest.approx(0.00042)

    text = render_markdown(rows, price_in_per_m=0.3, price_out_per_m=1.2)
    assert "上限估算，不是服务账单" in text
    assert "$0.0100（1/2）" in text


def test_category_matrix_and_markdown_mark_missing_values():
    rows = [run("react", category="read"), run("team", category="read", completeness="partial"),
            run("team", category="gap", prompt_tokens=None, completion_tokens=None,
                cache_hit_tokens=None, cache_miss_tokens=None)]
    assert completeness_by_category(rows)["read"] == {"react": (1, 1), "team": (0, 1)}
    text = render_markdown(rows)
    assert "| gap | — | 1/1 |" in text
    assert "未返回" in text
    assert "不含回答质量评分" in text


def test_cli_refuses_to_overwrite_and_requires_both_prices(tmp_path):
    import json

    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(run("react"), ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "report.md"
    summarize_main(["--runs", str(runs), "--output", str(output)])
    assert "react" in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        summarize_main(["--runs", str(runs), "--output", str(output)])
    with pytest.raises(SystemExit):
        summarize_main(["--runs", str(runs), "--price-in", "0.3"])
