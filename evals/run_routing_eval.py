"""在留出集上比较固定、规则与成本感知路由，并生成原始数据和报告。"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals.calibrate_router import ensure_model
from evals.run_eval import build_runner
from scholaragent import config
from scholaragent.metrics import MetricsCollector
from scholaragent.routing import AdaptiveRunner, CostAwareRouter, GlobalUtilityRouter, ROUTING_MODES, RuleRouter
from scholaragent.routing_evaluation import attach_quality, write_report


def load_holdout_tasks(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip() and not line.lstrip().startswith("#")]
    tasks = [row for row in rows if row.get("split") == "holdout"]
    if not tasks:
        raise ValueError("任务集中没有 holdout 划分")
    return tasks


def load_scores(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return {row["run_id"]: row for row in (json.loads(line) for line in handle if line.strip())}


def build_strategy(strategy, collector, policy_path=None):
    if strategy in ROUTING_MODES:
        return build_runner(strategy, metrics_collector=collector), None
    runners = {mode: build_runner(mode, metrics_collector=collector) for mode in ROUTING_MODES}
    if strategy == "rule":
        router = RuleRouter()
    elif strategy == "cost_aware" or strategy == "cost_quality_only" or strategy.startswith("sensitivity:"):
        router = CostAwareRouter(policy_path)
    elif strategy == "global_only":
        router = GlobalUtilityRouter(policy_path)
    else:
        raise ValueError(f"未知策略: {strategy}")
    return AdaptiveRunner(router, runners), router


def run_strategy(tasks, strategy, repeat, output_dir, policy_path=None):
    rows = []
    for task in tasks:
        collector = MetricsCollector(strategy)
        config.DATA_DIR = os.path.join(output_dir, "data", f"{strategy}_{task['id']}_{repeat}")
        runner, _ = build_strategy(strategy, collector, policy_path)
        collector.restart()
        error = None
        try:
            answer = runner.run(task["task"])
        except Exception as exc:
            answer = ""
            error = f"{type(exc).__name__}: {exc}"
        decision = getattr(runner, "last_decision", None)
        metrics = collector.finish().to_dict()
        total_tokens = None if metrics["prompt_tokens"] is None or metrics["completion_tokens"] is None else metrics["prompt_tokens"] + metrics["completion_tokens"]
        rows.append({
            "run_id": f"{task['id']}:{strategy}:{repeat}", "task_id": task["id"],
            "split": "holdout", "category": task["category"], "task": task["task"],
            "strategy": strategy, "repeat": repeat,
            "selected_mode": decision.mode if decision else strategy,
            "routing_decision": decision.to_dict() if decision else None,
            "metrics": metrics, "total_tokens": total_tokens,
            "answer": answer, "error": error,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=os.path.join("evals", "router_tasks.jsonl"))
    parser.add_argument("--policy", required=True, help="成本感知策略 JSON")
    parser.add_argument("--quality-only-policy", required=True, help="lambda 全为 0 训练出的消融策略")
    parser.add_argument("--global-policy", required=True, help="同一校准数据训练出的策略，用于无特征消融")
    parser.add_argument("--sensitivity-policy", action="append", default=[], metavar="名称=路径")
    parser.add_argument("--scores", default=None, help="独立人工评分 JSONL；缺失时报告明确标记未评分")
    parser.add_argument("--output-dir", default=os.path.join("evals", "results"))
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("正式结果至少需要 --repetitions 2")
    ensure_model()
    tasks = load_holdout_tasks(args.tasks)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"routing_{stamp}")
    os.makedirs(output_dir, exist_ok=False)
    strategies = ["react", "plan", "team", "rule", "cost_aware"]
    all_rows = []
    for repeat in range(1, args.repetitions + 1):
        for strategy in strategies:
            all_rows.extend(run_strategy(tasks, strategy, repeat, output_dir, args.policy))
        all_rows.extend(run_strategy(tasks, "cost_quality_only", repeat, output_dir, args.quality_only_policy))
        all_rows.extend(run_strategy(tasks, "global_only", repeat, output_dir, args.global_policy))
        for item in args.sensitivity_policy:
            if "=" not in item:
                parser.error("--sensitivity-policy 格式为 名称=路径")
            label, path = item.split("=", 1)
            all_rows.extend(run_strategy(tasks, f"sensitivity:{label}", repeat, output_dir, path))
    rows = attach_quality(all_rows, load_scores(args.scores))
    report_path = os.path.join(output_dir, "routing_report.md")
    write_report(rows, report_path)
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
