"""零模型调用地评估规则路由与固定任务复杂度标签的一致性。"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.routing import RuleRouter


EXPECTED_MODE = {"simple": "react", "medium": "plan", "complex": "team"}


def load_tasks(path):
    with open(path, encoding="utf-8") as handle:
        return [
            json.loads(line) for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def evaluate_tasks(tasks, router=None):
    router = router or RuleRouter()
    rows = []
    for task in tasks:
        expected = EXPECTED_MODE[task["category"]]
        decision = router.route(task["task"])
        rows.append({
            "task_id": task["id"],
            "split": task["split"],
            "category": task["category"],
            "expected_mode": expected,
            "selected_mode": decision.mode,
            "matches_category_label": decision.mode == expected,
            "reason": decision.reason,
            "features": decision.features,
        })
    return rows


def summarize_observations(path):
    if not path:
        return None
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[row["mode"]].append(row)
    summary = {}
    for mode, rows in sorted(groups.items()):
        summary[mode] = {
            "runs": len(rows),
            "llm_calls": sum(row["metrics"]["llm_calls"] for row in rows),
            "prompt_tokens": sum(row["metrics"]["prompt_tokens"] or 0 for row in rows),
            "completion_tokens": sum(row["metrics"]["completion_tokens"] or 0 for row in rows),
            "seconds": sum(row["metrics"]["seconds"] for row in rows),
            "errors": sum(bool(row.get("error")) for row in rows),
        }
    return summary


def write_report(rows, output_dir, observation_summary=None):
    os.makedirs(output_dir, exist_ok=False)
    raw_path = os.path.join(output_dir, "routing_decisions.jsonl")
    with open(raw_path, "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    matches = sum(row["matches_category_label"] for row in rows)
    confusion = Counter((row["category"], row["selected_mode"]) for row in rows)
    lines = [
        "# 规则路由离线评估",
        "",
        "> 本报告不调用模型。这里的准确率仅表示路由结果与任务集预设复杂度标签一致，",
        "> 不代表回答质量、事实正确率或成本感知学习策略的正式留出集效果。",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 任务数：{len(rows)}",
        f"- 标签一致数：{matches}",
        f"- 标签一致率：{matches / len(rows):.3%}",
        "",
        "## 路由混淆矩阵",
        "",
        "| 任务类别 | 选择模式 | 数量 |",
        "|---|---|---:|",
    ]
    for (category, mode), count in sorted(confusion.items()):
        lines.append(f"| {category} | {mode} | {count} |")

    if observation_summary:
        lines.extend([
            "",
            "## 已完成真实校准观测的成本画像",
            "",
            "> 观测尚未覆盖完整校准集，只用于说明各执行模式的实际成本差异。",
            "",
            "| 模式 | 运行数 | LLM 调用 | Prompt token | Completion token | 总耗时（秒） | 错误 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for mode, values in observation_summary.items():
            lines.append(
                f"| {mode} | {values['runs']} | {values['llm_calls']} | "
                f"{values['prompt_tokens']} | {values['completion_tokens']} | "
                f"{values['seconds']:.1f} | {values['errors']} |"
            )

    report_path = os.path.join(output_dir, "routing_report.md")
    with open(report_path, "x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=os.path.join("evals", "router_tasks.jsonl"))
    parser.add_argument("--observations", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = evaluate_tasks(load_tasks(args.tasks))
    report = write_report(rows, args.output_dir, summarize_observations(args.observations))
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
