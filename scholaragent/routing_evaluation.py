"""路由正式实验的汇总与报告，不生成或修改原始运行数据。"""

import json
from collections import Counter, defaultdict
from statistics import mean, pstdev


QUALITY_FIELDS = ("task_completion", "factual_correctness", "citation_validity", "output_completeness")


def attach_quality(rows, scores):
    """把独立人工评分附加到内存中的报告行，不回写原始 JSONL。"""
    enriched = []
    for row in rows:
        row = dict(row)
        score = scores.get(row["run_id"])
        if score is None:
            row["quality"] = None
        else:
            values = {field: float(score[field]) for field in QUALITY_FIELDS}
            if any(value < 0.0 or value > 1.0 for value in values.values()):
                raise ValueError(f"{row['run_id']} 的人工质量评分超出 [0, 1]")
            row["quality"] = (0.40 * values["task_completion"] + 0.30 * values["factual_correctness"]
                              + 0.20 * values["citation_validity"] + 0.10 * values["output_completeness"])
        enriched.append(row)
    return enriched


def _average(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return None if not values else mean(values)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    result = {}
    for strategy, group in grouped.items():
        result[strategy] = {
            "runs": len(group),
            "quality": _average(group, "quality"),
            "seconds": mean(row["metrics"]["seconds"] for row in group),
            "llm_calls": mean(row["metrics"]["llm_calls"] for row in group),
            "tool_calls": mean(row["metrics"]["tool_calls"] for row in group),
            "tokens": _average(group, "total_tokens"),
            "quality_std": (pstdev([row["quality"] for row in group if row["quality"] is not None])
                            if len([row for row in group if row["quality"] is not None]) > 1 else None),
        }
    return result


def write_report(rows, path):
    summary = summarize(rows)
    confusion = Counter(
        (row["strategy"], row["category"], row.get("selected_mode"))
        for row in rows if row.get("selected_mode")
    )
    lines = ["# 成本感知自适应路由实验报告", "", "## 汇总", "",
             "|策略|运行数|质量|耗时(s)|LLM 调用|工具调用|Token|质量波动|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy, values in sorted(summary.items()):
        display = lambda value: "未评分" if value is None else f"{value:.3f}"
        lines.append(
            f"| {strategy} | {values['runs']} | {display(values['quality'])} | "
            f"{values['seconds']:.3f} | {values['llm_calls']:.3f} | {values['tool_calls']:.3f} | "
            f"{display(values['tokens'])} | {display(values['quality_std'])} |"
        )
    lines += ["", "## 路由混淆矩阵", "", "|策略|任务类别|选择模式|次数|", "|---|---|---|---:|"]
    for (strategy, category, mode), count in sorted(confusion.items()):
        lines.append(f"| {strategy} | {category} | {mode} | {count} |")
    lines += ["", "## 逐任务路由理由", "", "|任务|策略|选择|理由|", "|---|---|---|---|"]
    for row in rows:
        if row.get("routing_decision"):
            reason = row["routing_decision"]["reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['task_id']} | {row['strategy']} | {row['selected_mode']} | {reason} |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(path.replace(".md", ".jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary
