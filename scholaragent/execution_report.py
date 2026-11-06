"""实验执行指标汇总：从 runs.jsonl 生成按模式、按题型的执行报告。

只汇总执行层面的事实(是否完成、停止原因、耗时、调用、token、缓存、费用),
不涉及回答质量;质量需要盲评后由 ``human_review`` 生成报告。

记账原则与运行时一致:
- 缺失的观测不当作 0,均值只在已返回的观测上计算,并标出覆盖数;
- ``reported_cost`` 只来自服务返回值;按单价估算的费用必须由调用方显式
  提供单价,并在报告里单独标注为“估算”,不与真实费用混写。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

MODE_ORDER = ("react", "plan", "team", "auto")


def _ordered(keys):
    return sorted(keys, key=lambda key: (MODE_ORDER.index(key) if key in MODE_ORDER
                                         else len(MODE_ORDER), str(key)))


def _observed(rows, field):
    return [row["metrics"][field] for row in rows
            if isinstance((row.get("metrics") or {}).get(field), (int, float))
            and not isinstance(row["metrics"][field], bool)]


def _mode(row):
    return row.get("mode") or row.get("requested_mode") or "unknown"


def _completeness(row):
    return (row.get("completion") or {}).get("completeness")


def execution_summary(rows, price_in_per_m=None, price_out_per_m=None):
    """按模式汇总执行指标。单价(美元/百万 token)两者都给出时才计算估算费用。"""
    groups = defaultdict(list)
    for row in rows:
        groups[_mode(row)].append(row)

    summary = {}
    for mode in _ordered(groups):
        group = groups[mode]
        seconds = _observed(group, "seconds")
        prompt = _observed(group, "prompt_tokens")
        completion = _observed(group, "completion_tokens")
        hit = _observed(group, "cache_hit_tokens")
        miss = _observed(group, "cache_miss_tokens")
        cost = _observed(group, "reported_cost")
        stop_reasons = Counter(
            reason for row in group
            for reason in (row.get("completion") or {}).get("stop_reasons") or [])

        estimated = None
        if price_in_per_m is not None and price_out_per_m is not None:
            paired = [(row["metrics"]["prompt_tokens"], row["metrics"]["completion_tokens"])
                      for row in group
                      if isinstance((row.get("metrics") or {}).get("prompt_tokens"), int)
                      and isinstance(row["metrics"].get("completion_tokens"), int)]
            if paired:
                per_run = [(p * price_in_per_m + c * price_out_per_m) / 1e6 for p, c in paired]
                estimated = {"total": sum(per_run), "mean": mean(per_run),
                             "observed": len(per_run)}

        summary[mode] = {
            "runs": len(group),
            "completed": sum(row.get("status") == "completed" for row in group),
            "failed": sum(row.get("status") == "failed" for row in group),
            "cancelled": sum(row.get("status") == "cancelled" for row in group),
            "complete": sum(_completeness(row) == "complete" for row in group),
            "stop_reasons": dict(stop_reasons),
            "seconds": {"mean": mean(seconds) if seconds else None,
                        "median": median(seconds) if seconds else None,
                        "observed": len(seconds)},
            "llm_calls": _mean_block(_observed(group, "llm_calls")),
            "tool_calls": _mean_block(_observed(group, "tool_calls")),
            "prompt_tokens": _mean_block(prompt),
            "completion_tokens": _mean_block(completion),
            "cache_hit_rate": (sum(hit) / (sum(hit) + sum(miss))
                               if hit and miss and len(hit) == len(miss) and sum(hit) + sum(miss)
                               else None),
            "cache_observed": min(len(hit), len(miss)),
            "reported_cost": {"total": sum(cost) if cost else None,
                              "mean": mean(cost) if cost else None,
                              "observed": len(cost)},
            "estimated_cost": estimated,
        }
    return summary


def _mean_block(values):
    return {"mean": mean(values) if values else None, "observed": len(values)}


def completeness_by_category(rows):
    """题型 × 模式的完整完成数:{category: {mode: (complete, runs)}}。"""
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        cell = table[row.get("category") or "unknown"][_mode(row)]
        cell[1] += 1
        cell[0] += _completeness(row) == "complete"
    return {category: {mode: tuple(cell) for mode, cell in modes.items()}
            for category, modes in table.items()}


def _fmt(value, pattern="{:.1f}", missing="未返回"):
    return missing if value is None else pattern.format(value)


def render_markdown(rows, title="实验执行报告", price_in_per_m=None, price_out_per_m=None):
    summary = execution_summary(rows, price_in_per_m, price_out_per_m)
    models = sorted({str((row.get("model_metadata") or {}).get("model")
                         or ((row.get("metrics") or {}).get("llm_usage_by_role") or {})
                         .get("general", {}).get("model") or "未记录")
                     for row in rows})
    lines = [f"# {title}", "",
             f"共 {len(rows)} 次运行；模型：{', '.join(models)}。"
             "本报告只含执行指标，不含回答质量评分。", "",
             "## 按模式", "",
             "|模式|运行|执行完成|完整完成|中位耗时(s)|平均 LLM 调用|平均工具调用|"
             "平均输入 token|平均输出 token|缓存命中率|服务返回费用(合计)|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, item in summary.items():
        cost = item["reported_cost"]
        cost_text = ("未返回" if cost["total"] is None
                     else f"${cost['total']:.4f}（{cost['observed']}/{item['runs']}）")
        lines.append(
            f"| {mode} | {item['runs']} | {item['completed']} | {item['complete']} | "
            f"{_fmt(item['seconds']['median'], '{:.0f}')} | "
            f"{_fmt(item['llm_calls']['mean'])} | {_fmt(item['tool_calls']['mean'])} | "
            f"{_fmt(item['prompt_tokens']['mean'], '{:,.0f}')} | "
            f"{_fmt(item['completion_tokens']['mean'], '{:,.0f}')} | "
            f"{_fmt(item['cache_hit_rate'], '{:.0%}')} | {cost_text} |")

    if price_in_per_m is not None and price_out_per_m is not None:
        lines += ["", f"### 按单价估算的费用（输入 ${price_in_per_m}/M、输出 ${price_out_per_m}/M）",
                  "", "缓存命中部分也按输入原价计算，因此是上限估算，不是服务账单。", "",
                  "|模式|估算合计|估算每次|覆盖|", "|---|---:|---:|---:|"]
        for mode, item in summary.items():
            est = item["estimated_cost"]
            if est is None:
                lines.append(f"| {mode} | 未知 | 未知 | 0/{item['runs']} |")
            else:
                lines.append(f"| {mode} | ${est['total']:.4f} | ${est['mean']:.4f} | "
                             f"{est['observed']}/{item['runs']} |")

    lines += ["", "## 停止原因", ""]
    for mode, item in summary.items():
        reasons = "、".join(f"{reason} × {count}"
                           for reason, count in sorted(item["stop_reasons"].items())) or "无"
        lines.append(f"- {mode}：失败 {item['failed']}，取消 {item['cancelled']}；{reasons}")

    by_category = completeness_by_category(rows)
    modes = list(summary)
    lines += ["", "## 题型 × 模式（完整完成/运行）", "",
              "|题型|" + "|".join(modes) + "|", "|---|" + "---:|" * len(modes)]
    for category in sorted(by_category):
        cells = []
        for mode in modes:
            done, total = by_category[category].get(mode, (0, 0))
            cells.append(f"{done}/{total}" if total else "—")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += ["", "耗时、token、费用只在已返回的观测上统计；缺失不当作 0。"
              "单次小样本结果只用于诊断，不构成模式优劣结论。", ""]
    return "\n".join(lines)
