"""Offline blind scoring and claim review; never rewrite original run evidence."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

from .evidence import ClaimEvidence, EvidenceLedger, SourceAnchor
from .routing_evaluation import QUALITY_FIELDS


def _fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _unique_rows(rows):
    rows = [dict(row) for row in rows]
    ids = [row.get("run_id") for row in rows]
    if not rows or any(not isinstance(run_id, str) or not run_id for run_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("运行记录必须非空且 run_id 唯一")
    return rows


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_blind_review(rows, output_dir, seed=0):
    """打乱呈现顺序，隐藏模式、指标、轨迹和含模式名的 run_id。"""
    rows = _unique_rows(rows)
    order = list(rows)
    random.Random(seed).shuffle(order)
    packets, scores, claims, key = [], [], [], {}
    for index, row in enumerate(order, 1):
        review_id = f"review-{index:04d}"
        key[review_id] = row["run_id"]
        evidence = row.get("evidence") or {}
        packets.append({
            "review_id": review_id,
            "task": row.get("task"),
            "acceptance": row.get("acceptance") or [],
            "answer": row.get("answer") or "",
            "completeness": (row.get("completion") or {}).get("completeness", "unknown"),
            "anchors": evidence.get("anchors") or [],
            "claims": evidence.get("claims") or [],
        })
        scores.append({"review_id": review_id, "reviewer": None, "scorer_type": None,
                       "note": "", **{field: None for field in QUALITY_FIELDS}})
        claims.extend({"review_id": review_id, "claim_id": claim["id"],
                       "decision": None, "reviewer": None, "note": ""}
                      for claim in evidence.get("claims") or [])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "review_packets.jsonl", packets)
    _write_jsonl(output / "scores.blind.template.jsonl", scores)
    _write_jsonl(output / "claims.blind.template.jsonl", claims)
    with (output / "key.private.json").open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": "blind-review-v1", "input_sha256": _fingerprint(rows),
                   "review_to_run": key}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output / "README.md").write_text(
        "# 人工盲评材料\n\n"
        "只向评分者提供 review_packets.jsonl 和两份 template；key.private.json 由实验维护者保留。\n"
        "评分为 [0,1]，四项分别衡量任务完成、事实正确、原文支持和输出完整。\n"
        "保留原回答，回答措辞本身可能透露执行方式；这不是双盲实验。\n"
        "scorer_type 填 author 或 independent；independent 应由作者以外的评分者完成。\n"
        "reviewer 和 note 必填；引用存在不等于原文支持。核对论文页码和摘录后，"
        "逐条 claim 填 verified 或 rejected。尚未核对的条目不要提交。\n"
        "缺失评分始终保留为未评分，失败运行也应纳入评分。\n",
        encoding="utf-8",
    )
    return output


def apply_human_review(rows, key, scores=(), claim_reviews=()):
    """验证评分与来源核验，返回派生记录；输入 rows 不变。"""
    rows = _unique_rows(rows)
    if key.get("schema_version") != "blind-review-v1" or key.get("input_sha256") != _fingerprint(rows):
        raise ValueError("评分映射与原始运行证据不匹配")
    mapping = key.get("review_to_run") or {}
    run_ids = {row["run_id"] for row in rows}
    if set(mapping.values()) != run_ids or len(mapping) != len(run_ids):
        raise ValueError("评分映射必须一一覆盖原始 run_id")
    score_map, claims_by_run = {}, defaultdict(list)
    for score in scores:
        run_id = mapping[score["review_id"]]
        if run_id in score_map:
            raise ValueError("同一运行存在重复评分")
        if (score.get("scorer_type") not in {"author", "independent"}
                or not str(score.get("reviewer") or "").strip()
                or not str(score.get("note") or "").strip()):
            raise ValueError("评分必须记录 scorer_type、reviewer 和核对说明")
        values = {}
        for field in QUALITY_FIELDS:
            value = score.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{field} 必须是 [0,1] 的有限数值")
            values[field] = float(value)
        score_map[run_id] = {**dict(score), **values, "run_id": run_id}
    seen = set()
    for review in claim_reviews:
        run_id = mapping[review["review_id"]]
        pair = (run_id, review["claim_id"])
        if pair in seen:
            raise ValueError("同一结论存在重复核验")
        seen.add(pair)
        claims_by_run[run_id].append(review)
    derived = []
    for row in rows:
        # JSON 深拷贝：人工核验只改派生结果，原始模型产物保持不变。
        row = json.loads(json.dumps(row, ensure_ascii=False))
        score = score_map.get(row["run_id"])
        row["score_state"] = (
            "independently_scored" if score and score["scorer_type"] == "independent"
            else "author_scored" if score else "unscored"
        )
        row["quality_score"] = score
        if claims_by_run[row["run_id"]]:
            evidence = row.get("evidence") or {}
            ledger = EvidenceLedger(
                anchors=[SourceAnchor.from_dict(anchor) for anchor in evidence.get("anchors", [])],
                claims=[ClaimEvidence(**claim) for claim in evidence.get("claims", [])],
            )
            for review in claims_by_run[row["run_id"]]:
                ledger.review_claim(review["claim_id"], review.get("decision"),
                                    str(review.get("reviewer") or ""), str(review.get("note") or ""))
            row["evidence"] = ledger.to_dict()
        derived.append(row)
    return derived


def review_summary(rows):
    """只报告已有观测和评分，不用零分替代缺失，不外推实验结论。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("strategy") or row.get("mode") or "unknown"].append(row)
    summary = {}
    for mode, group in groups.items():
        scored = [row["quality_score"] for row in group if row.get("quality_score")]
        metrics = [row.get("metrics") or {} for row in group]
        def observed(field):
            values = [item[field] for item in metrics if item.get(field) is not None]
            return {"mean": mean(values) if values else None, "observed": len(values), "runs": len(group)}
        qualities = {field: mean(score[field] for score in scored) if scored else None
                     for field in QUALITY_FIELDS}
        summary[mode] = {
            "runs": len(group), "scored": len(scored),
            "independently_scored": sum(row.get("score_state") == "independently_scored" for row in group),
            "failed": sum(row.get("status") == "failed" for row in group),
            "cancelled": sum(row.get("status") == "cancelled" for row in group),
            "partial": sum((row.get("completion") or {}).get("completeness") == "partial" for row in group),
            "quality": qualities,
            "seconds": observed("seconds"), "llm_calls": observed("llm_calls"),
            "request_attempts": observed("request_attempts"),
            "prompt_tokens": observed("prompt_tokens"), "completion_tokens": observed("completion_tokens"),
        }
    return summary


def write_review_report(rows, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = review_summary(rows)
    material_hashes = defaultdict(set)
    missing_snapshots = []
    missing_sources = []
    version_mismatches = []
    unexpected_sources = []
    for row in rows:
        snapshots = row.get("source_snapshots") or []
        expected = set(row.get("pinned_sources") or [])
        actual = {snapshot["arxiv_id"] for snapshot in snapshots}
        if expected and not snapshots:
            missing_snapshots.append(row["run_id"])
        for source in sorted(expected - actual):
            missing_sources.append({"run_id": row["run_id"], "arxiv_id": source})
            other_versions = sorted(item for item in actual if item.split("v", 1)[0] == source.split("v", 1)[0])
            if other_versions:
                version_mismatches.append({"run_id": row["run_id"], "expected": source,
                                           "observed": other_versions})
        for source in sorted(actual - expected) if expected else []:
            unexpected_sources.append({"run_id": row["run_id"], "arxiv_id": source})
        for snapshot in snapshots:
            material_hashes[(row.get("task_id"), snapshot["arxiv_id"])].add(snapshot["sha256"])
    conflicts = [{"task_id": task, "arxiv_id": source, "sha256": sorted(hashes)}
                 for (task, source), hashes in material_hashes.items() if len(hashes) > 1]
    hash_status = "not_observed" if not material_hashes else "conflicting" if conflicts else "observed_consistent"
    sources = {"status": "conflicting" if conflicts else "incomplete" if missing_sources or unexpected_sources else hash_status,
               "hash_status": hash_status, "conflicts": conflicts,
               "runs_without_pdf_snapshot": missing_snapshots, "missing_sources": missing_sources,
               "version_mismatches": version_mismatches, "unexpected_sources": unexpected_sources}
    _write_jsonl(output / "reviewed_runs.jsonl", rows)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output / "sources.audit.json").open("x", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = ["# 真实任务人工核对报告", "",
             "本报告为派生结果，原始 runs.jsonl 未改动。缺失评分不是零分；"
             "未完成任务和失败运行保留在分母中。", "",
             "|模式|运行|已评分/独立评分|失败/取消/部分完成|完成度|事实正确|引用支持|输出完整|",
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for mode, item in summary.items():
        quality = " | ".join("未评分" if item["quality"][field] is None
                             else f"{item['quality'][field]:.3f}" for field in QUALITY_FIELDS)
        lines.append(f"| {mode} | {item['runs']} | {item['scored']}/{item['independently_scored']} | "
                     f"{item['failed']}/{item['cancelled']}/{item['partial']} | {quality} |")
    lines += ["", "## 资源消耗（仅平均已返回观测，括号为覆盖数/运行数）", ""]
    for mode, item in summary.items():
        measurements = []
        for field in ("seconds", "llm_calls", "request_attempts", "prompt_tokens", "completion_tokens"):
            observation = item[field]
            value = "未知" if observation["mean"] is None else f"{observation['mean']:.2f}"
            measurements.append(f"{field}={value}（{observation['observed']}/{observation['runs']}）")
        lines.append(f"- {mode}: " + "；".join(measurements))
    lines += ["", "## 输入材料一致性", "",
              f"实际 PDF 哈希检查：{sources['status']}；冲突 {len(conflicts)} 项；"
              f"尚无 PDF 快照的运行 {len(missing_snapshots)} 项；"
              f"缺少指定文献 {len(missing_sources)} 项，版本偏差 {len(version_mismatches)} 项，"
              f"额外材料 {len(unexpected_sources)} 项。详情见 sources.audit.json。",
              "仅有检索元数据的任务可能没有 PDF；缺少快照不等于材料已经一致。", "",
              "单次、小样本结果仅用于诊断；多次重复、独立评分和一致材料是正式比较的前提。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output
