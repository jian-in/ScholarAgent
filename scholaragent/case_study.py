"""公开案例运行记录：统一执行三种模式并保留可复验证据。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .artifacts import ArtifactCollector
from .metrics import MetricsCollector
from .routing_evaluation import QUALITY_FIELDS


CASE_MODES = ("react", "plan", "team")


def load_case(path) -> dict:
    """读取一个 UTF-8 JSON 案例定义。"""
    with Path(path).open(encoding="utf-8") as handle:
        case = json.load(handle)
    case_id = case.get("id") if isinstance(case, dict) else None
    if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", case_id):
        raise ValueError("案例 id 只能使用安全的 ASCII 小写标识")
    rubric = case.get("rubric")
    if (not isinstance(rubric, dict)
            or set(rubric) != set(QUALITY_FIELDS)
            or any(not isinstance(rubric[field], str) or not rubric[field].strip()
                   for field in QUALITY_FIELDS)):
        raise ValueError("案例 rubric 必须包含四项非空人工评分说明")
    modes = case.get("modes")
    if (not isinstance(modes, list) or not modes or len(modes) != len(set(modes))
            or any(mode not in CASE_MODES for mode in modes)):
        raise ValueError("案例 modes 必须是 react、plan、team 的非空无重复子集")
    return case


class TraceSummary:
    """把实时进度压缩为可公开的结构化轨迹摘要。"""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        line = " ".join(str(text or "").split())
        tool_call = re.match(r"^(.*调用工具\s+)([A-Za-z0-9_.-]+)\(.*\)$", line)
        if tool_call:
            line = f"{tool_call.group(1)}{tool_call.group(2)}(<参数已省略>)"
        for marker, replacement in (
            ("模型想法:", "模型想法:<已省略内部推理>"),
            ("工具返回:", "工具返回:<已省略返回正文>"),
            ("最终回答:", "最终回答:<完整内容见 answer 字段>"),
        ):
            if marker in line:
                prefix = line.split(marker, 1)[0]
                line = prefix + replacement
                break
        if len(line) > 320:
            line = line[:319] + "…"
        self.lines.append(line)


def _portable_artifacts(artifacts: ArtifactCollector, state_dir) -> dict:
    data = artifacts.to_dict()
    state_path = Path(state_dir).resolve()
    for section in ("papers", "notes", "memories"):
        portable_items = []
        for original in data[section]:
            item = dict(original)
            raw_path = item.get("path")
            if raw_path:
                try:
                    item["path"] = Path(raw_path).resolve().relative_to(state_path).as_posix()
                except (OSError, ValueError):
                    item["path"] = "<outside-state-omitted>"
            portable_items.append(item)
        data[section] = portable_items
    return data


def run_case(case: dict, runner_factory, modes=CASE_MODES, state_root=None) -> list[dict]:
    """运行一个案例并返回完整回答、摘要轨迹、指标与产物。"""
    rows = []
    original_data_dir = config.DATA_DIR
    try:
        for mode in modes:
            if state_root is not None:
                config.DATA_DIR = str(Path(state_root) / case["id"] / mode)
            active_state_dir = config.DATA_DIR
            metrics = MetricsCollector(mode)
            trace = TraceSummary()
            artifacts = ArtifactCollector()
            runner = runner_factory(
                mode,
                metrics_collector=metrics,
                on_progress=trace,
                artifacts=artifacts,
            )
            metrics.restart()
            error = None
            try:
                answer = runner.run(case["task"])
            except Exception as exc:
                answer = ""
                error = f"{type(exc).__name__}: {exc}"
            rows.append({
                "schema_version": "case-run-v1",
                "run_id": f"{case['id']}:{mode}:1",
                "case_id": case["id"],
                "task": case["task"],
                "rubric": dict(case.get("rubric") or {}),
                "mode": mode,
                "answer": answer,
                "trace": list(trace.lines),
                "metrics": metrics.finish().to_dict(),
                "artifacts": _portable_artifacts(artifacts, active_state_dir),
                "quality": None,
                "error": error,
            })
    finally:
        config.DATA_DIR = original_data_dir
    return rows


def write_case_bundle(rows: list[dict], output_dir) -> dict[str, Path]:
    """把原始运行与独立人工评分模板写入一个不可覆盖的新目录。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runs_path = output / "runs.jsonl"
    scores_path = output / "scores.template.jsonl"

    with runs_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with scores_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            score = {"run_id": row["run_id"]}
            score.update({field: None for field in QUALITY_FIELDS})
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")

    return {"runs": runs_path, "scores": scores_path}
