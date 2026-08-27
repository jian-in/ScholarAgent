"""公开案例运行记录：统一执行三种模式并保留可复验证据。"""

from __future__ import annotations

import json
import inspect
import re
from pathlib import Path

from .artifacts import ArtifactCollector
from .experiments import utc_now, write_evidence_bundle
from .metrics import MetricsCollector
from .routing_evaluation import QUALITY_FIELDS
from .workspace import TemporaryWorkspace, default_workspace


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
    for mode in modes:
        started_at = utc_now()
        active_state_dir = (
            Path(state_root) / case["id"] / mode
            if state_root is not None
            else default_workspace().root
        )
        workspace = TemporaryWorkspace(active_state_dir) if state_root is not None else default_workspace()
        metrics = MetricsCollector(mode)
        trace = TraceSummary()
        artifacts = ArtifactCollector(workspace)
        kwargs = {
            "metrics_collector": metrics,
            "on_progress": trace,
            "artifacts": artifacts,
            "workspace": workspace,
        }
        try:
            parameters = inspect.signature(runner_factory).parameters
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in parameters.values()):
                kwargs = {key: value for key, value in kwargs.items()
                          if key in parameters}
        except (TypeError, ValueError):
            pass
        runner = runner_factory(mode, **kwargs)
        metrics.restart()
        error = None
        result = None
        try:
            answer = runner.run(case["task"])
        except Exception as exc:
            answer = ""
            error = f"{type(exc).__name__}: {exc}"
        result = getattr(runner, "last_result", None)
        if result is not None:
            answer = result.answer if not error else answer
            error = error or result.error
        run_id = f"{case['id']}:{mode}:1"
        result_events = []
        routing = None
        evidence = {}
        workflow = None
        source_format = None
        status = "failed" if error else "completed"
        if result is not None:
            result_events = [dict(event) for event in result.events]
            # 证据行的稳定 id 优先于运行时内部随机 id，便于评分表关联。
            for event in result_events:
                event["run_id"] = run_id
            routing = result.routing
            evidence = dict(result.evidence)
            workflow = result.workflow
            source_format = result.source_format
            status = result.status
        rows.append({
            "schema_version": "case-run-v1",
            "run_id": run_id,
            "case_id": case["id"],
            "task": case["task"],
            "rubric": dict(case.get("rubric") or {}),
            "mode": mode,
            "status": status,
            "answer": answer,
            "trace": list(trace.lines),
            "metrics": metrics.finish().to_dict(),
            "artifacts": _portable_artifacts(artifacts, workspace.root),
            "evidence": evidence,
            "workflow": workflow,
            "source_format": source_format,
            "routing": routing,
            "events": result_events,
            "started_at": started_at,
            "ended_at": utc_now(),
            "quality": None,
            "error": error,
        })
    return rows


def write_case_bundle(rows: list[dict], output_dir) -> dict[str, Path]:
    """把原始运行与独立人工评分模板写入一个不可覆盖的新目录。"""
    rows = list(rows)
    first = rows[0] if rows else {}
    task = {
        "id": first.get("case_id") or first.get("task"),
        "task": first.get("task"),
    }
    if first.get("rubric"):
        task["rubric"] = dict(first["rubric"])
    started = next((row.get("started_at") for row in rows if row.get("started_at")), None)
    paths = write_evidence_bundle(
        rows,
        output_dir,
        experiment_id=first.get("case_id") or Path(output_dir).name,
        tasks=[task] if task["task"] is not None else [],
        modes=[row.get("mode") for row in rows if row.get("mode")],
        strategy_version="case-run-v2",
        started_at=started,
    )
    return paths
