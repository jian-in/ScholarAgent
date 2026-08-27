"""Deterministic checks for replayable run artifacts.

The auditor deliberately stays side-effect free.  It accepts either a
``RunResult`` or its JSON-compatible mapping so the same checks can be used
by the runtime, evaluation scripts, and post-hoc artifact verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


TERMINAL_EVENTS = frozenset({"completed", "failed", "cancelled"})
VALID_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class AuditReport:
    """可序列化的运行产物审计结果。"""

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = list(self.errors)
        data["warnings"] = list(self.warnings)
        data["checks"] = dict(self.checks)
        return data


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    return None


def audit_run_result(result: Any) -> AuditReport:
    """审计一次运行结果的身份、终态和证据一致性。

    这些检查只依赖已落盘的 JSON 字段，不执行模型、工具或外部网络调用。
    缺少新版本字段的旧产物会收到 warning，但不会因兼容性直接判失败。
    """

    data = _mapping(result)
    if data is None:
        return AuditReport(
            ok=False,
            errors=("result 必须是 RunResult 或 mapping",),
            checks={"result_shape": False},
        )

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    run_id = data.get("run_id")
    checks["run_id_present"] = isinstance(run_id, str) and bool(run_id.strip())
    if not checks["run_id_present"]:
        errors.append("run_id 缺失或为空")

    status = data.get("status")
    checks["status_valid"] = status in VALID_STATUSES
    if not checks["status_valid"]:
        errors.append(f"status 无效: {status!r}")

    raw_events = data.get("events", ())
    checks["events_sequence"] = isinstance(raw_events, (list, tuple))
    if not checks["events_sequence"]:
        errors.append("events 必须是列表或元组")
        raw_events = ()

    events = list(raw_events)
    event_run_ids_ok = True
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            event_run_ids_ok = False
            errors.append(f"event[{index}] 不是 mapping")
            continue
        event_run_id = event.get("run_id")
        if event_run_id != run_id:
            event_run_ids_ok = False
            errors.append(
                f"event[{index}].run_id 与 run_id 不一致: "
                f"{event_run_id!r} != {run_id!r}"
            )
    checks["event_run_ids"] = event_run_ids_ok
    terminal_positions = [
        index for index, event in enumerate(events)
        if isinstance(event, Mapping) and event.get("type") in TERMINAL_EVENTS
    ]
    checks["single_terminal_event"] = len(terminal_positions) == 1
    if len(terminal_positions) != 1:
        errors.append(
            f"终态事件数量应为 1，实际为 {len(terminal_positions)}"
        )
    else:
        terminal_type = events[terminal_positions[0]].get("type")
        checks["terminal_matches_status"] = terminal_type == status
        if terminal_type != status:
            errors.append(
                f"终态事件 {terminal_type!r} 与 status {status!r} 不一致"
            )
        checks["terminal_is_last"] = terminal_positions[0] == len(events) - 1
        if not checks["terminal_is_last"]:
            errors.append("终态事件不是事件流最后一项")
    if not events:
        warnings.append("events 为空，无法判断事件类型顺序")

    evidence = data.get("evidence")
    if evidence is None:
        warnings.append("缺少 evidence 字段（旧产物兼容）")
        checks["evidence_valid"] = True
    elif not isinstance(evidence, Mapping):
        checks["evidence_valid"] = False
        errors.append("evidence 必须是 mapping")
    else:
        validation_errors = evidence.get("validation_errors", ())
        checks["evidence_valid"] = not bool(validation_errors)
        if validation_errors:
            errors.append(
                "evidence 校验失败: "
                + "; ".join(str(item) for item in validation_errors)
            )
        if not evidence.get("schema_version"):
            warnings.append("evidence 缺少 schema_version")

    return AuditReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
    )


def audit_evidence_bundle(bundle_dir: str | Path) -> AuditReport:
    """审计 ``manifest.json`` + ``runs.jsonl`` 组成的只读证据包。"""

    root = Path(bundle_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    manifest_path = root / "manifest.json"
    runs_path = root / "runs.jsonl"
    checks["manifest_present"] = manifest_path.is_file()
    checks["runs_present"] = runs_path.is_file()
    if not checks["manifest_present"]:
        errors.append("证据包缺少 manifest.json")
    if not checks["runs_present"]:
        errors.append("证据包缺少 runs.jsonl")
    if errors:
        return AuditReport(False, tuple(errors), tuple(warnings), checks)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"manifest.json 无法解析: {type(exc).__name__}")
        manifest = {}
    checks["manifest_shape"] = isinstance(manifest, Mapping)
    if not checks["manifest_shape"]:
        errors.append("manifest.json 必须是对象")
        manifest = {}
    elif manifest.get("schema_version") != "experiment-bundle-v1":
        errors.append(
            "manifest.schema_version 无效: "
            f"{manifest.get('schema_version')!r}"
        )
    if not manifest.get("experiment_id"):
        warnings.append("manifest 缺少 experiment_id")

    rows: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(runs_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                errors.append(f"runs.jsonl 第 {line_number} 行不是对象")
                continue
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"runs.jsonl 无法解析: {type(exc).__name__}")

    run_ids = [str(row.get("run_id") or "") for row in rows]
    checks["run_ids_unique"] = all(run_id and run_ids.count(run_id) == 1 for run_id in run_ids)
    if not checks["run_ids_unique"]:
        errors.append("证据包的 run_id 必须存在且唯一")

    row_checks_ok = True
    for index, row in enumerate(rows):
        candidate = dict(row)
        if "status" not in candidate:
            terminal = next(
                (
                    event.get("type")
                    for event in candidate.get("events", ())
                    if isinstance(event, Mapping) and event.get("type") in TERMINAL_EVENTS
                ),
                None,
            )
            if terminal:
                candidate["status"] = terminal
        report = audit_run_result(candidate)
        if not report.ok:
            row_checks_ok = False
            errors.extend(f"run[{index}]: {error}" for error in report.errors)
        warnings.extend(f"run[{index}]: {warning}" for warning in report.warnings)
    checks["run_results"] = row_checks_ok
    return AuditReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
    )


__all__ = ["AuditReport", "audit_evidence_bundle", "audit_run_result"]
