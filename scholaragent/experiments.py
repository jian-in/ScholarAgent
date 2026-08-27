"""实验清单与不可变证据包。

这个模块把“运行了什么、用的哪个版本、产生了什么”收进一个小而稳定的
接口。原始运行与人工评分始终分文件保存；写入目标已经存在时直接失败，
避免重跑悄悄覆盖证据。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .routing_evaluation import QUALITY_FIELDS


EXPERIMENT_SCHEMA_VERSION = "experiment-bundle-v1"
SCORE_STATES = ("unscored", "author_scored", "independently_scored")
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|"
    r"password|secret|private[_-]?key)",
    re.IGNORECASE,
)


def utc_now() -> str:
    """返回不含本机时区歧义的 UTC 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def task_set_hash(tasks: Iterable[Mapping[str, Any]]) -> str:
    """按输入顺序对任务清单做稳定 SHA-256 哈希。"""
    normalized = [dict(task) for task in tasks]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY.search(key):
        return None
    if isinstance(value, Mapping):
        result = {}
        for child_key, child_value in value.items():
            if _SECRET_KEY.search(str(child_key)):
                continue
            clean = _sanitize(child_value, str(child_key))
            if clean is not None:
                result[str(child_key)] = clean
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sanitize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """只保留非秘密、可 JSON 序列化的配置快照。"""
    clean = _sanitize(config or {})
    return clean if isinstance(clean, dict) else {}


def git_commit(repo_root: str | Path | None = None) -> str | None:
    """读取当前 Git 提交；非 Git 环境返回 ``None`` 而不伪造版本。"""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _unique_modes(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    modes = []
    for row in rows:
        mode = row.get("mode") or row.get("selected_mode") or row.get("strategy")
        if mode and mode not in modes:
            modes.append(str(mode))
    return modes


def _score_state(score: Mapping[str, Any] | None) -> str:
    if not score:
        return "unscored"
    scorer = str(score.get("scorer_type") or score.get("scorer") or "author").lower()
    if scorer in {"independent", "third_party", "third-party", "external", "blind"}:
        return "independently_scored"
    return "author_scored"


@dataclass(frozen=True)
class ExperimentManifest:
    """证据包的不可变来源描述。"""

    schema_version: str
    experiment_id: str
    task_set_hash: str
    git_commit: str | None
    model: str | None
    config: Mapping[str, Any]
    modes: tuple[str, ...]
    strategy_version: str | None
    started_at: str
    ended_at: str | None
    score_states: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modes"] = list(self.modes)
        data["config"] = dict(self.config)
        data["score_states"] = dict(self.score_states)
        return data


def build_manifest(
    experiment_id: str,
    *,
    tasks: Iterable[Mapping[str, Any]],
    model: str | None = None,
    config: Mapping[str, Any] | None = None,
    modes: Iterable[str] | None = None,
    strategy_version: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    score_states: Mapping[str, int] | None = None,
    repo_root: str | Path | None = None,
) -> ExperimentManifest:
    """构造一份只含可复验元数据的实验清单。"""
    task_rows = [dict(task) for task in tasks]
    selected_modes = list(dict.fromkeys(str(mode) for mode in (modes or ())))
    return ExperimentManifest(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=str(experiment_id),
        task_set_hash=task_set_hash(task_rows),
        git_commit=git_commit(repo_root),
        model=str(model) if model is not None else None,
        config=sanitize_config(config),
        modes=tuple(selected_modes),
        strategy_version=strategy_version if strategy_version is not None else None,
        started_at=started_at or utc_now(),
        ended_at=ended_at,
        score_states={state: int((score_states or {}).get(state, 0)) for state in SCORE_STATES},
    )


def _tasks_from_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for row in rows:
        task_id = row.get("task_id") or row.get("case_id") or row.get("task")
        marker = str(task_id)
        if marker in seen:
            continue
        seen.add(marker)
        result.append({
            "id": task_id,
            "task": row.get("task"),
            **({"split": row["split"]} if row.get("split") is not None else {}),
        })
    return result


def _score_map(scores: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None):
    if scores is None:
        return {}
    values = scores.values() if isinstance(scores, Mapping) else scores
    result = {}
    for score in values:
        score = dict(score)
        run_id = score.get("run_id")
        if not run_id:
            raise ValueError("评分记录缺少 run_id")
        if run_id in result:
            raise ValueError(f"评分记录包含重复 run_id: {run_id}")
        result[str(run_id)] = score
    return result


def _score_template(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"run_id": row["run_id"], **{field: None for field in QUALITY_FIELDS}}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _summary(rows: list[Mapping[str, Any]], manifest: ExperimentManifest) -> str:
    lines = [
        f"# 实验汇总：{manifest.experiment_id}",
        "",
        "本目录由 `scholaragent.experiments` 生成。原始运行、结构化事件、指标、产物摘要和错误保存在 `runs.jsonl`；评分单独保存。",
        "",
        f"- 清单版本：`{manifest.schema_version}`",
        f"- 任务集哈希：`{manifest.task_set_hash}`",
        f"- Git 提交：`{manifest.git_commit or '未能读取'}`",
        f"- 模型：`{manifest.model or '未记录'}`",
        f"- 模式/策略：{', '.join(manifest.modes) or '未记录'}",
        f"- 策略版本：`{manifest.strategy_version or '未记录'}`",
        f"- 开始：`{manifest.started_at}`",
        f"- 结束：`{manifest.ended_at or '运行中或未记录'}`",
        "",
        "## 运行与评分状态",
        "",
        "|运行|模式|状态|错误|回答长度|LLM 调用|工具调用|",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        error = str(row.get("error") or "")[:120].replace("|", "\\|")
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('mode') or row.get('strategy') or ''} | "
            f"{row.get('score_state', 'unscored')} | {error or '—'} | "
            f"{len(str(row.get('answer') or ''))} | {metrics.get('llm_calls', '未记录')} | "
            f"{metrics.get('tool_calls', '未记录')} |"
        )
    lines += [
        "",
        "评分状态：`unscored` 未评分，`author_scored` 作者评分，"
        "`independently_scored` 独立评分；没有独立评分时不得把作者评分表述为第三方结论。",
        "",
    ]
    return "\n".join(lines)


def write_manifest_file(manifest: ExperimentManifest, path: str | Path) -> Path:
    """以不可覆盖方式写出单独的实验清单文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, manifest.to_dict())
    return target


def write_evidence_bundle(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    experiment_id: str | None = None,
    tasks: Iterable[Mapping[str, Any]] | None = None,
    model: str | None = None,
    config: Mapping[str, Any] | None = None,
    modes: Iterable[str] | None = None,
    strategy_version: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    scores: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    """创建一份新的证据包；目标目录存在时拒绝写入。"""
    materialized = [dict(row) for row in rows]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    task_rows = [dict(task) for task in (tasks if tasks is not None else _tasks_from_rows(materialized))]
    score_rows = _score_map(scores)
    states = Counter(
        _score_state(score_rows.get(str(row.get("run_id"))))
        for row in materialized
    )
    final_rows = []
    for row in materialized:
        row["score_state"] = _score_state(score_rows.get(str(row.get("run_id"))))
        final_rows.append(row)
    manifest = build_manifest(
        experiment_id or output.name,
        tasks=task_rows,
        model=model,
        config=config,
        modes=modes or _unique_modes(final_rows),
        strategy_version=strategy_version,
        started_at=started_at,
        ended_at=ended_at or utc_now(),
        score_states=states,
        repo_root=repo_root,
    )

    manifest_path = output / "manifest.json"
    runs_path = output / "runs.jsonl"
    template_path = output / "scores.template.jsonl"
    summary_path = output / "summary.md"
    _write_json(manifest_path, manifest.to_dict())
    _write_jsonl(runs_path, final_rows)
    _write_jsonl(template_path, (_score_template(row) for row in final_rows))
    paths = {
        "directory": output,
        "manifest": manifest_path,
        "runs": runs_path,
        "scores": template_path,
        "summary": summary_path,
    }
    if score_rows:
        score_path = output / "scores.jsonl"
        _write_jsonl(score_path, score_rows.values())
        paths["scored"] = score_path
    summary_path.write_text(_summary(final_rows, manifest) + "\n", encoding="utf-8", newline="\n")
    return paths
