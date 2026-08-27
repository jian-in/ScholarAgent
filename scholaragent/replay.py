"""把本地 JSONL 证据包投影为 Web 可消费的回放结果。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .markdown_lite import render_markdown


CASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
SCORE_STATES = ("unscored", "author_scored", "independently_scored")


class SavedCaseStore:
    """只读本地案例适配器；默认读取仓库内公开案例目录。"""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "evals" / "case_results"

    def _case_dir(self, case_id: str) -> Path:
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
            raise KeyError("案例不存在")
        target = (self.root / case_id).resolve()
        root = self.root.resolve()
        if root not in target.parents:
            raise KeyError("案例不存在")
        if not target.is_dir() or not (target / "runs.jsonl").is_file():
            raise KeyError("案例不存在")
        return target

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _manifest(self, case_dir: Path) -> dict[str, Any]:
        path = case_dir / "manifest.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _score_state(score):
        if not score:
            return "unscored"
        scorer = str(score.get("scorer_type") or score.get("scorer") or "author").lower()
        return (
            "independently_scored"
            if scorer in {"independent", "third_party", "third-party", "external", "blind"}
            else "author_scored"
        )

    @staticmethod
    def _scores(case_dir):
        path = case_dir / "scores.jsonl"
        if not path.is_file():
            return {}
        try:
            rows = SavedCaseStore._read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(row["run_id"]): row for row in rows if row.get("run_id")}

    @staticmethod
    def _score_states(rows, manifest, case_dir):
        states = manifest.get("score_states")
        if isinstance(states, dict):
            return {state: int(states.get(state, 0)) for state in SCORE_STATES}
        scores = SavedCaseStore._scores(case_dir)
        counts = {state: 0 for state in SCORE_STATES}
        for row in rows:
            counts[SavedCaseStore._score_state(scores.get(str(row.get("run_id"))))] += 1
        return counts

    def list_cases(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        result = []
        for case_dir in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not case_dir.is_dir() or not CASE_ID.fullmatch(case_dir.name):
                continue
            runs_path = case_dir / "runs.jsonl"
            if not runs_path.is_file():
                continue
            try:
                rows = self._read_jsonl(runs_path)
            except (OSError, json.JSONDecodeError):
                continue
            manifest = self._manifest(case_dir)
            result.append({
                "id": case_dir.name,
                "title": manifest.get("title") or manifest.get("experiment_id") or case_dir.name,
                "runs": len(rows),
                "score_states": self._score_states(rows, manifest, case_dir),
            })
        return result

    @staticmethod
    def _legacy_events(row: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = str(row.get("run_id") or "saved-run")
        mode = str(row.get("mode") or row.get("selected_mode") or "react")
        events = [
            {"type": "run_started", "run_id": run_id, "timestamp": None, "mode": mode,
             "payload": {"source": "saved_case"}},
            {"type": "mode_selected", "run_id": run_id, "timestamp": None, "mode": mode,
             "payload": {"mode": mode, "reason": "公开案例兼容回放"}},
        ]
        for message in row.get("trace") or []:
            events.append({
                "type": "legacy_progress",
                "run_id": run_id,
                "timestamp": None,
                "mode": mode,
                "payload": {"message": str(message)},
            })
        terminal = "failed" if row.get("error") else (row.get("status") or "completed")
        if terminal not in {"cancelled", "failed", "completed"}:
            terminal = "completed"
        events.append({
            "type": terminal,
            "run_id": run_id,
            "timestamp": None,
            "mode": mode,
            "payload": {"source": "saved_case"},
        })
        return events

    @classmethod
    def _project(cls, row: dict[str, Any], score_state: str | None = None) -> dict[str, Any]:
        run_id = str(row.get("run_id") or "saved-run")
        mode = str(row.get("mode") or row.get("selected_mode") or "react")
        status = row.get("status") or ("failed" if row.get("error") else "completed")
        events = [dict(event) for event in (row.get("events") or cls._legacy_events(row))]
        for event in events:
            event["run_id"] = run_id
        answer = str(row.get("answer") or "")
        return {
            "source": "saved_case",
            "run_id": run_id,
            "case_id": row.get("case_id"),
            "task": row.get("task"),
            "requested_mode": row.get("requested_mode") or mode,
            "mode": mode,
            "status": status,
            "cancelled": status == "cancelled",
            "answer": answer,
            "answer_html": render_markdown(answer),
            "routing": row.get("routing") or row.get("routing_decision"),
            "metrics": row.get("metrics"),
            "artifacts": row.get("artifacts") or {},
            "evidence": row.get("evidence") or {},
            "workflow": row.get("workflow"),
            "source_format": row.get("source_format"),
            "events": events,
            "trace": row.get("trace") or [],
            "score_state": score_state or row.get("score_state") or (
                "author_scored" if row.get("quality") is not None else "unscored"
            ),
            "error": row.get("error"),
        }

    def get_case(self, case_id: str, run_id: str | None = None) -> dict[str, Any]:
        case_dir = self._case_dir(case_id)
        rows = self._read_jsonl(case_dir / "runs.jsonl")
        if not rows:
            raise KeyError("案例没有运行记录")
        manifest = self._manifest(case_dir)
        scores = self._scores(case_dir)
        projected = [
            self._project(
                row,
                self._score_state(scores[str(row.get("run_id"))])
                if str(row.get("run_id")) in scores else None,
            )
            for row in rows
        ]
        selected = projected[0]
        if run_id is not None:
            selected = next((row for row in projected if row["run_id"] == run_id), None)
            if selected is None:
                raise KeyError("运行记录不存在")
        return {
            "source": "saved_case",
            "case_id": case_id,
            "title": manifest.get("title") or manifest.get("experiment_id") or case_id,
            "score_states": self._score_states(rows, manifest, case_dir),
            "runs": projected,
            "selected": selected,
        }
