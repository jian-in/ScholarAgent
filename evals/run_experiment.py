"""从版本化实验清单生成不可覆盖证据包。

示例：
    python evals/run_experiment.py --manifest evals/experiments/demo.json \
        --output evals/case_results/demo-v1

清单可以选择固定模式、规则路由或成本感知路由；真正的运行统一进入
``scholaragent.runtime.ExecutionRuntime``，输出由 experiments 模块负责。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent import config
from scholaragent.llm import LLMClient
from scholaragent.experiments import utc_now, write_evidence_bundle
from scholaragent.routing import CostAwareRouter, RuleRouter
from scholaragent.runtime import RUNTIME_MODES, create_runtime, detect_ollama
from scholaragent.workspace import TemporaryWorkspace


DEFINITION_VERSION = "experiment-definition-v1"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


def load_definition(path: str | Path) -> dict:
    """读取并校验实验清单；任务文件路径相对清单文件解析。"""
    definition_path = Path(path).resolve()
    with definition_path.open(encoding="utf-8") as handle:
        definition = json.load(handle)
    if not isinstance(definition, dict) or definition.get("schema_version") != DEFINITION_VERSION:
        raise ValueError(f"实验清单必须使用 {DEFINITION_VERSION}")
    experiment_id = definition.get("id")
    if not isinstance(experiment_id, str) or not SAFE_ID.fullmatch(experiment_id):
        raise ValueError("实验 id 只能使用安全的 ASCII 小写标识")
    tasks_path = Path(str(definition.get("tasks") or ""))
    if not tasks_path.is_absolute():
        tasks_path = definition_path.parent / tasks_path
    definition = dict(definition)
    definition["_path"] = str(definition_path)
    definition["_tasks_path"] = str(tasks_path.resolve())
    strategy = definition.get("strategy", "fixed")
    if strategy not in {"fixed", "rule", "cost_aware", "auto"}:
        raise ValueError("strategy 必须是 fixed、rule、cost_aware 或 auto")
    modes = definition.get("modes", list(RUNTIME_MODES))
    if not isinstance(modes, list) or not modes or any(mode not in (*RUNTIME_MODES, "auto") for mode in modes):
        raise ValueError("modes 必须是 react、plan、team、auto 的非空列表")
    definition["modes"] = list(dict.fromkeys(modes))
    return definition


def load_tasks(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(not isinstance(row, dict) or not str(row.get("task") or "").strip() for row in rows):
        raise ValueError("实验任务文件必须包含至少一条带 task 的对象")
    return [dict(row) for row in rows]


def _event_rows(result, run_id):
    return [
        {**dict(event), "run_id": run_id}
        for event in getattr(result, "events", ())
    ]


def _run_row(task, result, run_id, label):
    return {
        "schema_version": "experiment-run-v1",
        "run_id": run_id,
        "task_id": task.get("id") or run_id,
        "split": task.get("split"),
        "category": task.get("category"),
        "acceptance": task.get("acceptance") or [],
        "pinned_sources": task.get("sources") or [],
        "task": task["task"],
        "requested_mode": result.requested_mode,
        "mode": result.mode,
        "strategy": label,
        "status": result.status,
        "answer": result.answer,
        "events": _event_rows(result, run_id),
        "metrics": result.metrics.to_dict(),
        "artifacts": dict(result.artifacts),
        "evidence": dict(result.evidence),
        "workflow": result.workflow,
        "source_format": result.source_format,
        "completion": dict(getattr(result, "completion", {}) or {}),
        "routing": result.routing,
        "error": result.error,
        "score_state": "unscored",
    }


def experiment_tasks(definition):
    tasks = load_tasks(definition.get("_tasks_path") or definition["tasks"])
    ids = [task.get("id") for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("任务 id 必须唯一")
    selected = definition.get("task_ids")
    if selected is not None:
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
            raise ValueError("task_ids 必须为非空且唯一的列表")
        if set(selected) - set(ids):
            raise ValueError("task_ids 包含未知任务")
        tasks = [task for task in tasks if task.get("id") in selected]
    for task in tasks:
        sources = task.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(source, str) or not re.fullmatch(
                r"\d{4}\.\d{4,5}v\d+", source) for source in sources):
            raise ValueError("sources 必须使用带版本的 arXiv 编号")
    return tasks


def _source_snapshots(result, workspace):
    snapshots = []
    for paper in (result.artifacts or {}).get("papers", []):
        arxiv_id = paper.get("arxiv_id") or ""
        if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv_id):
            continue
        path = workspace.paper_path(arxiv_id)
        if path.is_file():
            content = path.read_bytes()
            snapshots.append({"arxiv_id": arxiv_id, "sha256": hashlib.sha256(content).hexdigest(),
                              "bytes": len(content)})
    return snapshots


def _configure_router(runtime, strategy, definition):
    if strategy == "rule":
        runtime.router = RuleRouter()
    elif strategy == "cost_aware":
        policy = definition.get("policy")
        runtime.router = CostAwareRouter(str(policy) if policy else None)


def execute_experiment(definition: dict, output_dir: str | Path, runtime_factory=create_runtime) -> dict[str, Path]:
    """执行清单并生成证据包；私有工作区与公开输出分离。"""
    if definition.get("schema_version") != DEFINITION_VERSION:
        raise ValueError(f"实验清单必须使用 {DEFINITION_VERSION}")
    experiment_id = definition["id"]
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"实验输出目录已存在: {output}")
    state_root = Path(str(output) + ".state")
    if state_root.exists():
        raise FileExistsError(f"实验私有状态目录已存在: {state_root}")
    tasks = experiment_tasks(definition)
    parameters = definition.get("config") or {}
    step_budget = parameters.get("agent_max_steps")
    if step_budget is not None and (
        isinstance(step_budget, bool) or not isinstance(step_budget, int) or step_budget < 1
    ):
        raise ValueError("agent_max_steps 必须是正整数")
    strategy = definition.get("strategy", "fixed")
    labels = definition.get("modes", list(RUNTIME_MODES)) if strategy == "fixed" else [strategy]
    if strategy == "auto":
        labels = ["auto"]
    started_at = utc_now()
    rows = []
    state_root.mkdir(parents=True, exist_ok=False)
    for task_index, task in enumerate(tasks, 1):
        for label in labels:
            mode = label if strategy == "fixed" or label == "auto" else "auto"
            run_id = f"{experiment_id}:{label}:{task_index}"
            workspace = TemporaryWorkspace(state_root / re.sub(r"[^a-zA-Z0-9._-]+", "_", run_id))
            kwargs = {
                "workspace": workspace,
                "conversation": False,
                "auto_recall": False,
                "team_require_full_paper": definition.get("team_require_full_paper", True),
            }
            if definition.get("demo"):
                kwargs["demo"] = True
            runtime = runtime_factory(**kwargs)
            if definition.get("model"):
                for client in (getattr(runtime, "research_llm", None), getattr(runtime, "summary_llm", None)):
                    if isinstance(client, LLMClient):
                        client.model = definition["model"]
            if step_budget is not None:
                runtime.agent.max_steps = step_budget
                runtime.runners["plan"].agent.max_steps = step_budget
            _configure_router(runtime, strategy, definition)
            text = task["task"]
            if task.get("sources"):
                text += "\n固定文献版本：" + ", ".join(task["sources"]) + "。只依据这些版本，重要结论标注来源锚点。"
            result = runtime.run(text, mode=mode, run_id=run_id,
                                 pinned_sources=task.get("sources") or ())
            row = _run_row(task, result, run_id, label)
            row["source_snapshots"] = _source_snapshots(result, workspace)
            metadata = runtime.llm.metadata() if hasattr(runtime.llm, "metadata") else {}
            row["model_metadata"] = {key: metadata.get(key) for key in ("model", "provider", "role")}
            row["step_budgets"] = {
                "react": runtime.agent.max_steps, "plan_worker": runtime.runners["plan"].agent.max_steps,
                "team_searcher": runtime.runners["team"].searcher.max_steps,
                "team_reader": runtime.runners["team"].reader.max_steps,
                "team_writer": runtime.runners["team"].writer.max_steps,
            }
            rows.append(row)
    models = list(dict.fromkeys(row["model_metadata"].get("model") for row in rows))
    return write_evidence_bundle(
        rows,
        output,
        experiment_id=experiment_id,
        tasks=tasks,
        model=models[0] if len(models) == 1 else "mixed:" + ",".join(str(model) for model in models),
        config={
            **dict(definition.get("config") or {}),
            "strategy": strategy,
            "modes": labels,
            "declared_model": definition.get("model"),
            "source_policy": "enforced-pinned-arxiv-version-and-observed-pdf-sha256",
        },
        modes=[row["mode"] for row in rows],
        strategy_version=definition.get("strategy_version") or "experiment-run-v1",
        started_at=started_at,
        ended_at=utc_now(),
    )


def _ensure_model():
    if config.LLM_API_KEY:
        return
    model = detect_ollama(prefer=config.LLM_MODEL)
    if not model:
        raise RuntimeError("没有可用模型：请配置 API Key 或启动 Ollama")
    config.LLM_BASE_URL = "http://localhost:11434/v1"
    config.LLM_API_KEY = "ollama"
    config.LLM_MODEL = model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="experiment-definition-v1 JSON")
    parser.add_argument("--output", required=True, help="新的不可覆盖证据包目录")
    parser.add_argument("--dry-run", action="store_true", help="只检查清单并打印运行数量，不连接模型")
    args = parser.parse_args(argv)
    definition = load_definition(args.manifest)
    if args.dry_run:
        tasks = experiment_tasks(definition)
        count = len(definition["modes"]) if definition.get("strategy", "fixed") == "fixed" else 1
        print(json.dumps({"experiment_id": definition["id"], "tasks": len(tasks),
                          "runs": len(tasks) * count, "demo": bool(definition.get("demo")),
                          "model": definition.get("model") or "configured-at-execution"}, ensure_ascii=False))
        return
    if not definition.get("demo"):
        _ensure_model()
    paths = execute_experiment(definition, args.output)
    print(f"证据包: {paths['directory']}")
    print(f"汇总: {paths['summary']}")


if __name__ == "__main__":
    main()
