"""实验清单命令的离线端到端行为。"""

import json

from evals.run_experiment import execute_experiment
from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import create_runtime


def test_experiment_manifest_generates_evidence_and_markdown_once(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps({
        "id": "offline-1", "task": "离线演示", "split": "demo",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    definition = {
        "schema_version": "experiment-definition-v1",
        "id": "offline-demo-v1",
        "tasks": str(tasks_path),
        "strategy": "fixed",
        "modes": ["react"],
        "model": "offline-scripted",
    }

    def factory(**kwargs):
        return create_runtime(
            llm=ScriptedLLM([{"content": "离线回答", "tool_calls": []}]),
            **kwargs,
        )

    paths = execute_experiment(definition, tmp_path / "bundle", runtime_factory=factory)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    row = json.loads(paths["runs"].read_text(encoding="utf-8").strip())

    assert manifest["experiment_id"] == "offline-demo-v1"
    assert manifest["model"] == "offline-scripted"
    assert row["answer"] == "离线回答"
    assert row["events"][-1]["type"] == "completed"
    assert "offline-demo-v1" in paths["summary"].read_text(encoding="utf-8")
