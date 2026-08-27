"""公开案例包的行为测试。"""

import json
from pathlib import Path

import pytest

from scholaragent.case_study import load_case, run_case, write_case_bundle
from scholaragent.llm import ScriptedLLM


class FakeRunner:
    def __init__(self, mode, collector, on_progress, artifacts):
        self.mode = mode
        self.collector = collector
        self.on_progress = on_progress
        self.artifacts = artifacts

    def run(self, task):
        self.collector.record_llm_call({"prompt_tokens": 10, "completion_tokens": 5})
        self.collector.record_tool_call()
        self.on_progress("[第 1 步] 模型想法:这段内部推理不应进入公开轨迹")
        self.on_progress("[第 1 步] 调用工具 save_note({'title': 'ReAct 要点'})")
        self.on_progress("[第 1 步] 工具返回:这里是一段很长的论文原文")
        self.artifacts.record(
            "save_note",
            {"title": "ReAct 要点", "content": "推理与行动交替"},
            "笔记「ReAct 要点」已保存",
        )
        answer = f"{self.mode} 完整回答：ReAct 让推理与行动交替进行。"
        self.on_progress(f"[第 1 步] 最终回答:{answer}")
        return answer


def fake_runner_factory(mode, metrics_collector, on_progress, artifacts):
    return FakeRunner(mode, metrics_collector, on_progress, artifacts)


def test_run_case_keeps_evidence_but_summarizes_internal_trace():
    case = {
        "id": "react-method",
        "task": "说明 ReAct 如何结合推理与行动。",
        "rubric": {
            "task_completion": "回答研究问题",
            "factual_correctness": "准确说明交替机制",
            "citation_validity": "给出可核验论文标识",
            "output_completeness": "包含方法与局限",
        },
    }

    rows = run_case(case, fake_runner_factory)

    assert [row["mode"] for row in rows] == ["react", "plan", "team"]
    assert all(row["case_id"] == "react-method" for row in rows)
    assert all(row["task"] == case["task"] for row in rows)
    assert rows[0]["answer"].startswith("react 完整回答")
    assert rows[0]["metrics"]["llm_calls"] == 1
    assert rows[0]["metrics"]["tool_calls"] == 1
    assert rows[0]["metrics"]["prompt_tokens"] == 10
    assert rows[0]["artifacts"]["counts"]["notes"] == 1
    assert rows[0]["quality"] is None
    assert rows[0]["error"] is None

    trace = "\n".join(rows[0]["trace"])
    assert "内部推理不应进入公开轨迹" not in trace
    assert "很长的论文原文" not in trace
    assert "完整回答：" not in trace
    assert "ReAct 要点" not in trace
    assert "模型想法:<已省略" in trace
    assert "调用工具 save_note(<参数已省略>)" in trace
    assert "工具返回:<已省略" in trace
    assert "最终回答:<完整内容见 answer 字段>" in trace


def test_write_case_bundle_separates_raw_runs_from_human_score_template(tmp_path):
    case = {
        "id": "react-method",
        "task": "说明 ReAct 如何结合推理与行动。",
        "rubric": {
            "task_completion": "回答研究问题",
            "factual_correctness": "准确说明交替机制",
            "citation_validity": "给出可核验论文标识",
            "output_completeness": "包含方法与局限",
        },
    }
    rows = run_case(case, fake_runner_factory, modes=("react",))
    output_dir = tmp_path / "react-method-run"

    paths = write_case_bundle(rows, output_dir)

    raw = [json.loads(line) for line in paths["runs"].read_text(encoding="utf-8").splitlines()]
    scores = [json.loads(line) for line in paths["scores"].read_text(encoding="utf-8").splitlines()]
    assert raw[0]["answer"].startswith("react 完整回答")
    assert raw[0]["run_id"] == "react-method:react:1"
    assert scores == [{
        "run_id": "react-method:react:1",
        "task_completion": None,
        "factual_correctness": None,
        "citation_validity": None,
        "output_completeness": None,
    }]

    with pytest.raises(FileExistsError):
        write_case_bundle(rows, output_dir)


def test_run_case_records_one_mode_failure_and_continues():
    class SometimesBrokenRunner(FakeRunner):
        def run(self, task):
            if self.mode == "plan":
                raise RuntimeError("模拟规划失败")
            return super().run(task)

    def factory(mode, metrics_collector, on_progress, artifacts):
        return SometimesBrokenRunner(mode, metrics_collector, on_progress, artifacts)

    rows = run_case(
        {"id": "react-method", "task": "说明 ReAct 方法。", "rubric": {}},
        factory,
    )

    assert [row["mode"] for row in rows] == ["react", "plan", "team"]
    assert rows[1]["answer"] == ""
    assert rows[1]["error"] == "RuntimeError: 模拟规划失败"
    assert rows[2]["answer"].startswith("team 完整回答")


def test_existing_runner_factory_exposes_case_study_evidence_seam(tmp_path, monkeypatch):
    from evals import run_eval
    from scholaragent import config

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        run_eval,
        "LLMClient",
        lambda: ScriptedLLM([{"content": "离线完整回答", "tool_calls": []}]),
    )

    rows = run_case(
        {"id": "offline", "task": "离线任务", "rubric": {}},
        run_eval.build_runner,
        modes=("react",),
    )

    assert rows[0]["error"] is None, rows[0]
    assert rows[0]["answer"] == "离线完整回答"
    assert rows[0]["metrics"]["llm_calls"] == 1
    assert rows[0]["metrics"]["prompt_tokens"] is None
    assert rows[0]["trace"] == ["[第 1 步] 最终回答:<完整内容见 answer 字段>"]


def test_run_case_isolates_each_modes_runtime_state(tmp_path):
    observed = {}

    def factory(mode, metrics_collector, on_progress, artifacts, workspace):
        observed[mode] = workspace.root
        return FakeRunner(mode, metrics_collector, on_progress, artifacts)

    run_case(
        {"id": "react-method", "task": "说明 ReAct 方法。", "rubric": {}},
        factory,
        state_root=tmp_path / "state",
    )

    assert observed == {
        "react": (tmp_path / "state" / "react-method" / "react").resolve(),
        "plan": (tmp_path / "state" / "react-method" / "plan").resolve(),
        "team": (tmp_path / "state" / "react-method" / "team").resolve(),
    }


def test_repository_react_case_has_fixed_prompt_and_complete_rubric():
    from pathlib import Path

    path = Path(__file__).parents[1] / "evals" / "cases" / "react_method.json"
    case = load_case(path)

    assert case["schema_version"] == "case-definition-v1"
    assert case["id"] == "react-method-evidence-v1"
    assert case["modes"] == ["react", "plan", "team"]
    assert "2210.03629" in case["task"]
    assert set(case["rubric"]) == {
        "task_completion",
        "factual_correctness",
        "citation_validity",
        "output_completeness",
    }


def test_load_case_rejects_identifier_that_can_escape_state_root(tmp_path):
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps({
        "schema_version": "case-definition-v1",
        "id": "../private",
        "title": "不安全案例",
        "task": "任务",
        "modes": ["react"],
        "rubric": {},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="案例 id"):
        load_case(path)


def test_load_case_requires_all_independent_human_score_fields(tmp_path):
    path = tmp_path / "incomplete-rubric.json"
    path.write_text(json.dumps({
        "schema_version": "case-definition-v1",
        "id": "incomplete-rubric",
        "title": "不完整评分",
        "task": "任务",
        "modes": ["react", "plan", "team"],
        "rubric": {"task_completion": "完成任务"},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="rubric"):
        load_case(path)


def test_load_case_rejects_unknown_execution_mode(tmp_path):
    path = tmp_path / "unknown-mode.json"
    path.write_text(json.dumps({
        "schema_version": "case-definition-v1",
        "id": "unknown-mode",
        "title": "错误模式",
        "task": "任务",
        "modes": ["react", "magic"],
        "rubric": {
            "task_completion": "完成",
            "factual_correctness": "正确",
            "citation_validity": "引用",
            "output_completeness": "完整",
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="modes"):
        load_case(path)


def test_run_case_does_not_publish_absolute_artifact_paths(tmp_path):
    rows = run_case(
        {
            "id": "react-method",
            "task": "说明 ReAct 方法。",
            "modes": ["react"],
            "rubric": {},
        },
        fake_runner_factory,
        modes=("react",),
        state_root=tmp_path / "private-state",
    )

    artifact_path = rows[0]["artifacts"]["notes"][0]["path"]
    assert not Path(artifact_path).is_absolute()
    assert artifact_path.replace("\\", "/") == "notes/research_notes.md"


def test_case_study_command_writes_fixed_case_bundle_with_injected_runner(tmp_path):
    from evals.run_case_study import execute_case_study

    case_path = Path(__file__).parents[1] / "evals" / "cases" / "react_method.json"
    output_dir = tmp_path / "case-run"

    paths = execute_case_study(case_path, output_dir, runner_factory=fake_runner_factory)

    rows = [json.loads(line) for line in paths["runs"].read_text(encoding="utf-8").splitlines()]
    assert [row["mode"] for row in rows] == ["react", "plan", "team"]
    assert all(row["case_id"] == "react-method-evidence-v1" for row in rows)
    assert paths["scores"].exists()
    assert paths["state"].is_dir()


def test_execute_case_study_keeps_private_state_outside_public_output(tmp_path):
    from evals.run_case_study import execute_case_study

    case_path = Path(__file__).parents[1] / "evals" / "cases" / "react_method.json"
    output_dir = tmp_path / "public-bundle"
    state_dir = tmp_path / "private-state"

    paths = execute_case_study(
        case_path,
        output_dir,
        runner_factory=fake_runner_factory,
        state_dir=state_dir,
    )

    assert paths["state"] == state_dir
    assert state_dir.is_dir()
    # 公开目录只包含证据包文件，不混入私有运行状态
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "manifest.json",
        "runs.jsonl",
        "scores.template.jsonl",
        "summary.md",
    ]
    # 默认行为不变：不传 state_dir 时仍是输出目录旁的同名 .state
    default_paths = execute_case_study(
        case_path,
        tmp_path / "default-bundle",
        runner_factory=fake_runner_factory,
    )
    assert default_paths["state"] == tmp_path / "default-bundle.state"
