"""Blind scoring packets, review validation and immutable derived reports."""

import copy
import json
from pathlib import Path

import pytest

from evals import run_experiment
from evals.review_results import main as review_main
from scholaragent.evidence import EvidenceLedger
from scholaragent.human_review import apply_human_review, prepare_blind_review, write_review_report
from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import create_runtime


ROOT = Path(__file__).resolve().parents[1]


def run_rows():
    ledger = EvidenceLedger()
    ledger.add_anchor(id="S001", kind="text", source="paper", page=1, excerpt="事实")
    ledger.register_answer("事实。[S001]")
    return [{"run_id": f"experiment:{mode}:1", "task_id": "t1", "task": "研究任务",
             "mode": mode, "strategy": mode, "answer": "事实。[S001]",
             "acceptance": ["逐条核对"], "evidence": ledger.to_dict(),
             "status": "completed" if mode == "react" else "failed",
             "completion": {"completeness": "complete" if mode == "react" else "partial"},
             "metrics": {"seconds": 1, "llm_calls": 2, "request_attempts": 3,
                         "prompt_tokens": None, "completion_tokens": None}}
            for mode in ("react", "plan")]


def prepared(tmp_path):
    rows = run_rows()
    output = prepare_blind_review(rows, tmp_path / "review")
    key = json.loads((output / "key.private.json").read_text(encoding="utf-8"))
    first = next(iter(key["review_to_run"]))
    score = {"review_id": first, "reviewer": "reviewer-a", "scorer_type": "independent", "note": "已核对原文",
             "task_completion": 0.8, "factual_correctness": 0.9, "citation_validity": 0.7, "output_completeness": 0.8}
    return rows, key, score


def test_blind_packets_hide_modes_run_ids_metrics_and_keep_sources(tmp_path):
    rows = run_rows()
    output = prepare_blind_review(rows, tmp_path / "review")
    packets = [json.loads(line) for line in (output / "review_packets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("run_id" not in row and "mode" not in row and "metrics" not in row for row in packets)
    assert packets[0]["anchors"][0]["excerpt"] == "事实"
    assert "experiment:" not in (output / "review_packets.jsonl").read_text(encoding="utf-8")
    assert "key.private.json" in (output / "README.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_blind_review(rows, output)


def test_scores_and_claim_review_only_change_derived_rows(tmp_path):
    rows, key, score = prepared(tmp_path)
    original = copy.deepcopy(rows)
    claim_review = {"review_id": score["review_id"], "claim_id": "C001", "reviewer": "reviewer-a",
                    "decision": "verified", "note": "逐字核对"}
    derived = apply_human_review(rows, key, [score], [claim_review])
    reviewed = next(row for row in derived if row["run_id"] == key["review_to_run"][score["review_id"]])
    assert reviewed["score_state"] == "independently_scored"
    assert reviewed["evidence"]["summary"]["verified_claims"] == 1
    assert rows == original
    assert sum(row["score_state"] == "unscored" for row in derived) == 1


@pytest.mark.parametrize("invalid", [None, True, -1, 2, float("nan"), float("inf")])
def test_missing_or_invalid_score_never_becomes_zero(tmp_path, invalid):
    rows, key, score = prepared(tmp_path)
    score["factual_correctness"] = invalid
    with pytest.raises(ValueError):
        apply_human_review(rows, key, [score])


def test_review_mapping_rejects_changed_original_runs(tmp_path):
    rows, key, score = prepared(tmp_path)
    rows[0]["answer"] = "changed"
    with pytest.raises(ValueError, match="不匹配"):
        apply_human_review(rows, key, [score])


def test_duplicate_scores_and_reviews_are_rejected(tmp_path):
    rows, key, score = prepared(tmp_path)
    with pytest.raises(ValueError, match="重复评分"):
        apply_human_review(rows, key, [score, score])
    review = {"review_id": score["review_id"], "claim_id": "C001", "reviewer": "r",
              "decision": "verified", "note": "checked"}
    with pytest.raises(ValueError, match="重复核验"):
        apply_human_review(rows, key, claim_reviews=[review, review])


def test_unscored_and_failed_runs_stay_in_report_denominators(tmp_path):
    rows, key, _ = prepared(tmp_path)
    derived = apply_human_review(rows, key)
    output = write_review_report(derived, tmp_path / "report")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["plan"]["runs"] == 1 and summary["plan"]["failed"] == 1
    assert summary["plan"]["quality"]["factual_correctness"] is None
    assert summary["plan"]["prompt_tokens"]["observed"] == 0
    assert "未评分" in (output / "report.md").read_text(encoding="utf-8")


def test_review_cli_writes_separate_packets_and_report(tmp_path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(row) for row in run_rows()), encoding="utf-8")
    original = runs.read_bytes()
    review_main(["prepare", "--runs", str(runs), "--output", str(tmp_path / "review")])
    review_main(["report", "--runs", str(runs), "--key", str(tmp_path / "review/key.private.json"),
                 "--output", str(tmp_path / "report")])
    assert runs.read_bytes() == original
    assert (tmp_path / "report/reviewed_runs.jsonl").is_file()


def test_research_manifests_have_twelve_tasks_and_three_pilot_tasks():
    full = run_experiment.load_definition(ROOT / "evals/experiments/research_full.json")
    pilot = run_experiment.load_definition(ROOT / "evals/experiments/research_pilot.json")
    tasks = run_experiment.experiment_tasks(full)
    assert len(tasks) == 12
    assert {task["category"] for task in tasks} == {"locate", "read", "compare", "gap"}
    assert all(task["acceptance"] and task["sources"] for task in tasks)
    assert len(run_experiment.experiment_tasks(pilot)) == 3
    assert all(task["split"] == "pilot" for task in run_experiment.experiment_tasks(pilot))


def test_dry_run_does_not_create_output_or_connect_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_experiment, "_ensure_model", lambda: pytest.fail("model connection during dry-run"))
    output = tmp_path / "output"
    run_experiment.main(["--manifest", str(ROOT / "evals/experiments/research_pilot.json"),
                         "--output", str(output), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["runs"] == 9
    assert not output.exists() and not Path(str(output) + ".state").exists()


def test_experiment_preserves_actual_model_budget_and_acceptance(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "t1", "task": "固定材料任务", "sources": ["2210.03629v1"],
                                 "acceptance": ["逐条核对"]}), encoding="utf-8")
    definition = {"schema_version": run_experiment.DEFINITION_VERSION, "id": "offline",
                  "tasks": str(tasks), "modes": ["react"], "config": {"agent_max_steps": 5}}
    def factory(**kwargs):
        return create_runtime(llm=ScriptedLLM([{"content": "回答", "tool_calls": []}], model="actual-scripted"), **kwargs)
    output = run_experiment.execute_experiment(definition, tmp_path / "out", runtime_factory=factory)
    row = json.loads(output["runs"].read_text(encoding="utf-8"))
    manifest = json.loads(output["manifest"].read_text(encoding="utf-8"))
    assert row["step_budgets"]["react"] == 5
    assert row["acceptance"] == ["逐条核对"]
    assert row["pinned_sources"] == ["2210.03629v1"]
    assert row["completion"]["completeness"] == "complete"
    assert manifest["model"] == "actual-scripted"


def test_source_snapshot_hash_uses_owned_workspace_material(tmp_path):
    from types import SimpleNamespace
    from scholaragent.workspace import TemporaryWorkspace
    import hashlib

    workspace = TemporaryWorkspace(tmp_path)
    path = workspace.paper_path("2210.03629v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture PDF bytes")
    result = SimpleNamespace(artifacts={"papers": [{"arxiv_id": "2210.03629v1"}]})
    snapshot = run_experiment._source_snapshots(result, workspace)[0]
    assert snapshot["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "path" not in snapshot


def test_material_conflicts_are_visible_in_derived_report(tmp_path):
    rows = run_rows()
    for index, row in enumerate(rows):
        row["pinned_sources"] = ["2210.03629v1"]
        row["source_snapshots"] = [{"arxiv_id": "2210.03629v1", "sha256": str(index)}]
    output = write_review_report(rows, tmp_path / "out")
    audit = json.loads((output / "sources.audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "conflicting" and len(audit["conflicts"]) == 1


@pytest.mark.parametrize("observed_second", [None, "1512.03385v2", "1512.03385"])
def test_each_pinned_source_is_checked_for_missing_or_wrong_version(tmp_path, observed_second):
    rows = run_rows()
    for row in rows:
        row["pinned_sources"] = ["2210.03629v1", "1512.03385v1"]
        row["source_snapshots"] = [{"arxiv_id": "2210.03629v1", "sha256": "same"}]
        if observed_second:
            row["source_snapshots"].append({"arxiv_id": observed_second, "sha256": "same-too"})
    output = write_review_report(rows, tmp_path / "out")
    audit = json.loads((output / "sources.audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "incomplete"
    assert audit["hash_status"] == "observed_consistent"
    assert len(audit["missing_sources"]) == 2
    assert {item["arxiv_id"] for item in audit["missing_sources"]} == {"1512.03385v1"}
    assert len(audit["version_mismatches"]) == (2 if observed_second else 0)
    assert len(audit["unexpected_sources"]) == (2 if observed_second else 0)
    assert audit["runs_without_pdf_snapshot"] == []


def test_exact_pinned_materials_are_consistent(tmp_path):
    rows = run_rows()
    for row in rows:
        row["pinned_sources"] = ["2210.03629v1", "1512.03385v1"]
        row["source_snapshots"] = [{"arxiv_id": source, "sha256": source}
                                   for source in row["pinned_sources"]]
    output = write_review_report(rows, tmp_path / "out")
    audit = json.loads((output / "sources.audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "observed_consistent"
    assert not audit["missing_sources"] and not audit["version_mismatches"]


def test_job_store_and_saved_replay_preserve_partial_completion():
    import webapp
    from scholaragent.replay import SavedCaseStore

    store = webapp.JobStore()
    job_id = store.create("任务", "react")
    completion = {"completeness": "partial", "stop_reasons": ["budget_exhausted"]}
    result = {"answer": "阶段结果", "completion": completion}
    store.finish(job_id, result)
    assert store.snapshot(job_id)["completion"] == completion
    assert SavedCaseStore._project(result)["completion"] == completion
