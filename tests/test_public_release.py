from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {
    ".bat",
    ".css",
    ".example",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def _tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    paths = result.stdout.decode("utf-8").split("\0")
    return [
        PROJECT_ROOT / path
        for path in paths
        if path and Path(path).suffix.lower() in PUBLIC_TEXT_SUFFIXES
    ]


def _tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return set(result.stdout.decode("utf-8").split("\0"))


def test_public_release_has_no_private_persona_wording() -> None:
    forbidden_terms = ("\u54c8\u5409\u7c73", "\u4e3b\u4eba")
    offenders: list[str] = []

    for path in _tracked_text_files():
        content = path.read_text(encoding="utf-8")
        matched = [term for term in forbidden_terms if term in content]
        if matched:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            offenders.append(f"{relative}: {', '.join(matched)}")

    assert not offenders, "公开文件含有禁止的私人称呼：\n" + "\n".join(offenders)


def test_public_release_excludes_local_private_files() -> None:
    tracked = _tracked_paths()

    assert ".env" not in tracked
    assert "docs/进度报告_2026-07-26夜间.md" not in tracked
    assert not any(path.startswith("data/") for path in tracked)
    assert not any(path.startswith("evals/results/") for path in tracked)


def test_public_release_includes_contribution_guide() -> None:
    guide = PROJECT_ROOT / "CONTRIBUTING.md"

    assert guide.is_file(), "公开仓库需要 CONTRIBUTING.md"
    content = guide.read_text(encoding="utf-8")
    for required_text in ("pytest", "Pull Request", "MIT"):
        assert required_text in content


def test_public_release_includes_security_policy() -> None:
    policy = PROJECT_ROOT / "SECURITY.md"

    assert policy.is_file(), "公开仓库需要 SECURITY.md"
    content = policy.read_text(encoding="utf-8")
    for required_text in ("安全漏洞", ".env", "GitHub Security Advisory"):
        assert required_text in content


def test_public_release_includes_community_governance_files() -> None:
    required_files = {
        "CODE_OF_CONDUCT.md": ("Contributor Covenant", "SECURITY.md"),
        "CHANGELOG.md": ("Keep a Changelog",),
        "MAINTAINERS.md": ("维护范围", "SECURITY.md"),
        ".github/ISSUE_TEMPLATE/bug_report.md": ("复现",),
        ".github/ISSUE_TEMPLATE/feature_request.md": ("ReAct",),
        ".github/PULL_REQUEST_TEMPLATE.md": ("pytest",),
        ".github/workflows/ci.yml": ("pytest", "requirements.txt"),
    }

    for relative, required_text in required_files.items():
        path = PROJECT_ROOT / relative
        assert path.is_file(), f"公开仓库需要 {relative}"
        content = path.read_text(encoding="utf-8")
        for text in required_text:
            assert text in content, f"{relative} 缺少关键内容: {text}"


def test_readme_leads_with_public_project_positioning() -> None:
    introduction = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")[:600]

    assert "Agent 工程的透明实验台" in introduction
    assert "毕业设计项目" not in introduction


def test_readme_exposes_reproducible_evidence_and_project_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for required_text in (
        "200+ 项离线测试",
        "尚未产出正式成本对比结论",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ):
        assert required_text in readme


def test_public_case_bundles_are_valid_and_sanitized() -> None:
    """已发布的案例证据包必须可解析、路径可移植、评分完整且无路径泄漏。"""
    import json
    import re

    bundles = sorted(
        path for path in (PROJECT_ROOT / "evals" / "case_results").iterdir()
        if path.is_dir()
    )
    assert bundles, "公开案例目录 evals/case_results/ 至少需要一个证据包"

    for bundle in bundles:
        runs = bundle / "runs.jsonl"
        assert runs.is_file(), f"{runs.relative_to(PROJECT_ROOT)} 缺失"
        rows = [
            json.loads(line)
            for line in runs.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows, f"{runs.relative_to(PROJECT_ROOT)} 为空"
        assert [row["mode"] for row in rows] == ["react", "plan", "team"], (
            f"{runs.relative_to(PROJECT_ROOT)} 必须覆盖三种模式"
        )

        raw_json = runs.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:\\\\", raw_json), (
            f"{runs.relative_to(PROJECT_ROOT)} 泄漏了 Windows 绝对路径"
        )
        assert "/mnt/" not in raw_json, (
            f"{runs.relative_to(PROJECT_ROOT)} 泄漏了挂载路径"
        )

        for row in rows:
            assert row["run_id"].startswith(row["case_id"])
            assert isinstance(row["metrics"], dict)
            assert row["rubric"]
            for section in ("papers", "notes", "memories"):
                for item in row["artifacts"][section]:
                    path = item.get("path")
                    if path:
                        assert not Path(path).is_absolute(), (
                            f"{row['run_id']} 产物路径不是可移植相对路径: {path}"
                        )

        scores = bundle / "scores.jsonl"
        assert scores.is_file(), f"{scores.relative_to(PROJECT_ROOT)} 缺失"
        score_rows = [
            json.loads(line)
            for line in scores.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_run = {score["run_id"]: score for score in score_rows}
        assert set(by_run) == {row["run_id"] for row in rows}
        for score in score_rows:
            for field in ("task_completion", "factual_correctness",
                          "citation_validity", "output_completeness"):
                assert field in score and 0.0 <= float(score[field]) <= 1.0
