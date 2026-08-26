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


def test_readme_leads_with_public_project_positioning() -> None:
    introduction = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")[:600]

    assert "Agent 工程的透明实验台" in introduction
    assert "毕业设计项目" not in introduction


def test_readme_exposes_reproducible_evidence_and_project_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for required_text in (
        "100+ 项离线测试",
        "尚未产出正式成本对比结论",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ):
        assert required_text in readme
