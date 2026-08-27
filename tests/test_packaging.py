"""打包元数据与安装后入口的契约。"""

from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_pyproject_declares_runtime_metadata_and_two_console_scripts():
    path = ROOT / "pyproject.toml"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.13"' in text
    assert 'name = "scholaragent"' in text
    assert 'version = "0.1.0rc1"' in text
    assert 'scholaragent = "main:main"' in text
    assert 'scholaragent-web = "webapp:main"' in text
    assert "openai" in text and "httpx" in text and "pypdf" in text


def test_ci_covers_supported_python_matrix_and_wheel_smoke():
    path = ROOT / ".github" / "workflows" / "ci.yml"
    text = path.read_text(encoding="utf-8")
    for version in ('"3.10"', '"3.11"', '"3.12"'):
        assert version in text
    assert "python -m build" in text
    assert "pip install" in text


def test_installed_entrypoint_targets_are_importable():
    import main
    import webapp

    assert callable(main.main)
    assert callable(webapp.main)


@pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(), reason="等待打包元数据")
def test_release_candidate_version_is_pep440_compatible():
    from packaging.version import Version

    assert str(Version("0.1.0rc1")) == "0.1.0rc1"
