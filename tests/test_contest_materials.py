"""比赛材料只陈述可核验事实，并准备第二个真实案例。"""

import json
from pathlib import Path

from scholaragent.case_study import load_case


ROOT = Path(__file__).parents[1]


def test_second_case_is_a_valid_public_real_paper_definition():
    case = load_case(ROOT / "evals" / "cases" / "resnet_method.json")
    assert case["id"] == "resnet-method-evidence-v1"
    assert "1512.03385" in case["task"]
    assert "计算机视觉" in case["task"]
    assert case["modes"] == ["react", "plan", "team"]


def test_contest_materials_keep_pending_experiments_explicit():
    matrix = (ROOT / "docs" / "contest" / "证据矩阵.md").read_text(encoding="utf-8")
    source = (ROOT / "docs" / "contest" / "项目介绍源稿.md").read_text(encoding="utf-8")
    assert "待补数据" in matrix
    assert "尚未运行" in source
    assert "独立评分" in matrix and "独立评分" in source
    assert "上传" in source
