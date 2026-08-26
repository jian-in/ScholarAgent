"""M6 评测框架的离线测试。

运行方式(项目根目录下):
    python -m pytest tests -q    (推荐)
    python tests/test_m6_eval.py (没装 pytest 时直接跑)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.evaluate import Evaluator, load_tasks, score_answer, write_report

SAMPLE_TASKS = """# 注释行要被跳过
{"id": "t1", "task": "算算术", "expect": ["96"], "modes": ["react", "plan"]}
{"id": "t2", "task": "写综述", "expect": ["ReAct"], "modes": ["team"]}
{"id": "t3", "task": "查论文", "expect": ["2210"]}
"""


def _write_tasks(tmp):
    path = os.path.join(tmp, "tasks.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TASKS)
    return path


def test_load_tasks_filters_by_mode():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_tasks(tmp)
        assert [t["id"] for t in load_tasks(path)] == ["t1", "t2", "t3"]
        # 按模式过滤;没写 modes 的任务默认全模式适用
        assert [t["id"] for t in load_tasks(path, "react")] == ["t1", "t3"]
        assert [t["id"] for t in load_tasks(path, "team")] == ["t2", "t3"]


def test_score_answer():
    assert score_answer("结果是 96", ["96"]) == 1.0
    assert score_answer("ReAct(arXiv 2210.03629)", ["react", "2210"]) == 1.0  # 大小写不敏感
    assert score_answer("只命中一半 2210", ["2210", "不存在"]) == 0.5
    assert score_answer("啥也没有", ["96"]) == 0.0
    assert score_answer("任何回答", []) == 0.0  # 没有期望关键词就没法给分
    # 归一化:千分位逗号/空格不能骗过评分器(真实评测踩过的坑)
    assert score_answer("一共有 525,600 分钟", ["525600"]) == 1.0
    assert score_answer("答案是 5 2 5 6 0 0", ["525600"]) == 1.0
    # 归一化只在数字之间生效:不能把 "22, 10" 粘成 "2210" 造成误判
    assert score_answer("检索到 22, 10 篇论文", ["2210"]) == 0.0
    # 期望关键词支持备选列表:命中任意一个写法就算命中
    assert score_answer("It combines reasoning and acting",
                        [["推理", "reasoning"], ["行动", "acting"]]) == 1.0


class FakeRunner:
    """答对一题、崩一题的假被测对象。"""

    def __init__(self):
        self.calls = 0

    def run(self, task):
        self.calls += 1
        if "崩溃" in task:
            raise RuntimeError("模拟执行崩溃")
        return "回答:结果是 96"


def test_evaluator_scores_and_survives_crash():
    tasks = [
        {"id": "ok", "task": "算算术", "expect": ["96"]},
        {"id": "boom", "task": "这题会崩溃", "expect": ["随便"]},
    ]
    rows = Evaluator(FakeRunner(), mode="react", verbose=False).run(tasks)

    assert rows[0]["score"] == 1.0
    assert rows[1]["score"] == 0.0                 # 崩溃记 0 分
    assert "执行出错" in rows[1]["answer_preview"]  # 但要留下现场
    assert len(rows) == 2                          # 一题崩溃不影响后面的题


def test_write_report():
    with tempfile.TemporaryDirectory() as tmp:
        rows = [
            {"id": "a", "mode": "react", "score": 1.0, "seconds": 2.0,
             "answer_preview": "好"},
            {"id": "b", "mode": "react", "score": 0.5, "seconds": 3.0,
             "answer_preview": "带|竖线\n和换行"},
        ]
        report = os.path.join(tmp, "r", "report.md")
        avg = write_report(rows, report)

        assert abs(avg - 0.75) < 1e-9
        text = open(report, encoding="utf-8").read()
        assert "0.75" in text and "| a |" in text
        assert os.path.exists(report.replace(".md", ".jsonl"))  # 原始数据也要落盘


if __name__ == "__main__":
    # 不依赖 pytest 的极简测试运行器
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"通过:{name}")
    print("全部测试通过")
