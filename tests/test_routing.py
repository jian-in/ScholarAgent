"""阶段 B：特征、规则路由与自动委派的离线测试。"""

import pytest

from scholaragent.routing import AdaptiveRunner, RuleRouter, TaskFeatureExtractor


def test_features_cover_empty_bilingual_long_and_negated_tasks():
    extractor = TaskFeatureExtractor()
    assert extractor.extract("")["task_length"] == 0.0

    bilingual = extractor.extract(
        "Search literature, download the full text and compare multiple papers"
    )
    assert bilingual["requires_literature_search"] == 1.0
    assert bilingual["requires_full_text"] == 1.0
    assert bilingual["requires_multi_paper_comparison"] == 1.0
    assert bilingual["action_goal_count"] == 3.0

    negated = extractor.extract("不需要检索论文，也不要下载全文，只总结已有内容")
    assert negated["requires_literature_search"] == 0.0
    assert negated["requires_full_text"] == 0.0
    assert negated["requires_review"] == 1.0
    assert extractor.extract("x" * 5000)["task_length"] == 1.0


def test_rule_router_selects_simple_medium_and_complex_tasks():
    router = RuleRouter()
    assert router.route("查询 ReAct 论文的发表年份").mode == "react"
    assert router.route("先检索相关论文，再阅读方法部分并保存笔记").mode == "plan"
    decision = router.route("检索并阅读全文，比较三篇论文，撰写研究脉络综述和开放问题")
    assert decision.mode == "team"
    assert "全文精读" in decision.reason


def test_rule_router_threshold_boundary_is_deterministic():
    router = RuleRouter()
    assert router.route("检索一篇论文并总结").mode == "react"
    assert router.route("检索一篇论文、阅读方法并总结").mode == "plan"


class RecordingRunner:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def run(self, task):
        self.calls.append(task)
        return self.answer


def test_adaptive_runner_executes_only_selected_runner():
    runners = {mode: RecordingRunner(mode) for mode in ("react", "plan", "team")}
    adaptive = AdaptiveRunner(RuleRouter(), runners)

    assert adaptive.run("查询论文年份") == "react"
    assert runners["react"].calls == ["查询论文年份"]
    assert runners["plan"].calls == []
    assert runners["team"].calls == []


def test_adaptive_runner_does_not_retry_other_modes_after_error():
    class FailingRunner:
        def run(self, task):
            raise RuntimeError("react failed")

    team = RecordingRunner("team")
    adaptive = AdaptiveRunner(
        RuleRouter(), {"react": FailingRunner(), "plan": RecordingRunner("plan"), "team": team}
    )
    with pytest.raises(RuntimeError, match="react failed"):
        adaptive.run("查询论文年份")
    assert team.calls == []
