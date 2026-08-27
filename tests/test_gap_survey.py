from scholaragent.gap_survey import (
    GAP_TRACKS,
    build_gap_survey_plan,
    is_fast_gap_survey_task,
    is_gap_survey_task,
    missing_synthesis_sections,
)
from scholaragent.llm import ScriptedLLM
from scholaragent.planner import Planner
from scholaragent.routing import RuleRouter
from scholaragent.runtime import create_runtime
from scholaragent.workflow import WorkflowRegistry


GAP_TASK = """
补齐当前研究的四项资料缺口：
1. 推理能力，重点是 CoT、ToT 等前沿论文；
2. Agent 工具使用的机制研究；
3. 2024-2025 年最新进展的系统性综述；
4. 硬件与推理效率优化。
请全部开始调研，并让每篇论文都说明核心问题、方法、实验结论和局限。
"""

FAST_GAP_TASK = GAP_TASK + "\n请采用快速摘要筛选：不下载 PDF，不做全文精读，先给出四个方向的候选和摘要级核心。"
DEEP_GAP_TASK = GAP_TASK + "\n请深度补齐：每个方向下载论文并精读到全文，给出页码级证据。"


def test_gap_task_is_detected_and_expanded_into_four_independent_tracks():
    assert is_gap_survey_task(GAP_TASK)
    assert is_fast_gap_survey_task(GAP_TASK)

    plan = build_gap_survey_plan(GAP_TASK)

    assert len(plan) == len(GAP_TRACKS) == 4
    assert all("至少" in step and "核心问题" in step for step in plan)
    assert "CoT" in plan[0] and "ToT" in plan[0]
    assert "Tool Use" in plan[1]
    assert "2024-2025" in plan[2]
    assert "inference optimization" in plan[3]


def test_a_request_to_explain_one_gap_is_not_promoted_to_a_four_track_run():
    assert not is_gap_survey_task("请说明目前资料缺口，并给出下一步建议。")


def test_fast_gap_mode_avoids_pdf_reading_and_is_explicit_about_abstract_evidence():
    assert is_fast_gap_survey_task(FAST_GAP_TASK)

    plan = build_gap_survey_plan(FAST_GAP_TASK, depth="fast")

    assert len(plan) == 4
    assert all("摘要" in step and "不下载" in step for step in plan)
    assert all("全文精读" not in step for step in plan)


def test_fast_gap_step_limits_steps_and_tool_schema():
    class Registry:
        def __init__(self, names):
            self.names = tuple(names)

        def subset(self, names):
            return Registry(names)

    class RecordingAgent:
        def __init__(self):
            self.max_steps = 15
            self.tools = Registry(("arxiv_search", "download_paper", "read_paper", "recall"))
            self.seen = None

        def run(self, prompt):
            self.seen = (self.max_steps, self.tools.names)
            return "摘要级结果"

    agent = RecordingAgent()
    planner = Planner(llm=None, agent=agent)

    planner._execute_step(
        FAST_GAP_TASK,
        build_gap_survey_plan(FAST_GAP_TASK, depth="fast"),
        [],
        1,
        "",
    )

    assert agent.seen == (6, ("arxiv_search", "recall"))
    assert agent.max_steps == 15
    assert agent.tools.names == ("arxiv_search", "download_paper", "read_paper", "recall")


def test_gap_task_routes_to_plan_without_waiting_for_model_planning():
    decision = RuleRouter().route(GAP_TASK)

    assert decision.mode == "plan"
    assert "四个方向" in decision.reason


def test_planner_uses_deterministic_gap_plan_instead_of_generic_plan_call():
    planner = Planner(llm=None, agent=None)

    plan = planner._make_plan(GAP_TASK)

    assert plan == build_gap_survey_plan(GAP_TASK)


def test_auto_gap_run_executes_all_four_tracks_before_synthesis(tmp_path):
    replies = []
    for index in range(4):
        replies.extend([
            {"content": f"方向 {index + 1} 核心问题、核心方法、实验结论、局限", "tool_calls": []},
            {"content": '{"ok": true}', "tool_calls": []},
        ])
    replies.append({
        "content": "\n".join(
            f"{track.title}：核心问题；核心方法；实验与结论；局限性。"
            for track in GAP_TRACKS
        ),
        "tool_calls": [],
    })
    runtime = create_runtime(
        llm=ScriptedLLM(replies),
        workspace=tmp_path,
        conversation=False,
        auto_recall=False,
    )

    result = runtime.run(DEEP_GAP_TASK, mode="auto")

    assert result.status == "completed"
    assert result.mode == "plan"
    assert result.metrics.llm_calls == 9
    assert [
        event["type"] for event in result.events if event["type"] == "mode_selected"
    ] == ["mode_selected"]


def test_fast_auto_gap_run_skips_four_reflection_calls(tmp_path):
    structured = "\n".join(
        f"{track.title}：核心问题；核心方法；实验与结论；局限性。摘要级证据。"
        for track in GAP_TRACKS
    )
    replies = [
        {"content": f"方向 {index + 1} 摘要级核心问题、核心方法、实验结论、局限", "tool_calls": []}
        for index in range(4)
    ]
    replies.append({"content": structured, "tool_calls": []})
    runtime = create_runtime(
        llm=ScriptedLLM(replies),
        workspace=tmp_path,
        conversation=False,
        auto_recall=False,
    )

    result = runtime.run(FAST_GAP_TASK, mode="auto")

    assert result.status == "completed"
    assert result.mode == "plan"
    assert result.metrics.llm_calls == 5


def test_gap_task_has_an_explicit_workflow_contract(tmp_path):
    selection = WorkflowRegistry.default().select(GAP_TASK)

    assert selection.name == "gap-survey"
    assert "gap-matrix" in selection.spec.outputs

    runtime = create_runtime(
        llm=ScriptedLLM([{"content": "不会执行", "tool_calls": []}]),
        workspace=tmp_path,
        conversation=False,
        auto_recall=False,
    )
    result = runtime.run(GAP_TASK, mode="react")
    assert result.workflow == "gap-survey"


def test_gap_synthesis_prompt_requires_core_cards_and_keeps_longer_handoffs():
    structured = "\n".join(
        f"{track.title}：核心问题；核心方法；实验与结论；局限性。"
        for track in GAP_TRACKS
    )
    llm = ScriptedLLM([{"content": structured, "tool_calls": []}])
    planner = Planner(llm=llm, agent=None)
    results = ["核心问题 方法 实验 结论 局限\n" + "x" * 4000 for _ in GAP_TRACKS]

    planner._synthesize(DEEP_GAP_TASK, build_gap_survey_plan(DEEP_GAP_TASK), results)

    system_prompt = llm.last_messages[0]["content"]
    user_prompt = llm.last_messages[1]["content"]
    assert "四路资料缺口补全" in system_prompt
    assert "核心卡片" in system_prompt
    # 汇总材料是保头保尾(60/30)截断:头部说明和结尾的长段证据都必须
    # 保留,中间允许被压缩;连续保留量必须显著大于普通任务的预算,
    # 这正是本测试守护的"gap 任务要走更长交接"的回归点。
    assert "核心问题 方法 实验 结论 局限" in user_prompt
    assert "x" * 1500 in user_prompt
    assert user_prompt.strip().endswith("x")


def test_gap_synthesis_repairs_an_answer_that_is_still_too_generic():
    incomplete = "目前资料仍有不足，建议继续关注四个方向。"
    repaired = "\n".join(
        f"{track.title}：核心问题；核心方法；实验与结论；局限性。"
        for track in GAP_TRACKS
    )
    llm = ScriptedLLM([
        {"content": incomplete, "tool_calls": []},
        {"content": repaired, "tool_calls": []},
    ])
    planner = Planner(llm=llm, agent=None)

    answer = planner._synthesize(
        DEEP_GAP_TASK,
        build_gap_survey_plan(DEEP_GAP_TASK),
        ["各方向已有论文核心证据" for _ in GAP_TRACKS],
    )

    assert answer == repaired
    assert "结构审查" in llm.last_messages[0]["content"]
    assert missing_synthesis_sections(answer) == []
