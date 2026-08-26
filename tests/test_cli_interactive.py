"""CLI 交互模式解析与执行委派的离线测试。"""

import main


def test_parse_cli_flags_modes():
    demo, mode, args = main.parse_cli_flags(["--plan", "复杂任务"])
    assert demo is False
    assert mode == "plan"
    assert args == ["复杂任务"]

    demo, mode, args = main.parse_cli_flags(["--team", "主题"])
    assert mode == "team"
    assert args == ["主题"]

    demo, mode, args = main.parse_cli_flags(["--auto", "综述"])
    assert mode == "auto"

    demo, mode, args = main.parse_cli_flags(["普通任务"])
    assert mode == "react"
    assert args == ["普通任务"]


def test_parse_cli_flags_rejects_conflicts():
    try:
        main.parse_cli_flags(["--plan", "--team", "x"])
        raise AssertionError("应拒绝多模式")
    except ValueError as exc:
        assert "不能同时" in str(exc)

    try:
        main.parse_cli_flags(["--demo", "--plan"])
        raise AssertionError("demo 不应带 plan")
    except ValueError as exc:
        assert "demo" in str(exc).lower() or "演示" in str(exc)


def test_parse_interactive_line_commands():
    assert main.parse_interactive_line("q", "react")["kind"] == "quit"
    assert main.parse_interactive_line("/help", "react")["kind"] == "help"
    assert main.parse_interactive_line("/mode", "plan")["mode"] == "plan"

    switched = main.parse_interactive_line("/team", "react")
    assert switched["kind"] == "switch"
    assert switched["mode"] == "team"

    tasked = main.parse_interactive_line("/plan 调研 ReAct", "react")
    assert tasked["kind"] == "task"
    assert tasked["mode"] == "plan"
    assert tasked["task"] == "调研 ReAct"
    assert tasked["switched"] is True

    plain = main.parse_interactive_line("现在几点", "auto")
    assert plain["kind"] == "task"
    assert plain["mode"] == "auto"
    assert plain["task"] == "现在几点"
    assert plain["switched"] is False

    bad = main.parse_interactive_line("/nope", "react")
    assert bad["kind"] == "error"


def test_run_task_with_mode_delegates(monkeypatch):
    calls = []

    class FakeRunner:
        def __init__(self, name):
            self.name = name

        def run(self, task):
            calls.append((self.name, task))
            return f"from-{self.name}"

    def fake_build_runners(agent, on_progress=None, should_stop=None, artifacts=None):
        return {
            "react": FakeRunner("react"),
            "plan": FakeRunner("plan"),
            "team": FakeRunner("team"),
        }

    class FakeAdaptive:
        def __init__(self, router, runners):
            self.runners = runners
            self.last_decision = type(
                "D",
                (),
                {
                    "mode": "plan",
                    "reason": "测试",
                    "predicted_utility": {"react": 0.1, "plan": 0.9, "team": 0.2},
                },
            )()

        def run(self, task):
            calls.append(("auto", task))
            return "from-auto"

    monkeypatch.setattr(main, "build_runners", fake_build_runners)
    monkeypatch.setattr(main, "AdaptiveRunner", FakeAdaptive)
    monkeypatch.setattr(main, "CostAwareRouter", lambda path: object())

    agent = object()
    assert main.run_task_with_mode(agent, "plan", "任务A") == ("from-plan", "plan")
    assert main.run_task_with_mode(agent, "team", "任务B") == ("from-team", "team")
    assert main.run_task_with_mode(agent, "auto", "任务C") == ("from-auto", "plan")
    assert ("plan", "任务A") in calls
    assert ("team", "任务B") in calls
    assert ("auto", "任务C") in calls


def test_print_mode_answer_skips_react(capsys):
    main._print_mode_answer("react", "react", "只应被 agent 自己打印")
    main._print_mode_answer("auto", "react", "auto 落到 react 也不重复")
    main._print_mode_answer("plan", "plan", "计划答案")
    main._print_mode_answer("auto", "team", "团队答案")
    out = capsys.readouterr().out
    assert "只应被 agent 自己打印" not in out
    assert "auto 落到 react 也不重复" not in out
    assert "计划答案" in out
    assert "团队答案" in out


def test_run_task_with_mode_missing_utility(monkeypatch):
    """predicted_utility 缺 key 时不应抛 KeyError。"""

    class FakeAdaptive:
        def __init__(self, router, runners):
            self.last_decision = type(
                "D", (), {"mode": "team", "reason": "x", "predicted_utility": {}}
            )()

        def run(self, task):
            return "ok"

    monkeypatch.setattr(
        main,
        "build_runners",
        lambda *a, **k: {"react": object(), "plan": object(), "team": object()},
    )
    monkeypatch.setattr(main, "AdaptiveRunner", FakeAdaptive)
    monkeypatch.setattr(main, "CostAwareRouter", lambda path: object())
    answer, executed = main.run_task_with_mode(object(), "auto", "t")
    assert answer == "ok"
    assert executed == "team"
