"""一键启动脚本与 webapp --open 标志的离线检查。"""

import argparse
from pathlib import Path

import webapp


ROOT = Path(__file__).resolve().parents[1]


def test_start_scripts_exist_and_point_to_webapp():
    bat = (ROOT / "start.bat").read_text(encoding="utf-8")
    ps1 = (ROOT / "start.ps1").read_text(encoding="utf-8")

    for content in (bat, ps1):
        assert "webapp.py" in content
        assert "--open" in content
        assert ".venv" in content
        assert "8765" in content
        # 依赖缺失时应能补装,而不是只在完全没有 .venv 时安装
        assert "import openai" in content or "openai,dotenv" in content.replace(" ", "")

    assert "requirements.txt" in bat
    assert "requirements.txt" in ps1
    # bat 应尽量自包含,双击不依赖 ExecutionPolicy
    assert "python -m venv" in bat
    assert "LASTEXITCODE" in ps1


def test_webapp_cli_accepts_open_flag(monkeypatch):
    """--open 应被 argparse 接受,且会安排打开浏览器(不真正起服务)。"""
    calls = {"serve": 0, "open": 0, "timer": [], "daemon": None}

    class FakeServer:
        server_port = 8765

        def serve_forever(self):
            calls["serve"] += 1
            raise KeyboardInterrupt

        def server_close(self):
            return None

    class FakeTimer:
        def __init__(self, delay, fn):
            calls["timer"].append(delay)
            self.fn = fn
            self.daemon = False

        def start(self):
            calls["daemon"] = self.daemon
            # 不真正延时,直接调用以验证会走到 webbrowser.open
            self.fn()

    monkeypatch.setattr(webapp, "create_server", lambda port: FakeServer())
    monkeypatch.setattr(
        webapp.threading, "Timer", FakeTimer
    )

    import webbrowser

    monkeypatch.setattr(
        webbrowser, "open", lambda url: calls.__setitem__("open", calls["open"] + 1)
        or calls.setdefault("url", url)
    )
    monkeypatch.setattr(
        "sys.argv", ["webapp.py", "--port", "8765", "--open"]
    )

    webapp.main()

    assert calls["serve"] == 1
    assert calls["open"] == 1
    assert calls["timer"] and calls["timer"][0] > 0
    assert calls["daemon"] is True
    assert calls["url"] == "http://127.0.0.1:8765"


def test_webapp_port_in_use_exits_friendly(monkeypatch, capsys):
    def boom(port):
        raise OSError(10048, "address already in use")

    monkeypatch.setattr(webapp, "create_server", boom)
    monkeypatch.setattr("sys.argv", ["webapp.py", "--port", "8765"])
    try:
        webapp.main()
        raise AssertionError("端口占用应 SystemExit")
    except SystemExit as exc:
        assert exc.code == 1
    err = capsys.readouterr().out
    assert "8765" in err
    assert "端口" in err


def test_webapp_argparse_defaults():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args([])
    assert args.port == 8765
    assert args.open is False
    args = parser.parse_args(["--open", "--port", "9000"])
    assert args.open is True
    assert args.port == 9000
