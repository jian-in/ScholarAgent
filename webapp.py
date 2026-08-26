"""ScholarAgent 本地工作台：静态网页与仅本机可访问的 JSON API。"""

import argparse
import json
import mimetypes
import os
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter, time
from urllib.parse import urlparse

from main import build_agent, build_runners, detect_ollama
from scholaragent import config
from scholaragent.artifacts import ArtifactCollector
from scholaragent.markdown_lite import render_markdown
from scholaragent.routing import AdaptiveRunner, CostAwareRouter


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
VALID_MODES = ("react", "plan", "team", "auto")
# 完成/失败/取消的任务保留一段时间,方便前端最后一次拉取完整日志
JOB_TTL_SECONDS = 600
MAX_LOG_CHARS = 500
# 前端超时提示阈值(秒);不强制杀任务,只提示用户可取消
SLOW_HINT_SECONDS = 90
# 后端软超时(秒):协作式请求停止,下一步边界生效
DEFAULT_SOFT_TIMEOUT_SECONDS = 900


def _clip_log(text: str, limit: int = MAX_LOG_CHARS) -> str:
    text = str(text).replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…(已截断)"


class JobStore:
    """进程内任务表:支持后台执行 + 进度日志轮询 + 协作式取消。"""

    TERMINAL = frozenset(("done", "error", "cancelled"))

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}

    def create(self, task: str, mode: str) -> str:
        job_id = uuid.uuid4().hex
        now = time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "task": task,
                "requested_mode": mode,
                "status": "queued",
                "cancel_requested": False,
                "logs": [],
                "answer": None,
                "answer_html": None,
                "artifacts": None,
                "error": None,
                "mode": None,
                "routing": None,
                "metrics": None,
                "seconds": None,
                "elapsed": 0.0,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
            }
        return job_id

    def append_log(self, job_id: str, message: str):
        entry = {
            "ts": round(time(), 3),
            "message": _clip_log(message),
        }
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["logs"].append(entry)
            # 防止极端长任务把内存撑爆
            if len(job["logs"]) > 500:
                job["logs"] = job["logs"][-400:]
            job["updated_at"] = time()
            if job["status"] == "queued":
                job["status"] = "running"

    def mark_running(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                now = time()
                job["status"] = "running"
                job["started_at"] = now
                job["updated_at"] = now

    def request_cancel(self, job_id: str) -> bool:
        """标记取消;返回 False 表示任务不存在或已结束。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job["status"] in self.TERMINAL:
                return False
            job["cancel_requested"] = True
            job["updated_at"] = time()
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))

    def finish(self, job_id: str, result: dict, cancelled: bool = False):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            answer = result.get("answer")
            job["status"] = "cancelled" if cancelled else "done"
            job["answer"] = answer
            job["answer_html"] = result.get("answer_html")
            if job["answer_html"] is None and answer is not None:
                job["answer_html"] = render_markdown(str(answer))
            job["artifacts"] = result.get("artifacts")
            job["mode"] = result.get("mode")
            job["routing"] = result.get("routing")
            job["metrics"] = result.get("metrics")
            job["seconds"] = result.get("seconds")
            job["updated_at"] = time()

    def fail(self, job_id: str, error: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = error
            job["updated_at"] = time()

    def snapshot(self, job_id: str, after: int = 0) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            logs = job["logs"]
            after = max(0, int(after))
            now = time()
            started = job["started_at"] or job["created_at"]
            elapsed = round(now - started, 1) if job["status"] not in self.TERMINAL else (
                job["seconds"] if job["seconds"] is not None else round(job["updated_at"] - started, 1)
            )
            return {
                "id": job["id"],
                "status": job["status"],
                "task": job["task"],
                "requested_mode": job["requested_mode"],
                "mode": job["mode"],
                "routing": job["routing"],
                "metrics": job["metrics"],
                "seconds": job["seconds"],
                "elapsed": elapsed,
                "cancel_requested": job["cancel_requested"],
                "slow": (
                    job["status"] not in self.TERMINAL
                    and elapsed >= SLOW_HINT_SECONDS
                ),
                "answer": job["answer"],
                "answer_html": job.get("answer_html"),
                "artifacts": job.get("artifacts"),
                "error": job["error"],
                "log_count": len(logs),
                "logs": list(logs[after:]),
                "next_after": len(logs),
            }

    def cleanup(self):
        cutoff = time() - JOB_TTL_SECONDS
        with self._lock:
            stale = [
                job_id for job_id, job in self._jobs.items()
                if job["status"] in self.TERMINAL and job["updated_at"] < cutoff
            ]
            for job_id in stale:
                del self._jobs[job_id]


class LocalWorkspace:
    """复用既有执行器，不在网页层复制 Agent、Planner 或 Team 流程。"""

    def __init__(self):
        self._model_lock = threading.Lock()
        self.jobs = JobStore()

    def status(self):
        return {
            "model": config.LLM_MODEL,
            "api_key_configured": bool(config.LLM_API_KEY),
            "policy_available": os.path.exists(config.ROUTER_POLICY_PATH),
            "policy_path": config.ROUTER_POLICY_PATH,
        }

    def _ensure_model(self):
        if config.LLM_API_KEY:
            return
        local_model = detect_ollama(prefer=config.LLM_MODEL)
        if not local_model:
            raise RuntimeError("没有可用模型：请配置 .env 的 LLM_API_KEY，或启动 Ollama。")
        config.LLM_BASE_URL = "http://localhost:11434/v1"
        config.LLM_API_KEY = "ollama"
        config.LLM_MODEL = local_model

    def run(self, task: str, mode: str, on_progress=None, should_stop=None):
        from scholaragent.agent import CANCELLED_ANSWER

        task = task.strip()
        if not task:
            raise ValueError("任务不能为空。")
        if mode not in VALID_MODES:
            raise ValueError("未知执行模式。")

        artifacts = ArtifactCollector()
        # 同一进程内复用全局模型配置；串行化配置初始化，避免并发请求相互覆盖。
        with self._model_lock:
            self._ensure_model()
            agent = build_agent(demo=False, artifacts=artifacts)
            # 工作台默认静音终端 print,进度只走 on_progress
            if hasattr(agent, "verbose"):
                agent.verbose = False
            runners = build_runners(
                agent,
                on_progress=on_progress,
                should_stop=should_stop,
                artifacts=artifacts,
            )
            for runner in runners.values():
                if hasattr(runner, "verbose"):
                    runner.verbose = False
            if mode == "auto":
                runner = AdaptiveRunner(
                    CostAwareRouter(config.ROUTER_POLICY_PATH), runners
                )
            else:
                runner = runners[mode]

        if on_progress:
            on_progress(f"[工作台] 开始执行 · 模式 {mode.upper()}")
        started_at = perf_counter()
        answer = runner.run(task)
        seconds = perf_counter() - started_at
        decision = getattr(runner, "last_decision", None)
        selected_mode = decision.mode if decision else mode
        metrics_owner = runners.get(selected_mode)
        metrics = getattr(metrics_owner, "last_metrics", None)
        cancelled = (
            answer == CANCELLED_ANSWER
            or (should_stop is not None and should_stop())
        )
        artifact_summary = artifacts.to_dict()
        if on_progress:
            if cancelled:
                on_progress(f"[工作台] 任务已取消 · 已运行 {seconds:.1f}s")
            else:
                on_progress(f"[工作台] 执行完成 · 实际模式 {selected_mode.upper()}")
            counts = artifact_summary["counts"]
            if any(counts.values()):
                on_progress(
                    "[产物] "
                    f"论文 {counts['papers']} · 阅读 {counts['read']} · "
                    f"笔记 {counts['notes']} · 记忆 {counts['memories']}"
                )
        return {
            "answer": answer,
            "answer_html": render_markdown(str(answer or "")),
            "artifacts": artifact_summary,
            "seconds": round(seconds, 3),
            "mode": selected_mode,
            "routing": decision.to_dict() if decision else None,
            "metrics": metrics.to_dict() if metrics else None,
            "cancelled": cancelled,
        }

    def start_job(self, task: str, mode: str) -> str:
        task = task.strip()
        if not task:
            raise ValueError("任务不能为空。")
        if mode not in VALID_MODES:
            raise ValueError("未知执行模式。")
        self.jobs.cleanup()
        job_id = self.jobs.create(task, mode)
        soft_timeout = DEFAULT_SOFT_TIMEOUT_SECONDS
        timeout_notified = {"done": False}

        def worker():
            self.jobs.mark_running(job_id)
            started = time()

            def on_progress(message: str):
                self.jobs.append_log(job_id, message)

            def should_stop():
                if self.jobs.is_cancel_requested(job_id):
                    return True
                elapsed = time() - started
                if elapsed >= soft_timeout:
                    if not timeout_notified["done"]:
                        timeout_notified["done"] = True
                        self.jobs.request_cancel(job_id)
                        on_progress(
                            f"[超时] 已运行 {int(elapsed)}s，超过软限制 "
                            f"{soft_timeout}s，正在协作式停止"
                        )
                    return True
                return False

            try:
                result = self.run(
                    task, mode, on_progress=on_progress, should_stop=should_stop
                )
                self.jobs.finish(
                    job_id, result, cancelled=bool(result.get("cancelled"))
                )
            except Exception as exc:
                if self.jobs.is_cancel_requested(job_id):
                    self.jobs.finish(
                        job_id,
                        {
                            "answer": "(任务已取消)",
                            "seconds": round(time() - started, 3),
                            "mode": mode,
                            "routing": None,
                            "metrics": None,
                        },
                        cancelled=True,
                    )
                else:
                    self.jobs.fail(job_id, f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=worker, daemon=True, name=f"job-{job_id[:8]}")
        thread.start()
        return job_id

    def cancel_job(self, job_id: str) -> dict:
        ok = self.jobs.request_cancel(job_id)
        if not ok:
            snap = self.jobs.snapshot(job_id)
            if snap is None:
                raise KeyError("任务不存在或已过期。")
            return {
                "ok": False,
                "status": snap["status"],
                "message": "任务已结束，无法取消。",
            }
        self.jobs.append_log(job_id, "[取消] 已收到停止请求，将在下一步边界生效")
        return {
            "ok": True,
            "status": "cancelling",
            "message": "已请求取消，将在当前步骤结束后停止。",
        }

    def job_status(self, job_id: str, after: int = 0):
        snap = self.jobs.snapshot(job_id, after=after)
        if snap is None:
            raise KeyError("任务不存在或已过期。")
        return snap


class WorkspaceRequestHandler(BaseHTTPRequestHandler):
    workspace = LocalWorkspace()

    def log_message(self, format, *args):
        # 不把每个浏览器静态资源请求刷进用户终端。
        return

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 65536:
            raise ValueError("请求内容无效或过大。")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._send_json(HTTPStatus.OK, self.workspace.status())
            return
        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/"):].strip("/")
            if not job_id or "/" in job_id:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "任务不存在。"})
                return
            after = 0
            if parsed.query:
                for part in parsed.query.split("&"):
                    if part.startswith("after="):
                        try:
                            after = int(part.split("=", 1)[1])
                        except ValueError:
                            after = 0
            try:
                self._send_json(HTTPStatus.OK, self.workspace.job_status(job_id, after=after))
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return

        requested = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB_DIR / requested).resolve()
        if WEB_DIR not in target.parents and target != WEB_DIR:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/jobs":
            try:
                payload = self._read_json_body()
                job_id = self.workspace.start_job(
                    str(payload.get("task", "")),
                    str(payload.get("mode", "auto")),
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, {"job_id": job_id})
            return

        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path[len("/api/jobs/"):-len("/cancel")].strip("/")
            if not job_id or "/" in job_id:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "任务不存在。"})
                return
            try:
                # cancel 允许空 body
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length > 0:
                    self.rfile.read(length)
                result = self.workspace.cancel_job(job_id)
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return

        # 兼容旧同步接口(测试与脚本仍可用)
        if path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json_body()
            result = self.workspace.run(
                str(payload.get("task", "")), str(payload.get("mode", "auto"))
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._send_json(HTTPStatus.OK, result)


def create_server(port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), WorkspaceRequestHandler)


def main():
    parser = argparse.ArgumentParser(description="启动 ScholarAgent 本地工作台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        help="启动后自动用系统默认浏览器打开工作台",
    )
    args = parser.parse_args()
    try:
        server = create_server(args.port)
    except OSError as exc:
        print(
            f"无法监听 127.0.0.1:{args.port}（{exc}）。"
            "端口可能被占用，可换端口：python webapp.py --port 8766 --open"
        )
        raise SystemExit(1) from exc
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"ScholarAgent 本地工作台：{url}")
    print("仅监听本机回环地址；按 Ctrl+C 停止。")
    if args.open:
        import webbrowser

        # 稍后再开浏览器,避免服务还没 bind 好就连不上;daemon 以免拖住退出
        timer = threading.Timer(0.6, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
