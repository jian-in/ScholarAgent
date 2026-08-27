"""ScholarAgent 本地工作台：静态网页与仅本机可访问的 JSON API。"""

import argparse
import inspect
import json
import mimetypes
import os
import sys
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter, time
from urllib.parse import urlparse

from scholaragent import config
from scholaragent.artifacts import ArtifactCollector
from scholaragent.events import RunEvent, event_message
from scholaragent.markdown_lite import render_markdown
from scholaragent.ocr import default_ocr
from scholaragent.llm import LLMClient
from scholaragent.routing import AdaptiveRunner, CostAwareRouter
from scholaragent.replay import SavedCaseStore
from scholaragent.runtime import (
    build_agent,
    build_runners,
    detect_ollama,
    execute_runners,
    list_ollama_models,
)
from scholaragent.workspace import TemporaryWorkspace, default_workspace


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
if not WEB_DIR.is_dir():
    # setuptools 的 data-files 会把源码 Web 目录安装到环境前缀下；源码运行
    # 仍优先使用仓库内的 web/，安装后的 console script 则走这里。
    for candidate in (Path(sys.prefix) / "web", Path(sys.prefix) / "share" / "scholaragent" / "web"):
        if candidate.is_dir():
            WEB_DIR = candidate
            break
VALID_MODES = ("react", "plan", "team", "auto")
# 完成/失败/取消的任务保留一段时间,方便前端最后一次拉取完整日志
JOB_TTL_SECONDS = 600
MAX_LOG_CHARS = 500
# 前端超时提示阈值(秒);不强制杀任务,只提示用户可取消
SLOW_HINT_SECONDS = 90
# 后端软超时(秒):协作式请求停止,下一步边界生效
DEFAULT_SOFT_TIMEOUT_SECONDS = getattr(config, "JOB_SOFT_TIMEOUT_SECONDS", 900)
MODEL_ROUTING_MODES = ("single", "split")


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

    def create(self, task: str, mode: str, model_routing: str = "single") -> str:
        job_id = uuid.uuid4().hex
        now = time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "task": task,
                "requested_mode": mode,
                "model_routing": model_routing,
                "status": "queued",
                "cancel_requested": False,
                "logs": [],
                "events": [],
                "answer": None,
                "answer_html": None,
                "artifacts": None,
                "error": None,
                "mode": None,
                "routing": None,
                "metrics": None,
                "workflow": None,
                "source_format": None,
                "evidence": None,
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

    def append_event(self, job_id: str, event: RunEvent | dict):
        """保存结构化事件，并生成旧前端可读的 message 投影。"""
        data = event.to_dict() if isinstance(event, RunEvent) else dict(event)
        message = event_message(event) if isinstance(event, RunEvent) else str(
            data.get("message") or data.get("type") or ""
        )
        entry = {
            "ts": data.get("timestamp") or round(time(), 3),
            "message": _clip_log(message),
            "event": data,
        }
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["logs"].append(entry)
            job["events"].append(data)
            if len(job["logs"]) > 500:
                job["logs"] = job["logs"][-400:]
                job["events"] = job["events"][-400:]
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
            job["workflow"] = result.get("workflow")
            job["source_format"] = result.get("source_format")
            job["evidence"] = result.get("evidence")
            job["model_routing"] = result.get(
                "model_routing", job.get("model_routing", "single")
            )
            job["seconds"] = result.get("seconds")
            job["updated_at"] = time()

    def fail(self, job_id: str, error: str, result: dict | None = None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = error
            if result:
                job["answer"] = result.get("answer")
                job["answer_html"] = result.get("answer_html")
                job["artifacts"] = result.get("artifacts")
                job["mode"] = result.get("mode")
                job["routing"] = result.get("routing")
                job["metrics"] = result.get("metrics")
                job["workflow"] = result.get("workflow")
                job["source_format"] = result.get("source_format")
                job["evidence"] = result.get("evidence")
                job["model_routing"] = result.get(
                    "model_routing", job.get("model_routing", "single")
                )
                job["seconds"] = result.get("seconds")
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
                "model_routing": job.get("model_routing", "single"),
                "mode": job["mode"],
                "routing": job["routing"],
                "metrics": job["metrics"],
                "workflow": job["workflow"],
                "source_format": job["source_format"],
                "evidence": job["evidence"],
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
                "events": list(job.get("events", [])[after:]),
                # 旧 logs 与结构化 events 数量不一定相同；给新前端一份
                # 独立全量事件投影，避免复用 logs 游标时漏掉时间线节点。
                "all_events": list(job.get("events", [])),
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

    def __init__(self, saved_cases=None):
        self._model_lock = threading.Lock()
        self.jobs = JobStore()
        self.saved_cases = saved_cases or SavedCaseStore()
        self._cloud_profile = self._capture_cloud_profile()
        configured_routing = getattr(config, "MODEL_ROUTING_MODE", "split")
        self._routing_mode = (
            configured_routing if configured_routing in MODEL_ROUTING_MODES else "split"
        )
        self._local_model_preference = (
            config.LLM_MODEL if self._is_ollama_url(config.LLM_BASE_URL) else ""
        )

    @staticmethod
    def _is_ollama_url(value):
        normalized = str(value or "").lower().rstrip("/")
        return normalized in {
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
        }

    @classmethod
    def _capture_cloud_profile(cls):
        base_url = str(getattr(config, "CLOUD_LLM_BASE_URL", "") or "").strip()
        model = str(getattr(config, "CLOUD_LLM_MODEL", "") or "").strip()
        api_key = str(getattr(config, "CLOUD_LLM_API_KEY", "") or "").strip()
        if not base_url and not cls._is_ollama_url(config.LLM_BASE_URL):
            base_url = str(config.LLM_BASE_URL or "").strip()
        if not model and not cls._is_ollama_url(config.LLM_BASE_URL):
            model = str(config.LLM_MODEL or "").strip()
        if not api_key:
            current_key = str(config.LLM_API_KEY or "").strip()
            if current_key.lower() != "ollama":
                api_key = current_key
        if api_key.lower() == "ollama":
            api_key = ""
        return {"base_url": base_url, "model": model, "api_key": api_key}

    def _local_profile(self, models=None):
        """选出本机总结模型；只返回内部使用的连接配置。"""
        # 如果当前配置已经明确指向 Ollama，或用户刚刚验证并选择过本地
        # 模型，就直接复用该名字。任务启动不应为了再次确认同一个模型而
        # 阻塞在 Ollama 的 HTTP 探测上；模型真正不可用时由调用结果如实报错。
        preferred = self._local_model_preference
        if not preferred and self._is_ollama_url(config.LLM_BASE_URL):
            preferred = str(config.LLM_MODEL or "").strip()
        if preferred:
            return {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "model": preferred,
                "provider": "ollama",
            }

        models = list_ollama_models() if models is None else models
        candidates = [
            item for item in models
            if item.get("name") and item.get("supports_tools", True)
        ]
        if not candidates:
            return None
        selected = next(
            (item for item in candidates if item.get("name") == preferred),
            candidates[0],
        )
        return {
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "model": str(selected["name"]),
            "provider": "ollama",
        }

    @staticmethod
    def _public_profile(profile):
        return {
            "provider": str(profile.get("provider") or "unknown"),
            "model": str(profile.get("model") or ""),
        }

    def _resolve_model_routing(self, requested=None, strict=False, local_models=None):
        """解析一次任务的模型分工；内部 profile 永远不会返回给网页。"""
        desired = requested or self._routing_mode
        if desired not in MODEL_ROUTING_MODES:
            raise ValueError("模型分工只支持 single 或 split。")

        local = self._local_profile(local_models)
        cloud = self._cloud_profile
        cloud_ready = all(cloud.values())
        if desired == "split" and cloud_ready and local:
            research = {**cloud, "provider": "cloud", "role": "research"}
            summary = {**local, "role": "summary"}
            return {
                "mode": "split",
                "requested_mode": desired,
                "reason": "云端负责外部检索与工具决策，本地 Ollama 负责证据摘要、反思和最终汇总。",
                "roles": {
                    "research": self._public_profile(research),
                    "summary": self._public_profile(summary),
                },
                "profiles": {"research": research, "summary": summary},
            }

        if desired == "split" and strict:
            missing = []
            if not cloud_ready:
                missing.append("云端模型档案")
            if not local:
                missing.append("本地 Ollama 聊天模型")
            raise ValueError("双模型分工不可用，缺少：" + "、".join(missing))

        # 降级或显式单模型都复用当前 LLM_* 配置；运行前由 _ensure_model
        # 负责在没有 Key 时自动探测 Ollama。
        provider = "ollama" if self._is_ollama_url(config.LLM_BASE_URL) else "cloud"
        single = {
            "base_url": config.LLM_BASE_URL,
            "api_key": config.LLM_API_KEY,
            "model": config.LLM_MODEL,
            "provider": provider,
            "role": "single",
        }
        reason = (
            "当前使用单模型。"
            if desired == "single"
            else "双模型分工暂不可用，已降级为当前单模型；请检查云端档案和 Ollama。"
        )
        return {
            "mode": "single",
            "requested_mode": desired,
            "reason": reason,
            "roles": {
                "research": self._public_profile(single),
                "summary": self._public_profile(single),
            },
            "profiles": {"research": single, "summary": single},
        }

    @staticmethod
    def _public_routing(spec):
        return {
            "mode": spec["mode"],
            "requested_mode": spec.get("requested_mode", spec["mode"]),
            "reason": spec["reason"],
            "roles": {
                role: dict(profile)
                for role, profile in spec["roles"].items()
            },
        }

    def model_routing(self, local_models=None):
        """返回网页显示的当前分工，不包含任何连接密钥。"""
        with self._model_lock:
            spec = self._resolve_model_routing(local_models=local_models)
            return {
                **self._public_routing(spec),
                "available": spec["mode"] == "split",
                "soft_timeout_seconds": DEFAULT_SOFT_TIMEOUT_SECONDS,
            }

    def select_model_routing(self, mode):
        """切换后续新任务的模型分工。"""
        mode = str(mode or "").strip().lower()
        if mode not in MODEL_ROUTING_MODES:
            raise ValueError("模型分工只支持 single 或 split。")
        with self._model_lock:
            # 显式开启 split 时必须两端都准备好，避免用户以为已经分工。
            self._resolve_model_routing(mode, strict=(mode == "split"))
            self._routing_mode = mode
        return self.model_catalog()

    @staticmethod
    def _make_llm(profile):
        return LLMClient(
            base_url=profile["base_url"],
            api_key=profile["api_key"],
            model=profile["model"],
            provider=profile["provider"],
            role=profile.get("role", "general"),
        )

    def _prepare_execution_routing(self, requested=None):
        """为新任务绑定模型客户端；绑定后切换不会影响该任务。"""
        desired = requested or self._routing_mode
        if desired == "single":
            self._ensure_model()
        spec = self._resolve_model_routing(
            desired,
            strict=requested is not None,
        )
        if desired == "split" and spec["mode"] == "single":
            # 默认 split 缺少配置时允许兼容运行，但先确保单模型真的可用。
            self._ensure_model()
            spec = self._resolve_model_routing("single")
        return spec

    def list_saved_cases(self):
        return self.saved_cases.list_cases()

    def replay_case(self, case_id, run_id=None):
        return self.saved_cases.get_case(case_id, run_id=run_id)

    def status(self):
        policy_path = default_workspace().router_policy_path
        return {
            "model": config.LLM_MODEL,
            "provider": "ollama" if self._is_ollama_url(config.LLM_BASE_URL) else "cloud",
            "api_key_configured": bool(config.LLM_API_KEY),
            "policy_available": policy_path.exists(),
            "policy_path": str(policy_path),
            "ocr": default_ocr().diagnostics(),
            "model_routing": self.model_routing(),
        }

    def model_catalog(self):
        """返回网页模型选择器所需的安全目录，不包含任何密钥。"""
        local_models = list_ollama_models()
        options = []
        for item in local_models:
            options.append({
                **item,
                "id": f"ollama|{item['name']}",
                "provider": "ollama",
                "label": f"本地 · {item['name']}",
            })
        cloud = self._cloud_profile
        cloud_ready = all(cloud.values())
        if cloud_ready:
            options.append({
                "id": f"cloud|{cloud['model']}",
                "provider": "cloud",
                "model": cloud["model"],
                "label": f"云端 · {cloud['model']}",
                "supports_tools": True,
            })
        current_provider = "ollama" if self._is_ollama_url(config.LLM_BASE_URL) else "cloud"
        return {
            "current": {
                "id": f"{current_provider}|{config.LLM_MODEL}",
                "provider": current_provider,
                "model": config.LLM_MODEL,
            },
            "options": options,
            "ollama_available": bool(local_models),
            "cloud_configured": cloud_ready,
            "routing": self.model_routing(local_models=local_models),
        }

    def select_model(self, provider, model=None):
        """切换后续新任务使用的模型；正在运行的任务继续使用其已创建的客户端。"""
        provider = str(provider or "").strip().lower()
        model = str(model or "").strip()
        if provider == "ollama":
            candidates = {item["name"]: item for item in list_ollama_models()}
            selected = candidates.get(model)
            if selected is None:
                raise ValueError("本机 Ollama 中没有这个模型，或 Ollama 当前不可用。")
            if not selected.get("supports_tools", True):
                raise ValueError("该模型不声明工具调用能力，不能用于 ScholarAgent 调研任务。")
            with self._model_lock:
                config.LLM_BASE_URL = "http://127.0.0.1:11434/v1"
                config.LLM_API_KEY = "ollama"
                config.LLM_MODEL = model
                self._local_model_preference = model
                self._routing_mode = "single"
        elif provider == "cloud":
            cloud = self._cloud_profile
            if not all(cloud.values()):
                raise ValueError("云端模型档案未配置，请在 .env 中配置 CLOUD_LLM_BASE_URL、CLOUD_LLM_MODEL 和密钥。")
            with self._model_lock:
                config.LLM_BASE_URL = cloud["base_url"]
                config.LLM_API_KEY = cloud["api_key"]
                config.LLM_MODEL = cloud["model"]
                self._routing_mode = "single"
        else:
            raise ValueError("未知模型提供方，只支持 ollama 或 cloud。")
        return self.model_catalog()

    def _ensure_model(self):
        if config.LLM_API_KEY:
            return
        local_model = detect_ollama(prefer=config.LLM_MODEL)
        if not local_model:
            raise RuntimeError("没有可用模型：请配置 .env 的 LLM_API_KEY，或启动 Ollama。")
        config.LLM_BASE_URL = "http://localhost:11434/v1"
        config.LLM_API_KEY = "ollama"
        config.LLM_MODEL = local_model

    def run(self, task: str, mode: str, on_progress=None, should_stop=None,
            event_sink=None, model_routing=None):
        from scholaragent.agent import CANCELLED_ANSWER

        task = task.strip()
        if not task:
            raise ValueError("任务不能为空。")
        if mode not in VALID_MODES:
            raise ValueError("未知执行模式。")

        run_id = uuid.uuid4().hex
        workspace = TemporaryWorkspace(default_workspace().runs_dir / run_id)
        artifacts = ArtifactCollector(workspace)
        # 同一进程内串行创建客户端，避免模型切换与新任务初始化互相覆盖。
        with self._model_lock:
            routing_spec = self._prepare_execution_routing(model_routing)
            profiles = routing_spec["profiles"]
            research_llm = self._make_llm(profiles["research"])
            summary_llm = (
                research_llm
                if routing_spec["mode"] == "single"
                else self._make_llm(profiles["summary"])
            )
            agent_kwargs = {
                "demo": False,
                "artifacts": artifacts,
                "workspace": workspace,
                "research_llm": research_llm,
                "summary_llm": summary_llm,
            }
            try:
                parameters = inspect.signature(build_agent).parameters
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in parameters.values()):
                    agent_kwargs = {key: value for key, value in agent_kwargs.items()
                                    if key in parameters}
            except (TypeError, ValueError):
                pass
            agent = build_agent(**agent_kwargs)
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

        if on_progress:
            public_routing = self._public_routing(routing_spec)
            research = public_routing["roles"]["research"]
            summary = public_routing["roles"]["summary"]
            on_progress(
                "[工作台] 模型分工 · "
                f"调研 {research['provider']}/{research['model']} · "
                f"总结 {summary['provider']}/{summary['model']}"
            )
            if routing_spec["mode"] != routing_spec.get("requested_mode"):
                on_progress(f"[工作台] {routing_spec['reason']}")
            on_progress(f"[工作台] 开始执行 · 模式 {mode.upper()}")
        result = execute_runners(
            task,
            mode,
            runners,
            CostAwareRouter(str(workspace.router_policy_path)),
            workspace,
            artifacts,
            run_id=run_id,
            event_sink=event_sink,
            on_progress=on_progress,
            should_stop=should_stop,
        )
        result_dict = result.to_dict()
        result_dict["model_routing"] = self._public_routing(routing_spec)
        answer = result.answer
        seconds = result.metrics.seconds
        selected_mode = result.mode
        cancelled = result.status == "cancelled"
        if on_progress:
            if cancelled:
                on_progress(f"[工作台] 任务已取消 · 已运行 {seconds:.1f}s")
            elif result.status == "failed":
                on_progress(f"[工作台] 执行失败 · {result.error}")
            else:
                on_progress(f"[工作台] 执行完成 · 实际模式 {selected_mode.upper()}")
            counts = result_dict["artifacts"]["counts"]
            if any(counts.values()):
                on_progress(
                    "[产物] "
                    f"论文 {counts['papers']} · 阅读 {counts['read']} · "
                    f"笔记 {counts['notes']} · 记忆 {counts['memories']}"
                )
        result_dict["answer_html"] = render_markdown(str(answer or ""))
        return result_dict

    def start_job(self, task: str, mode: str, model_routing=None) -> str:
        task = task.strip()
        if not task:
            raise ValueError("任务不能为空。")
        if mode not in VALID_MODES:
            raise ValueError("未知执行模式。")
        self.jobs.cleanup()
        # 排队接口不做 Ollama/网络探测：Windows 下 HTTP 客户端初始化可能
        # 较慢，不能让“提交任务”本身被模型发现阻塞。只冻结用户当下选择，
        # 真正的 profile 绑定在后台线程中完成。
        with self._model_lock:
            selected_routing = model_routing or self._routing_mode
            # 记住这是用户显式选择还是默认继承:显式选择 split 必须严格
            # 校验(避免用户以为已经分工);默认继承 split 而两端未就绪时,
            # 由 run 按设计降级为单模型并在状态中说明原因。
            user_selected_routing = model_routing is not None
            if selected_routing not in MODEL_ROUTING_MODES:
                raise ValueError("模型分工只支持 single 或 split。")
        job_id = self.jobs.create(task, mode, model_routing=selected_routing)
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
                    task,
                    mode,
                    on_progress=on_progress,
                    should_stop=should_stop,
                    event_sink=lambda event: self.jobs.append_event(job_id, event),
                    model_routing=(
                        selected_routing if user_selected_routing else None),
                )
                if result.get("status") == "failed":
                    self.jobs.fail(job_id, result.get("error") or "执行失败", result=result)
                else:
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
                            "model_routing": {
                                "mode": selected_routing,
                                "roles": {},
                                "reason": "任务在模型调用前被取消。",
                            },
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
        if path == "/api/models":
            self._send_json(HTTPStatus.OK, self.workspace.model_catalog())
            return
        if path == "/api/model-routing":
            self._send_json(HTTPStatus.OK, self.workspace.model_routing())
            return
        if path == "/api/cases":
            self._send_json(HTTPStatus.OK, {"cases": self.workspace.list_saved_cases()})
            return
        if path.startswith("/api/cases/"):
            case_id = path[len("/api/cases/"):].strip("/")
            run_id = None
            for part in parsed.query.split("&"):
                if part.startswith("run_id="):
                    run_id = part.split("=", 1)[1]
            try:
                self._send_json(HTTPStatus.OK, self.workspace.replay_case(case_id, run_id))
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
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
        if path == "/api/model":
            try:
                payload = self._read_json_body()
                result = self.workspace.select_model(
                    payload.get("provider"), payload.get("model")
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
            return
        if path == "/api/model-routing":
            try:
                payload = self._read_json_body()
                result = self.workspace.select_model_routing(payload.get("mode"))
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
            return
        if path == "/api/jobs":
            try:
                payload = self._read_json_body()
                job_id = self.workspace.start_job(
                    str(payload.get("task", "")),
                    str(payload.get("mode", "auto")),
                    payload.get("model_routing"),
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
