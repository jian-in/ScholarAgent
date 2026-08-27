# -*- coding: utf-8 -*-
"""M6 评测入口:对指定模式跑一遍评测任务集,输出报告。

用法(项目根目录):
    python evals/run_eval.py --mode react            全部 react 任务
    python evals/run_eval.py --mode plan --limit 3   规划模式只跑前 3 题
    python evals/run_eval.py --mode team             团队模式(耗时较长)

模型选择与 main.py 一致:.env 里的 API Key 优先,否则自动用本机 Ollama。
结果写到 evals/results/report_<模式>_<时间戳>.md(+ 同名 .jsonl 原始数据)。
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scholaragent import config  # noqa: E402
from scholaragent.evaluate import Evaluator, load_tasks, write_report  # noqa: E402
from scholaragent.experiments import build_manifest, utc_now, write_manifest_file  # noqa: E402
from scholaragent.llm import LLMClient  # noqa: E402
from scholaragent.runtime import (  # noqa: E402
    create_runtime,
    detect_ollama,
)
from scholaragent.workspace import TemporaryWorkspace  # noqa: E402


def isolate_eval_state(stamp: str):
    """把评测的笔记/记忆/论文缓存重定向到本次运行的独立目录。

    不隔离的话:mem-1 上一次运行存的记忆会让这一次"免检得分",
    评测垃圾还会永久混进用户真实的研究笔记 —— 实验必须可复现、可清理。
    """
    return TemporaryWorkspace(os.path.join("evals", "results", f"data_{stamp}"))


def build_runner(mode: str, metrics_collector=None, on_progress=None, artifacts=None,
                 workspace=None):
    """按模式组装被测对象:三者都有 run(task)->str 接口。"""
    runtime = create_runtime(
        llm=LLMClient(),
        workspace=workspace,
        artifacts=artifacts,
        conversation=False,
        auto_recall=False,
    )
    for runner in runtime.runners.values():
        if hasattr(runner, "verbose"):
            runner.verbose = False
    return runtime.runner(
        mode,
        metrics_collector=metrics_collector,
        on_progress=on_progress,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["react", "plan", "team"],
                        default="react")
    parser.add_argument("--tasks", default=os.path.join("evals", "tasks.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题(0=全部)")
    args = parser.parse_args()

    # 模型选择:优先 .env 的 Key,否则自动探测本机 Ollama
    if not config.LLM_API_KEY:
        local_model = detect_ollama()
        if not local_model:
            print("没有可用模型:请配置 .env 的 LLM_API_KEY,或安装并启动 Ollama")
            sys.exit(1)
        config.LLM_BASE_URL = "http://localhost:11434/v1"
        config.LLM_API_KEY = "ollama"
        config.LLM_MODEL = local_model

    tasks = load_tasks(args.tasks, mode=args.mode)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"模式:{args.mode}  模型:{config.LLM_MODEL}  任务数:{len(tasks)}\n")

    stamp = time.strftime("%m%d_%H%M%S")
    started_at = utc_now()
    workspace = isolate_eval_state(stamp)

    rows = Evaluator(
        build_runner(args.mode, workspace=workspace), mode=args.mode
    ).run(tasks)

    report = os.path.join("evals", "results", f"report_{args.mode}_{stamp}.md")
    avg = write_report(rows, report)
    manifest = build_manifest(
        Path(report).stem,
        tasks=tasks,
        model=config.LLM_MODEL,
        config={"mode": args.mode, "limit": args.limit},
        modes=(args.mode,),
        strategy_version="fixed-eval-v2",
        started_at=started_at,
        ended_at=utc_now(),
        score_states={"unscored": len(rows)},
    )
    write_manifest_file(manifest, Path(report).with_suffix(".manifest.json"))
    print(f"\n平均得分:{avg:.2f}  报告:{report}")


if __name__ == "__main__":
    main()
