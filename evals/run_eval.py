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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import RESEARCH_SYSTEM_PROMPT, detect_ollama  # noqa: E402
from scholaragent import config  # noqa: E402
from scholaragent.agent import Agent  # noqa: E402
from scholaragent.evaluate import Evaluator, load_tasks, write_report  # noqa: E402
from scholaragent.llm import LLMClient  # noqa: E402
from scholaragent.memory import MemoryStore  # noqa: E402
from scholaragent.metrics import InstrumentedLLM, InstrumentedTool  # noqa: E402
from scholaragent.planner import Planner  # noqa: E402
from scholaragent.team import ResearchTeam  # noqa: E402
from scholaragent.tool import ToolRegistry  # noqa: E402
from scholaragent.tools import BUILTIN_TOOLS  # noqa: E402
from scholaragent.tools.memory_tools import RecallTool, RememberTool  # noqa: E402


def isolate_eval_state(stamp: str):
    """把评测的笔记/记忆/论文缓存重定向到本次运行的独立目录。

    不隔离的话:mem-1 上一次运行存的记忆会让这一次"免检得分",
    评测垃圾还会永久混进用户真实的研究笔记 —— 实验必须可复现、可清理。
    """
    config.DATA_DIR = os.path.join("evals", "results", f"data_{stamp}")


def build_runner(mode: str, metrics_collector=None):
    """按模式组装被测对象:三者都有 run(task)->str 接口。"""
    llm = LLMClient()
    if metrics_collector is not None:
        llm = InstrumentedLLM(llm, metrics_collector)
    # 记忆工具要用"现在"的 DATA_DIR 重新构建(BUILTIN_TOOLS 里的
    # 那两个在 import 时就绑定了真实 data/ 路径,不能直接用)
    store = MemoryStore()
    tools = [t for t in BUILTIN_TOOLS if t.name not in ("remember", "recall")]
    tools += [RememberTool(store), RecallTool(store)]
    if metrics_collector is not None:
        tools = [InstrumentedTool(tool, metrics_collector) for tool in tools]
    registry = ToolRegistry(tools)
    if mode == "react":
        # 评测用的 Agent 不带会话记忆:每道题独立作答,分数才可比
        return Agent(llm, registry, system_prompt=RESEARCH_SYSTEM_PROMPT,
                     max_steps=15, verbose=False)
    if mode == "plan":
        worker = Agent(llm, registry, system_prompt=RESEARCH_SYSTEM_PROMPT,
                       max_steps=15, verbose=False)
        return Planner(llm, worker, verbose=False)
    if mode == "team":
        return ResearchTeam(llm, registry, verbose=False)
    raise ValueError(f"未知模式:{mode}")


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
    isolate_eval_state(stamp)  # 必须在 build_runner 之前:隔离评测读写的数据

    rows = Evaluator(build_runner(args.mode), mode=args.mode).run(tasks)

    report = os.path.join("evals", "results", f"report_{args.mode}_{stamp}.md")
    avg = write_report(rows, report)
    print(f"\n平均得分:{avg:.2f}  报告:{report}")


if __name__ == "__main__":
    main()
