"""对固定校准集运行三种模式，保存不可变的原始路由观测。"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals.run_eval import build_runner, isolate_eval_state
from main import detect_ollama
from scholaragent import config
from scholaragent.metrics import MetricsCollector
from scholaragent.routing import FEATURE_VERSION, ROUTING_MODES, TaskFeatureExtractor


def load_calibration_tasks(path):
    tasks = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task = json.loads(line)
            if task.get("split") == "calibration":
                tasks.append(task)
    if not tasks:
        raise ValueError("任务集中没有 calibration 划分")
    return tasks


def ensure_model():
    if config.LLM_API_KEY:
        return
    model = detect_ollama()
    if not model:
        raise RuntimeError("没有可用模型：请配置 API Key 或启动 Ollama")
    config.LLM_BASE_URL = "http://localhost:11434/v1"
    config.LLM_API_KEY = "ollama"
    config.LLM_MODEL = model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=os.path.join("evals", "router_tasks.jsonl"))
    parser.add_argument("--output", required=True, help="新的原始 JSONL 路径，已存在则拒绝覆盖")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions 必须至少为 1")
    if os.path.exists(args.output):
        parser.error("原始校准结果不可覆盖，请指定新路径")

    ensure_model()
    tasks = load_calibration_tasks(args.tasks)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    isolate_eval_state(f"router_calibration_{stamp}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    extractor = TaskFeatureExtractor()

    with open(args.output, "x", encoding="utf-8") as handle:
        for repeat in range(1, args.repetitions + 1):
            for task in tasks:
                for mode in ROUTING_MODES:
                    collector = MetricsCollector(mode)
                    runner = build_runner(mode, metrics_collector=collector)
                    collector.restart()  # 排除执行器构建时间，只记录本次任务执行。
                    error = None
                    try:
                        answer = runner.run(task["task"])
                    except Exception as exc:
                        answer = ""
                        error = f"{type(exc).__name__}: {exc}"
                    row = {
                        "run_id": f"{task['id']}:{mode}:{repeat}",
                        "task_id": task["id"],
                        "split": "calibration",
                        "category": task["category"],
                        "task": task["task"],
                        "mode": mode,
                        "repeat": repeat,
                        "feature_version": FEATURE_VERSION,
                        "features": extractor.extract(task["task"]),
                        "metrics": collector.finish().to_dict(),
                        "answer": answer,
                        "error": error,
                        "quality": None,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"{row['run_id']}: {row['metrics']['seconds']:.2f}s")


if __name__ == "__main__":
    main()
