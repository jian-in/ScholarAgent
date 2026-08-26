"""评测层(M6):给智能体出卷子、打分、出报告。

论文实验的地基。设计思路:

    任务集   evals/tasks.jsonl,每行一道题:任务描述 + 期望关键词
    打分     关键词命中率(0~1)。简单、可复现、无需人工;
             局限也很明确(答案对但换了说法会漏判),论文里如实讨论,
             再抽样人工复核即可 —— 自动初筛 + 人工复核是标准做法
    对比     同一批任务分别跑 ReAct / 规划 / 团队三种模式,
             就是现成的消融实验(ablation study)

Agent、Planner、ResearchTeam 都暴露 run(task) -> str 接口,
所以评测器不关心被测对象是谁 —— 接口统一的又一次红利。
"""

import json
import os
import re
import time

ALL_MODES = ("react", "plan", "team")


def load_tasks(path: str, mode: str = None) -> list:
    """读取评测任务集(JSONL);给定 mode 时只留适用该模式的任务。"""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task = json.loads(line)
            if mode and mode not in task.get("modes", list(ALL_MODES)):
                continue
            tasks.append(task)
    return tasks


def score_answer(answer: str, expect_keywords: list) -> float:
    """关键词命中率:期望关键词有多大比例出现在回答里。

    两个实测踩过的坑,决定了下面的实现细节:
    1. 千分位:模型答"525,600 分钟"会因逗号漏判 —— 归一化时去掉
       数字之间的逗号/空格。注意只处理"数字之间":全局删逗号会把
       "22, 10"粘成"2210"造成反向误判。
    2. 措辞:同一个意思有多种说法(如 推理/reasoning)——
       期望关键词可以写成列表,命中其中任意一个就算命中。
    """
    if not expect_keywords:
        return 0.0
    answer = (answer or "").lower()
    normalized = re.sub(r"(?<=\d)[,，\s](?=\d)", "", answer)

    def hit(keyword) -> bool:
        alternatives = keyword if isinstance(keyword, list) else [keyword]
        return any(str(a).lower() in answer or str(a).lower() in normalized
                   for a in alternatives)

    return sum(1 for kw in expect_keywords if hit(kw)) / len(expect_keywords)


class Evaluator:
    """对一个 runner(有 run(task)->str 方法的对象)跑一遍任务集。"""

    def __init__(self, runner, mode: str = "react", verbose: bool = True):
        self.runner = runner
        self.mode = mode
        self.verbose = verbose

    def run(self, tasks: list) -> list:
        rows = []
        for i, task in enumerate(tasks, 1):
            if self.verbose:
                print(f"[评测 {i}/{len(tasks)}] {task['id']}: {task['task'][:40]}")
            start = time.time()
            try:
                answer = self.runner.run(task["task"])
            except Exception as exc:
                # 单题崩溃记 0 分继续,不能让一道题毁掉整场评测
                answer = f"(执行出错:{type(exc).__name__}: {exc})"
            seconds = round(time.time() - start, 1)
            score = score_answer(answer, task.get("expect", []))
            rows.append({
                "id": task["id"],
                "mode": self.mode,
                "score": round(score, 2),
                "seconds": seconds,
                "answer_preview": str(answer)[:200],
            })
            if self.verbose:
                print(f"  得分 {score:.2f},耗时 {seconds}s")
        return rows


def write_report(rows: list, path: str):
    """把评测结果写成 markdown 报告 + 同名 JSONL 原始数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path.replace(".md", ".jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    total_time = sum(r["seconds"] for r in rows)
    lines = [
        f"# 评测报告(模式:{rows[0]['mode'] if rows else '?'})",
        "",
        f"- 任务数:{len(rows)}  平均得分:**{avg:.2f}**  总耗时:{total_time:.0f}s",
        "",
        "| 任务 | 得分 | 耗时(s) | 回答摘要 |",
        "|------|------|---------|----------|",
    ]
    for r in rows:
        preview = r["answer_preview"][:60].replace("\n", " ").replace("|", ",")
        lines.append(f"| {r['id']} | {r['score']} | {r['seconds']} | {preview} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return avg
