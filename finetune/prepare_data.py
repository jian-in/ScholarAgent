# -*- coding: utf-8 -*-
"""生成 QLoRA 微调数据(Alpaca 格式)。

用法(项目根目录):python finetune/prepare_data.py
输出:finetune/data/scholar_sft.json

样本三个来源:
    1. 内置种子样本(计划拆解 / 反思质检两类任务的标准示范)
    2. data/notes/、data/memory/ 里系统真实运行留下的领域语料
    3. 你手工往 SEED_* 列表里补充的样本(质量比数量重要!)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.planner import PLAN_SYSTEM_PROMPT, REFLECT_SYSTEM_PROMPT

# ―― 种子样本:计划拆解 ――――――――――――――――――――――――――――――
# instruction 用和线上完全相同的系统提示词,保证训练/推理一致
SEED_PLANS = [
    ("调研 LLM Agent 的记忆机制研究进展",
     '["检索 LLM Agent memory 相关论文", "下载并精读最相关的 2 篇", "总结各家方法并保存笔记"]'),
    ("帮我算 365*24",
     '["用计算器算出 365*24"]'),
    ("查一下 ReAct 论文的核心贡献并记下来",
     '["检索 ReAct 论文", "下载并阅读引言与方法部分", "提炼核心贡献保存到笔记"]'),
    ("对比 ReAct 和 Plan-and-Solve 两种智能体范式",
     '["分别检索两篇论文", "各自精读方法部分", "从规划方式与纠错机制两个维度对比", "写成对比笔记"]'),
]

# ―― 种子样本:反思质检 ――――――――――――――――――――――――――――――
SEED_REFLECTS = [
    ("步骤目标:检索 ReAct 论文\n执行结果:找到了 ReAct: Synergizing Reasoning and "
     "Acting in Language Models(arXiv 2210.03629),另附 3 篇相关论文",
     '{"ok": true}'),
    ("步骤目标:总结论文核心贡献\n执行结果:这篇论文很不错,值得一读",
     '{"ok": false, "advice": "结果没有给出任何具体贡献,需列出方法要点并注明出处"}'),
    ("步骤目标:下载论文 2210.03629\n执行结果:论文 2210.03629 已就绪(618 KB,共 33 页)",
     '{"ok": true}'),
]


def collect_domain_corpus():
    """收集系统真实运行产生的领域语料(仅作参考语料,不直接当样本)。"""
    corpus = []
    notes = os.path.join("data", "notes", "research_notes.md")
    if os.path.exists(notes):
        with open(notes, encoding="utf-8") as f:
            corpus.append({"type": "notes", "text": f.read()[:5000]})
    memories = os.path.join("data", "memory", "memories.jsonl")
    if os.path.exists(memories):
        with open(memories, encoding="utf-8") as f:
            corpus.append({"type": "memories", "text": f.read()[:5000]})
    return corpus


def main():
    samples = []
    for task, plan in SEED_PLANS:
        samples.append({
            "instruction": PLAN_SYSTEM_PROMPT,
            "input": task,
            "output": plan,
        })
    for case, verdict in SEED_REFLECTS:
        samples.append({
            "instruction": REFLECT_SYSTEM_PROMPT,
            "input": case,
            "output": verdict,
        })

    out_dir = os.path.join("finetune", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scholar_sft.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    corpus = collect_domain_corpus()
    print(f"已生成 {len(samples)} 条训练样本 -> {out_path}")
    print(f"另收集到 {len(corpus)} 份领域语料(可用于扩充样本)")
    print("提醒:种子样本只是起步,请人工扩充并逐条核对质量后再训练。")


if __name__ == "__main__":
    main()
