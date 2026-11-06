# 真实任务实验

离线测试用的是假模型，只能证明“模型这样回复时，程序这样处理”。这份文档说明怎么用真实模型、
真实论文把三种模式跑一遍，以及系统在跑到一半出问题时会怎么处理。

## 跑不完时会怎样

- `max_steps` 包含最后一步总结。步数快用完时，Agent 不再调用工具，只整理已有材料。
- 总结失败、返回空内容，或者模型还想继续调用工具时，系统直接用已有的工具结果生成一份“阶段报告”，
  不再额外请求模型。报告里写清楚拿到了什么、哪里失败了、为什么停下。
- 运行结果分两层：`status` 表示有没有跑完（`completed / failed / cancelled`），
  `completion.completeness` 表示答案完不完整（`complete / partial`）。
  Plan、Team 的某个子步骤只拿到部分结果时，最终结果也会标成“部分完成”，不会被后面的总结盖掉。
- 用户取消时，取消前的阶段报告保存在 `completion.partial_reports`，工作台里可以展开查看。
- `llm_calls` 是模型调用次数（含失败），`request_attempts` 是实际发出的请求次数（含重试）。
  某次调用重试过、且服务没返回完整用量时，token 总数记为未知，不拿最后一次的数字充数。

## 结论怎么对到原文

工具读到的内容会生成来源编号（如 `S001`），模型在最终回答里每条重要结论后面标上编号：

```text
该机制交织推理与行动。[S001]
```

系统逐行登记这些结论：
- 引用了存在的编号：记为“有引用、待人工确认”。程序只检查编号对得上，不判断原文是否真的支持这句话。
- 没有引用：记为“无法判断”。
- 引用了不存在的编号：记为“引用错误”，这次运行会被标成证据校验未通过。
- 用 `原文：“...”` 格式直接摘录时，程序会检查这段话是否真的出现在读到的原文里。

工作台可以展开每条结论，查看来源、页码和摘录。只有人工在工作台里确认过，结论才会变成“已核验”；
模型自己的输出永远不会被自动标成已核验。

## 实验怎么跑

任务在 `evals/research_tasks.jsonl`，共 12 道，分四类，每类 3 道：定位论文、精读、比较、找证据缺口。
题目都围绕 ReAct 和 ResNet 两篇论文的固定版本（`2210.03629v1`、`1512.03385v1`）。每道题分别用
ReAct、Plan、Team 三种模式各跑一次，每次运行使用独立的工作区。

先不联网检查清单：

```powershell
.venv/Scripts/python.exe evals/run_experiment.py --manifest evals/experiments/research_full.json --output evals/results/research-full-v1 --dry-run
```

应显示 12 道题 / 36 次运行。去掉 `--dry-run` 就会真正开始跑，需要先在 `.env` 里配置好模型。
`research_pilot.json` 是只有 3 道题的试跑版本，适合先确认流程能跑通。

几点说明：
- 任务指定了论文版本，下载和阅读工具会强制使用这个版本；版本不对或者不在清单里的论文会被拒绝。
  普通调用也可以这样用：`runtime.run(task, mode="team", pinned_sources=["2210.03629v1"])`。
- 下载到的 PDF 会记录 SHA-256，方便核对每次运行读的是不是同一份材料。
- ReAct 和 Plan 每步最多 15 次，Team 的三个角色沿用各自默认的步数上限，实际数值写在每条运行记录里。
- 结果是一个模型、每道题跑一次，适合用来发现问题、粗略比较，不能当作“哪种模式更好”的定论。

## 看结果

跑完后生成执行报告，内容包括完成情况、停止原因、耗时、调用次数、token、缓存命中率和花费：

```powershell
.venv/Scripts/python.exe evals/summarize_runs.py --runs evals/results/research-full-v1/runs.jsonl
```

花费优先使用服务返回的实际金额（`usage.cost`）。服务没有返回时，可以用 `--price-in` / `--price-out`
（美元/百万 token）按单价估算，报告会标明“估算”。

## 给回答打分（可选）

想给回答质量打分时，可以生成隐藏了模式名称的评分材料，避免打分时先入为主：

```powershell
.venv/Scripts/python.exe evals/review_results.py prepare --runs evals/results/research-full-v1/runs.jsonl --output evals/results/research-full-review-v1
```

在 `scores.blind.template.jsonl` 里按任务完成、事实正确、引用支持、输出完整四项打分（0 到 1），
另存为 `scores.blind.jsonl`，再用 `review_results.py report` 生成带分数的报告。
`key.private.json` 记录了哪条回答对应哪种模式，打分的人不要看这个文件。
