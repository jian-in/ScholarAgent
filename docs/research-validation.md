# 阶段结果、原文证据链与真实任务验证

本轮改动补工程与证据入口，不声明科研质量或成本已经优于任何基线。真实模型实验与人工评分尚未执行。

## 1. 预算与停止语义

- Agent 的 `max_steps` 包含总结调用。最后一轮只整理已有材料，不再执行工具。
- 收尾模型失败、返回空内容或继续请求工具时，直接用已有工具结果生成阶段报告，不追加新模型请求。
- 报告包含已获得材料、工具失败、必要阅读流程尚未完成的情况，以及停止原因。
- `RunResult.status` 仍表示 `completed / failed / cancelled` 的执行终态；`completion.completeness` 单独表示 `complete / partial`。
- Plan 或 Team 子步骤出现部分结果时，最终汇总也保留部分完成说明。当前采用保守聚合：后续摘要不会自动抹掉前序缺口。
- 取消返回明确停止信号，取消前的阶段报告保存在 `completion.partial_reports`，工作台可展开查看。
- `llm_calls` 统计逻辑模型调用（含失败），`request_attempts` 统计真实 SDK 请求尝试。SDK 内部重试保持关闭。
- 一次调用经过重试、空响应或失败，且各次 usage 未完整返回时，总 token 为未知，不把最后一次响应的 token 当作总成本。

验收：短预算、收尾失败和取消都有结果或停止说明；执行加收尾不超过步数预算；异常后上下文恢复；唯一终态事件保留。

## 2. 结论到来源

工具产物中的来源锚点进入运行级账本，ID 冲突时重新分配；返回模型的工具观察会附带实际运行级 ID。
模型最终回答应在每条重要结论的同一行标注引用，例如：

```text
该机制交织推理与行动。[S001]
```

最终回答按行登记候选结论。引用到已存在锚点的结论仍为 `partial + unreviewed`，不是自动认证。
没有引用的行记录为 `not_assessable`；未知引用记录为 `unsupported` 并触发结构校验错误。代码块不当作事实结论登记。

可选的直接摘录使用 `原文：“...”` 格式；程序检查其是否出现在已记录的锚点摘录中。锚点只保留有限长度的原文片段，
截取范围外的引文需要补充来源，结构通过与语义判断须分开。

工作台可以展开每条候选结论，查看来源、页码、摘录和人工核验状态。`verified / rejected` 必须通过人工核验入口登记核验者与说明。
当前候选提取是确定性的逐行登记，不声称已经具备完整的语义主张识别；人工需检查遗漏、引用范围与原文支持程度。

验收：来源可定位，缺失引用与不匹配摘录可见；模型产物不自动变成 `verified`；多工具与多子步骤不会因重置编号串用来源。

## 3. 固定真实任务集与试跑

`evals/research_tasks.jsonl` 包含 12 题：定位、精读、比较、证据缺口四类各 3 题。
先用已有 ReAct 与 ResNet 案例的两个固定 arXiv v1 版本校验流程；这不是跨学科泛化评测。
3 题为 pilot，9 题为 holdout；不要使用 holdout 调整规则后再宣称其为留出结果。

实验设置：ReAct/Plan/Team 三模式，独立工作区，关闭会话继承和自动长期记忆召回。
ReAct 与 Plan worker 的上限设为 15；Team 保留不同角色的原有预算，实际预算写入每条运行记录。
因此本轮是执行方式及其默认预算的比较，不是严格等调用预算实验。`team_require_full_paper=false` 避免定位题强制读取整篇论文；
需要精读的题仍在任务要求和人工验收项中明确约束。

先做不联网的清单检查：

```powershell
.venv/Scripts/python.exe evals/run_experiment.py --manifest evals/experiments/research_pilot.json --output evals/results/research-pilot-v1 --dry-run
.venv/Scripts/python.exe evals/run_experiment.py --manifest evals/experiments/research_full.json --output evals/results/research-full-v1 --dry-run
```

分别应显示 3 题 / 9 次运行，以及 12 题 / 36 次运行。dry-run 不创建工作区，也不连接模型。
确认模型、额度与文献网络访问后，移除 `--dry-run` 才执行真实实验。清单可设置 `model`；未设置时使用执行环境模型，并记录实际模型名称。
固定版本写入各模式的实际任务文本，下载到的 PDF 记录 SHA-256；材料不一致或未取得全文的情况保留在报告中。
这批记录依旧只是小样本诊断，正式比较需重复运行并核对材料一致性。

## 4. 离线盲评与派生报告

从实验 `runs.jsonl` 创建独立的评分材料：

```powershell
.venv/Scripts/python.exe evals/review_results.py prepare --runs evals/results/research-pilot-v1/runs.jsonl --output evals/results/research-pilot-review-v1
```

- `review_packets.jsonl`：隐藏模式、运行 ID、成本和轨迹，保留任务、验收项、原回答及原文锚点。
- `scores.blind.template.jsonl`：任务完成、事实正确、引用支持、输出完整四项空白评分。
- `claims.blind.template.jsonl`：逐条原文核验模板。
- `key.private.json`：维护者保留的映射与原始记录哈希，不给评分者。

原回答的措辞可能透露角色分工，因此这是隐藏模式元数据的单盲材料，不宣称双盲。
评分者另存已完成条目为 `scores.blind.jsonl`、`claims.blind.jsonl`。未核对的条目留空并从提交文件中省略；
每条提交均需真实核验者与说明。评分值应为 `[0,1]` 的有限数值；作者评分与独立评分必须如实区分。

```powershell
.venv/Scripts/python.exe evals/review_results.py report --runs evals/results/research-pilot-v1/runs.jsonl --key evals/results/research-pilot-review-v1/key.private.json --scores evals/results/research-pilot-review-v1/scores.blind.jsonl --claims evals/results/research-pilot-review-v1/claims.blind.jsonl --output evals/results/research-pilot-report-v1
```

输出 `reviewed_runs.jsonl`、`report.md`、`summary.json` 和 `sources.audit.json`。不改原始运行文件，不覆盖已有目录。
缺失评分不是零分；失败、取消和部分完成的运行保留在分母中。资源均值附带已知观测的覆盖数，未知 token 不参与假精确汇总。
只有原文支持程度经过人工核对，才可以说对应结论已核验；结构校验通过只证明记录自洽。

## 本地离线验收范围

预算与停止场景、跨子步骤锚点冲突、最终回答引用、人工核验身份、非法/重复评分、原始文件不可变、清单 dry-run 和材料哈希都有离线回归。
真实文献下载、模型回答质量、跨学科泛化以及独立人工评分留待单独实验，当前文档不填入未观察的成绩。
