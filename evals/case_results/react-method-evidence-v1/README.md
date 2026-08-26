# 案例证据包：react-method-evidence-v1

本目录是固定公开案例 **ReAct 论文文献调研**（`evals/cases/react_method.json`）在
ReAct / Plan / Team 三种执行模式下的一次真实运行证据，可复验、可评分。

## 复现

```powershell
# 需要可用的 OpenAI 兼容 API（.env 配置 LLM_API_KEY）或本机 Ollama
# 运行记录与评分模板写入新的不可覆盖目录；私有状态隔离到 evals/results/ 下
.venv\Scripts\python evals\run_case_study.py `
  --case evals\cases\react_method.json `
  --output evals\case_results\react-method-evidence-v1-new `
  --state-dir evals\results\case_states\react-method-evidence-v1-new
```

运行环境：2026-08-26，模型 `deepseek-v4-flash`（OpenAI 兼容接口，经本机 HTTP 代理访问
arXiv 与 API）。不同模型、网络与版本下结果会不同——这正是可复现证据的意义：每次运行
都留下独立的 `runs.jsonl`，不覆盖旧记录。

## 文件

| 文件 | 内容 |
|---|---|
| `runs.jsonl` | 三模式原始运行：任务、回答、轨迹摘要、指标、可移植产物路径 |
| `scores.template.jsonl` | 空的人工评分模板（四项 rubric） |
| `scores.jsonl` | 已填写评分（作者初步自评，非独立第三方评分） |

## 汇总

| 模式 | 质量 | 耗时(s) | LLM 调用 | 工具调用 | Prompt token | 回答长度 |
|---|---:|---:|---:|---:|---:|---:|
| react | 0.175 | 73 | 15 | 19 | 144325 | 0（未产出） |
| plan | 0.945 | 279 | 49 | 44 | 351114 | 5125 |
| team | 0.885 | 171 | 47 | 51 | 1258143 | 3926 |

质量 = 0.40×任务完成 + 0.30×事实正确 + 0.20×引用有效 + 0.10×输出完整，与
`scholaragent/routing_evaluation.py` 的加权口径一致。

## 诚实声明

- **react 未完成**：15 步上限全部用于搜索、下载与分段阅读 33 页论文，未留预算撰写
  最终回答（`answer` 为空，见 `runs.jsonl` 的 trace）。这是步数预算约束的真实表现，
  不是故障；调大 `max_steps` 或缩短论文可复现成功路径。
- **评分为作者初步自评**：按 rubric 逐项打分，用于展示实验台工作流；正式路由实验的
  独立人工评分表仍在进行（见 `evals/` 路由实验与 `docs/case-study.md`）。
- 三个模式都真实调用了 `arxiv_search` / `download_paper` / `read_paper` 等工具并下载了
  论文 PDF（私有状态在 `evals/results/`，不入库），回答基于论文原文，非凭空编造。
