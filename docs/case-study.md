# 案例研究：ReAct 论文文献调研（三模式对比）

> 本文引用可公开复验的证据包 [`evals/case_results/react-method-evidence-v1/`](../evals/case_results/react-method-evidence-v1/)，
> 不复制私有运行日志。运行日期 2026-08-26，模型 `deepseek-v4-flash`
> （OpenAI 兼容接口），论文为 ReAct（arXiv:2210.03629）。

## 1. 为什么选这个案例

案例定义见 [`evals/cases/react_method.json`](../evals/cases/react_method.json)：

- **可核验**：目标论文有明确 arXiv 编号（2210.03629），检索、下载、全文阅读都有
  客观成败标准；
- **任务有难度梯度**：需要「检索 → 下载 → 分段阅读长论文 → 提炼机制 → 指出局限」，
  能区分三种执行模式的真实行为差异；
- **内置诚实出口**：任务明确要求「若全文读取失败，明确说明证据缺口」，失败也是证据。

## 2. 三种执行模式的真实行为

| 模式 | 执行结构 | 本次真实轨迹（摘要见 `runs.jsonl`） |
|---|---|---|
| ReAct | 单循环「思考→行动→观察」，最多 15 步 | 搜索→下载→连续 8 次分段阅读 33 页论文→存笔记→写记忆；**步数耗尽，未产出最终回答** |
| Plan | 计划（5 个子任务）→ 逐步执行 → 反思 → 汇总 | 按「基本信息→摘要引言证据→方法机制→局限→汇总」逐步推进，每步阅读、摘录、存笔记，最终给出结构化回答 |
| Team | 检索员→精读员→写作员三角色流水线 | 检索、精读全文并交叉核验后续工作（Pre-Act 等），写作员汇总为综述式回答 |

三个模式都真实下载了论文 PDF（各模式状态隔离在 `evals/results/`，不入库），
轨迹摘要与回答全文见 `runs.jsonl`。

## 3. 结果与初步评分

四项 rubric（任务完成 / 事实正确 / 引用有效 / 输出完整）加权口径与
`scholaragent/routing_evaluation.py` 一致：

| 模式 | 质量 | 耗时(s) | LLM 调用 | 工具调用 | 回答 |
|---|---:|---:|---:|---:|---|
| react | 0.175 | 73 | 15 | 19 | 未产出（预算耗尽） |
| plan | 0.945 | 279 | 49 | 44 | 5125 字符，含原文逐字引文与页号 |
| team | 0.885 | 171 | 47 | 51 | 3926 字符，综述式，含后续工作对比 |

**关键观察（基于本次运行，不推广为普遍结论）：**

1. **步数预算是 ReAct 的显性约束**：33 页论文全文阅读消耗了全部 15 步，ReAct
   未完成回答。Plan 通过「先计划再执行」把阅读拆分进多个子任务并穿插存笔记，
   避免了同一问题。
2. **Plan 回答可核验度最高**：机制说明引用了论文摘要与 Introduction 的逐字原文
   （如 "reasoning traces and task-specific actions in an interleaved manner"）、
   Section 2 的动作空间定义，并给出页号与 arXiv 版本；局限部分三条均出自论文正文，
   且对「引文被截断」的证据缺口做了如实说明。
3. **Team 覆盖面更广但间接引用更多**：加入了与 Pre-Act / RAISE / ReSpAct 的对比，
   其中后续工作结论基于检索到的摘要，文档已声明「细节未逐页精读」。

## 4. 诚实边界

- 本案例评分为**作者初步自评**，用于展示实验台评分工作流；正式成本/质量对比结论
  需要独立人工评分表与留出集实验（见 `evals/` 路由实验，尚未完成）。
- react 模式的失败是真实行为，被完整记录（`answer` 为空、`error` 为 null、
  trace 显示步数耗尽），未被隐藏或重跑美化。
- 本次运行经本机 HTTP 代理访问 arXiv；若网络不可达，`arxiv_search` 会按设计自动
  兜底到 OpenAlex 的 arXiv 索引，`download_paper` 失败则如实产生证据缺口。

## 5. 复现

```powershell
.venv\Scripts\python evals\run_case_study.py `
  --case evals\cases\react_method.json `
  --output evals\case_results\react-method-evidence-v1-new `
  --state-dir evals\results\case_states\react-method-evidence-v1-new
```

每次运行生成不可覆盖的新目录，保留时间戳化的原始记录；评分模板
（`scores.template.jsonl`）与已评分文件（`scores.jsonl`）分离，避免把评分写回原始运行。
