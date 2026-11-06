# 能力与设计

项目的设计理念、完整能力清单、与同类工具的对比和目录结构。项目概览见 [README](../README.md)。

## 项目定位

ScholarAgent 不是一个只把问题转发给大模型的聊天壳,而是一个面向
**科研文献调研流程**的可解释 Agent 原型:把大语言模型放进受控的工具链、
记忆、规划和多角色协作框架中,让它能够按步骤完成「检索论文 → 下载阅读
→ 记录结论 → 回忆已有发现 → 汇总回答」这一类任务。

一句话概括:

> 本项目从零实现了一个轻量级科研调研智能体框架,使大语言模型能够在
> 工具白名单、步数上限和可复现评测约束下,通过 ReAct 循环、任务规划、
> 长期记忆和多智能体流水线辅助完成文献调研。

## 核心理念

1. **大模型负责决策,程序负责执行边界**:模型只生成工具名和参数,真正的
   搜索、下载、阅读、计算、记忆写入都由 Python 工具完成;工具错误以文字
   回传给模型纠错,而不是让程序直接崩溃。
2. **复杂任务要分层完成**:简单任务直接走 ReAct 循环,复杂任务先规划再
   执行并反思,完整调研任务交给检索员、精读员、写作员三角色流水线。
3. **结果要能沉淀、能复验**:研究笔记和长期记忆落到本地文件,评测任务集和
   报告用于比较不同模式的质量与耗时,避免只靠主观演示评价系统。

## 设计原则

1. **分层单向依赖**:上层只依赖下层接口,换模型/加工具不动核心代码。
2. **错误回传而非崩溃**:工具出错的信息交还给模型,让它自己纠错重试。
3. **一切可离线验证**:假模型 ScriptedLLM 让测试和演示零成本。

## 能干什么

- 按关键词检索 arXiv 论文,在官方接口繁忙时可用 OpenAlex 的 arXiv 索引兜底。
- 按 arXiv 编号下载 PDF,并按页/字符分段阅读长论文,避免一次塞爆上下文。
- 读取扫描版 PDF：没有文字层的页面会自动尝试 Tesseract OCR，结果带 `[OCR]`
  标记和可审计的页码来源锚点；依赖不可用时会保留可解释的降级信息。
- 保存和读取研究笔记,把重要结论写入长期记忆,再用手写 BM25 检索召回。
- 调用计算器和时钟等基础工具,演示模型如何把外部结果纳入回答。
- 在 `--plan` 模式下执行「计划 → 单步 ReAct 执行 → 反思重试 → 汇总」。
- 在 `--team` 模式下执行「检索员 → 精读员 → 写作员」的多智能体流水线。
- 自动探测本机 Ollama 聊天模型，并在网页工作台切换本地模型或已配置的云端模型。
- 对明确的资料缺口任务自动拆成推理、工具使用、近期综述、推理效率四个方向，
  快捷入口默认先做摘要级快速筛选，也可切换到深度精读；每个方向要求论文核心卡片
  和来源证据，再汇总横向对比与剩余缺口。网页提供“快速/深度补齐四项缺口”快捷入口，
  详见 [`docs/architecture/资料缺口调研说明.md`](architecture/资料缺口调研说明.md)。
- 用 `ScriptedLLM` 离线演示/测试 Agent 循环,用 `evals/` 对比三种模式在同一批任务上的表现。

### 外部思路与可选依赖

本项目还吸收了 [nature-skills](https://github.com/Yuan1z0825/nature-skills)
中“短路由器 + 静态工作流清单 + 来源锚点 + 确定性审查”的可复验思路，
但没有复制其源码、资源或整套技能目录。具体边界和 Apache-2.0 使用注意事项见
[`docs/architecture/nature-skills-吸收说明.md`](architecture/nature-skills-吸收说明.md)。

扫描版 PDF 的阅读增强见
[`docs/architecture/ocr-增强说明.md`](architecture/ocr-增强说明.md)：
项目会优先读取 PDF 文字层；页面没有文字层时，自动尝试调用本机的
Tesseract + `pdftoppm`，并在阅读结果中标记 `[OCR]`，同时把该页证据锚点降为
`medium` 置信度。OCR 是可选外部依赖，不会把用户机器上的二进制文件打进仓库。

## 与同类工具的差异

下表只陈述可在本仓库复验的事实，不声明性能领先：

| 维度 | 单模型聊天壳 | 黑盒编排框架 | ScholarAgent |
|---|---|---|---|
| 执行模式选择 | 无模式概念，链路由人工拆 | 通常由开发者硬编码调用链 | ReAct/Plan/Team 三模式 + `--auto` 自动路由，决策附带特征与人话理由 |
| 路由可解释性 | 不适用 | 视框架，多为隐式 | 规则路由零模型调用、版本化特征，固定 36 题标签一致性可复验 |
| 成本感知 | 无 | 少数提供 token 统计 | reward 显式建模质量−延迟−调用−token；服务没返回 token 就记为未知，不估算 |
| 执行过程可见性 | 只见最终回答 | 见中间消息 | 执行级指标（耗时/LLM 调用/工具调用/token）+ 轨迹 + 产物收集 |
| 失败处理 | 报错即止或静默 | 异常吞掉或回调 | 失败如实保留：步数耗尽、工具错误回传模型纠错、策略损坏安全回退 |
| 依赖重量 | 极轻但无工具链 | 框架级依赖 | 4 个运行时依赖；ReAct 核心循环约 200 行，从零实现 |
| 证据可复验 | 不可 | 视实现 | 案例证据包 + 390+ 项离线测试 + 双系统 CI |

### 五个技术点与复验入口

1. **从零实现的轻量 Agent 循环**：不依赖 LangChain 等编排封装，
   「思考→行动→观察」循环含步数保险丝与坏参数容错。
   复验：`scholaragent/agent.py`；`python main.py --demo`（离线）。
2. **可解释的三模式路由**：确定性特征 `task-features-v1` + 规则基线
   `rule-router-v1` + 可安全回退的学习路由，决策输出特征、理由与策略版本。
   复验：`scholaragent/routing.py`；
   `python evals/evaluate_rule_router.py --tasks evals/router_tasks.jsonl`（零模型调用）。
3. **执行级指标与产物追踪**：每次运行记录真实 LLM/工具调用、token 与耗时，
   并收集论文、笔记、记忆产物；token 为 `None` 时绝不伪造。
   复验：`scholaragent/metrics.py`、`scholaragent/artifacts.py`；
   工作台 Auto 模式运行后的指标面板。
4. **失败可见与安全回退**：案例中 ReAct 模式 15 步耗尽未产出回答被完整保留；
   路由策略缺失/损坏时打印原因并回退规则路由。
   复验：`docs/case-study.md` 第 3 节 react 行；`tests/test_cost_aware_routing.py`。
5. **来源感知与产物审计**：按任务意图和输入来源选择声明式工作流；论文阅读工具
   输出稳定的页码/段落锚点，运行结果和证据包可离线检查 `run_id`、唯一终态事件
   与证据校验状态。
   复验：`scholaragent/workflows/manifests.json`、`scholaragent/evidence.py`、
   `scholaragent/audit.py`；`python -m pytest tests/test_workflow_contracts.py tests/test_artifact_audit.py -q`。

## 目录结构

```
ScholarAgent/
├── main.py                  入口:交互模式 / 单次任务 / --plan / --team / --auto / --demo
├── webapp.py                本地工作台服务(纯标准库 http.server,127.0.0.1:8765)
├── web/                     工作台前端(原生 JS:任务提交、轨迹、指标、产物查看)
├── scholaragent/            核心包(自底向上分层)
│   ├── config.py            配置层:从 .env 读取 API Key 等
│   ├── llm.py               模型层:唯一碰大模型 API 的地方(含测试用假模型)
│   ├── tool.py              工具层:Tool 基类 + ToolRegistry 登记处
│   ├── workspace.py         运行级工作区:论文/笔记/记忆/证据路径
│   ├── events.py            运行事件与取消上下文
│   ├── runtime.py           CLI/Web/评测共用的统一组装与 RunResult
│   ├── workflow.py          静态工作流清单与来源格式路由
│   ├── evidence.py          来源锚点、主张—证据关系与证据账本
│   ├── audit.py             运行结果与证据包的确定性审计
│   ├── ocr.py               Tesseract + PDF 栅格化 OCR 适配器(可选)
│   ├── experiments.py       实验清单与不可覆盖证据包
│   ├── agent.py             智能体层:思考→行动→观察 的 ReAct 循环
│   ├── memory.py            记忆层:会话记忆(按轮裁剪)+
│   │                        长期记忆(JSONL 持久化 + 手写 BM25 检索)
│   ├── planner.py           规划层:计划→执行→反思→汇总(M3)
│   ├── team.py              多智能体:检索员→精读员→写作员流水线(M4)
│   ├── routing.py           路由层:特征提取 + 规则路由 + 成本感知路由(--auto)
│   ├── router_training.py   离线岭回归奖励模型拟合(训练不调用模型)
│   ├── routing_evaluation.py 路由实验汇总:质量加权与混淆矩阵
│   ├── evaluate.py          评测层:任务集加载、关键词打分、报告生成(M6)
│   ├── metrics.py           执行级指标:耗时/LLM 调用/工具调用/token 采集
│   ├── artifacts.py         产物收集:论文、笔记、记忆按运行归档
│   ├── case_study.py        固定案例运行与可公开证据包写出
│   └── tools/               内置工具:arXiv 搜索、论文下载/分段阅读、
│                            研究笔记、记忆存取、计算器、时钟
├── evals/                   评测任务集(tasks.jsonl / router_tasks.jsonl)
│                            与评测入口:run_eval / run_case_study
│                            / evaluate_rule_router / calibrate_router
│                            / train_router / run_routing_eval
├── archive/                 归档:QLoRA 微调脚手架(未训练)与参赛材料
├── tests/                   离线测试,不联网不花钱(核心循环、文献工具、
│                            记忆、规划、多智能体、路由、发布闸门等)
├── data/                    运行时生成:下载的论文 PDF、研究笔记、长期记忆
│                            (已被 .gitignore 忽略,不入库)
├── requirements.txt         依赖清单
└── .env.example             配置模板(复制成 .env 后填 Key)
```

## 里程碑

| 阶段 | 内容 | 状态 |
|------|------|------|
| M0 | 核心框架:模型层 + 工具层 + ReAct 循环 + 测试 | ✅ 已完成 |
| M1 | 文献工具集:arXiv 搜索、PDF 下载与分段阅读、研究笔记 | ✅ 已完成 |
| M2 | 记忆系统:多轮对话记忆 + 长期记忆(BM25 检索,预留向量接口) | ✅ 已完成 |
| M3 | 规划能力:复杂任务分解、计划—执行—反思(`--plan`) | ✅ 已完成 |
| M4 | 多智能体:检索员 / 精读员 / 写作员分工协作(`--team`) | ✅ 已完成 |
| M5 | 本地模型:无 Key 自动使用本机 Ollama,全流程已实测;QLoRA 微调脚手架已归档(未实际训练,见 archive/finetune/) | ✅ 基础完成 |
| M6 | 评测:任务集与三模式对比实验(12 道真实任务已跑) | ✅ 已完成 |

> 硬件说明:RTX 4060 8G 可跑 7B 模型的 int4 量化推理,
> 以及 1.5B~3B 模型的 QLoRA 微调,M5 阶段够用。

