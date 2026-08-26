# 更新日志

本文件记录值得使用者注意的变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)；在发布首个正式版之前，
以里程碑（M0–M6）与参赛版本组织条目。

## [Unreleased]

### 新增

- 开源治理配套：Issue / PR 模板、GitHub Actions CI（双系统离线测试）、
  行为准则（CODE_OF_CONDUCT）、维护说明（MAINTAINERS）、路线图（docs/ROADMAP.md）。
- 五分钟演示指南（docs/demo-guide.md）与参赛材料文字稿（docs/contest/）。
- README 增加“与同类工具的差异”对比表与各技术点的复验入口。

## 2026-08-26 · OSCHINA 2026 参赛基线

### 新增

- 成本感知自适应路由（`--auto` 与第四种工作台入口）：版本化任务特征
  `task-features-v1`、可解释规则路由 `rule-router-v1`（固定 36 题标签一致）、
  带安全回退的 `CostAwareRouter` 学习路由与离线训练/校准/留出评测入口。
- 首个可公开复验的真实案例证据包：ReAct 论文（arXiv:2210.03629）三模式
  对比运行与初步评分（`docs/case-study.md`、`evals/case_results/`）。
- 公开发布纪律测试（`tests/test_public_release.py`）：隐私措辞、私有文件
  排除、贡献指南与安全策略闸门。

### 说明

- 29/54 条真实校准观测仅用于成本诊断；完整学习策略实验与正式成本对比结论
  尚未产出，未在本版本中声明。

## 2026-07-26 · 毕业设计里程碑 M0–M6

- **M0 核心框架**：模型层（OpenAI 兼容 + 测试用 ScriptedLLM）、工具层
  （ToolRegistry 白名单）、ReAct 循环、离线测试。
- **M1 文献工具集**：arXiv 搜索（OpenAlex 兜底）、PDF 下载与分段阅读、
  研究笔记。
- **M2 记忆系统**：按轮裁剪的会话记忆 + JSONL 长期记忆与手写 BM25 检索。
- **M3 规划模式**：计划 → 单步执行 → 反思重试 → 汇总（`--plan`）。
- **M4 团队模式**：检索员 → 精读员 → 写作员三角色流水线（`--team`）。
- **M5 本地模型**：无 Key 自动探测本机 Ollama；QLoRA 微调脚手架就绪。
- **M6 评测框架**：任务集、关键词评分、三模式消融对比。
- 本地工作台（`webapp.py` + `start.bat` 一键启动）。
