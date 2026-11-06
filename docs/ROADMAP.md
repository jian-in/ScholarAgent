# 路线图

## 已完成

- ReAct / Plan / Team 三种执行模式，以及按任务自动选择模式的 `--auto`
- arXiv 检索、PDF 下载与分段阅读、扫描版 OCR、笔记与长期记忆
- 来源编号与结论登记、固定论文版本、运行指标（耗时、调用、token、缓存、花费）
- 成本护栏：步数与字符预算、超时与重试、连续失败熔断、软超时
- 本地工作台：模式与模型切换、双模型分工、事件时间线、历史案例回放
- 390+ 项离线测试，CI 覆盖 Windows / Ubuntu × Python 3.10–3.12，ruff 检查
- `pip install` 打包，已发布 `v0.1.0`；工具插件接口
- 英文 README
- 12 道真实任务 × 3 种模式的实验，并修复了实验中发现的问题

## 接下来可能做

- 把工具层暴露成 MCP server，让 Claude、Cursor 等客户端直接调用
- 导出 OpenTelemetry trace，在 Jaeger / Langfuse 里查看运行过程
- 更多文献源：Semantic Scholar、OpenAlex 原生接口
- 更多公开案例：不同学科、不同的失败情况

## 暂停

学习路由的完整实验（校准集打分、训练策略、多策略对比）原本是按论文标准规划的，工作量大，
对项目本身的能力没有影响，2026-11 起暂停。相关脚本仍然可以运行。
在做完之前，项目不声称学习路由比规则路由更好。

## 欢迎贡献

1. 新增一个基础工具（如单位换算、DOI 解析），附离线测试
2. 为 `evals/tasks.jsonl` 补充新的调研任务
3. 打磨本地工作台：任务状态、错误提示、产物预览

修改流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)，维护范围见 [MAINTAINERS.md](../MAINTAINERS.md)。
