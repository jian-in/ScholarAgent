# 维护说明

本文件说明项目当前由谁维护、维护范围和响应预期，供贡献者与评审者判断
项目的可持续性。

## 维护者

- Bofei Jian（项目作者，MIT 许可证持有人）

## 维护范围

**接受维护的部分**：

- `scholaragent/` 核心包与 `main.py` / `webapp.py` 入口的行为问题
- `tests/` 离线测试与 CI
- `evals/` 评测脚本的正确性与可复现性
- 文档与演示路径（README、docs/）

**明确的边界**：

- 模型本身的能力（幻觉、知识时效）不是本项目的缺陷范围；与模型行为相关的
  Issue 请附完整轨迹与复现命令，便于区分是路由、工具还是模型的问题。
- `data/`、`evals/results/` 是本地运行产物，维护者不为其内容提供支持。
- 云端 API 的配额、计费与网络可达性问题由使用者自理；Ollama 本地模型仅承诺
  “能被自动探测并接入”，不承诺特定模型的效果。

## 响应预期

- 安全漏洞：按 [SECURITY.md](SECURITY.md) 的私密渠道报告，争取 7 天内响应。
- 功能 Issue / PR：以业余时间维护，目标 14 天内给出初步回应；暂无 SLA 承诺。
- 破坏 CI 或发布闸门测试（`tests/test_public_release.py`）的改动优先处理。

## 版本与发布

- 发布前必须：`python -m pytest -q` 全绿、`git diff --check` 通过、
  发布扫描不含密钥/私人信息/运行数据。
- 值得记录的变更写入 [CHANGELOG.md](CHANGELOG.md)。
- 后续计划见 [docs/ROADMAP.md](docs/ROADMAP.md)。
