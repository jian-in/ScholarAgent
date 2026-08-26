# 参与贡献

感谢你愿意改进 ScholarAgent。项目优先接受能够提升可复现性、文献调研质量、
运行稳定性或开发者体验的改动。

## 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

首次运行可先执行不需要 API Key 的离线演示：

```powershell
.\.venv\Scripts\python.exe main.py --demo
```

## 修改流程

1. 从最新主分支创建独立分支。
2. 先补充一个能够复现问题或描述新行为的测试，确认它会失败。
3. 用最小改动让测试通过，再运行完整 `pytest`。
4. 提交前执行 `git diff --check`，并检查没有加入运行数据或本地配置。
5. 发起 Pull Request，说明问题、方案、验证命令和结果。

## Pull Request 检查清单

- [ ] 改动范围单一，提交信息能够说明目的。
- [ ] 新行为有测试覆盖，完整测试通过。
- [ ] README、命令示例和实现保持一致。
- [ ] 未提交 `.env`、API Key、论文下载文件、评测运行产物或私人记录。
- [ ] 新增依赖有明确用途，并确认许可证兼容。

## 许可证

提交贡献即表示你同意按项目的 [MIT License](LICENSE) 发布对应代码和文档。
