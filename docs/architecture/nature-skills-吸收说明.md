# nature-skills 吸收说明

## 结论

本项目参考了 [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills)
的公开工作流组织方式，用来强化 ScholarAgent 的科研文献调研证据链。
本次实现是“吸收设计思想并独立实现”，没有把对方仓库整包安装进项目，也没有复制
其源码、图片、模板或技能目录。

## 吸收了什么

| 参考方向 | ScholarAgent 中的落点 |
|---|---|
| 短路由器先判断任务类型，再按来源格式分流 | `scholaragent/workflow.py` 的 `WorkflowRegistry` 与 `detect_source_format` |
| 用静态清单声明 always-load、按需参考、输出和验证器 | `scholaragent/workflows/manifests.json`，只解析声明，不执行清单中的代码 |
| 论文阅读区分来源边界和定位信息 | `scholaragent/evidence.py` 的 `SourceAnchor`、`ClaimEvidence`、`EvidenceLedger` |
| 产物生成后做确定性检查，而不是只相信模型输出 | `scholaragent/audit.py` 的运行结果/证据包审计 |
| 检索、精读、交付、归档分阶段 | `literature-search`、`paper-reading`、`paper-card`、`literature-pipeline` 工作流契约；未实现的完整产物明确标为 `contract` |

运行结果会记录工作流名、来源格式和证据账本；论文分段阅读时，页眉会转成稳定的
`S001`、`S002` 等来源锚点。这样评测、案例证据包和 Web 回放看到的是同一份来源信息。

## 许可证与比赛使用

上游仓库根目录声明为 Apache License 2.0。该许可证通常允许个人或组织复制、修改、
展示、再许可和分发，也没有“不得参赛”或“仅限非商业”的限制。因此，**仅使用其公开
思路并由本项目独立实现，通常可以用于比赛**；但许可证许可不等于自动满足比赛主办方
的原创性、第三方依赖披露或作品归属规则。

如果未来直接复制上游代码、文档或资源，发布/提交前至少要：

1. 随分发物保留 Apache-2.0 许可证文本；
2. 保留原版权、专利、商标和归属声明；
3. 修改过的文件加入醒目的修改说明；
4. 如果对应目录带有 `NOTICE`，一并保留其中要求的声明；
5. 逐项核对内嵌资源和依赖的许可证，不能把上游项目名称或商标写成官方背书。

本次没有直接复制上游代码，因此当前项目不因这次参考而新增上游代码归属义务；为了
学术和比赛透明，仍建议在报名材料或答辩中写明“参考 nature-skills 的声明式科研工作流
与来源审计思想”，并同时遵守比赛自己的作品原创性条款。以上是工程合规判断，不替代
主办方规则或法律意见。

## 复验入口

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_contracts.py tests/test_artifact_audit.py -q
```

直接查看上游许可证：
<https://raw.githubusercontent.com/Yuan1z0825/nature-skills/main/LICENSE>
