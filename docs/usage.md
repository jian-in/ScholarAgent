# 使用指南

安装、运行模式和本地工作台的完整说明。项目概览见 [README](../README.md)。

## 五分钟启动（源码与安装两条路径）

源码运行适合开发和复验：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py --demo
.venv\Scripts\python webapp.py --open
```

安装候选版本适合验证入口和 wheel：

```powershell
.venv\Scripts\python -m pip install ".[dev]"
.venv\Scripts\scholaragent.exe --demo
.venv\Scripts\scholaragent-web.exe --open
```

`--demo` 使用内置脚本模型，不需要 API Key；真实调研仍需在本地 `.env` 配置
OpenAI 兼容接口或运行 Ollama。当前版本为 `v0.1.0`，尚未发布到 PyPI。

## 快速开始

> **五分钟看懂项目**：评审/演示的完整黄金路径（含离线兜底）见
> [`docs/demo-guide.md`](demo-guide.md)；下面是逐条命令版。

以下命令在 **PowerShell**(Windows 自带终端)中执行:

```powershell
# 1. 创建虚拟环境并安装依赖(只需一次)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 2. 先跑离线演示,看懂 Agent 循环(不需要 API Key)
.venv\Scripts\python main.py --demo

# 3. 复制配置模板,然后用记事本打开 .env 填入 API Key(保存时保持 UTF-8 编码)
Copy-Item .env.example .env

# 4. 和真模型对话(装了 Ollama 的机器可跳过第 3 步,会自动用本地模型)
.venv\Scripts\python main.py "现在几点了?顺便帮我算 365*24"

# 5. 复杂任务用规划模式,完整调研用团队模式
.venv\Scripts\python main.py --plan "调研 ReAct 论文的核心思想并记笔记"
.venv\Scripts\python main.py --team "LLM agent 的记忆机制"

# 6. 跑测试 / 跑评测
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python evals/run_eval.py --mode react --limit 4
```

> **为什么用 `.venv\Scripts\python` 而不是"激活虚拟环境"**:免激活写法在
> PowerShell/cmd 里都能直接跑,不受 PowerShell 执行策略限制,电脑上装了
> 多个 Python 也绝不会用错解释器,对初学者最稳。
>
> 如果你更喜欢激活的方式:PowerShell 执行 `.venv\Scripts\Activate.ps1`
> (若提示"禁止运行脚本",先执行一次
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`);
> cmd 执行 `.venv\Scripts\activate.bat`;Git Bash 执行
> `source .venv/Scripts/activate`。注意:**每开一个新终端都要重新激活**,
> 只有第 1 步的创建和安装是一次性的。
> cmd 用户把第 3 步换成 `copy .env.example .env`。

## 本地工作台

本地工作台把现有执行器接到浏览器，不会把 API Key 发送给前端，也不会监听局域网。

**一键启动（推荐）**：在资源管理器中双击项目根目录的 `start.bat`
（或在 PowerShell 中执行 `.\start.ps1`）。脚本会自动：

1. 若无 `.venv` 则创建并安装 `requirements.txt`
2. 若无 `.env` 则从 `.env.example` 复制一份
3. 启动工作台并打开浏览器 `http://127.0.0.1:8765`

也可手动启动：

```powershell
.venv\Scripts\python webapp.py --open
```

页面可选择 ReAct、Plan、Team、Auto；Auto 会显示实际选择的模式、策略版本、理由与耗时。
默认任务分工是“云端检索 · 本地总结”；模型分工和单模型都只影响新任务。后台任务的
900 秒是协作式软超时，不是硬杀限制，可通过 `SCHOLARAGENT_JOB_SOFT_TIMEOUT_SECONDS`
调整。按 `Ctrl+C` 停止本地服务。

### 模型选择与双模型分工

网页工作台顶部提供模型选择器：它会发现本机 Ollama 的聊天模型，也会显示 `.env`
中配置的云端 OpenAI 兼容模型。切换只影响之后创建的新任务，正在运行的任务继续使用
启动时已经绑定的模型。运行结果会展示 LLM 调用次数、工具调用次数以及服务实际返回的
输入/输出/总 token；接口没有返回 token 时显示“Token 未返回”，不做估算。配置说明见
[`docs/architecture/模型切换与token统计.md`](architecture/模型切换与token统计.md)。
工作台默认还提供“双模型分工”：云端模型负责外部论文检索和工具决策，本地 Ollama
负责证据摘要、反思和最终汇总；每次运行的事件时间线会显示实际角色与模型。可在页面
一键切回单模型，详细边界见
[`docs/architecture/模型分工与超时策略.md`](architecture/模型分工与超时策略.md)。
