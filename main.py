"""ScholarAgent 入口。

用法(在项目根目录下):
    python main.py                      交互模式(.env 配了 API Key 用云端模型;
                                        没配但装了 Ollama 时零配置用本地模型)
    python main.py "帮我算 (3+5)*12"     单次执行一个任务(ReAct 模式)
    python main.py --plan "复杂任务"     规划模式:计划→执行→反思→汇总(M3)
    python main.py --team "调研主题"     团队模式:检索员→精读员→写作员(M4)
    python main.py --auto "调研主题"     自动选择 ReAct / Plan / Team(M7)
    python main.py --demo               离线演示模式,不需要 API Key
"""

import sys

# 当输出被重定向到文件/管道时(如 python main.py > log.txt、部分 IDE 运行器),
# Windows 默认按 GBK 编码写出,模型回复里的 emoji、✅ 等 GBK 表示不了的字符
# 会抛 UnicodeEncodeError,这里统一切到 UTF-8 规避。
# (直接在终端打印没这个问题:Python 走的是 Windows 的 Unicode 控制台接口)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from scholaragent import config
from scholaragent.agent import Agent
from scholaragent.llm import LLMClient, ScriptedLLM
from scholaragent.memory import ConversationMemory, MemoryStore
from scholaragent.planner import Planner
from scholaragent.routing import AdaptiveRunner, CostAwareRouter
from scholaragent.team import ResearchTeam
from scholaragent.tool import ToolRegistry
from scholaragent.tools import BUILTIN_TOOLS

def detect_ollama(prefer: str = None):
    """探测本机的 Ollama 服务,返回一个"能对话且支持工具调用"的模型名。

    M5 的"零配置本地模型":没配 API Key 但装了 Ollama 的机器,
    直接用本地模型跑,不花钱、不联网、数据不出本机。

    不能盲选列表第一个:机器上常同时拉着 embedding 专用模型(nomic-embed
    等,根本不会聊天)或不支持工具调用的模型 —— 按 Ollama 返回的
    capabilities 元数据过滤(老版本没有该字段时从宽放行),这也是
    "用服务的元数据做能力探测"的一个小教学点。
    """
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        candidates = []
        for m in resp.json().get("models", []):
            name = m.get("name")
            if not name:
                continue
            caps = m.get("capabilities")
            if caps is not None and ("completion" not in caps
                                     or "tools" not in caps):
                continue  # 不会对话或不支持工具调用,跳过
            if any(x in name.lower() for x in ("embed", "rerank")):
                continue  # 名字兜底过滤,防老版本漏掉 capabilities
            candidates.append(name)
        if not candidates:
            return None
        # .env 里指定过 LLM_MODEL 且本机恰好有,就尊重用户的选择
        if prefer in candidates:
            return prefer
        return candidates[0]
    except Exception:
        return None  # 服务没开/没装,都不算错误,回落到其他方式


# 离线演示的"剧本":假模型先调计算器,再给最终回答,
# 让你不花一分钱就能看到 Agent 循环的完整流程。
DEMO_SCRIPT = [
    {
        "content": "这个乘法我需要用计算器算一下。",
        "tool_calls": [
            {"id": "call_demo_1", "name": "calculator",
             "arguments": {"expression": "(3+5)*12"}},
        ],
    },
    {"content": "计算完成:(3+5)*12 = 96。", "tool_calls": []},
]


# 文献调研场景的系统提示词:告诉模型有哪些工具、按什么流程用。
# 提示词也是"代码"—— 它直接决定 Agent 的行为质量,值得反复打磨。
RESEARCH_SYSTEM_PROMPT = (
    "你是一个严谨的科研文献调研助手。建议的工作流程:"
    "先用 arxiv_search 检索相关论文;需要细读时用 download_paper 下载,"
    "再用 read_paper 分段阅读(结尾会提示下一页,可多次调用翻页);"
    "重要发现随手用 save_note 记录;任务开始时可用 read_notes 回顾之前的笔记。"
    "值得跨对话记住的结论用 remember 存入长期记忆,想不起来的事用 recall 检索。"
    "回答时必须注明信息来自哪篇论文(标题 + arXiv 编号);"
    "论文里没有的内容不要编造;遇到计算用 calculator,问时间用 current_time。"
)


def build_agent(demo: bool, artifacts=None) -> Agent:
    # 让 remember/recall 工具和 Agent 的自动回忆共用同一个 MemoryStore 实例:
    # 一处写入、处处立即可见,不依赖文件同步的兜底机制
    store = MemoryStore()
    from scholaragent.tools.memory_tools import RecallTool, RememberTool
    tools = [t for t in BUILTIN_TOOLS if t.name not in ("remember", "recall")]
    registry = ToolRegistry(
        tools + [RememberTool(store), RecallTool(store)],
        artifacts=artifacts,
    )

    llm = ScriptedLLM(DEMO_SCRIPT) if demo else LLMClient()
    # 文献任务一轮要检索+下载+翻好几页,步数上限给宽一些。
    # M2:交互模式共用一份会话记忆(多轮对话)和长期记忆(自动回忆开启)
    return Agent(llm, registry, system_prompt=RESEARCH_SYSTEM_PROMPT, max_steps=15,
                 conversation=ConversationMemory(), long_memory=store,
                 auto_recall=True, tool_call_limits={"arxiv_search": 3})


def build_runners(agent: Agent, on_progress=None, should_stop=None,
                  artifacts=None) -> dict:
    """组装三种既有执行器，供 CLI 与自动路由共用。"""
    if on_progress is not None:
        agent.on_progress = on_progress
    if should_stop is not None:
        agent.should_stop = should_stop
    if artifacts is not None and getattr(agent, "tools", None) is not None:
        agent.tools.artifacts = artifacts
    worker = Agent(agent.llm, agent.tools,
                   system_prompt=RESEARCH_SYSTEM_PROMPT, max_steps=15,
                   tool_call_limits={"arxiv_search": 3}, metrics_mode="plan",
                   on_progress=on_progress, should_stop=should_stop)
    return {
        "react": agent,
        "plan": Planner(agent.llm, worker, on_progress=on_progress,
                        should_stop=should_stop),
        "team": ResearchTeam(agent.llm, agent.tools, on_progress=on_progress,
                             should_stop=should_stop),
    }


CLI_MODES = ("react", "plan", "team", "auto")
MODE_HELP = (
    "交互命令:\n"
    "  /react  [任务]   切换到 ReAct(默认),可附带本轮任务\n"
    "  /plan   [任务]   切换到规划模式\n"
    "  /team   [任务]   切换到团队模式\n"
    "  /auto   [任务]   切换到自动路由\n"
    "  /mode            查看当前模式\n"
    "  /help            显示本帮助\n"
    "  q / quit / exit  退出\n"
    "直接输入任务会用当前模式执行;无需退出重开进程。"
)


def parse_cli_flags(argv):
    """解析命令行标志,返回 (demo, mode, task_args)。"""
    args = list(argv)
    demo = "--demo" in args
    plan_mode = "--plan" in args
    team_mode = "--team" in args
    auto_mode = "--auto" in args
    task_args = [a for a in args if a not in ("--demo", "--plan", "--team", "--auto")]

    if sum((plan_mode, team_mode, auto_mode)) > 1:
        raise ValueError("--plan、--team 和 --auto 不能同时使用,请只选择一种模式")
    if demo and (plan_mode or team_mode or auto_mode):
        raise ValueError(
            "--plan/--team/--auto 需要真实模型,离线演示(--demo)不支持;"
            "配置 .env 的 API Key 或安装 Ollama 后,去掉 --demo 再试。"
        )
    if team_mode:
        mode = "team"
    elif plan_mode:
        mode = "plan"
    elif auto_mode:
        mode = "auto"
    else:
        mode = "react"
    return demo, mode, task_args


def parse_interactive_line(line: str, current_mode: str):
    """解析交互输入。

    返回 dict:
      kind: quit | help | mode | switch | task | empty | error
      mode / task / message 视 kind 而定
    """
    text = (line or "").strip()
    if not text:
        return {"kind": "empty"}
    lower = text.lower()
    if lower in {"q", "quit", "exit"}:
        return {"kind": "quit"}

    if text.startswith("/"):
        parts = text.split(None, 1)
        command = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if command in {"/help", "/?"}:
            return {"kind": "help", "message": MODE_HELP}
        if command == "/mode":
            return {
                "kind": "mode",
                "mode": current_mode,
                "message": f"当前模式: {current_mode}",
            }
        if command in {"/react", "/plan", "/team", "/auto"}:
            mode = command[1:]
            if rest:
                return {"kind": "task", "mode": mode, "task": rest, "switched": True}
            return {
                "kind": "switch",
                "mode": mode,
                "message": f"已切换到 {mode} 模式。直接输入任务即可。",
            }
        return {
            "kind": "error",
            "message": f"未知命令 {command}。输入 /help 查看可用命令。",
        }

    return {"kind": "task", "mode": current_mode, "task": text, "switched": False}


def run_task_with_mode(agent: Agent, mode: str, task: str) -> tuple[str, str]:
    """按模式执行一次任务。

    返回 (answer, executed_mode)。executed_mode 在 auto 下是路由实际选中的模式,
    其它情况与请求模式相同。路由信息会打印到 stdout。
    """
    mode = (mode or "react").lower()
    if mode not in CLI_MODES:
        raise ValueError(f"未知模式:{mode}")

    runners = build_runners(agent)
    if mode == "auto":
        adaptive = AdaptiveRunner(
            CostAwareRouter(config.ROUTER_POLICY_PATH), runners
        )
        answer = adaptive.run(task)
        decision = adaptive.last_decision
        executed = (decision.mode if decision else "react") or "react"
        utility = 0.0
        if decision and isinstance(decision.predicted_utility, dict):
            utility = float(decision.predicted_utility.get(executed, 0.0) or 0.0)
        reason = decision.reason if decision else "无决策"
        print(f"路由: {executed}（{reason}；预测效用 {utility:.2f}）")
        return answer, executed

    runner = runners[mode]
    return runner.run(task), mode


def _print_mode_answer(request_mode: str, executed_mode: str, answer: str) -> None:
    """按模式打印最终回答,避免 ReAct 与 auto→react 重复输出。

    ReAct 在 verbose 下会自己 _log 最终回答;plan/team 只返回字符串。
    auto 若落到 react 同样已打印,不再回显;落到 plan/team 时补打一次。
    """
    if not answer:
        return
    if request_mode == "react" or executed_mode == "react":
        return
    print("\n" + answer)


def interactive_loop(agent: Agent, initial_mode: str = "react"):
    """多轮交互:支持 /plan /team /auto 等模式切换,无需重开进程。"""
    mode = initial_mode if initial_mode in CLI_MODES else "react"
    print(f"已连接模型:{config.LLM_MODEL}")
    print(f"当前模式:{mode}(输入 /help 看命令,q 退出)")
    while True:
        try:
            line = input(f"\n你[{mode}]:").strip()
        except (EOFError, KeyboardInterrupt):
            break

        parsed = parse_interactive_line(line, mode)
        kind = parsed["kind"]
        if kind == "empty":
            continue
        if kind == "quit":
            break
        if kind in {"help", "mode", "switch", "error"}:
            if kind == "switch":
                mode = parsed["mode"]
            print(parsed["message"])
            continue

        # kind == task
        if parsed.get("switched"):
            mode = parsed["mode"]
            print(f"(本轮使用 {mode} 模式)")
        task = parsed["task"]
        try:
            answer, executed = run_task_with_mode(agent, mode, task)
            _print_mode_answer(mode, executed, answer)
        except KeyboardInterrupt:
            print("\n(本轮任务已中断,可以继续提问)")
        except Exception as exc:
            print(f"本轮调用失败:{type(exc).__name__}: {exc}")
            print("请检查网络 / API Key / 账户余额后重试。")


def main():
    try:
        demo, mode, task_args = parse_cli_flags(sys.argv[1:])
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if not demo and not config.LLM_API_KEY:
        # 没配 Key 时的优先级:本机 Ollama > 提示配置 > 离线演示
        local_model = detect_ollama(prefer=config.LLM_MODEL)
        if local_model:
            print(f"未配置 API Key,检测到本机 Ollama,使用本地模型:{local_model}")
            print("(想换模型:在 .env 的 LLM_MODEL 里写模型名即可)\n")
            config.LLM_BASE_URL = "http://localhost:11434/v1"
            config.LLM_API_KEY = "ollama"  # Ollama 不校验 Key,占位即可
            config.LLM_MODEL = local_model
        elif task_args or mode != "react":
            # 用户明确给了任务/模式却没有任何可用模型:直接说清楚,
            # 不能悄悄改跑演示题,答非所问比报错更迷惑人
            print("检测到任务,但没有可用的模型:")
            print("  方式一:配置云端 API——复制 .env.example 为 .env,填入 LLM_API_KEY")
            print("  方式二:本地模型——安装 Ollama 并 ollama pull 一个模型")
            print("或者运行 python main.py --demo 观看不需要模型的离线演示。")
            sys.exit(1)
        else:
            print("未检测到 API Key 和本机 Ollama,自动进入离线演示模式。\n")
            demo = True

    if not demo and not config.LLM_API_KEY.isascii():
        # 中文等非 ASCII 字符没法放进 HTTP 认证头,与其让报错穿透到
        # httpx 内部变成天书,不如在门口就拦下并说人话
        print("LLM_API_KEY 含有非 ASCII 字符(是不是还没把占位文字换成真实 Key?)")
        print("请检查 .env,填入形如 sk-xxxx 的真实 API Key。")
        sys.exit(1)

    agent = build_agent(demo)

    if demo:
        print("=== 离线演示:假模型按剧本走一遍 思考→调用工具→回答 的流程 ===\n")
        agent.run("帮我算 (3+5)*12")
        print("\n演示结束。把 .env.example 复制成 .env 并填入 API Key 后,")
        print("再运行 python main.py 就可以和真模型对话了。")
        return

    if task_args:
        # 命令行直接给了任务,执行一次就退出(兼容 --plan/--team/--auto)
        try:
            answer, executed = run_task_with_mode(agent, mode, " ".join(task_args))
            _print_mode_answer(mode, executed, answer)
        except KeyboardInterrupt:
            print("\n(本轮任务已中断)")
            sys.exit(130)
        except Exception as exc:
            print(f"调用失败:{type(exc).__name__}: {exc}")
            print("请检查网络 / API Key / 账户余额后重试。")
            sys.exit(1)
        return

    # 无任务参数:进入交互循环;若用 --plan/--team/--auto 启动则作为初始模式
    interactive_loop(agent, initial_mode=mode)


if __name__ == "__main__":
    main()
