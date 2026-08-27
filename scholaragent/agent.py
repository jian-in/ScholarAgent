"""智能体层:思考→行动→观察 的核心循环(ReAct 模式)。

一次 run() 的生命周期:

    用户任务 → [模型思考] → 需要用工具吗?
        需要 → 执行工具 → 把结果塞回对话 → 回到 [模型思考]
        不需要 → 模型给出的文字就是最终答案,循环结束

max_steps 是保险丝:防止模型陷入"无限调工具"的死循环,
把 API 费用烧光。这是所有 Agent 框架都有的标配防护。
"""

from . import config
from .events import RunContext
from .llm import assistant_message
from .tool import STOP_RETRY_PREFIX, ToolRegistry, ToolResult, adapt_tool_result

DEFAULT_SYSTEM_PROMPT = (
    "你是一个严谨的智能助手。遇到需要计算、查询等无法凭空回答的问题时,"
    "优先调用合适的工具,拿到结果后再作答;不要编造工具没有给出的信息。"
)

# 协作式取消:在下一步边界抛出,让工作台能干净收尾
CANCELLED_ANSWER = "(任务已取消)"

# 旧工具结果被压缩后打上的标记(判断"已压缩过"就靠它,保证幂等)
COMPRESSED_MARK = "较早的工具结果已压缩"

# 云端调研模型完成工具调用后，交给本地模型的固定交接指令。
# 这条边界放在 Agent 循环里，而不是只写在系统提示词里，才能确保
# 总结调用确实不再拿到外部工具清单。
LOCAL_SUMMARY_INSTRUCTION = (
    "外部调研阶段已经结束，现在进入内部总结阶段。"
    "请只根据上方已经获得的工具证据整理本步骤摘要，保留必要的出处和不确定性；"
    "不要调用工具，不要补充材料中没有的事实。"
)


class JobCancelled(Exception):
    """用户请求取消当前任务(协作式,不强制杀线程)。"""


class Agent:
    def __init__(self, llm, tools, system_prompt=DEFAULT_SYSTEM_PROMPT,
                 max_steps=10, verbose=True,
                 conversation=None, long_memory=None, auto_recall=False,
                 tool_call_limits=None, required_tool_completions=None,
                 min_final_chars=0, metrics_mode="react",
                 on_progress=None, should_stop=None, summary_llm=None):
        self.llm = llm            # 只要求有 chat() 方法,真假模型皆可
        # 可选的内部总结模型。None 或与主模型相同表示保持单模型行为。
        self.summary_llm = summary_llm
        self.tools = tools        # ToolRegistry 实例
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.verbose = verbose
        # 可选进度回调:CLI 继续 print,本地工作台用来推步骤日志
        self.on_progress = on_progress
        # 可选停止探测:返回 True 时在下一步边界退出
        self.should_stop = should_stop
        # M2 记忆(都是可选的,不传就保持 M0 的"每次全新对话"行为):
        self.conversation = conversation  # ConversationMemory:多轮对话
        self.long_memory = long_memory    # MemoryStore:长期记忆
        self.auto_recall = auto_recall    # 任务开始前自动检索长期记忆(RAG)
        # 限制单轮内某个工具的调用次数,防止模型用近义参数反复撞同一服务。
        self.tool_call_limits = dict(tool_call_limits or {})
        # 某些长流程不能相信模型口头宣称完成，必须由工具状态确认。
        self.required_tool_completions = tuple(required_tool_completions or ())
        self.min_final_chars = max(0, int(min_final_chars))
        self.metrics_mode = metrics_mode
        self.last_metrics = None
        self._active_context = None

    def run(self, task: str, context: RunContext = None) -> str:
        """执行一个任务,返回最终回答。"""
        owns_context = context is None
        context = context or RunContext(
            mode=self.metrics_mode,
            should_stop=self.should_stop,
        )
        self._active_context = context
        self._context_external = not owns_context
        self.tools.start_run()
        metrics = context.metrics
        # 有会话记忆就接着上次聊,没有就开新对话。
        # 注意 list(...) 复制:必须在副本上追加消息,任务被中断(如 Ctrl+C)时
        # 才不会把"只有工具调用、没有工具结果"的半截轮次漏进记忆 ——
        # 那种残缺历史发给 API 会永久报错,整个会话就废了
        if self.conversation:
            messages = list(self.conversation.load(self.system_prompt))
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

        # 自动回忆(RAG 的"检索"半步):把长期记忆里相关的内容连同出处
        # 附在任务后,并明确告知模型这些内容仅供参考
        original_task = task
        if self.auto_recall and self.long_memory:
            hits = self.long_memory.search(task, top_k=3)
            if hits:
                recalled = "\n".join(
                    f"- {h.get('text', '')}"
                    + (f"(出处:{h['source']})" if h.get("source") else "")
                    for h in hits)
                self._log(f"[记忆] 自动检索到 {len(hits)} 条相关长期记忆")
                task = (f"{task}\n\n(以下是从长期记忆自动检索到的相关内容,"
                        f"仅供参考,以工具实际查到的信息为准:\n{recalled})")

        user_message = {"role": "user", "content": task}
        messages.append(user_message)

        answer = f"(已达到最大步数 {self.max_steps},任务中止。可以换个问法或调大 max_steps)"
        disabled_tools = set()
        tool_call_counts = {}
        remembered_arguments = {}
        had_tool_evidence = False
        summary_handed_off = False
        for step in range(1, self.max_steps + 1):
            if self._stop_requested():
                self._log("[取消] 用户已请求停止,在下一步边界退出")
                answer = CANCELLED_ANSWER
                break
            # ① 思考:把完整对话历史 + 工具清单交给模型。
            # 发送前先压缩较早的工具结果:它们每一步都被整段重发,
            # 是 prompt token 的最大浪费(尤其 read_paper 的论文原文)
            self._compress_history(messages)
            reply = context.chat(
                self.llm,
                messages,
                tools=self.tools.schemas(exclude=disabled_tools),
                step=step,
            )
            messages.append(assistant_message(reply))

            # ② 判断:模型不再要求调工具,说明它认为任务完成了
            if not reply["tool_calls"]:
                answer = (reply["content"] or "").strip()
                pending = self.tools.pending_completions(
                    self.required_tool_completions)
                if pending:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"任务还未完成：工具 {', '.join(pending)} 尚未确认全文读完。"
                            "请忽略刚才的完成判断，继续按工具记录的游标阅读；"
                            "只有工具明确返回“全文读完”后才能总结。"
                        ),
                    })
                    continue

                if (
                    had_tool_evidence
                    and not summary_handed_off
                    and self.summary_llm is not None
                    and self.summary_llm is not self.llm
                ):
                    # 这是一次明确的模型交接：主模型负责外部调研，
                    # 总结模型只看到已产生的证据，且拿不到任何工具 schema。
                    summary_handed_off = True
                    self._log("[模型] 外部调研完成，切换本地模型总结")
                    messages.append({
                        "role": "user",
                        "content": LOCAL_SUMMARY_INSTRUCTION,
                    })
                    summary_reply = context.chat(
                        self.summary_llm,
                        messages,
                        tools=None,
                        step="summary",
                    )
                    messages.append(assistant_message(summary_reply))
                    summarized = (summary_reply.get("content") or "").strip()
                    if summarized:
                        answer = summarized

                if not answer:
                    messages.append({
                        "role": "user",
                        "content": (
                            "上一条回复为空。若任务尚未完成，请继续调用工具；"
                            "若已完成，请给出完整的文字结论。"
                        ),
                    })
                    continue
                if len(answer) < self.min_final_chars:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"最终回答只有 {len(answer)} 个字符，精读结论尚不完整。"
                            "请直接写完研究问题、方法、关键结论和局限性，"
                            "不要只说将要总结。"
                        ),
                    })
                    continue
                self._log(f"[第 {step} 步] 最终回答:{answer}")
                break

            # ③ 行动 + 观察:逐个执行模型点名的工具,结果回填进对话
            had_tool_evidence = True
            if reply["content"]:
                self._log(f"[第 {step} 步] 模型想法:{reply['content']}")
            for tc in reply["tool_calls"]:
                if self._stop_requested():
                    self._log("[取消] 用户已请求停止,跳过剩余工具调用")
                    answer = CANCELLED_ANSWER
                    break
                self._log(f"[第 {step} 步] 调用工具 {tc['name']}({tc['arguments']})")
                tool_name = tc["name"]
                limit = self.tool_call_limits.get(tool_name)
                used = tool_call_counts.get(tool_name, 0)
                if tool_name in disabled_tools:
                    tool_result = ToolResult(
                        # 保留旧前缀只是模型可见的兼容文案；是否停用由
                        # ``stop_retry`` 字段决定，不能从文案反推控制流。
                        text=(f"{STOP_RETRY_PREFIX} 工具 {tool_name} 在本轮已停用。"
                              "请根据已有信息作答并说明资料缺口。"),
                        success=False,
                        stop_retry=True,
                        diagnostic={"kind": "tool_disabled"},
                    )
                    result = tool_result.text
                elif limit is not None and used >= limit:
                    disabled_tools.add(tool_name)
                    tool_result = ToolResult(
                        text=(f"{STOP_RETRY_PREFIX} 工具 {tool_name} 已达到本轮调用上限 {limit} 次。"
                              "请根据已有信息作答并说明资料缺口。"),
                        success=False,
                        stop_retry=True,
                        diagnostic={"kind": "tool_call_limit", "limit": limit},
                    )
                    result = tool_result.text
                elif tc.get("error"):
                    # 参数在模型层就没解析成功,直接把错误当结果回传给模型
                    tool_result = ToolResult(
                        text=tc["error"],
                        success=False,
                        diagnostic={"kind": "invalid_arguments"},
                    )
                    result = tc["error"]
                else:
                    arguments = self.tools.complete_required_arguments(
                        tool_name, tc["arguments"], remembered_arguments)
                    remembered_arguments.update(arguments)
                    tool_call_counts[tool_name] = used + 1
                    # 外部旧代码可能在实例上 monkey-patch ``call``；保留
                    # 这条 seam，同时让正常 ToolRegistry 走结构化结果入口。
                    custom_call = (
                        self.tools.__dict__.get("call")
                        or getattr(type(self.tools), "call", None) is not ToolRegistry.call
                    )
                    if custom_call:
                        if context is not None:
                            context.emit("tool_started", name=tool_name, arguments=arguments)
                            context.metrics.record_tool_call()
                        tool_result = adapt_tool_result(self.tools.call(tool_name, arguments))
                        if context is not None:
                            context.emit(
                                "tool_completed",
                                name=tool_name,
                                success=tool_result.success,
                                stop_retry=tool_result.stop_retry,
                                observation_preview=tool_result.text[:240],
                                diagnostic=tool_result.diagnostic,
                            )
                    else:
                        tool_result = self.tools.call_result(
                            tool_name, arguments, context=context)
                    result = tool_result.text
                    if tool_result.stop_retry:
                        disabled_tools.add(tool_name)
                self._log(f"[第 {step} 步] 工具返回:{result}")
                messages.append({
                    "role": "tool",
                    # tool_call_id 让模型知道这段结果对应它的哪一次调用
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            if answer == CANCELLED_ANSWER:
                break

        # 无论正常结束还是撞上步数上限,都要把对话写回会话记忆。
        # 写回前把注入的回忆内容从 user 消息里剥掉:注入只服务于本轮,
        # 落进历史会导致同样的记忆逐轮重复累积、白白吃掉裁剪预算
        if self.conversation:
            user_message["content"] = original_task
            self.conversation.save(messages)
        self.last_metrics = (
            metrics.finish(self.tools.run_tool_calls)
            if owns_context else metrics.snapshot()
        )
        self._active_context = None
        self._context_external = False
        return answer

    def _compress_history(self, messages: list):
        """把较早的工具结果原地压缩成"保头 + 保尾"的短摘要。

        每一步循环都会把整个历史重发给模型,一条 6000 字符的论文片段
        读完后还要在后续每一步重复计费。最近几条保持原文(模型正要
        用它继续阅读/引用),更早的压到预算内 —— 消息只改内容、
        不删不重排,tool_call_id 配对保持完整,API 不会报错。
        压缩原地生效,因此写回会话记忆的也是压缩后历史。
        """
        limit = config.AGENT_CONTEXT_OLD_OBSERVATION_CHARS
        if limit <= 0:
            return  # 显式关闭
        keep_recent = config.AGENT_CONTEXT_RECENT_OBSERVATIONS
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if keep_recent > 0:
            tool_indices = tool_indices[:-keep_recent]
        compressed, saved_chars = 0, 0
        for i in tool_indices:
            content = messages[i].get("content")
            if (not isinstance(content, str) or len(content) <= limit
                    or COMPRESSED_MARK in content):
                continue  # 短消息、已压缩过的,都不动
            head, tail = int(limit * 0.6), int(limit * 0.3)
            messages[i]["content"] = (
                f"{content[:head]}\n…({COMPRESSED_MARK},"
                "完整原文可在需要时重新调用工具获取,"
                f"关键结论应已存入笔记)…\n{content[-tail:]}")
            compressed += 1
            saved_chars += len(content) - len(messages[i]["content"])
        if compressed:
            self._log(f"[上下文] 已压缩 {compressed} 条较早的工具结果"
                      f"(约省 {saved_chars} 字符)")

    def _stop_requested(self) -> bool:
        if self._active_context is not None:
            return self._active_context.is_cancelled()
        if not self.should_stop:
            return False
        try:
            return bool(self.should_stop())
        except Exception:
            return False

    def _log(self, text: str):
        # 传入结构化运行上下文时，终端输出交给 CLI/Web 适配器；没有上下文
        # 的旧直接调用仍保留原有 verbose 行为。
        if self.verbose and not getattr(self, "_context_external", False):
            print(text)
        if self.on_progress:
            try:
                self.on_progress(text)
            except Exception:
                # 进度回调绝不能拖垮主循环(例如前端已断开)
                pass
