"""规划层(M3):计划 → 执行 → 反思。

ReAct 循环擅长"走一步看一步",但复杂任务(比如"调研某方向近三年
进展并写综述")容易走着走着迷路。规划层的思路是先谋后动:

    1. 计划   把任务拆成有序步骤(模型输出 JSON 数组)
    2. 执行   每一步交给内层 Agent 用 ReAct + 工具完成,
              并能看到前面步骤的结果摘要
    3. 反思   每步完成后让模型当"质检员"打分,不合格就带着
              改进意见重试一次 —— 这是 Agent 自我纠错的第二层
              (第一层是工具报错回传,见 tool.py)
    4. 汇总   所有步骤完成后,综合成最终回答

和 ReAct 的分工:简单任务直接 ReAct(快),复杂任务走规划层(稳)。
两者共用同一个模型层和工具层 —— 分层设计的红利在这里兑现。
"""

import json

from . import config
from .events import RunContext
from .gap_survey import (
    FAST_GAP_MAX_STEPS,
    FAST_GAP_HANDOFF_MAX_CHARS,
    FAST_GAP_TOOL_NAMES,
    GAP_HANDOFF_MAX_CHARS,
    build_gap_survey_plan,
    is_fast_gap_survey_task,
    is_gap_survey_task,
    missing_synthesis_sections,
    synthesis_instruction,
)

PLAN_SYSTEM_PROMPT = (
    "你是任务规划师。把用户任务拆解成有序的执行步骤。"
    "只输出一个 JSON 数组,每个元素是一句话描述的一个步骤,"
    "步骤可以依赖前面步骤的结果,最多 6 步;任务简单就只写 1 步。"
    '示例:["检索 LLM Agent 相关论文", "下载并阅读最相关的一篇", "总结要点并保存笔记"]'
)

REFLECT_SYSTEM_PROMPT = (
    "你是执行质量检查员。判断下面这一步的执行结果是否真正达成了该步骤的目标。"
    '只输出 JSON:达成输出 {"ok": true};'
    '未达成输出 {"ok": false, "advice": "一句话改进建议"}。'
    "注意:结果不完美但可用也算达成,不要吹毛求疵。"
)

SYNTHESIZE_SYSTEM_PROMPT = (
    "你是总结者。根据任务目标和各步骤的执行结果,写出条理清晰的最终回答。"
    "信息要注明出处(论文标题/arXiv 编号),各步骤没查到的内容不要编造。"
    "涉及论文时,不能只写宏观趋势；对每篇代表论文至少交代研究问题、核心方法、"
    "实验或评价结果、局限性和证据位置。"
)


def parse_plan(text: str):
    """从模型输出里提取步骤列表;解析失败返回 None(调用方自行兜底)。

    模型不一定听话地只输出 JSON,常见的是前后夹杂解释文字,
    所以先截取第一个 [ 到最后一个 ] 之间的部分再解析。
    """
    if not text:
        return None
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    steps = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    return steps[:6] or None  # 最多 6 步,防模型过度拆解


def _clip(text: str, limit: int = 400) -> str:
    """截断长文本,给步骤间传递的"结果摘要"设预算。

    保头 + 保尾、砍中间(与 team._clip 同一套 60/30 预算):
    结果的背景/方法多在开头,而结论、局限和"已存笔记"等收尾信息
    常在结尾,只保开头会把最关键的尾部整段砍掉。
    """
    text = str(text)
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.3)
    return text[:head] + "\n…(中间已截断)…\n" + text[-tail:]


class Planner:
    def __init__(self, llm, agent, verbose=True, on_progress=None,
                 should_stop=None, summary_llm=None):
        self.llm = llm      # 用于计划和外部调研决策的模型(chat() 接口)
        self.summary_llm = summary_llm or llm  # 双模型时由本地模型负责内部判断
        self.agent = agent  # 用于执行单步的内层 Agent(ReAct + 工具)
        self.verbose = verbose
        self.on_progress = on_progress
        self.should_stop = should_stop
        # 内层 Agent 与规划层共用同一进度/停止通道
        if on_progress is not None and getattr(agent, "on_progress", None) is None:
            agent.on_progress = on_progress
        if should_stop is not None and getattr(agent, "should_stop", None) is None:
            agent.should_stop = should_stop
        self.last_metrics = None

    def run(self, task: str, context: RunContext = None) -> str:
        """执行一个复杂任务:计划 → 逐步执行(带反思重试)→ 汇总。"""
        from .agent import CANCELLED_ANSWER

        owns_context = context is None
        context = context or RunContext(mode="plan", should_stop=self.should_stop)
        self._active_context = context
        self._context_external = not owns_context
        try:
            if self._stop_requested():
                self._log("[取消] 用户已请求停止")
                return CANCELLED_ANSWER

            steps = self._make_plan(task, context)
            self._log("[计划] " + " -> ".join(steps))
            fast_gap = is_fast_gap_survey_task(task)
            if fast_gap:
                self._log("[计划] 快速摘要模式：跳过每步反思，保留最终结构审查")

            results = []
            for idx, step in enumerate(steps, 1):
                if self._stop_requested():
                    self._log("[取消] 用户已请求停止,中止后续计划步骤")
                    return CANCELLED_ANSWER
                self._log(f"[执行 {idx}/{len(steps)}] {step}")
                result = self._execute_step(task, steps, results, idx, advice="",
                                            context=context)
                if result == CANCELLED_ANSWER:
                    return CANCELLED_ANSWER

                if not fast_gap:
                    ok, advice = self._reflect(step, result, context)
                    if not ok:
                        if self._stop_requested():
                            self._log("[取消] 用户已请求停止,跳过重试")
                            return CANCELLED_ANSWER
                        self._log(f"[反思] 第 {idx} 步未达标,带着建议重试:{advice}")
                        result = self._execute_step(task, steps, results, idx, advice,
                                                    context=context)
                        if result == CANCELLED_ANSWER:
                            return CANCELLED_ANSWER
                results.append(result)

            if len(steps) == 1:
                return results[0]  # 单步任务,执行结果就是答案,不必再总结
            if self._stop_requested():
                self._log("[取消] 用户已请求停止,跳过汇总")
                return CANCELLED_ANSWER
            return self._synthesize(task, steps, results, context)
        finally:
            self.last_metrics = (
                context.metrics.finish() if owns_context else context.metrics.snapshot()
            )
            self._active_context = None
            self._context_external = False

    # ―― 内部方法:计划 / 执行 / 反思 / 汇总 ――――――――――――――――

    def _make_plan(self, task: str, context: RunContext = None) -> list:
        if is_gap_survey_task(task):
            return build_gap_survey_plan(task)
        reply = self._chat(context, self.llm, [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ], step="plan")
        steps = parse_plan(reply["content"])
        if not steps:
            # 计划解析失败不致命:退化成"整个任务当一步"——
            # 执行方式与纯 ReAct 相同,只是仍会多做一次反思质检
            self._log("[计划] 解析失败,退化为单步执行")
            return [task]
        return steps

    def _execute_step(self, task, steps, results, idx, advice,
                      context: RunContext = None) -> str:
        context_lines = [f"总任务:{task}", "完整计划:"]
        context_lines += [f"  {i}. {s}" for i, s in enumerate(steps, 1)]
        if results:
            context_lines.append("已完成步骤的结果摘要:")
            context_lines += [f"  第{i}步:{_clip(r)}"
                              for i, r in enumerate(results, 1)]
        context_lines.append(f"现在请只执行第 {idx} 步:{steps[idx - 1]}")
        if advice:
            context_lines.append(f"(上次执行未达标,改进建议:{advice})")
        prompt = "\n".join(context_lines)
        if not is_fast_gap_survey_task(task) or not hasattr(self.agent, "max_steps"):
            if context is None:
                return self.agent.run(prompt)
            return context.invoke(self.agent, prompt)

        original_max_steps = self.agent.max_steps
        original_tools = getattr(self.agent, "tools", None)
        self.agent.max_steps = min(original_max_steps, FAST_GAP_MAX_STEPS)
        if hasattr(original_tools, "subset"):
            self.agent.tools = original_tools.subset(list(FAST_GAP_TOOL_NAMES))
        try:
            if context is None:
                return self.agent.run(prompt)
            return context.invoke(self.agent, prompt)
        finally:
            self.agent.max_steps = original_max_steps
            self.agent.tools = original_tools

    def _reflect(self, step: str, result: str, context: RunContext = None):
        reply = self._chat(context, self.summary_llm, [
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"步骤目标:{step}\n执行结果:{_clip(result, 1500)}"},
        ], step="reflect")
        text = reply["content"] or ""
        # 和 parse_plan 同一个道理:模型常把 JSON 包进 ```围栏或解释文字里,
        # 先截取第一个 { 到最后一个 } 再解析,否则反思环节会被静默禁用
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return True, ""  # 完全不含 JSON,放行不阻塞主流程
        try:
            verdict = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return True, ""
        if not isinstance(verdict, dict) or "ok" not in verdict:
            # 没有明确的 ok 字段就不采信"未达标",避免带着空建议白白重试
            return True, ""
        return bool(verdict["ok"]), str(verdict.get("advice", ""))

    def _synthesize(self, task, steps, results, context: RunContext = None) -> str:
        lines = [f"任务:{task}", "各步骤执行结果:"]
        if is_fast_gap_survey_task(task):
            result_limit = FAST_GAP_HANDOFF_MAX_CHARS
        else:
            # 普通任务的汇总材料预算可配置:默认 4000 字符/步。
            # 曾用 1200 导致最终回答只能基于被砍剩的碎片来写,
            # 是"输出看起来不完整"的主要来源;加大后单次汇总成本仍可控
            result_limit = (
                GAP_HANDOFF_MAX_CHARS if is_gap_survey_task(task)
                else config.PLAN_SYNTHESIS_STEP_CHARS)
        lines += [f"第{i}步({s}):\n{_clip(r, result_limit)}"
                  for i, (s, r) in enumerate(zip(steps, results), 1)]
        system_prompt = SYNTHESIZE_SYSTEM_PROMPT
        extra_instruction = synthesis_instruction(task)
        if extra_instruction:
            system_prompt += extra_instruction
        reply = self._chat(context, self.summary_llm, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(lines)},
        ], step="synthesize")
        answer = (reply["content"] or "").strip()
        if not is_gap_survey_task(task):
            return answer

        missing = missing_synthesis_sections(answer)
        if not missing:
            return answer
        self._log("[汇总审查] 结构不完整，要求模型补齐: " + "、".join(missing))
        repair_reply = self._chat(context, self.summary_llm, [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n这是一次结构审查修订：上一版汇总缺少必要方向或论文核心字段。"
                      "请基于已有步骤证据完整重写，不要新增材料中没有的事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原任务:{task}\n上一版汇总:\n{_clip(answer, GAP_HANDOFF_MAX_CHARS)}\n"
                    f"结构审查缺少: {', '.join(missing)}\n"
                    "请按四个方向、论文核心卡片、横向对比和剩余缺口的格式重写。"
                ),
            },
        ], step="synthesize_repair")
        repaired = (repair_reply["content"] or "").strip()
        if repaired:
            remaining = missing_synthesis_sections(repaired)
            if remaining:
                return (
                    repaired
                    + "\n\n> 结构审查提示：仍缺少 "
                    + "、".join(remaining)
                    + "；相关方向暂不能视为完成。"
                )
            return repaired
        return answer + "\n\n> 结构审查提示：本轮未能补齐 " + "、".join(missing) + "。"

    @staticmethod
    def _chat(context, llm, messages, step=None):
        if context is None:
            return llm.chat(messages)
        return context.chat(llm, messages, step=step)

    def _stop_requested(self) -> bool:
        if getattr(self, "_active_context", None) is not None:
            return self._active_context.is_cancelled()
        if not self.should_stop:
            return False
        try:
            return bool(self.should_stop())
        except Exception:
            return False

    def _log(self, text: str):
        if self.verbose and not getattr(self, "_context_external", False):
            print(text)
        if self.on_progress:
            try:
                self.on_progress(text)
            except Exception:
                pass
