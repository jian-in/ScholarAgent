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
    """截断长文本,给步骤间传递的"结果摘要"设预算。"""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…(已截断)"


class Planner:
    def __init__(self, llm, agent, verbose=True, on_progress=None, should_stop=None):
        self.llm = llm      # 用于计划/反思/汇总的模型(chat() 接口)
        self.agent = agent  # 用于执行单步的内层 Agent(ReAct + 工具)
        self.verbose = verbose
        self.on_progress = on_progress
        self.should_stop = should_stop
        # 内层 Agent 与规划层共用同一进度/停止通道
        if on_progress is not None and getattr(agent, "on_progress", None) is None:
            agent.on_progress = on_progress
        if should_stop is not None and getattr(agent, "should_stop", None) is None:
            agent.should_stop = should_stop

    def run(self, task: str) -> str:
        """执行一个复杂任务:计划 → 逐步执行(带反思重试)→ 汇总。"""
        from .agent import CANCELLED_ANSWER

        if self._stop_requested():
            self._log("[取消] 用户已请求停止")
            return CANCELLED_ANSWER

        steps = self._make_plan(task)
        self._log("[计划] " + " -> ".join(steps))

        results = []
        for idx, step in enumerate(steps, 1):
            if self._stop_requested():
                self._log("[取消] 用户已请求停止,中止后续计划步骤")
                return CANCELLED_ANSWER
            self._log(f"[执行 {idx}/{len(steps)}] {step}")
            result = self._execute_step(task, steps, results, idx, advice="")
            if result == CANCELLED_ANSWER:
                return CANCELLED_ANSWER

            ok, advice = self._reflect(step, result)
            if not ok:
                if self._stop_requested():
                    self._log("[取消] 用户已请求停止,跳过重试")
                    return CANCELLED_ANSWER
                self._log(f"[反思] 第 {idx} 步未达标,带着建议重试:{advice}")
                result = self._execute_step(task, steps, results, idx, advice)
                if result == CANCELLED_ANSWER:
                    return CANCELLED_ANSWER
            results.append(result)

        if len(steps) == 1:
            return results[0]  # 单步任务,执行结果就是答案,不必再总结
        if self._stop_requested():
            self._log("[取消] 用户已请求停止,跳过汇总")
            return CANCELLED_ANSWER
        return self._synthesize(task, steps, results)

    # ―― 内部方法:计划 / 执行 / 反思 / 汇总 ――――――――――――――――

    def _make_plan(self, task: str) -> list:
        reply = self.llm.chat([
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ])
        steps = parse_plan(reply["content"])
        if not steps:
            # 计划解析失败不致命:退化成"整个任务当一步"——
            # 执行方式与纯 ReAct 相同,只是仍会多做一次反思质检
            self._log("[计划] 解析失败,退化为单步执行")
            return [task]
        return steps

    def _execute_step(self, task, steps, results, idx, advice) -> str:
        context_lines = [f"总任务:{task}", "完整计划:"]
        context_lines += [f"  {i}. {s}" for i, s in enumerate(steps, 1)]
        if results:
            context_lines.append("已完成步骤的结果摘要:")
            context_lines += [f"  第{i}步:{_clip(r)}"
                              for i, r in enumerate(results, 1)]
        context_lines.append(f"现在请只执行第 {idx} 步:{steps[idx - 1]}")
        if advice:
            context_lines.append(f"(上次执行未达标,改进建议:{advice})")
        return self.agent.run("\n".join(context_lines))

    def _reflect(self, step: str, result: str):
        reply = self.llm.chat([
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"步骤目标:{step}\n执行结果:{_clip(result, 1500)}"},
        ])
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

    def _synthesize(self, task, steps, results) -> str:
        lines = [f"任务:{task}", "各步骤执行结果:"]
        lines += [f"第{i}步({s}):\n{_clip(r, 1200)}"
                  for i, (s, r) in enumerate(zip(steps, results), 1)]
        reply = self.llm.chat([
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ])
        return reply["content"] or ""

    def _stop_requested(self) -> bool:
        if not self.should_stop:
            return False
        try:
            return bool(self.should_stop())
        except Exception:
            return False

    def _log(self, text: str):
        if self.verbose:
            print(text)
        if self.on_progress:
            try:
                self.on_progress(text)
            except Exception:
                pass
