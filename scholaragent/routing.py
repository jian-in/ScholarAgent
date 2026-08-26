"""成本感知路由的稳定数据接口。

具体路由实现按阶段逐步加入；本模块先固定决策与策略文件的公共格式，避免
训练、评测和推理对特征版本产生静默分歧。
"""

import re
import json
from dataclasses import asdict, dataclass
from typing import Callable, Mapping


FEATURE_VERSION = "task-features-v1"
POLICY_FORMAT_VERSION = "cost-aware-router-v1"
ROUTING_MODES = ("react", "plan", "team")
FEATURE_NAMES = (
    "task_length",
    "action_goal_count",
    "requires_literature_search",
    "requires_full_text",
    "requires_multi_paper_comparison",
    "requires_review",
    "requires_memory",
    "has_complex_constraints",
    "bias",
)


@dataclass(frozen=True)
class RoutingDecision:
    mode: str
    predicted_utility: Mapping[str, float]
    features: Mapping[str, float]
    reason: str
    policy_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def empty_policy_document() -> dict:
    """返回策略 JSON 的版本化骨架，供训练脚本填充真实实验数据。"""
    return {
        "format_version": POLICY_FORMAT_VERSION,
        "feature_version": FEATURE_VERSION,
        "modes": list(ROUTING_MODES),
        "normalization": {},
        "weights": {},
        "models": {},
        "trained_at": None,
        "sample_summary": {},
    }


class TaskFeatureExtractor:
    """从任务文本提取确定性、可解释且无需模型调用的特征。"""

    _ACTION_PATTERNS = {
        "search": r"检索|搜索|查找|文献|论文|search|literature",
        "read": r"下载|阅读|精读|通读|全文|download|read|full text",
        "compare": r"比较|对比|多篇|多篇论文|compare|comparison|multiple papers",
        "review": r"综述|总结|脉络|开放问题|review|survey|summari[sz]e",
        "memory": r"保存|笔记|记忆|存储|save|note|memory",
    }
    _FULL_TEXT_PATTERN = r"下载|通读|全文|download|full text"
    _NEGATION = r"(?:不需要|无需|不用|不要|不必|no need to|without)"
    _CONSTRAINT_PATTERNS = (
        r"先.{0,30}(?:再|然后|之后)",
        r"(?:近|过去)\s*\d+\s*年",
        r"\d+\s*(?:篇|篇论文|篇文献|papers?)",
        r"分别|依次|限定|至少|不超过|between\s+\d{4}\s+and\s+\d{4}",
    )

    def _mentioned(self, task: str, pattern: str) -> bool:
        match = re.search(pattern, task, flags=re.IGNORECASE)
        if not match:
            return False
        before = task[max(0, match.start() - 16):match.start()]
        return not re.search(self._NEGATION + r"\s*$", before, re.IGNORECASE)

    def extract(self, task: str) -> dict[str, float]:
        task = task or ""
        actions = {
            name: self._mentioned(task, pattern)
            for name, pattern in self._ACTION_PATTERNS.items()
        }
        full_text = self._mentioned(task, self._FULL_TEXT_PATTERN)
        features = {
            "task_length": min(len(task) / 1000.0, 1.0),
            "action_goal_count": float(sum(actions.values())),
            "requires_literature_search": float(actions["search"]),
            "requires_full_text": float(full_text),
            "requires_multi_paper_comparison": float(actions["compare"]),
            "requires_review": float(actions["review"]),
            "requires_memory": float(actions["memory"]),
            "has_complex_constraints": float(any(
                re.search(pattern, task, re.IGNORECASE)
                for pattern in self._CONSTRAINT_PATTERNS
            )),
            "bias": 1.0,
        }
        return {name: features[name] for name in FEATURE_NAMES}


class RuleRouter:
    """透明的冷启动路由基线，阈值集中在本类而非入口代码。"""

    policy_version = "rule-router-v1"

    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor or TaskFeatureExtractor()

    def route(self, task: str) -> RoutingDecision:
        features = self.feature_extractor.extract(task)
        full_reading = features["requires_full_text"]
        comparison = features["requires_multi_paper_comparison"]
        review = features["requires_review"]

        if full_reading and (comparison or review):
            mode = "team"
            reason = "全文精读与" + ("多论文比较" if comparison else "综述任务")
        elif features["has_complex_constraints"] or features["action_goal_count"] >= 3:
            mode = "plan"
            reason = "存在明确步骤依赖或多个行动目标"
        else:
            mode = "react"
            reason = "单目标或低复杂度任务"

        utility = {candidate: 0.0 for candidate in ROUTING_MODES}
        utility[mode] = 1.0
        return RoutingDecision(
            mode=mode,
            predicted_utility=utility,
            features=features,
            reason=reason,
            policy_version=self.policy_version,
        )


class AdaptiveRunner:
    """只负责路由和委派，执行器仍遵守既有 ``run(task) -> str`` 契约。"""

    def __init__(self, router, runners: Mapping[str, object]):
        missing = [mode for mode in ROUTING_MODES if mode not in runners]
        if missing:
            raise ValueError(f"缺少执行器: {', '.join(missing)}")
        self.router = router
        self.runners = dict(runners)
        self.last_decision = None

    def run(self, task: str) -> str:
        decision = self.router.route(task)
        if decision.mode not in self.runners:
            raise ValueError(f"路由器返回了未知模式:{decision.mode}")
        self.last_decision = decision
        runner = self.runners[decision.mode]
        run: Callable[[str], str] = getattr(runner, "run", runner)
        return run(task)


class CostAwareRouter:
    """加载离线训练的线性奖励策略；不可用时安全回退规则路由。"""

    def __init__(self, policy_path=None, feature_extractor=None, alpha=0.0):
        self.feature_extractor = feature_extractor or TaskFeatureExtractor()
        self.alpha = max(0.0, float(alpha))
        self.fallback = RuleRouter(self.feature_extractor)
        self.policy = None
        self.last_warning = None
        if policy_path:
            self.load_policy(policy_path)
        else:
            self.last_warning = "未提供策略文件"

    def load_policy(self, policy_path: str) -> bool:
        try:
            with open(policy_path, encoding="utf-8") as handle:
                policy = json.load(handle)
            self._validate_policy(policy)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.policy = None
            self.last_warning = f"策略不可用: {exc}"
            return False
        self.policy = policy
        self.last_warning = None
        return True

    @staticmethod
    def _validate_policy(policy: Mapping) -> None:
        if policy.get("format_version") != POLICY_FORMAT_VERSION:
            raise ValueError("策略格式版本不兼容")
        if policy.get("feature_version") != FEATURE_VERSION:
            raise ValueError("特征版本不兼容")
        if policy.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("特征名称或顺序不兼容")
        models = policy.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("策略缺少模型参数")
        for mode in ROUTING_MODES:
            coefficients = models.get(mode, {}).get("coefficients")
            if (not isinstance(coefficients, list)
                    or len(coefficients) != len(FEATURE_NAMES)):
                raise ValueError(f"模式 {mode} 的系数不完整")

    def route(self, task: str) -> RoutingDecision:
        if self.policy is None:
            decision = self.fallback.route(task)
            warning = self.last_warning or "策略不可用"
            return RoutingDecision(
                mode=decision.mode,
                predicted_utility=decision.predicted_utility,
                features=decision.features,
                reason=f"{decision.reason}；已回退规则路由（{warning}）",
                policy_version=decision.policy_version,
            )

        features = self.feature_extractor.extract(task)
        vector = [features[name] for name in FEATURE_NAMES]
        utility = {}
        for mode in ROUTING_MODES:
            model = self.policy["models"][mode]
            value = sum(a * b for a, b in zip(model["coefficients"], vector))
            utility[mode] = value + self.alpha * self._uncertainty(
                vector, model.get("covariance_inverse")
            )
        mode = max(ROUTING_MODES, key=lambda candidate: (utility[candidate], -ROUTING_MODES.index(candidate)))
        return RoutingDecision(
            mode=mode,
            predicted_utility=utility,
            features=features,
            reason="离线校准的成本感知线性奖励模型预测效用最高",
            policy_version=self.policy["format_version"],
        )

    @staticmethod
    def _uncertainty(vector, inverse) -> float:
        if not inverse:
            return 0.0
        try:
            quadratic = sum(
                vector[row] * inverse[row][column] * vector[column]
                for row in range(len(vector))
                for column in range(len(vector))
            )
        except (IndexError, TypeError):
            return 0.0
        return max(0.0, quadratic) ** 0.5


class GlobalUtilityRouter(CostAwareRouter):
    """去除任务特征的消融：只按各模式全局平均效用选择。"""

    def route(self, task: str) -> RoutingDecision:
        if self.policy is None:
            return super().route(task)
        features = self.feature_extractor.extract(task)
        bias_index = FEATURE_NAMES.index("bias")
        utility = {
            mode: self.policy["models"][mode]["coefficients"][bias_index]
            for mode in ROUTING_MODES
        }
        mode = max(ROUTING_MODES, key=lambda candidate: (utility[candidate], -ROUTING_MODES.index(candidate)))
        return RoutingDecision(
            mode=mode,
            predicted_utility=utility,
            features=features,
            reason="仅按校准集全局平均效用选择，未使用任务特征",
            policy_version=self.policy["format_version"] + "-global-only",
        )
