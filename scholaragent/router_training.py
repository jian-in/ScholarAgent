"""成本感知路由的离线训练，不调用模型也不修改原始实验结果。"""

from collections import defaultdict
from statistics import mean, pstdev

from .routing import FEATURE_NAMES, FEATURE_VERSION, ROUTING_MODES, empty_policy_document


DEFAULT_WEIGHTS = {"lambda_time": 0.15, "lambda_calls": 0.10, "lambda_token": 0.0}


def _normalizer(values):
    center = mean(values)
    scale = pstdev(values)
    return {"mean": center, "std": scale, "effective_std": scale or 1.0}


def _normalize(value, params):
    return (value - params["mean"]) / params["effective_std"]


def _solve(matrix, target):
    """高斯消元求小型岭回归方程，避免为 9 个特征引入重量级依赖。"""
    size = len(target)
    augmented = [list(matrix[row]) + [target[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("岭回归矩阵不可逆")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _inverse(matrix):
    size = len(matrix)
    return [_solve(matrix, [1.0 if row == column else 0.0 for row in range(size)])
            for column in range(size)]


def _fit_ridge(rows, rewards, ridge):
    size = len(FEATURE_NAMES)
    covariance = [[ridge if row == column else 0.0 for column in range(size)] for row in range(size)]
    target = [0.0] * size
    for row, reward in zip(rows, rewards):
        vector = [float(row["features"][name]) for name in FEATURE_NAMES]
        for left in range(size):
            target[left] += vector[left] * reward
            for right in range(size):
                covariance[left][right] += vector[left] * vector[right]
    return _solve(covariance, target), _inverse(covariance)


def _quality(row, scores):
    if row.get("quality") is not None:
        return float(row["quality"])
    score = scores.get(row["run_id"], {})
    required = ("task_completion", "factual_correctness", "citation_validity", "output_completeness")
    if any(name not in score for name in required):
        raise ValueError(f"运行 {row['run_id']} 缺少人工质量评分")
    values = {name: float(score[name]) for name in required}
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise ValueError(f"运行 {row['run_id']} 的质量评分超出 [0, 1]")
    return (0.40 * values["task_completion"] + 0.30 * values["factual_correctness"]
            + 0.20 * values["citation_validity"] + 0.10 * values["output_completeness"])


def fit_policy(observations, scores=None, weights=None, ridge=1.0, seed=0, trained_at=None):
    """从校准集观测拟合每个模式的线性效用模型。"""
    rows = list(observations)
    scores = scores or {}
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if not rows:
        raise ValueError("没有校准观测")
    if any(row.get("split") != "calibration" for row in rows):
        raise ValueError("训练只允许使用 calibration 划分")
    if any(row.get("feature_version") != FEATURE_VERSION for row in rows):
        raise ValueError("观测特征版本不兼容")
    if any(row.get("mode") not in ROUTING_MODES for row in rows):
        raise ValueError("观测中存在未知模式")

    metrics = [row["metrics"] for row in rows]
    seconds = [float(item["seconds"]) for item in metrics]
    calls = [float(item["llm_calls"]) for item in metrics]
    tokens = [
        None if item.get("prompt_tokens") is None or item.get("completion_tokens") is None
        else float(item["prompt_tokens"] + item["completion_tokens"])
        for item in metrics
    ]
    if weights["lambda_token"] and any(value is None for value in tokens):
        raise ValueError("token 缺失，不能在 lambda_token 非零时伪造成本")
    normalization = {"seconds": _normalizer(seconds), "llm_calls": _normalizer(calls)}
    if all(value is not None for value in tokens):
        normalization["tokens"] = _normalizer(tokens)
    else:
        normalization["tokens"] = None

    grouped = defaultdict(list)
    rewards = defaultdict(list)
    for row, metric, token_count in zip(rows, metrics, tokens):
        reward = (_quality(row, scores)
                  - weights["lambda_time"] * _normalize(float(metric["seconds"]), normalization["seconds"])
                  - weights["lambda_calls"] * _normalize(float(metric["llm_calls"]), normalization["llm_calls"]))
        if weights["lambda_token"]:
            reward -= weights["lambda_token"] * _normalize(token_count, normalization["tokens"])
        grouped[row["mode"]].append(row)
        rewards[row["mode"]].append(reward)

    policy = empty_policy_document()
    policy.update({
        "feature_names": list(FEATURE_NAMES),
        "normalization": normalization,
        "weights": weights,
        "trained_at": trained_at,
        "seed": seed,
        "sample_summary": {"runs": len(rows), "per_mode": {}},
    })
    for mode in ROUTING_MODES:
        if not grouped[mode]:
            raise ValueError(f"模式 {mode} 没有校准样本")
        coefficients, covariance_inverse = _fit_ridge(grouped[mode], rewards[mode], float(ridge))
        policy["models"][mode] = {
            "coefficients": coefficients,
            "covariance_inverse": covariance_inverse,
        }
        policy["sample_summary"]["per_mode"][mode] = len(grouped[mode])
    return policy
