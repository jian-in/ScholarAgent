"""从已保存的校准观测与独立人工评分表训练成本感知路由策略。"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scholaragent.router_training import fit_policy
from scholaragent import config


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip() and not line.lstrip().startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--scores", required=True, help="独立人工评分 JSONL，以 run_id 对应观测")
    parser.add_argument("--output", default=os.path.join("data", "router", "policy.json"))
    parser.add_argument("--lambda-time", type=float, default=config.ROUTER_LAMBDA_TIME)
    parser.add_argument("--lambda-calls", type=float, default=config.ROUTER_LAMBDA_CALLS)
    parser.add_argument("--lambda-token", type=float, default=config.ROUTER_LAMBDA_TOKEN)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trained-at", default=None, help="ISO 时间；测试可固定以验证确定性")
    args = parser.parse_args()
    if args.ridge <= 0:
        parser.error("--ridge 必须大于 0")

    scores = {row["run_id"]: row for row in load_jsonl(args.scores)}
    policy = fit_policy(
        load_jsonl(args.observations), scores=scores,
        weights={
            "lambda_time": args.lambda_time,
            "lambda_calls": args.lambda_calls,
            "lambda_token": args.lambda_token,
        },
        ridge=args.ridge,
        seed=args.seed,
        trained_at=args.trained_at or datetime.now(timezone.utc).isoformat(),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"已写入策略: {args.output}")


if __name__ == "__main__":
    main()
