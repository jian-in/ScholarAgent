"""把一个或多个 runs.jsonl 汇总成执行指标报告(Markdown)。

    python evals/summarize_runs.py --runs evals/results/research-full-v1/runs.jsonl
    python evals/summarize_runs.py --runs a.jsonl --runs b.jsonl --output report.md \\
        --price-in 0.30 --price-out 1.20

只读原始运行记录，不修改它们；报告不含质量评分。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholaragent.execution_report import render_markdown  # noqa: E402


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", required=True, help="runs.jsonl，可重复")
    parser.add_argument("--output", help="写入新文件；省略则打印到标准输出")
    parser.add_argument("--title", default="实验执行报告")
    parser.add_argument("--price-in", type=float, help="输入单价，美元/百万 token（仅用于估算）")
    parser.add_argument("--price-out", type=float, help="输出单价，美元/百万 token（仅用于估算）")
    args = parser.parse_args(argv)
    if (args.price_in is None) != (args.price_out is None):
        parser.error("--price-in 与 --price-out 需要同时提供")

    text = render_markdown(load_rows(args.runs), args.title, args.price_in, args.price_out)
    if args.output:
        output = Path(args.output)
        with output.open("x", encoding="utf-8") as handle:  # 不覆盖已有报告
            handle.write(text)
        print(output)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
