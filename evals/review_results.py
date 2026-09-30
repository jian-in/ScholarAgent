"""Prepare blind human review packets or build a separately scored report offline."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholaragent.human_review import apply_human_review, prepare_blind_review, write_review_report


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "report"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing output is preserved")
    parser.add_argument("--key")
    parser.add_argument("--scores")
    parser.add_argument("--claims")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = load_rows(args.runs)
    if args.action == "prepare":
        output = prepare_blind_review(rows, args.output, args.seed)
    else:
        if not args.key:
            parser.error("report requires --key")
        key = json.loads(Path(args.key).read_text(encoding="utf-8"))
        derived = apply_human_review(rows, key, load_rows(args.scores) if args.scores else (),
                                    load_rows(args.claims) if args.claims else ())
        output = write_review_report(derived, args.output)
    print(output)


if __name__ == "__main__":
    main()
