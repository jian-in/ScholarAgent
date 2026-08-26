"""运行一个固定公开案例，保存三模式的可复验证据包。"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals.calibrate_router import ensure_model  # noqa: E402
from evals.run_eval import build_runner  # noqa: E402
from scholaragent.case_study import load_case, run_case, write_case_bundle  # noqa: E402


def execute_case_study(case_path, output_dir, runner_factory=build_runner,
                       state_dir=None):
    """执行案例文件并写出不可覆盖的运行记录与评分模板。

    state_dir 显式指定私有运行状态的存放位置（下载论文、笔记、记忆），
    以便把可公开的证据包与不可公开的原始状态彻底分开。
    """
    case = load_case(case_path)
    output = Path(output_dir)
    state = Path(state_dir) if state_dir else Path(str(output) + ".state")
    if output.exists():
        raise FileExistsError(f"案例输出目录已存在: {output}")
    state.mkdir(parents=True, exist_ok=False)

    rows = run_case(
        case,
        runner_factory,
        modes=tuple(case["modes"]),
        state_root=state,
    )
    paths = write_case_bundle(rows, output)
    paths["state"] = state
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default=os.path.join("evals", "cases", "react_method.json"),
        help="固定案例定义 JSON",
    )
    parser.add_argument("--output", required=True, help="新的案例证据包目录")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="私有运行状态目录(下载论文/笔记/记忆)；默认在输出目录旁生成 <output>.state",
    )
    args = parser.parse_args(argv)

    ensure_model()
    paths = execute_case_study(args.case, args.output, state_dir=args.state_dir)
    print(f"运行记录: {paths['runs']}")
    print(f"评分模板: {paths['scores']}")
    print(f"隔离状态: {paths['state']}")


if __name__ == "__main__":
    main()
