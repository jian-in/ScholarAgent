"""配置层:统一从环境变量 / .env 文件读取配置。

为什么用环境变量:API Key 属于机密,绝不能写进代码提交到 git。
.env 文件已被 .gitignore 忽略,只存在于你自己的电脑上。
"""

import os

from dotenv import load_dotenv

# 把项目根目录 .env 文件里的配置加载进环境变量(文件不存在也不会报错)。
# dotenv 要求 .env 必须是 UTF-8 编码;若被编辑器另存成了 GBK/ANSI,
# 这里给出人话提示,而不是让一大段 UnicodeDecodeError 砸在初学者脸上。
try:
    load_dotenv()
except UnicodeDecodeError:
    raise SystemExit(
        "读取 .env 失败:文件不是 UTF-8 编码。\n"
        "请用记事本或 VS Code 打开 .env,另存为 UTF-8 编码后重试。"
    )

# 任何 OpenAI 兼容接口都可以:DeepSeek / 通义千问 / 智谱 / 本地 Ollama
# 具体怎么换,见项目根目录的 .env.example
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 网页工作台的云端模型档案。它和当前 LLM_* 选择分开，便于在本地模型与
# 云端模型之间即时切换；没有配置时，网页只展示本机 Ollama 模型。
CLOUD_LLM_BASE_URL = os.getenv("CLOUD_LLM_BASE_URL", "")
CLOUD_LLM_API_KEY = os.getenv("CLOUD_LLM_API_KEY", "")
CLOUD_LLM_MODEL = os.getenv("CLOUD_LLM_MODEL", "")


def _positive_int(name: str, default: int) -> int:
    """读取正整数配置;手误写成空值或非数字时回落到稳妥默认值。"""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _nonnegative_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


# 网页后台任务的协作式软超时。它不是强制的系统限制；设为更大的正整数
# 可给长论文任务更多时间，任务仍可由用户主动取消。
JOB_SOFT_TIMEOUT_SECONDS = _positive_int(
    "SCHOLARAGENT_JOB_SOFT_TIMEOUT_SECONDS", 900)

# 工作台默认尝试双模型分工：云端调研 + 本地总结。缺少任一端时，
# Web 层会明确降级为单模型并在状态中说明原因。
MODEL_ROUTING_MODE = os.getenv("SCHOLARAGENT_MODEL_ROUTING", "split").strip().lower()
if MODEL_ROUTING_MODE not in {"single", "split"}:
    MODEL_ROUTING_MODE = "split"


# 长论文仍按块读取,但会话与角色交接需要有足够空间保留前文。
CONVERSATION_MAX_CHARS = _positive_int("CONVERSATION_MAX_CHARS", 60000)
TEAM_HANDOFF_MAX_CHARS = _positive_int("TEAM_HANDOFF_MAX_CHARS", 12000)
PAPER_READER_CHUNK_CHARS = _positive_int("PAPER_READER_CHUNK_CHARS", 6000)
PAPER_READER_MAX_STEPS = _positive_int("PAPER_READER_MAX_STEPS", 48)

# 可选扫描版 PDF OCR。默认会从 PATH、项目所在磁盘的 Tesseract 目录和
# Windows 常见安装目录发现命令；没有依赖时保持原有“如实降级”行为。
OCR_LANGUAGE = os.getenv("SCHOLARAGENT_OCR_LANGUAGE", "chi_sim+eng")
OCR_DPI = _positive_int("SCHOLARAGENT_OCR_DPI", 200)
OCR_TIMEOUT = _positive_int("SCHOLARAGENT_OCR_TIMEOUT", 120)
OCR_PSM = _nonnegative_int("SCHOLARAGENT_OCR_PSM", 3)

# 上下文压缩:Agent 循环每一步都会把完整历史重发给模型,旧工具结果
# (尤其 read_paper 的论文原文)反复重发是 prompt token 的最大浪费。
# 最近 RECENT 条工具结果保留原文,更早的压到 OLD_CHARS 字符(保头保尾);
# OLD_CHARS 设为 0 可完全关闭压缩。
AGENT_CONTEXT_RECENT_OBSERVATIONS = _positive_int(
    "AGENT_CONTEXT_RECENT_OBSERVATIONS", 2)
AGENT_CONTEXT_OLD_OBSERVATION_CHARS = _nonnegative_int(
    "AGENT_CONTEXT_OLD_OBSERVATION_CHARS", 600)

# Plan 模式写最终汇总时,每个步骤结果提供的最大字符数(保头保尾)。
# 太小会导致汇总"看不到"步骤里真正的结论,输出显得残缺。
PLAN_SYNTHESIS_STEP_CHARS = _positive_int("PLAN_SYNTHESIS_STEP_CHARS", 4000)


def _nonnegative_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


ROUTER_POLICY_PATH = os.getenv("ROUTER_POLICY_PATH", "data/router/policy.json")
ROUTER_LAMBDA_TIME = _nonnegative_float("ROUTER_LAMBDA_TIME", 0.15)
ROUTER_LAMBDA_CALLS = _nonnegative_float("ROUTER_LAMBDA_CALLS", 0.10)
ROUTER_LAMBDA_TOKEN = _nonnegative_float("ROUTER_LAMBDA_TOKEN", 0.0)

# 项目根目录(config.py 的上上级),以及数据目录:
# 下载的论文 PDF、研究笔记都存在 data/ 下,已被 .gitignore 忽略
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
