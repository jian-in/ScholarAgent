"""运行级工作区：集中管理所有运行时文件路径。

工作区把原来散落在配置、工具和评测脚本里的路径规则收拢到一个小而稳定的
对象里。生产默认使用 ``data/``，每次评测或并发任务可以传入自己的根目录，
从而不需要修改全局配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config


_DEFAULT_POLICY = Path("data") / "router" / "policy.json"


@dataclass(frozen=True)
class Workspace:
    """一个运行级文件工作区。

    ``root`` 对应旧布局中的 ``data/`` 目录。所有路径属性都返回绝对
    ``Path``，方便跨线程、跨当前工作目录使用；创建目录仍由真正写入的
    适配器按需完成。
    """

    root: Path
    policy_path: Path | None = None

    def __init__(self, root: str | Path | None = None,
                 policy_path: str | Path | None = None):
        raw_root = Path(root if root is not None else config.DATA_DIR)
        object.__setattr__(self, "root", raw_root.expanduser().resolve())
        if policy_path is None:
            object.__setattr__(self, "policy_path", None)
        else:
            object.__setattr__(self, "policy_path", Path(policy_path).expanduser().resolve())

    @classmethod
    def from_config(cls) -> "Workspace":
        """从当前配置创建工作区，但不修改配置。"""
        configured = Path(config.ROUTER_POLICY_PATH)
        if configured.is_absolute():
            return cls(config.DATA_DIR, policy_path=configured)
        if configured == _DEFAULT_POLICY:
            return cls(config.DATA_DIR)
        # 自定义相对策略路径沿用旧的“相对项目启动目录”语义；它是配置
        # 指定的外部路径，不把它误拼到每个临时工作区下面。
        return cls(config.DATA_DIR, policy_path=Path.cwd() / configured)

    @classmethod
    def temporary(cls, root: str | Path) -> "TemporaryWorkspace":
        return TemporaryWorkspace(root)

    @property
    def papers_dir(self) -> Path:
        return self.root / "papers"

    @property
    def notes_dir(self) -> Path:
        return self.root / "notes"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def router_dir(self) -> Path:
        return self.root / "router"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def papers_path(self) -> Path:
        return self.papers_dir

    @property
    def notes_path(self) -> Path:
        return self.notes_dir / "research_notes.md"

    @property
    def memory_path(self) -> Path:
        return self.memory_dir / "memories.jsonl"

    @property
    def router_policy_path(self) -> Path:
        return self.policy_path or (self.router_dir / "policy.json")

    def paper_path(self, arxiv_id: str) -> Path:
        """返回论文缓存路径。

        编号格式的安全校验仍由论文工具负责；这里仅负责把已校验的编号
        映射成旧布局兼容的文件名。
        """
        safe_name = str(arxiv_id).replace("/", "_")
        return self.papers_dir / f"{safe_name}.pdf"

    def ensure(self, *names: str) -> None:
        """按名称创建工作区子目录，未知名称直接报错。"""
        directories = {
            "root": self.root,
            "papers": self.papers_dir,
            "notes": self.notes_dir,
            "memory": self.memory_dir,
            "router": self.router_dir,
            "artifacts": self.artifacts_dir,
            "runs": self.runs_dir,
        }
        for name in names:
            try:
                directories[name].mkdir(parents=True, exist_ok=True)
            except KeyError as exc:
                raise ValueError(f"未知工作区目录: {name}") from exc


class LocalWorkspace(Workspace):
    """生产默认的本地文件工作区适配器。"""

    @classmethod
    def from_config(cls) -> "LocalWorkspace":
        base = Workspace.from_config()
        return cls(base.root, policy_path=base.policy_path)


class TemporaryWorkspace(Workspace):
    """测试/评测使用的显式临时文件工作区适配器。"""


def default_workspace() -> LocalWorkspace:
    """返回生产默认工作区；调用者可为每次运行保存独立实例。"""
    return LocalWorkspace.from_config()


def workspace_for(root: str | Path | Workspace | None = None) -> Workspace:
    """把可选路径归一化成工作区对象。"""
    if isinstance(root, Workspace):
        return root
    if root is None:
        return default_workspace()
    return TemporaryWorkspace(root)
