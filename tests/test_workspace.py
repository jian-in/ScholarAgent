"""运行级工作区和并发隔离测试。"""

from concurrent.futures import ThreadPoolExecutor

from scholaragent.artifacts import ArtifactCollector
from scholaragent.memory import MemoryStore
from scholaragent.tools.memory_tools import RememberTool
from scholaragent.tools.notes import ReadNotesTool, SaveNoteTool
from scholaragent.workspace import TemporaryWorkspace


def test_two_workspaces_do_not_share_notes_or_memory(tmp_path):
    first = TemporaryWorkspace(tmp_path / "first")
    second = TemporaryWorkspace(tmp_path / "second")

    def write(workspace, label):
        SaveNoteTool(workspace).run(title=label, content=f"笔记 {label}")
        RememberTool(workspace=workspace).run(f"记忆 {label}", source=label)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, (first, second), ("first", "second")))

    assert "first" in ReadNotesTool(first).run()
    assert "second" not in ReadNotesTool(first).run()
    assert "second" in ReadNotesTool(second).run()
    assert "first" not in ReadNotesTool(second).run()
    assert MemoryStore(workspace=first).search("first")
    assert MemoryStore(workspace=first).search("second") == []
    assert MemoryStore(workspace=second).search("second")


def test_artifact_collectors_are_bound_to_their_workspace(tmp_path):
    first = TemporaryWorkspace(tmp_path / "first")
    second = TemporaryWorkspace(tmp_path / "second")
    a = ArtifactCollector(first)
    b = ArtifactCollector(second)

    a.record("save_note", {"title": "A", "content": "a"}, "笔记已保存")
    b.record("save_note", {"title": "B", "content": "b"}, "笔记已保存")

    assert a.notes[0]["path"] == str(first.notes_path)
    assert b.notes[0]["path"] == str(second.notes_path)
    assert a.notes[0]["path"] != b.notes[0]["path"]
